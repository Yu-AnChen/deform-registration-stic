import copy
import itertools
import pathlib
import re
import shlex
import subprocess

import dask.array as da
import matplotlib.colors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance as sdistance
import textalloc as ta
import tqdm.contrib


def parse_roi_points(all_points):
    if not isinstance(all_points, str):
        return None
    return np.array(re.findall(r"-?\d+\.?\d+", all_points), dtype=float).reshape(-1, 2)


def ellipse_points_to_patch(
    vertex_1, vertex_2, co_vertex_1, co_vertex_2, patch_kwargs={}
):
    """
    Parameters
    ----------
    vertex_1, vertex_2, co_vertex_1, co_vertex_2: array like, in the form of (x-coordinate, y-coordinate)

    """
    v_and_co_v = np.array([vertex_1, vertex_2, co_vertex_1, co_vertex_2])
    centers = v_and_co_v.mean(axis=0)

    d = sdistance.cdist(v_and_co_v, v_and_co_v, metric="euclidean")
    width = d[0, 1]
    height = d[2, 3]

    vector_2 = v_and_co_v[1] - v_and_co_v[0]
    vector_2 /= np.linalg.norm(vector_2)

    angle = np.degrees(np.arccos([1, 0] @ vector_2))

    ellipse_patch = mpatches.Ellipse(
        centers, width=width, height=height, angle=angle, **patch_kwargs
    )
    return ellipse_patch


def add_mpatch(roi, patch_kwargs={}):
    roi = roi.copy()
    points = parse_roi_points(roi["all_points"])
    roi.loc["parsed_points"] = points

    roi_type = roi["type"]
    if roi_type in ["Point", "Line"]:
        roi_mpatch = mpatches.Polygon(points, closed=False, **patch_kwargs)
    elif roi_type in ["Rectangle", "Polygon", "Polyline"]:
        roi_mpatch = mpatches.Polygon(points, closed=True, **patch_kwargs)
    elif roi_type == "Ellipse":
        roi_mpatch = ellipse_points_to_patch(*points, patch_kwargs=patch_kwargs)
    else:
        roi_mpatch = None

    roi.loc["mpatch"] = roi_mpatch
    return roi


def load_roi_pkl(path):
    import pickle

    with open(path, "rb") as f:
        roi = pickle.load(f)

    if len(roi) == 0:
        return roi

    for cc in ["Name", "Text", "type"]:
        roi[cc] = roi[cc].fillna("").astype("str")

    is_label = roi["type"].apply(lambda x: x == "Rectangle") & roi["Text"].apply(
        lambda x: "LSP1" in x
    )
    is_scimap = roi["Text"].apply(lambda x: "scimap" in x.lower())

    roi["mpatch"] = roi["transformed_mpatch"]

    return roi.loc[~(is_label | is_scimap)].copy()


def img_base_shape(img_path):
    import tifffile

    shape = tifffile.TiffFile(img_path).series[0].levels[0].shape
    assert len(shape) in [2, 3]
    if len(shape) == 3:
        _shape = list(shape)
        _shape.pop(np.argmin(shape))
        shape = tuple(_shape)
    return shape


def pyramid_level_shapes(base_shape, tile_size=1024):
    assert len(base_shape) == 2
    shape = np.ceil(base_shape).astype("int")
    shapes = [shape]
    level = 1
    while np.any(np.greater(shapes[-1], tile_size)):
        shapes.append(np.ceil(np.divide(base_shape, 2**level)).astype(int))
        level += 1
    return shapes


def roi_extent(roi, pad_lower_right=10):
    verts = np.vstack([pp.get_verts() for pp in roi["mpatch"]])
    width_height = np.ceil(verts.max(axis=0)).astype("int") + pad_lower_right
    return width_height[::-1]


def roi_to_svg(roi, out_dir, shape=None, dpi=720, colorset="IF"):
    assert colorset in ["IF", "HE"]
    COLORS = {
        "IF": ["#00bfff", "#ffd700", "#ffffff", "#000000"],
        "HE": ["#f08d44", "#005b7f", "#ffffff", "#000000"],
    }
    facecolor, linecolor, textcolor, backdrop = COLORS[colorset]

    OUT_DIR = pathlib.Path(out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    facecolor = matplotlib.colors.hex2color(facecolor)

    verts = np.vstack([pp.get_verts() for pp in roi["mpatch"]])
    W, H = np.ceil(verts.max(axis=0)).astype("int") + 10

    if shape is not None:
        H, W = np.ceil(shape).astype("int")

    SHAPES = pyramid_level_shapes((H, W))

    DPI = dpi
    PX = 1 / DPI

    for idx, (_H, _W) in enumerate(tqdm.tqdm(SHAPES)):
        fig, ax = plt.subplots(figsize=(_W * PX, _H * PX))
        ax.set_xlim(np.add(-0.5, (0, W)))
        ax.set_ylim(np.add(-0.5, (H, 0)))

        texts = []
        for pp, nn in zip(roi["mpatch"], roi["Text"]):
            size_factor = 1 / 2 ** max((idx - 4), 0)
            LINEWIDTH = 1 * 16**2 / DPI * size_factor
            FONTSIZE = 12 * 16**2 / DPI * size_factor
            padding = 0.2 / 2 ** min(idx, 4)

            pp = copy.copy(pp)
            pp = mpatches.Polygon(pp.get_verts(), closed=True)
            pp.set_linewidth(LINEWIDTH)
            pp.set_edgecolor(facecolor)
            pp.set_facecolor((*facecolor, 0.2))

            ax.add_patch(copy.copy(pp))
            xx, yy = pp.get_verts()[pp.get_verts()[:, 1].argmin()]
            texts.append([xx, yy, nn])

        plt.axis("off")
        ax.set_position([0, 0, 1, 1])

        ta.allocate(
            ax,
            [tt[0] for tt in texts],
            [tt[1] for tt in texts],
            [tt[2] for tt in texts],
            x_scatter=verts[:, 0],
            y_scatter=verts[:, 1],
            textsize=FONTSIZE,
            textcolor=textcolor,
            linecolor=linecolor,
            linewidth=0.75 * LINEWIDTH,
            bbox=dict(
                facecolor=backdrop,
                edgecolor=backdrop,
                alpha=0.5,
                boxstyle=f"square,pad={padding}",
            ),
            fontfamily="Arial",
        )

        plt.savefig(OUT_DIR / f"roi-{idx}.svg", transparent=True)
        plt.close(fig)
    return SHAPES


def dask_array_to_rsvg_arg(arr):
    yy, xx = np.indices(arr.numblocks).reshape(2, -1)
    # FIXME hardcoded tile size
    top, left = yy * 1024, xx * 1024
    height, width = np.array(list(itertools.product(*arr.chunks))).T

    # filename = "{}_{}_{}.{}".format(level, tx, ty, EXT)
    args = []
    for tt, ll, hh, ww, yi, xi in zip(top, left, height, width, yy, xx):
        args.append(
            f"--top=-{tt} --left=-{ll} --page-height={hh} --page-width={ww} --output=LEVEL_{xi}_{yi}.png"
        )
    return args


def pyramid_to_rsvg_arg(pyramid):
    all_args = []
    for idx, level in enumerate(pyramid):
        args = dask_array_to_rsvg_arg(level)
        all_args.append([aa.replace("LEVEL", str(idx)) for aa in args])
    return all_args


def configure_rendered_level_shapes(roi_shape, img_shape):
    """reduce canvas size removing bottom and right regions without ROIs"""
    # in current minerva story, if tile shape isn't the same as the given tile
    # from other layers, this tile will be distorted when rendered - round up to
    # avoid it

    # this actually needs to be done for _all_ the pyramid levels, dorp using
    # roi shape for now
    rshapes = pyramid_level_shapes(roi_shape)
    ishapes = pyramid_level_shapes(img_shape)

    shapes = []
    for ss, ii in zip(rshapes, ishapes):
        ss = 1024 * np.ceil(np.divide(ss, 1024)).astype("int")
        ss = np.clip(ss, None, ii)
        shapes.append(ss)
    print(f"shape: {shapes[0]}; roi: {rshapes[0]}; img: {ishapes[0]}")
    return shapes


def render_roi(
    roi_path: str,
    out_dir: str,
    img_path: str,
    colorset: str = "IF",
    preview: bool = False,
):
    print("Processing", roi_path)

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    DPI = 720

    roi = load_roi_pkl(roi_path)
    if len(roi) == 0:
        print("No valid ROI found in", roi_path, "\n")
        return

    roi_shape = roi_extent(roi)
    img_shape = np.array([1_000_000] * 2, dtype="int")
    if img_path is not None:
        img_shape = img_base_shape(img_path)
    shapes = configure_rendered_level_shapes(roi_shape, img_shape)

    roi_to_svg(roi, out_dir=out_dir, shape=shapes[0], dpi=DPI, colorset=colorset)

    da_pyramid = [da.zeros(ss, chunks=1024) for ss in shapes]
    all_args = pyramid_to_rsvg_arg(da_pyramid)

    import platform

    find_command = "which"
    if platform.system() == "Windows":
        find_command = "where"
    rsvg_exe = (
        subprocess.check_output(
            shlex.split(f"conda run -n svg {find_command} rsvg-convert")
        )
        .decode()
        .strip()
    )
    rsvg_exe = pathlib.Path(rsvg_exe)

    for idx, args_level in enumerate(all_args):
        svg_level = out_dir / f"roi-{idx}.svg"
        if not (svg_level).exists():
            print(svg_level, "does not exist")
            continue

        if preview:
            if idx != (len(all_args) - 3):
                continue
            out_png_name = svg_level.name.replace(".svg", "-preview.png")

            subprocess.run(
                [
                    rsvg_exe,
                    svg_level,
                    "--format=png",
                    f"--output={out_png_name}",
                    f"--dpi-x={DPI}",
                    f"--dpi-y={DPI}",
                ],
                cwd=out_dir,
            )
            print(f"Rendered: {out_dir / out_png_name}")
            break

        _calls = []
        # svg to svg to png
        # rsvg-convert roi-4.svg --format svg | rsvg-convert --format png --output zzzzz.png
        for aa in tqdm.tqdm(args_level, f"Level {idx} / {len(all_args) - 1}"):
            flags, out_png_name = aa.split(" --output=")
            crop_svg = subprocess.Popen(
                [
                    rsvg_exe,
                    svg_level,
                    "--format=svg",
                    f"--dpi-x={DPI}",
                    f"--dpi-y={DPI}",
                    *shlex.split(flags),
                ],
                stdout=subprocess.PIPE,
                cwd=out_dir,
            )
            to_png = subprocess.Popen(
                [rsvg_exe, "--format=png", f"--output={out_png_name}"],
                stdin=crop_svg.stdout,
                cwd=out_dir,
            )
            crop_svg.stdout.close()  # enable write error in crop_svg if to_png dies
            _calls.append(to_png)

        _ = [pp.communicate() for pp in _calls]
    print("Done")
    print()


def run_batch(csv_path, print_args=True, dryrun=False, **kwargs):
    import csv
    import inspect
    import pprint
    import types

    from fire.parser import DefaultParseValue

    func = render_roi

    if print_args:
        _args = [str(vv) for vv in inspect.signature(func).parameters.values()]
        print(f"\nFunction args\n{pprint.pformat(_args, indent=4)}\n")
    _arg_types = inspect.get_annotations(func)
    arg_types = {}
    for k, v in _arg_types.items():
        if isinstance(v, types.UnionType):
            v = v.__args__[0]
        arg_types[k] = v

    with open(csv_path) as f:
        csv_kwargs = [
            {
                kk: arg_types[kk](DefaultParseValue(vv))
                for kk, vv in rr.items()
                if (kk in arg_types) & (vv is not None)
            }
            for rr in csv.DictReader(f)
        ]

    if dryrun:
        for kk in csv_kwargs:
            pprint.pprint({**kwargs, **kk}, sort_dicts=False)
            print()
        return

    for kk in csv_kwargs:
        func(**{**kwargs, **kk})


def main():
    import fire

    fire.Fire({"run": render_roi, "run-batch": run_batch})


if __name__ == "__main__":
    import sys

    sys.exit(main())

# ---------------------------------------------------------------------------- #
#                                  dev section                                 #
# ---------------------------------------------------------------------------- #


# ------------------------- render full image as png ------------------------- #
def _dev():
    DPI = 720
    for idx in tqdm.trange(6):
        # if idx != 3:
        #     continue
        func = subprocess.run
        # func = " ".join
        _ = func(
            [
                "conda",
                "run",
                "-n",
                "svg",
                "rsvg-convert",
                "--dpi-x",
                str(int(DPI)),
                "--dpi-y",
                str(int(DPI)),
                f"roi-{idx}.svg",
                ">",
                f"roi-{idx}.png",
                # "--page-width=1024",
                # "--page-height=1024",
            ]
        )


# ---------------------------- test on actual WSI ---------------------------- #
def _dev():
    import napari
    import numpy as np
    import palom
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = 2**64

    pyramid = [
        np.array(
            Image.open(
                f"/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20241025-deform-registration-stic/minerva-annotation-and-HE/LSP16150/mask/test3/roi-{i}.png"
            )
        )
        for i in range(5)
    ]
    reader = palom.reader.OmePyramidReader(
        "/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20241025-deform-registration-stic/img-data/LSP16150-nearby-he-RD-SS-005.ome.tif"
    )

    v = napari.Viewer()
    v.add_image([np.moveaxis(pp, 0, 2) for pp in reader.pyramid])
    v.add_image(pyramid)
