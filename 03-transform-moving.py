import cv2
import dask.array as da
import dask.diagnostics
import numpy as np
import palom
import skimage.transform
import tifffile
import zarr
from numcodecs import Zstd
from palom.cli import align_he


def _warp_coords_cv2(mx, row_slice, col_slice, out_dtype="float64"):
    assert mx.shape == (3, 3)
    xx, yy = (
        np.arange(*col_slice, dtype="float64"),
        np.arange(*row_slice, dtype="float64"),
    )
    grid = np.reshape(
        np.meshgrid(xx, yy, indexing="xy"),
        (2, 1, -1),
    ).T
    grid = cv2.transform(grid, mx[:2, :]).astype(out_dtype)

    return np.squeeze(grid).T.reshape(2, len(yy), len(xx))[::-1]


def warp_coords_cv2(mx, shape, dtype="float64"):
    return _warp_coords_cv2(mx, (0, shape[0]), (0, shape[1]), out_dtype=dtype)


def _wrap_cv2_large_proper(
    dform, img, mx, cval, sigma=0, module="cv2", block_info=None
):
    assert module in ["cv2", "skimage"]
    assert mx.shape == (3, 3)
    assert dform.ndim == 3

    _, H, W = dform.shape

    dtype = "float64"

    _, rslice, cslice = block_info[0]["array-location"]

    if np.all(mx == 0):
        dform = np.array(dform)
    else:
        mgrid = _warp_coords_cv2(mx, rslice, cslice, out_dtype=dtype)
        # remap functions in opencv convert coordinates into 16-bit integer; for
        # large image/coordinates, slice the appropiate image block and
        # re-position the coordinate origin is required
        dform = np.array(dform) + mgrid

    # add extra pixel for linear interpolation
    rmin, cmin = np.floor(dform.min(axis=(1, 2))).astype("int") - 1
    rmax, cmax = np.ceil(dform.max(axis=(1, 2))).astype("int") + 1

    if np.any(np.asarray([rmax, cmax]) <= 0):
        return np.full((H, W), fill_value=cval, dtype=img.dtype)

    rmin, cmin = np.clip([rmin, cmin], 0, None)
    rmax, cmax = np.clip([rmax, cmax], None, img.shape)

    dform -= np.reshape([rmin, cmin], (2, 1, 1))

    # cast mapping down to 32-bit float for speed and compatibility
    dform = dform.astype("float32")

    crop_img = np.array(img[rmin:rmax, cmin:cmax])

    if 0 in crop_img.shape:
        return np.full((H, W), fill_value=cval, dtype=img.dtype)

    if sigma != 0:
        pad = sigma * 4
        pad_rmin, pad_cmin = np.clip(np.subtract([rmin, cmin], pad), 0, None)
        pad_rmax, pad_cmax = np.clip(np.add([rmax, cmax], pad), None, img.shape)
        _crop_img = np.array(img[pad_rmin:pad_rmax, pad_cmin:pad_cmax])
        border_mode = cv2.BORDER_REPLICATE
        _crop_img = cv2.GaussianBlur(_crop_img, (0, 0), sigma, borderType=border_mode)
        crop_img = _crop_img[
            rmin - pad_rmin : rmin - pad_rmin + crop_img.shape[0],
            cmin - pad_cmin : cmin - pad_cmin + crop_img.shape[1],
        ]

    if 0 in img.shape:
        return np.full((H, W), fill_value=cval, dtype=img.dtype)
    if module == "cv2":
        return cv2.remap(
            crop_img, dform[1], dform[0], cv2.INTER_LINEAR, borderValue=cval
        )
    return skimage.transform.warp(
        crop_img, dform, preserve_range=True, cval=cval
    ).astype(crop_img.dtype)


def run_transform(
    file_path: str,
    out_path: str,
    ref_file_path: str,
    affine_mx_path: str,
    deformation_field_path: str,
    temp_zarr_store_dir: str = None,
    pre_filter_sigma: int = 1,
    pyramid_level: int = 0,
):
    r1 = align_he.get_reader(ref_file_path)(ref_file_path)
    r2 = align_he.get_reader(file_path)(file_path)

    matched_levels = palom.align_multi_res.match_levels(r1, r2)

    LEVEL = pyramid_level
    affine_level1, affine_level2 = matched_levels[4]
    out_level1, out_level2 = matched_levels[LEVEL]

    d_moving = r2.level_downsamples[affine_level2] / r2.level_downsamples[out_level2]
    d_ref = r1.level_downsamples[affine_level1] / r1.level_downsamples[out_level1]

    downscale_dform = 2 * d_ref

    Affine = skimage.transform.AffineTransform

    mx_d = Affine(scale=downscale_dform).params

    mx = np.loadtxt(affine_mx_path, delimiter=",")
    dform = tifffile.imread(deformation_field_path)

    tform = Affine(scale=1 / d_moving) + Affine(matrix=mx) + Affine(scale=d_ref)

    ddx, ddy = (
        (
            # FIXME confirm whether it's the right math!
            (np.linalg.inv(mx[:2, :2]) @ dform.reshape(2, -1)).T @ mx_d[:2, :2]
        )
        .T.reshape(dform.shape)
        .astype("float64")
    )

    padded_shape = r1.pyramid[out_level1].shape[1:3]
    mapping = da.zeros((2, *padded_shape), dtype="float64", chunks=1024)

    _tform = tform + Affine(scale=1 / downscale_dform)
    # add extra pixel for linear interpolation
    _mgrid = skimage.transform.warp_coords(_tform.inverse, np.add(ddy.shape, 1))

    _mgrid[:, : ddy.shape[0], : ddy.shape[1]] += np.array([ddy, ddx])

    gy_gx = da.array(
        [
            mapping.map_blocks(
                _wrap_cv2_large_proper,
                gg,
                mx=np.linalg.inv(Affine(scale=downscale_dform).params),
                cval=0,
                module="skimage",
                dtype="float64",
                drop_axis=0,
            )
            for gg in _mgrid
        ]
    )
    # the chunk size (256, 256, 3) isn't ideal to be loaded with dask; hard-code
    # the reading and axis swap
    _moving = r2.pyramid[out_level2]
    chunks = np.ceil(np.divide(2048, _moving.chunksize[1:3])) * np.array(
        _moving.chunksize[1:3]
    )
    store = None
    if temp_zarr_store_dir is not None:
        store = zarr.TempStore(dir=temp_zarr_store_dir)
    moving = zarr.group(store=store, overwrite=True)
    for idx, channel in enumerate(_moving):
        moving[idx] = zarr.empty(
            channel.shape,
            chunks=chunks.astype("int"),
            dtype=_moving.dtype,
            compressor=Zstd(),
        )
        with dask.diagnostics.ProgressBar():
            channel.to_zarr(moving[idx])

    mosaics = []
    for channel in moving.values():
        sr, sc = np.ceil(np.divide(channel.shape, 1000)).astype("int")
        cval = np.percentile(np.array(channel[::sr, ::sc]), 75).item()
        warped_moving = gy_gx.map_blocks(
            _wrap_cv2_large_proper,
            channel,
            mx=np.zeros((3, 3)),
            cval=cval,
            sigma=pre_filter_sigma,
            module="skimage",
            dtype="uint8",
            drop_axis=0,
        )
        mosaics.append(warped_moving)

    palom.pyramid.write_pyramid(
        mosaics,
        output_path=out_path,
        pixel_size=r1.pixel_size * r1.level_downsamples[out_level1],
        channel_names=list("RGB"),
        downscale_factor=4,
        compression="zlib",
        save_RAM=True,
        tile_size=1024,
    )


def run_batch(csv_path, print_args=True, dryrun=False, **kwargs):
    import csv
    import inspect
    import pprint
    import types

    from fire.parser import DefaultParseValue

    func = run_transform

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

    fire.Fire({"run": run_transform, "run-batch": run_batch})


"""

file_path = r"\\research.files.med.harvard.edu\HITS\lsp-data\cycif-production\110-BRCA-Mutant-Ovarian-Precursors\HE\CD0302.08 (7923).svs"
ref_file_path = r"\\research.files.med.harvard.edu\HITS\lsp-analysis\cycif-production\110-BRCA-Mutant-Ovarian-Precursors\STIC_Batch6_2023\p110_STIC\LSP19420\registration\LSP19420.ome.tif"
affine_mx_path = r"X:\cycif-production\110-BRCA-Mutant-Ovarian-Precursors\HE\registered\thumbnail\LSP19420-affine-matrix.csv"
deformation_field_path = r"X:\cycif-production\110-BRCA-Mutant-Ovarian-Precursors\HE\registered\deformation-field\LSP19420-elastix-deformation-field-xy.ome.tif"
out_path = r"\\research.files.med.harvard.edu\HITS\lsp-data\cycif-production\110-BRCA-Mutant-Ovarian-Precursors\HE\registered\CD0302.08 (7923)-registered-to-LSP19420-test-1.ome.tif"
temp_zarr_store_dir = r"C:\Temp\temp-zarr-store"

run_transform(
    file_path=file_path,
    out_path=out_path,
    ref_file_path=ref_file_path,
    affine_mx_path=affine_mx_path,
    deformation_field_path=deformation_field_path,
    temp_zarr_store_dir=temp_zarr_store_dir,
    pre_filter_sigma=1,
)
"""


if __name__ == "__main__":
    main()
