import itertools
import pathlib
import pickle
import re

import itk
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import scipy.spatial.distance as sdistance
import skimage.transform
import tqdm


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


def map_moving_points(points_xy, param_obj):
    return _map_points(points_xy, param_obj, is_from_moving=True)


def map_fixed_points(points_xy, param_obj):
    return _map_points(points_xy, param_obj, is_from_moving=False)


def _map_points(points_xy, param_obj, is_from_moving=True):
    import scipy.ndimage as ndi

    points = np.asarray(points_xy)
    assert points.shape[1] == 2
    assert points.ndim == 2

    shape = param_obj.GetParameterMap(0).get("Size")[::-1]
    shape = np.array(shape, dtype="int")

    deformation_field = itk.transformix_deformation_field(
        itk.GetImageFromArray(np.zeros(shape, dtype="uint8")), param_obj
    )
    dx, dy = np.moveaxis(deformation_field, 2, 0)

    if is_from_moving:
        inverted_fixed_point = itk.FixedPointInverseDisplacementFieldImageFilter(
            deformation_field,
            NumberOfIterations=10,
            Size=deformation_field.shape[:2][::-1],
        )
        dx, dy = np.moveaxis(inverted_fixed_point, 2, 0)

    my, mx = np.mgrid[: shape[0], : shape[1]].astype("float32")

    mapped_points_xy = np.vstack(
        [
            ndi.map_coordinates(mx + dx, np.fliplr(points).T, mode="nearest"),
            ndi.map_coordinates(my + dy, np.fliplr(points).T, mode="nearest"),
        ]
    ).T
    return mapped_points_xy


def transform_moving_xy(
    coords_xy,
    elastix_tform_paths: list,
    affine_before: skimage.transform.AffineTransform = None,
    affine_after: skimage.transform.AffineTransform = None,
):
    coords_xy = np.array(coords_xy)
    assert coords_xy.ndim == 2
    assert coords_xy.shape[1] == 2

    elastix_parameter = itk.ParameterObject.New()
    for ff in elastix_tform_paths:
        elastix_parameter.AddParameterFile(str(ff))

    if affine_before is not None:
        coords_xy = affine_before(coords_xy)
    deform_points = map_moving_points(coords_xy, elastix_parameter)
    if affine_after is not None:
        deform_points = affine_after(deform_points)
    return deform_points


all_files = pd.read_csv(
    r"/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20241025-deform-registration-stic/00-files-id.csv",
    index_col="LSP ID",
)
files = pd.read_csv(
    r"/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20241025-deform-registration-stic/00-files.csv",
    index_col="LSP ID",
)
he_ids = all_files["HE OMERO ID"].loc[files.index].astype("int").astype("str")

dform_dir = r"/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20241025-deform-registration-stic/reg-param/tform"
affine_dir = r"/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20241025-deform-registration-stic/thumbnail"
roi_dir = r"/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20241025-deform-registration-stic/roi/he"
out_dir = r"/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20241025-deform-registration-stic/roi/transformed-he"

inputs = []
for lspid, heid in he_ids.items():
    d1 = pathlib.Path(dform_dir) / f"{lspid}-tform-elastix-param-0.txt"
    d2 = pathlib.Path(dform_dir) / f"{lspid}-tform-elastix-param-1.txt"
    aa = pathlib.Path(affine_dir) / f"{lspid}-affine-matrix.csv"
    rr = next(pathlib.Path(roi_dir).glob(f"*-{heid}-rois.csv"))

    ds1 = files["ref-downsize-factor-roi"][lspid]
    ds2 = files["moving-downsize-factor-roi"][lspid]
    assert d1.exists(), d1
    assert d2.exists(), d2
    assert aa.exists(), aa
    inputs.append((rr, [d1, d2], aa, ds1, ds2, lspid, heid))


def process_roi_file(
    roi_path: str,
    elastix_tform_paths: list,
    affine_before: skimage.transform.AffineTransform = None,
    affine_after: skimage.transform.AffineTransform = None,
):
    roi = pd.read_csv(roi_path)
    if len(roi) == 0:
        return roi
    roi = roi.apply(add_mpatch, axis=1)

    has_mpatch = roi["mpatch"].notnull()
    verts = [pp.get_verts() for pp in roi["mpatch"][has_mpatch]]
    tformed_verts = transform_moving_xy(
        np.vstack(verts), elastix_tform_paths, affine_before, affine_after
    )
    slice_positions = [0] + list(np.cumsum([len(vv) for vv in verts]))
    tformed_mpatch = [
        mpatches.Polygon(tformed_verts[slice(*ss)], closed=True)
        for ss in itertools.pairwise(slice_positions)
    ]
    roi["transformed_mpatch"] = None
    roi.loc[has_mpatch, "transformed_mpatch"] = tformed_mpatch
    return roi


Affine = skimage.transform.AffineTransform
extra_downsize_before_elastix = 2

out_dir = pathlib.Path(out_dir)
out_dir.mkdir(exist_ok=True, parents=True)
for roi_path, elastix_tform_paths, aa, ds1, ds2, lspid, heid in tqdm.tqdm(inputs[:]):
    affine_before = (
        Affine(scale=(1 / ds2,) * 2)
        + Affine(matrix=np.loadtxt(aa, delimiter=","))
        + Affine(scale=(1 / extra_downsize_before_elastix,) * 2)
    )
    affine_after = Affine(scale=(extra_downsize_before_elastix,) * 2) + Affine(
        scale=(ds1,) * 2
    )
    tformed_roi = process_roi_file(roi_path, elastix_tform_paths, affine_before, affine_after)
    with open(out_dir / f"{lspid}-{heid}-rois-transformed.pkl", "wb") as f:
        pickle.dump(tformed_roi, f)
