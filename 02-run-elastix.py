import pathlib

import itk
import napari
import numpy as np
import skimage.exposure
import skimage.metrics
import tifffile


def get_default_crc_params(
    grid_size: float = 80.0,
    sample_region_size: float = 300.0,
    sample_number_of_pixels: int = 4_000,
    number_of_iterations: int = 1_000,
):
    parameter_object = itk.ParameterObject.New()
    # deformation
    p = parameter_object.GetDefaultParameterMap("bspline")

    del p["FinalGridSpacingInPhysicalUnits"]

    p["ASGDParameterEstimationMethod"] = ["DisplacementDistribution"]
    p["FixedImagePyramid"] = ["FixedSmoothingImagePyramid"]
    p["HowToCombineTransforms"] = ["Compose"]
    p["Interpolator"] = ["BSplineInterpolator"]
    p["MovingImagePyramid"] = ["MovingSmoothingImagePyramid"]
    p["Transform"] = ["RecursiveBSplineTransform"]

    # metrics: higher weight on the bending energy panelty to reduce distortion
    p["Metric"] = ["AdvancedMattesMutualInformation", "TransformBendingEnergyPenalty"]
    p["Metric0Weight"] = ["1.0"]
    p["Metric1Weight"] = ["100.0"]

    # these should be pixel-size & image size related
    p["NumberOfResolutions"] = ["4"]
    p["GridSpacingSchedule"] = [f"{2**i}" for i in range(0, 4)[::-1]]
    p["FinalGridSpacingInVoxels"] = [f"{grid_size}", f"{grid_size}"]
    p["NumberOfSamplesForExactGradient"] = [f"{10_000}"]
    p["NumberOfSpatialSamples"] = [f"{sample_number_of_pixels}"]
    # p["NumberOfSpatialSamples"] = [f"{5000 // 2**i}" for i in range(4)[::-1]]
    p["UseRandomSampleRegion"] = ["true"]
    p["SampleRegionSize"] = [f"{sample_region_size}"]
    p["NumberOfHistogramBins"] = ["32"]

    # number if iterations in gradient descent
    p["MaximumNumberOfIterations"] = [f"{number_of_iterations}"]

    # must set to write result image, could be a bug?!
    p["WriteResultImage"] = ["true"]
    # a bug in transformix_jacobian cannot handle using non-default output image
    # format (such as tif, default is nii)
    p["ResultImageFormat"] = ["nii"]

    return p


def run_one_setting(ref_path, moving_path, setting):
    ref = tifffile.imread(ref_path)
    moving = tifffile.imread(moving_path)

    mref = skimage.exposure.match_histograms(-1.0 * ref, moving).round().astype("uint8")
    return _run_one_setting(mref, moving, setting)


def _run_one_setting(ref, moving, setting):
    if setting is None:
        setting = {}
    elastix_parameter = itk.ParameterObject.New()
    elastix_parameter.AddParameterMap(
        elastix_parameter.GetDefaultParameterMap("rigid", 4)
    )
    elastix_parameter.AddParameterMap(get_default_crc_params(**setting))
    warpped_moving, transform_parameter = itk.elastix_registration_method(
        ref,
        moving,
        parameter_object=elastix_parameter,
        log_to_console=False,
    )
    return warpped_moving, transform_parameter, elastix_parameter


def try_conditions(ref_path, moving_path, conditions):
    warpped_moving, transform_parameter, elastix_parameter = None, None, None
    for cc in conditions:
        try:
            print(cc)
            warpped_moving, transform_parameter, elastix_parameter = run_one_setting(
                ref_path,
                moving_path,
                cc,
            )

        except RuntimeError:
            continue
        else:
            # check MI metrics of the warpped with the ref
            nmi = skimage.metrics.normalized_mutual_information
            mi_ori = nmi(tifffile.imread(ref_path), tifffile.imread(moving_path))
            mi_reg = nmi(tifffile.imread(ref_path), warpped_moving)
            print(cc, f"{mi_ori:.5f} vs {mi_reg:.5f}")
            if mi_reg < mi_ori:
                continue
            else:
                break
    return warpped_moving, transform_parameter, elastix_parameter


def write_parameter(param_obj, out_dir, prefix=None):
    assert isinstance(param_obj, itk.ParameterObject)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = prefix or ""
    out_paths = []
    for idx in range(param_obj.GetNumberOfParameterMaps()):
        out_path = out_dir / f"{prefix}elastix-param-{idx}.txt"
        itk.ParameterObject.New().WriteParameterFile(
            param_obj.GetParameterMap(idx), str(out_path)
        )
        out_paths.append(out_path)
    return out_paths


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
            ndi.map_coordinates(mx + dx, np.fliplr(points).T),
            ndi.map_coordinates(my + dy, np.fliplr(points).T),
        ]
    ).T
    return mapped_points_xy


import itertools  # noqa: E402

grid_size = 60
sizes = [320, 400, 500]
n_samples = [5000, 4000, 3000]

conditions = [
    {"sample_region_size": ss, "sample_number_of_pixels": ns, "grid_size": grid_size}
    for ss, ns in itertools.product(sizes, n_samples)
]


case_number = "LSP15327,LSP15331,LSP15335,LSP15339,LSP15343,LSP15347,LSP15355,LSP15359,LSP16136,LSP16138,LSP16141,LSP16144,LSP16147,LSP16150,LSP16153,LSP16156,LSP16159,LSP16162,LSP17722,LSP17726,LSP17730,LSP17734,LSP17758,LSP17762,LSP17766,LSP17786,LSP17790,LSP17802,LSP17818,LSP17822,LSP17850,LSP18301,LSP18305,LSP18309,LSP18313,LSP18317,LSP19412,LSP19416,LSP19420".split(
    ","
)

elastix_config_dir = "/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20241025-deform-registration-stic/reg-param/config"
elastix_tform_dir = "/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20241025-deform-registration-stic/reg-param/tform"


# ------------------ registration at ~20 um - better results ----------------- #
import palom  # noqa: E402

v = napari.Viewer()
for nn in case_number[:]:
    print(f"{nn}")
    ref_path = rf"/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20241025-deform-registration-stic/thumbnail/{nn}-ref.tif"
    moving_path = rf"/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20241025-deform-registration-stic/thumbnail/{nn}-moving.tif"

    ref = palom.img_util.cv2_downscale_local_mean(tifffile.imread(ref_path), 2)
    moving = palom.img_util.cv2_downscale_local_mean(tifffile.imread(moving_path), 2)

    mref = (
        skimage.exposure.match_histograms(-1.0 * ref, moving)
        .round()
        .astype(moving.dtype)
    )

    img, params_tform, params_reg = _run_one_setting(
        np.array(mref),
        np.array(moving),
        {"sample_region_size": 320, "sample_number_of_pixels": 5000, "grid_size": 60},
    )

    # write parameters to disk
    write_parameter(params_reg, elastix_config_dir, f"{nn}-")
    write_parameter(params_tform, elastix_tform_dir, f"{nn}-tform-")

    napari_kwargs = dict(blending="additive", visible=False, name=f"{nn}-{grid_size}")
    v.add_image(
        ref,
        colormap="bop blue",
        visible=False,
        name=f"{nn}",
        contrast_limits=(0, 20000),
    )
    v.add_image(-moving, colormap="bop purple", **napari_kwargs)
    v.add_image(-img, colormap="bop orange", **napari_kwargs)


# ------------------ registration at ~10 um - worse results ------------------ #
v = napari.Viewer()
for nn in case_number[:]:
    print(f"{nn}")
    ref_path = rf"/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20241025-deform-registration-stic/thumbnail/{nn}-ref.tif"
    moving_path = rf"/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20241025-deform-registration-stic/thumbnail/{nn}-moving.tif"

    img, params_tform, params_reg = try_conditions(
        ref_path=ref_path, moving_path=moving_path, conditions=conditions
    )

    # write parameters to disk
    # write_parameter(params_reg, elastix_config_dir, f"{nn}-")
    # write_parameter(params_tform, elastix_tform_dir, f"{nn}-tform-")

    napari_kwargs = dict(blending="additive", visible=False, name=f"{nn}-{grid_size}")
    v.add_image(
        tifffile.imread(ref_path),
        colormap="bop blue",
        visible=False,
        name=f"{nn}",
        contrast_limits=(0, 20000),
    )
    v.add_image(-tifffile.imread(moving_path), colormap="bop purple", **napari_kwargs)
    v.add_image(-img, colormap="bop orange", **napari_kwargs)

# ------------------------------- try one image ------------------------------ #
_ = run_one_setting(
    "/Users/yuanchen/projects/STalign/docs/img_data/test-elastix-img-pair/C07-ref.tif",
    "/Users/yuanchen/projects/STalign/docs/img_data/test-elastix-img-pair/C07-moving.tif",
    {"sample_region_size": 400, "sample_number_of_pixels": 6000},
)

ref = tifffile.imread(
    "/Users/yuanchen/projects/STalign/docs/img_data/test-elastix-img-pair/C04-ref.tif"
)
moving = tifffile.imread(
    "/Users/yuanchen/projects/STalign/docs/img_data/test-elastix-img-pair/C04-moving.tif"
)


# ---------------------------------------------------------------------------- #
#                       write deformation field as image                       #
# ---------------------------------------------------------------------------- #
dform_dir = pathlib.Path(
    r"/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20241025-deform-registration-stic/deformation-field"
)

def _to_deformation_field(elastix_parameter):
    shape = elastix_parameter.GetParameterMap(0).get("Size")[::-1]
    shape = np.array(shape, dtype="int")
    return itk.transformix_deformation_field(
        itk.GetImageFromArray(np.zeros(shape, dtype="uint8")), elastix_parameter
    )


def to_deformation_field(elastix_tform_paths):
    elastix_parameter = itk.ParameterObject.New()
    for ff in elastix_tform_paths:
        elastix_parameter.AddParameterFile(str(ff))

    return _to_deformation_field(elastix_parameter)


for nn in case_number[:]:
    print(nn)
    elastix_tform_paths = [
        rf"/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20241025-deform-registration-stic/reg-param/tform/{nn}-tform-elastix-param-0.txt",
        rf"/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20241025-deform-registration-stic/reg-param/tform/{nn}-tform-elastix-param-1.txt",
    ]

    dform = to_deformation_field(elastix_tform_paths)
    dform = np.moveaxis(dform, 2, 0)

    tifffile.imwrite(
        dform_dir / f"{nn}-elastix-deformation-field-xy.ome.tif",
        dform,
        compression="zlib",
    )
# ---------------------------------------------------------------------------- #
#                                   dev block                                  #
# ---------------------------------------------------------------------------- #


v = napari.Viewer()
v.add_image(ref, colormap="bop blue")
v.add_image(moving, blending="additive", colormap="bop purple")


parameter_object = itk.ParameterObject.New()

p_rigid = parameter_object.GetDefaultParameterMap("rigid", 4)
p_rigid["HowToCombineTransforms"] = ["Compose"]
parameter_object.WriteParameterFile(
    p_rigid,
    "/Users/yuanchen/projects/ITKElastix/explore/elastix-parameter-file/rigid.txt",
)
parameter_object.WriteParameterFile(
    get_default_crc_params(sample_region_size=300, sample_number_of_pixels=5000),
    "/Users/yuanchen/projects/ITKElastix/explore/elastix-parameter-file/crc-default.txt",
)


# ---------------------------------------------------------------------------- #
#                       read parameter file and register                       #
# ---------------------------------------------------------------------------- #
p_obj2 = itk.ParameterObject.New()
p_obj2.AddParameterFile(
    "/Users/yuanchen/projects/ITKElastix/explore/elastix-parameter-file/rigid.txt"
)
p_obj2.AddParameterFile(
    "/Users/yuanchen/projects/ITKElastix/explore/elastix-parameter-file/crc-default.txt"
)


result_image_2, _result_transform_parameters = itk.elastix_registration_method(
    ref,
    moving,
    parameter_object=p_obj2,
    log_to_console=False,
)

v.add_image(result_image_2, blending="additive", colormap="bop orange")


# --------------------------------- mask test -------------------------------- #
import palom  # noqa: E402

ref_mask = palom.img_util.entropy_mask(palom.img_util.cv2_downscale_local_mean(ref, 4))
moving_mask = palom.img_util.entropy_mask(
    palom.img_util.cv2_downscale_local_mean(moving, 4)
)

ref_mask = palom.img_util.repeat_2d(ref_mask, (4, 4))[: ref.shape[0], : ref.shape[1]]
moving_mask = palom.img_util.repeat_2d(moving_mask, (4, 4))[
    : moving.shape[0], : moving.shape[1]
]


p_obj2 = itk.ParameterObject.New()
p_obj2.AddParameterFile(
    "/Users/yuanchen/projects/ITKElastix/explore/elastix-parameter-file/rigid.txt"
)
p_obj2.AddParameterFile(
    "/Users/yuanchen/projects/ITKElastix/explore/elastix-parameter-file/crc-default.txt"
)
result_image_m, _result_transform_parameters = itk.elastix_registration_method(
    ref,
    palom.register_util.masked_match_histograms(moving, ref),
    parameter_object=p_obj2,
    log_to_console=False,
    # fixed_mask=itk.image_from_array(ref_mask.astype("uint8")),
    # moving_mask=itk.image_from_array(moving_mask.astype("uint8")),
)
v.add_image(result_image_m, blending="additive", colormap="bop orange")


# ---------------------------------------------------------------------------- #
#                     load transformation result parameters                    #
# ---------------------------------------------------------------------------- #
for idx in range(_result_transform_parameters.GetNumberOfParameterMaps()):
    parameter_object.WriteParameterFile(
        _result_transform_parameters.GetParameterMap(idx),
        f"/Users/yuanchen/projects/ITKElastix/explore/elastix-parameter-file/post-register/crc01-{idx}.txt",
    )

loaded_parameter = itk.ParameterObject.New()
loaded_parameter.AddParameterFile(
    "/Users/yuanchen/projects/ITKElastix/explore/elastix-parameter-file/post-register/crc01-0.txt"
)
loaded_parameter.AddParameterFile(
    "/Users/yuanchen/projects/ITKElastix/explore/elastix-parameter-file/post-register/crc01-1.txt"
)

warpped_moving = itk.transformix_filter(
    moving, transform_parameter_object=loaded_parameter
)
deformation_field = itk.transformix_deformation_field(
    itk.image_from_array(moving), loaded_parameter
)


# ---------------------------------------------------------------------------- #
#                           valis default parameters                           #
# ---------------------------------------------------------------------------- #
import numpy as np  # noqa: E402


def get_default_params_valis(img_shape, grid_spacing_ratio=0.025):
    """
    Get default parameters for registration with sitk.ElastixImageFilter

    See https://simpleelastix.readthedocs.io/Introduction.html
    for advice on parameter selection
    """
    parameter_object = itk.ParameterObject.New()
    p = parameter_object.GetDefaultParameterMap("bspline")
    p["Metric"] = [
        "AdvancedMattesMutualInformation",
        "TransformBendingEnergyPenalty",
    ]
    p["MaximumNumberOfIterations"] = ["1500"]  # Can try up to 2000
    p["FixedImagePyramid"] = ["FixedRecursiveImagePyramid"]
    p["MovingImagePyramid"] = ["MovingRecursiveImagePyramid"]
    p["Interpolator"] = ["BSplineInterpolator"]
    p["ImageSampler"] = ["RandomCoordinate"]
    p["MetricSamplingStrategy"] = ["None"]  # Use all points
    p["UseRandomSampleRegion"] = ["true"]
    p["ErodeMask"] = ["true"]
    p["NumberOfHistogramBins"] = ["32"]
    p["NumberOfSpatialSamples"] = ["3000"]
    p["NewSamplesEveryIteration"] = ["true"]
    p["SampleRegionSize"] = [str(min([img_shape[1] // 3, img_shape[0] // 3]))]
    p["Optimizer"] = ["AdaptiveStochasticGradientDescent"]
    p["ASGDParameterEstimationMethod"] = ["DisplacementDistribution"]
    p["HowToCombineTransforms"] = ["Compose"]
    grid_spacing_x = img_shape[1] * grid_spacing_ratio
    grid_spacing_y = img_shape[0] * grid_spacing_ratio
    grid_spacing = str(int(np.mean([grid_spacing_x, grid_spacing_y])))
    p["FinalGridSpacingInPhysicalUnits"] = [grid_spacing]
    p["WriteResultImage"] = ["false"]

    return p


parameter_object.WriteParameterFile(
    get_default_params_valis((1515, 1835), grid_spacing_ratio=0.01),
    "/Users/yuanchen/projects/ITKElastix/explore/elastix-parameter-file/valis-default.txt",
)

p_obj1 = itk.ParameterObject.New()
p_obj1.AddParameterFile(
    "/Users/yuanchen/projects/ITKElastix/explore/elastix-parameter-file/rigid.txt"
)
p_obj1.AddParameterFile(
    "/Users/yuanchen/projects/ITKElastix/explore/elastix-parameter-file/valis-default.txt"
)

# result_image_1, _result_transform_parameters = itk.elastix_registration_method(
#     ref,
#     moving,
#     parameter_object=p_obj1,
#     log_to_console=False,
# )

# v.add_image(result_image_1, blending="additive", colormap="bop orange")
