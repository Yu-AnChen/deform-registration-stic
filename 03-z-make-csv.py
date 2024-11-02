import pandas as pd
import pathlib

df = pd.read_csv(
    r"X:\cycif-production\110-BRCA-Mutant-Ovarian-Precursors\HE\registered\script\files-all.csv"
)

df["affine_mx_path"] = df["LSP ID"].apply(
    lambda x: next(
        pathlib.Path(
            r"X:\cycif-production\110-BRCA-Mutant-Ovarian-Precursors\HE\registered\thumbnail"
        ).glob(f"{x}-affine-matrix.csv")
    )
)
df["deformation_field_path"] = df["LSP ID"].apply(
    lambda x: next(
        pathlib.Path(
            r"X:\cycif-production\110-BRCA-Mutant-Ovarian-Precursors\HE\registered\deformation-field"
        ).glob(f"{x}-elastix-deformation-field-xy.ome.tif")
    )
)
df["out_path"] = df.apply(
    lambda x: pathlib.Path(
        r"X:\cycif-production\110-BRCA-Mutant-Ovarian-Precursors\HE\registered\zlib"
    )
    / (
        f"{x['LSP ID']}-nearby-he-"
        + ".".join(pathlib.Path(x["p2"]).name.split(".")[:-1])
        + ".ome.tif"
    ),
    axis=1,
)
df["temp_zarr_store_dir"] = r"C:\Temp\temp-zarr-store"
df["pre_filter_sigma"] = 1
df.columns
df["thumbnail_max_size"] = df["thumbnail_max_size"].fillna(2000).astype("int")
df.rename(columns={"p1": "ref_file_path", "p2": "file_path"}).to_csv(
    r"X:\cycif-production\110-BRCA-Mutant-Ovarian-Precursors\HE\registered\script\files-all-warp.csv",
    index=False,
)
