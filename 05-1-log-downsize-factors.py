import pandas as pd

import palom
from palom.cli import align_he


def get_downsize_factors(
    ref_file_path: str,
    moving_file_path: str,
    ref_thumbnail_level: int = 0,
):
    r1 = align_he.get_reader(ref_file_path)(ref_file_path)
    r2 = align_he.get_reader(moving_file_path)(moving_file_path)

    matched_levels = palom.align_multi_res.match_levels(r1, r2)

    affine_level1, affine_level2 = matched_levels[ref_thumbnail_level]
    out_level1, out_level2 = 0, 0

    d_moving = r2.level_downsamples[affine_level2] / r2.level_downsamples[out_level2]
    d_ref = r1.level_downsamples[affine_level1] / r1.level_downsamples[out_level1]

    return d_ref, d_moving


REF_DOWISIZE_FACTOR = 4

files = r"X:\cycif-production\110-BRCA-Mutant-Ovarian-Precursors\HE\registered\script\files-all.csv"
df = pd.read_csv(files)

d_factors = df[["p1", "p2"]].apply(
    lambda x: get_downsize_factors(x["p1"], x["p2"], REF_DOWISIZE_FACTOR), axis=1
)

df["ref-downsize-factor-roi"] = [dd[0] for dd in d_factors]
df["moving-downsize-factor-roi"] = [dd[1] for dd in d_factors]
df.to_csv(
    r"X:\cycif-production\110-BRCA-Mutant-Ovarian-Precursors\HE\registered\script\files-all.csv",
    index=False,
)
