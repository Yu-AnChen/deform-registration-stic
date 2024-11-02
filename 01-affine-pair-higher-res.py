import csv
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import palom
import skimage.transform
import tifffile
import tqdm
from fire.parser import DefaultParseValue
from palom.cli import align_he

OUT_DIR = pathlib.Path(r"X:\cycif-production\110-BRCA-Mutant-Ovarian-Precursors\HE\registered\thumbnail")

csv_path = r"/Users/yuanchen/HMS Dropbox/Yu-An Chen/000 local remote sharing/20241025-deform-registration-stic/00-files.csv"
csv_path = r"X:\cycif-production\110-BRCA-Mutant-Ovarian-Precursors\HE\registered\script\files-all.csv"

with open(csv_path) as f:
    csv_kwargs = [
        {kk: DefaultParseValue(vv) for kk, vv in rr.items() if vv is not None}
        for rr in csv.DictReader(f)
    ]

for current in tqdm.tqdm(csv_kwargs):

    p1 = current["p1"]
    p2 = current["p2"]
    thumbnail_max_size = current.get("thumbnail_max_size", 2000)
    nn = current["LSP ID"]

    r1 = align_he.get_reader(p1)(p1)
    r2 = align_he.get_reader(p2)(p2)

    matched_levels = palom.align_multi_res.match_levels(r1, r2)

    LEVEL1 = 4
    LEVEL2 = matched_levels[4][1]

    aligner = palom.align.get_aligner(
        r1,
        r2,
        level1=LEVEL1,
        level2=LEVEL2,
        channel1=0,
        channel2=1,
        # make thumbnail level pair based on pixel_size
        thumbnail_level1=None,
        thumbnail_channel1=0,
        thumbnail_channel2=1,
    )
    plt.close()
    _mx = palom.register_dev.search_then_register(
        np.asarray(aligner.ref_thumbnail),
        np.asarray(aligner.moving_thumbnail),
        n_keypoints=20_000,
        auto_mask=True,
        max_size=thumbnail_max_size,
    )
    aligner.coarse_affine_matrix = np.vstack([_mx, [0, 0, 1]])


    ref = aligner.ref_img.compute()
    img = aligner.moving_img.compute()
    moving = skimage.transform.warp(
        img,
        skimage.transform.AffineTransform(matrix=aligner.affine_matrix).inverse,
        preserve_range=True,
        output_shape=ref.shape,
        cval=np.percentile(img, 95),
    )
    np.savetxt(OUT_DIR / f"{nn}-affine-matrix.csv", aligner.affine_matrix, delimiter=",")
    tifffile.imwrite(OUT_DIR / f"{nn}-ref.tif", ref, compression="zlib")
    tifffile.imwrite(
        OUT_DIR / f"{nn}-moving.tif",
        np.floor(moving).astype(r2.pyramid[0].dtype),
        compression="zlib",
    )
    tifffile.imwrite(OUT_DIR / f"{nn}-moving-ori.tif", img, compression="zlib")
