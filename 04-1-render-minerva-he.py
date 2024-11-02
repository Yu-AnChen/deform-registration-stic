import pathlib

import cv2
import dask.array as da
import dask.diagnostics
import dask.diagnostics.profile
import numpy as np
import tifffile
import zarr


def render_rgb(img_path: str, out_dir: str):
    zimg = zarr.open(tifffile.imread(img_path, aszarr=True), mode="r")
    pyramid = [da.from_zarr(zimg[ii]) for ii in zimg]

    for pp in pyramid:
        chr, chc, chs = pp.chunksize
        assert chs == 3
        assert (chr == 1024) | (chr < 1024)
        assert (chc == 1024) | (chc < 1024)

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    def write_block(data, out_dir, level, block_info=None):
        yy, xx, _ = block_info[0]["chunk-location"]
        out_path = out_dir / f"{level}_{xx}_{yy}.jpg"
        cv2.imwrite(
            str(out_path),
            cv2.cvtColor(data, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 90],
        )

        return np.ones((1, 1, 3), dtype="uint8")

    for level, pp in enumerate(pyramid):
        with dask.diagnostics.ProgressBar():
            _ = pp.map_blocks(
                write_block, out_dir=out_dir, level=level, dtype="uint8"
            ).compute()


def run_batch(csv_path, print_args=True, dryrun=False, **kwargs):
    import csv
    import inspect
    import pprint
    import types

    from fire.parser import DefaultParseValue

    func = render_rgb

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

    fire.Fire({"run": render_rgb, "run-batch": run_batch})


if __name__ == "__main__":
    import sys

    sys.exit(main())
