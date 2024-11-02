import json
import pathlib


def update_exhibit(exhibit_path, out_exhibit_path=None):
    if out_exhibit_path is None:
        out_exhibit_path = exhibit_path
    with open(exhibit_path) as f:
        exhibit = json.load(f)
    _update_exhibit(exhibit)
    with open(out_exhibit_path, "w") as f:
        json.dump(exhibit, f)


def _update_exhibit(exhibit):
    exhibit["Groups"].append(
        {
            "Name": "H&E (nearby)",
            "Colors": ["6356bc"],
            "Channels": ["H&E"],
            "Descriptions": [""],
        },
    )

    exhibit["Channels"].append(
        {"Rendered": True, "Name": "H&E", "Path": "nearby-he"},
    )

    exhibit["Masks"].extend(
        [
            {
                "Path": "path-annotation-cycif",
                "Name": "Path Annotations (CyCIF)",
                "Channels": ["Annotations (CyCIF)"],
                "Colors": ["00bfff"],
            },
            {
                "Path": "path-annotation-he",
                "Name": "Path Annotations (H&E)",
                "Channels": ["Annotations (H&E)"],
                "Colors": ["f08d44"],
            },
        ]
    )
    return

curr = pathlib.Path(r"W:\cycif-production\110-BRCA-Mutant-Ovarian-Precursors\MINERVA-path-annotation-and-he")
files = curr.glob("*/exhibit.json")

for ff in files:
    print(ff)
    update_exhibit(ff)