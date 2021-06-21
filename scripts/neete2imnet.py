import sys
from pathlib import Path

imageneete_lbl = dict(
    n01440764="tench",
    n02102040="English springer",
    n02979186="cassette player",
    n03000684="chain saw",
    n03028079="church",
    n03394916="French horn",
    n03417042="garbage truck",
    n03425413="gas pump",
    n03445777="golf ball",
    n03888257="parachute",
)


# rootdir = "/content/data/imagenette2/train/"
rootdir = sys.argv[1]
for path in Path(rootdir).iterdir():
    if path.is_dir():
        name = path.name
        newname = "_".join(imageneete_lbl[name].split())
        folder = path.parent
        Path(folder / name).rename(folder / newname)

