"""Training data: folders of crops with the teacher's vector for each.

A folder is what `distill/serve.py --record` writes (NNNNNN_II.jpg + NNNNNN.npy per encoder call).
`Crops` pairs every jpg with its row; `bird_matrix()` is the teacher's name matrix for birds, the
classifier both teacher and student are scored with.
"""
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image

NAMES = Path(__file__).resolve().parents[2] / "bioscan" / "data" / "names" / "cache"
BIRDS = "bioclip-2.5-vith14-a4f7858b14c15128.npz"      # AviList v2026, 11,172 rows (gardener's bundled cache)


def bird_matrix() -> np.ndarray:
    return np.load(NAMES / BIRDS, allow_pickle=False)["matrix"].astype(np.float32)


def index(folders: list[str]) -> tuple[list[Path], np.ndarray]:
    """Every image and its teacher vector, in a stable order. A folder is either serve.py's record
    (NNNNNN_II.jpg + NNNNNN.npy) or embed.py's (files.txt + vecs.npy, images under img/SPECIES/)."""
    files, vecs = [], []
    for folder in folders:
        if (Path(folder) / "files.txt").exists():
            files += [Path(folder) / "img" / line for line in (Path(folder) / "files.txt").read_text().splitlines()]
            vecs.append(np.load(Path(folder) / "vecs.npy"))
            continue
        for npy in sorted(Path(folder).glob("[0-9]*.npy")):
            v = np.load(npy)
            jpgs = [npy.with_name(f"{npy.stem}_{i:02d}.jpg") for i in range(len(v))]
            assert all(j.exists() for j in jpgs), f"{npy}: crops missing"
            files += jpgs
            vecs.append(v)
    return files, np.concatenate(vecs).astype(np.float32)


class Crops(torch.utils.data.Dataset):
    def __init__(self, files: list[Path], vecs: np.ndarray, transform) -> None:
        self.files, self.vecs, self.transform = files, torch.from_numpy(vecs), transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int):
        with Image.open(self.files[i]) as im:
            return self.transform(im.convert("RGB")), self.vecs[i]


def split_species(files: list[Path], vecs: np.ndarray, every: int) -> tuple[tuple, tuple]:
    """(train, unseen): one species folder in `every` is kept out of training entirely. Picked by a hash
    of the folder name, so the same species are held out whichever data folders a run combines."""
    held = np.array([zlib.crc32(f.parent.name.encode()) % every == 0 for f in files])
    pick = lambda m: ([f for f, k in zip(files, m) if k], vecs[m])
    return pick(~held), pick(held)
