"""One distillation step: the student regresses the teacher's vectors (1 - cosine).

  ./run distill/train.py --data data/rec/pilot --val data/rec/golden --arch vit_small_patch16_224 --out runs/s1

Reports, per epoch, on --val: mean cosine to the teacher, and top-1 agreement with the teacher over
the bird name matrix (agreement, not accuracy: accuracy is `gardener-engine eval` through
distill/serve.py --student). --pretrained starts from timm's ImageNet weights (a download).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torchvision import transforms as T

from distill.data import Crops, bird_matrix, index, split_species
from distill.student import Student, norm, params_m, preprocess


def evaluate(model, loader, matrix, dev) -> dict:
    model.eval()
    cos, same, n = 0.0, 0, 0
    with torch.no_grad():
        for x, t in loader:
            s, t = model(x.to(dev)), t.to(dev)
            cos += float((s * t).sum(-1).sum())
            same += int(((s @ matrix.T).argmax(-1) == (t @ matrix.T).argmax(-1)).sum())
            n += len(t)
    return {"cosine": round(cos / n, 4), "top1_agree": round(same / n, 4), "n": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--arch", default="vit_small_patch16_224")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (smoke test)")
    ap.add_argument("--unseen-every", type=int, default=0, help="keep one species in N of --data out of training; report agreement on them (the first --data folder's photos)")
    ap.add_argument("--init", help="start from this student.pt (same --arch) instead of the backbone's own weights")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    aug = T.Compose([T.RandomResizedCrop(a.size, scale=(0.75, 1.0), ratio=(0.9, 1.1), interpolation=T.InterpolationMode.BICUBIC),
                     T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(*norm(a.arch))])
    files, vecs = index(a.data)
    unseen = None
    if a.unseen_every:
        (files, vecs), unseen = split_species(files, vecs, a.unseen_every)
        first = [Path(a.data[0]) in f.parents for f in unseen[0]]       # scored on the first folder's photos only,
        unseen = ([f for f, k in zip(unseen[0], first) if k], unseen[1][first])   # so runs that add folders stay comparable
    train = Crops(files, vecs, aug)
    val = Crops(*index(a.val), preprocess(a.size, a.arch))
    kw = dict(num_workers=a.workers, persistent_workers=a.workers > 0)
    tl = torch.utils.data.DataLoader(train, batch_size=a.bs, shuffle=True, drop_last=True, **kw)
    vl = torch.utils.data.DataLoader(val, batch_size=256, **kw)
    ul = torch.utils.data.DataLoader(Crops(*unseen, preprocess(a.size, a.arch)), batch_size=256, **kw) if unseen else None
    model = Student(a.arch, a.pretrained).to(dev)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location="cpu", weights_only=True)["state"])
    matrix = torch.from_numpy(bird_matrix()).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    total = a.max_steps or a.epochs * len(tl)
    warm = min(500, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    print(f"{a.arch} @{a.size}: {params_m(model):.1f} M params, {len(train)} train crops, {len(val)} val crops, {total} steps on {dev}", flush=True)
    step, log = 0, []
    for epoch in range(a.epochs):
        model.train()
        t0, seen, run = time.time(), 0, 0.0
        for x, t in tl:
            loss = (1 - (model(x.to(dev)) * t.to(dev)).sum(-1)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step, seen, run = step + 1, seen + len(t), run + loss.item()
            if step >= total:
                break
        row = {"epoch": epoch + 1, "step": step, "loss": round(run / max(1, seen // a.bs), 4), "img_per_s": round(seen / (time.time() - t0), 1),
               **evaluate(model, vl, matrix, dev)}
        if ul is not None:
            row["unseen_species"] = evaluate(model, ul, matrix, dev)
        log.append(row)
        print(json.dumps(row), flush=True)
        torch.save({"arch": a.arch, "size": a.size, "state": model.state_dict()}, out / "student.pt")
        (out / "log.json").write_text(json.dumps({"args": vars(a), "epochs": log}, indent=1))
        if step >= total:
            break


if __name__ == "__main__":
    main()
