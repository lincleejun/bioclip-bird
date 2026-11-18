"""gardener's engine service with the species encoder swapped or recorded.

  ./run distill/serve.py --port 8771 --record data/rec/golden --allow-root DIR   # teacher; saves every crop + its vector
  ./run distill/serve.py --port 8771 --student runs/x/student.pt --allow-root DIR  # student in the teacher's seat

Everything else (detector, crops, geo prior, name lists, eval CLI) is gardener's own, so
`gardener-engine --url http://127.0.0.1:8771 eval ...` measures the student end to end against the
same baseline as the teacher. The teacher is still loaded with --student (its name lists and
logit scale are reused); a teacher-free adapter comes when the student is worth shipping.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

CROP_EDGE = 448          # recorded crops are shrunk to this long edge: room for a 384 px student


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8771)
    ap.add_argument("--record", help="folder: every species crop as NNNNNN_II.jpg, each call's vectors as NNNNNN.npy")
    ap.add_argument("--student", help="checkpoint from distill/train.py")
    ap.add_argument("--slim", help="the teacher's image tower from distill/slim.py; the text tower is dropped")
    ap.add_argument("--half", action="store_true", help="with --slim: compute in float16 too (default: float16 on disk, float32 in memory)")
    ap.add_argument("--detector", help="distill/detectors.py NAME in OWLv2's seat")
    ap.add_argument("--det-threshold", type=float, default=0.25)
    ap.add_argument("--det-log", help="file: the detector's raw boxes, one JSON line per frame")
    ap.add_argument("--allow-root", action="append", default=[])
    a = ap.parse_args()

    import uvicorn
    from gardener import profile
    from gardener.service import app as svc
    from gardener.service import engine as eng

    def species(device: str):
        bioclip, lists = eng.load_species(device)
        teacher = bioclip.encode_images
        if a.slim:
            import torch
            from distill import slim
            bioclip.model.visual.load_state_dict(slim.load(a.slim))
            bioclip.model.transformer = bioclip.model.token_embedding = None     # proof that identification never runs the text tower
            if a.half:
                bioclip.model.visual.half()

                def encode_half(images):
                    with torch.no_grad():
                        f = bioclip.model.visual(torch.stack([bioclip.preprocess(im) for im in images]).to(device).half()).float()
                        return f / f.norm(dim=-1, keepdim=True)
                bioclip.encode_images = encode_half
        if a.student:
            import torch

            from distill.student import Student, preprocess
            ck = torch.load(a.student, map_location="cpu", weights_only=True)
            model = Student(ck["arch"]).to(device).eval()
            model.load_state_dict(ck["state"])
            pre = preprocess(ck["size"], ck["arch"])

            def encode(images):
                with torch.no_grad():
                    return model(torch.stack([pre(im.convert("RGB")) for im in images]).to(device))
            bioclip.encode_images = encode
        if a.record:
            out, calls = Path(a.record), [0]
            out.mkdir(parents=True, exist_ok=True)
            inner = bioclip.encode_images

            def encode_and_record(images):
                feats = inner(images)
                calls[0] += 1
                for i, im in enumerate(images):
                    im = im.convert("RGB")
                    im.thumbnail((CROP_EDGE, CROP_EDGE))
                    im.save(out / f"{calls[0]:06d}_{i:02d}.jpg", quality=92)
                np.save(out / f"{calls[0]:06d}.npy", feats.float().cpu().numpy().astype(np.float16))
                (out / "logit_scale.txt").write_text(repr(bioclip.logit_scale))
                return feats
            bioclip.encode_images = encode_and_record
        del teacher
        return eng.place_lists(bioclip, lists)

    loaders = eng.Loaders(species=species)
    if a.detector:
        from distill import detectors
        loaders = eng.Loaders(species=species, owlv2=lambda device: detectors.load(a.detector, device, a.det_threshold, a.det_log))
    engine = eng.Engine(loaders=loaders)
    uvicorn.run(svc.create_app(engine, allow_roots=a.allow_root, profiles=profile.load()), host="127.0.0.1", port=a.port)


if __name__ == "__main__":
    main()
