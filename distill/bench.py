"""How fast is this machine? Teacher images/s (forward) and student images/s (forward + backward).

  ./run distill/bench.py                      # prints one line per model; nothing is downloaded (random-init students)
"""
from __future__ import annotations

import sys
import time

import torch

from distill.student import DIM, Student, params_m

STUDENTS = [("vit_small_patch16_224", 224), ("vit_base_patch16_224", 224), ("convnext_tiny", 224), ("fastvit_sa24", 256),
            ("fastvit_ma36", 256), ("fastvit_ma36", 384), ("efficientnet_b3", 288), ("tiny_vit_21m_224", 224)]


def sync(dev: str) -> None:
    if dev == "mps":
        torch.mps.synchronize()


def rate(step, batch: int, dev: str, seconds: float = 12.0) -> float:
    for _ in range(3):
        step()
    sync(dev)
    n, t0 = 0, time.time()
    while time.time() - t0 < seconds:
        step()
        n += batch
    sync(dev)
    return n / (time.time() - t0)


def main() -> None:
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device {dev}, torch {torch.__version__}")
    if "--no-teacher" not in sys.argv:
        from gardener.service.adapters.bioclip import BioCLIP
        t = BioCLIP(dev)
        x = torch.randn(32, 3, 224, 224, device=dev)

        def fwd():
            with torch.no_grad():
                t.model.encode_image(x)
        print(f"teacher BioCLIP 2.5 ViT-H/14 @224  {params_m(t.model.visual):6.1f} M visual params   forward {rate(fwd, 32, dev):7.1f} img/s")
        del t
    for arch, size in STUDENTS:
        try:
            m = Student(arch).to(dev).train()
            opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
            x, y = torch.randn(64, 3, size, size, device=dev), torch.nn.functional.normalize(torch.randn(64, DIM, device=dev), dim=-1)

            def step():
                opt.zero_grad(set_to_none=True)
                (1 - (m(x) * y).sum(-1)).mean().backward()
                opt.step()
            r = rate(step, 64, dev)
            m.eval()

            def infer():
                with torch.no_grad():
                    m(x)
            print(f"student {arch:24} @{size}  {params_m(m):6.1f} M params   train {r:7.1f} img/s   forward {rate(infer, 64, dev, 6):7.1f} img/s")
        except Exception as e:                      # an arch this timm lacks, or an op MPS lacks: say so, keep going
            print(f"student {arch:24} @{size}  failed: {type(e).__name__}: {str(e)[:120]}")


if __name__ == "__main__":
    main()
