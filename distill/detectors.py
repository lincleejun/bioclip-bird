"""Other detectors in OWLv2's seat, for the detector bench (distill/serve.py --detector NAME).

NAME: dfine-nano, dfine-small, rtdetr-r18, rtdetr-r50 (Apache-2.0, safetensors, torch), yolov10n/s/m (AGPL-3.0,
ONNX, onnxruntime), owlv2-int8 (gardener's own detector, its weights read
from the int8 file that `./run distill/detectors.py` writes, in distill/slim.py's format), cascade-NAME (NAME first,
OWLv2 int8 only for a frame where NAME found nothing).
The COCO models keep only their "bird" class, so alone they find no mammals; a cascade hands those asks to OWLv2.

The engine keeps a box when its confidence reaches the vocabulary's floor ("a bird": 0.3, tuned for OWLv2) and
asks again at 0.1 when a frame gave nothing. A COCO score c is mapped onto that scale so order survives:
c >= T -> [0.3, 1], T/2 <= c < T -> [0.1, 0.3). T is --det-threshold. Every call's raw boxes go to LOG as JSON lines.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

HF = {"dfine-nano": "ustc-community/dfine-nano-coco", "dfine-small": "ustc-community/dfine-small-coco",
      "rtdetr-r18": "PekingU/rtdetr_v2_r18vd", "rtdetr-r50": "PekingU/rtdetr_v2_r50vd"}
ONNX = {"yolov10n": "onnx-community/yolov10n", "yolov10s": "onnx-community/yolov10s", "yolov10m": "onnx-community/yolov10m"}
BIRD_PROMPT, FLOOR, SECOND = "a bird", 0.3, 0.1


def mapped(c: float, t: float) -> float:
    return FLOOR + (1 - FLOOR) * (c - t) / (1 - t) if c >= t else SECOND + (FLOOR - SECOND) * (c - t / 2) / (t / 2)


class Coco:
    """detect / detect_batch as gardener's OWLv2 adapter has them; `boxes(image)` is the model itself."""

    def __init__(self, name: str, device: str, threshold: float = 0.25, log: str | None = None) -> None:
        self.name, self.t, self.log = name, threshold, Path(log) if log else None
        if name in HF:
            import torch
            from transformers import AutoImageProcessor, AutoModelForObjectDetection
            self.torch, self.device = torch, device
            self.processor = AutoImageProcessor.from_pretrained(HF[name])
            self.model = AutoModelForObjectDetection.from_pretrained(HF[name]).to(device).eval()
            self.bird = self.model.config.label2id["bird"]
        else:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            config = json.loads(Path(hf_hub_download(ONNX[name], "config.json")).read_text())
            self.bird = next(int(k) for k, v in config["id2label"].items() if v == "bird")
            self.session = ort.InferenceSession(hf_hub_download(ONNX[name], "onnx/model.onnx"), providers=["CPUExecutionProvider"])
            self.input = self.session.get_inputs()[0].name

    def boxes(self, image: Any) -> list[tuple[float, tuple[float, float, float, float]]]:
        """[(score, (x0, y0, x1, y1) in the image's pixels)] of birds scoring at least T/2."""
        image = image.convert("RGB")
        if self.name in HF:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                out = self.model(**inputs)
            r = self.processor.post_process_object_detection(out, threshold=self.t / 2, target_sizes=[(image.height, image.width)])[0]
            return [(float(s), tuple(float(v) for v in b)) for s, l, b in zip(r["scores"], r["labels"], r["boxes"]) if int(l) == self.bird]
        k = 640 / max(image.size)                                    # letterbox: long side 640, grey padding right and below
        small = image.resize((max(1, round(image.width * k)), max(1, round(image.height * k))))
        x = np.full((640, 640, 3), 114, np.uint8)
        x[:small.height, :small.width] = np.asarray(small)
        out = self.session.run(None, {self.input: (x.astype(np.float32) / 255).transpose(2, 0, 1)[None]})[0][0]      # (300, 6): box, score, class
        return [(float(s), (float(x0 / k), float(y0 / k), float(x1 / k), float(y1 / k))) for x0, y0, x1, y1, s, c in out if int(c) == self.bird and s >= self.t / 2]

    def detect_batch(self, images: list[Any], prompts: list[str], *, threshold: float, batch: int = 8) -> list[list[Any]]:
        from gardener.service.adapters.owlv2 import Detection, clip_to_frame
        out = []
        for image in images:
            found = self.boxes(image) if BIRD_PROMPT in prompts else []
            dets = [Detection(BIRD_PROMPT, mapped(s, self.t), c) for s, b in found if (c := clip_to_frame(b, *image.size)) is not None]
            if self.log:
                with open(self.log, "a") as f:
                    f.write(json.dumps({"size": image.size, "threshold": threshold, "boxes": [[round(s, 4), *map(round, b)] for s, b in found]}) + "\n")
            out.append(dets)
        return out

    def detect(self, image: Any, prompts: list[str], *, threshold: float) -> list[Any]:
        return self.detect_batch([image], prompts, threshold=threshold)[0]


OWL_INT8 = Path(__file__).resolve().parents[1] / "models" / "owlv2-base-patch16-ensemble-int8.safetensors"


class Cascade:
    """A cheap COCO detector first; OWLv2 only for a frame where it found nothing at the asked threshold (and for
    every non-bird ask, since the COCO model answers only "a bird"). OWLv2 is the text-free int8 file when it exists
    (same boxes as the int8 file, see OwlVision), else the int8 file. Each frame's route goes to LOG as JSON lines."""

    def __init__(self, first: str, device: str, threshold: float = 0.25, log: str | None = None) -> None:
        self.first, self.log = Coco(first, device, threshold), Path(log) if log else None
        self.owl = OwlVision(device) if OWL_VISION.exists() else owlv2_int8(device)

    def detect(self, image: Any, prompts: list[str], *, threshold: float) -> list[Any]:
        found = self.first.detect(image, prompts, threshold=threshold)
        fallback = not found
        if fallback:
            found = self.owl.detect(image, prompts, threshold=threshold)
        if self.log:
            with open(self.log, "a") as f:
                f.write(json.dumps({"size": image.size, "threshold": threshold, "fallback": fallback, "boxes": len(found)}) + "\n")
        return found

    def detect_batch(self, images: list[Any], prompts: list[str], *, threshold: float, batch: int = 8) -> list[list[Any]]:
        return [self.detect(im, prompts, threshold=threshold) for im in images]


def load(name: str, device: str, threshold: float, log: str | None) -> Any:
    """NAME: a COCO detector above, owlv2-int8, or cascade-NAME (NAME first, OWLv2 int8 when it finds nothing)."""
    if name.startswith("cascade-"):
        return Cascade(name.removeprefix("cascade-"), device, threshold, log)
    if name == "owlv2-vision-int8":
        return OwlVision(device)
    if name != "owlv2-int8":
        return Coco(name, device, threshold, log)
    return owlv2_int8(device)


def _owl_model(state: dict, text: bool) -> Any:
    """Owlv2ForObjectDetection built on the meta device and given `state` by assignment (no second copy); without
    `text` the text tower and projection are left out, so the state must not have them either."""
    import torch
    from gardener.service.adapters.owlv2 import MODEL_ID, REVISION
    from transformers import Owlv2Config, Owlv2ForObjectDetection
    with torch.device("meta"):              # structure only; the unpacked weights are assigned, never copied
        model = Owlv2ForObjectDetection(Owlv2Config.from_pretrained(MODEL_ID, revision=REVISION))
    if not text:
        model.owlv2.text_model, model.owlv2.text_projection = torch.nn.Identity(), torch.nn.Identity()
    dtypes = {k: v.dtype for k, v in model.state_dict().items()}
    model.load_state_dict({k: v.to(dtypes[k]) for k, v in state.items()}, assign=True)
    for name, b in [(n, b) for n, b in model.named_buffers() if b.is_meta]:     # non-persistent: computed, not stored
        owner, leaf = model.get_submodule(name.rpartition(".")[0]), name.rpartition(".")[2]
        if leaf == "position_ids":
            value = torch.arange(b.shape[-1]).expand(b.shape)
        elif leaf == "box_bias":
            value = model.compute_box_bias(model.num_patches_height, model.num_patches_width)
        else:
            raise AssertionError(f"buffer not in the file: {name}")
        owner.register_buffer(leaf, value.to(b.dtype), persistent=False)
    return model


def owlv2_int8(device: str, path: Path = OWL_INT8) -> Any:
    """gardener's OWLv2 adapter with its weights read from the int8 file (distill/slim.py format), not from the
    original checkpoint: the model is built from config.json, and the processor's tokenizer and image settings come
    from the same pinned Hugging Face revision (small text files)."""
    import torch
    from gardener.service.adapters.owlv2 import MODEL_ID, REVISION, OWLv2
    from transformers import Owlv2Processor

    from distill import slim
    det = OWLv2.__new__(OWLv2)
    det.torch, det.device = torch, device
    det.processor = Owlv2Processor.from_pretrained(MODEL_ID, revision=REVISION)
    det.model = _owl_model(slim.load(str(path)), text=True).to(device).eval()
    print(f"owlv2-int8: {path.name}, {path.stat().st_size / 1e6:.0f} MB", flush=True)
    return det


OWL_VISION = OWL_INT8.with_name("owlv2-base-patch16-ensemble-vision-int8.safetensors")


class OwlVision:
    """OWLv2 without its text tower: the image half (int8 file) plus the text vectors of a fixed prompt list, worked
    out once by the full model and stored in the file. Same detect / detect_batch as gardener's adapter; a prompt
    outside the stored list is an error. 156 MB -> 92 MB on disk, and no text encoding per call."""

    def __init__(self, device: str, path: Path = OWL_VISION) -> None:
        import torch
        from gardener.service.adapters.owlv2 import MODEL_ID, REVISION
        from safetensors import safe_open
        from transformers import Owlv2ImageProcessor

        from distill import slim
        self.torch, self.device = torch, device
        self.image_processor = Owlv2ImageProcessor.from_pretrained(MODEL_ID, revision=REVISION)
        with safe_open(str(path), "pt") as f:
            self.prompts = json.loads(f.metadata()["prompts"])
        state = slim.load(str(path))
        self.queries = state.pop("query_embeds").float().to(device)          # (prompts, 512), as the full model made them
        self.model = _owl_model(state, text=False).to(device).eval()
        print(f"owlv2-vision-int8: {path.name}, {path.stat().st_size / 1e6:.0f} MB, {len(self.prompts)} prompts", flush=True)

    def detect(self, image: Any, prompts: list[str], *, threshold: float) -> list[Any]:
        return self.detect_batch([image], prompts, threshold=threshold)[0]

    def detect_batch(self, images: list[Any], prompts: list[str], *, threshold: float, batch: int = 8) -> list[list[Any]]:
        from gardener.service.adapters.owlv2 import Detection, clip_to_frame
        from transformers.models.owlv2.modeling_owlv2 import Owlv2ObjectDetectionOutput
        torch, m = self.torch, self.model
        q = self.queries[[self.prompts.index(p) for p in prompts]]
        out = []
        for start in range(0, len(images), batch):
            part = images[start:start + batch]
            pixels = self.image_processor(images=part, return_tensors="pt")["pixel_values"].to(self.device)
            with torch.no_grad():                                           # Owlv2ForObjectDetection.forward minus the text
                fmap = m.image_embedder(pixel_values=pixels)[0]
                feats = fmap.reshape(fmap.shape[0], -1, fmap.shape[-1])
                queries = q[None].expand(len(part), -1, -1)
                logits, _ = m.class_predictor(feats, queries, torch.ones(queries.shape[:2], dtype=torch.bool, device=self.device))
                boxes = m.box_predictor(feats, fmap, False)
            res = self.image_processor.post_process_object_detection(
                Owlv2ObjectDetectionOutput(logits=logits, pred_boxes=boxes), threshold=threshold,
                target_sizes=torch.tensor([[max(im.size)] * 2 for im in part]))
            for image, r in zip(part, res):
                dets = [Detection(prompts[int(label)], float(score), c) for score, label, box in zip(r["scores"], r["labels"], r["boxes"])
                        if (c := clip_to_frame(tuple(float(v) for v in box), *image.size)) is not None]
                out.append(dets)
        return out


def write_vision(path: Path = OWL_VISION) -> None:
    """OWL_VISION from OWL_INT8: the int8 file without text weights, plus every gardener prompt's text vector."""
    import torch
    from gardener.service.adapters.owlv2 import MODEL_ID, REVISION
    from gardener.service.taxa import VOCAB
    from PIL import Image
    from safetensors.torch import load_file, save_file
    from transformers import Owlv2Processor
    prompts = sorted({p for v in VOCAB.values() for p in v})
    full = owlv2_int8("cpu")
    proc = Owlv2Processor.from_pretrained(MODEL_ID, revision=REVISION)
    with torch.no_grad():
        q = full.model(**proc(text=[prompts], images=[Image.new("RGB", (64, 64))], return_tensors="pt")).text_embeds[0]
    packed = {k: v for k, v in load_file(str(OWL_INT8)).items() if ".text_model." not in k and ".text_projection." not in k}
    packed["query_embeds"] = q.half().contiguous()
    save_file(packed, str(path), metadata={"prompts": json.dumps(prompts)})
    print(f"{path}: {path.stat().st_size / 1e6:.0f} MB, {len(prompts)} prompts")


if __name__ == "__main__":       # ./run distill/detectors.py [--vision]: write OWL_INT8 from gardener's OWLv2 checkpoint
    import sys

    import torch
    from gardener.service.adapters.owlv2 import OWLv2
    from safetensors.torch import save_file

    from distill import slim
    if "--vision" in sys.argv:
        write_vision()
        raise SystemExit
    state = {k: v.cpu() for k, v in OWLv2("cpu").model.state_dict().items()}
    OWL_INT8.parent.mkdir(exist_ok=True)
    save_file(slim.pack(state, int8=True), OWL_INT8)
    back = owlv2_int8("cpu").model.state_dict()
    assert back.keys() == state.keys() and all(torch.equal(back[k], v.to(state[k].dtype)) for k, v in slim.unpack(slim.pack(state, True)).items())
    print(f"{OWL_INT8}: {OWL_INT8.stat().st_size / 1e6:.0f} MB, identical to the in-memory round trip the bench used")
