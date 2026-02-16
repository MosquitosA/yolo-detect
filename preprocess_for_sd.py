#!/usr/bin/env python3
"""YOLO-based preprocessor for 3ds Max renders before Stable Diffusion processing.

The script separates objects into two groups:
1. preserve: objects where shape/detail should stay sharp.
2. rewrite: objects that may be heavily redrawn by SD for better aesthetics.

Outputs per image:
- preserve_mask.png: white mask of "keep sharp" objects
- rewrite_mask.png: white mask of "can redraw" objects
- inpaint_mask.png: same as rewrite mask (for SD inpainting)
- sd_base.png: base image where rewrite zones are softened
- prompt_hints.txt: detected class summary for prompt engineering
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class ClassPolicy:
    """Defines which classes are kept sharp or allowed to redraw."""

    preserve: Set[str] = field(default_factory=set)
    rewrite: Set[str] = field(default_factory=set)

    @staticmethod
    def from_json(path: Path) -> "ClassPolicy":
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        preserve = {str(v).strip().lower() for v in raw.get("preserve", []) if str(v).strip()}
        rewrite = {str(v).strip().lower() for v in raw.get("rewrite", []) if str(v).strip()}

        intersection = preserve & rewrite
        if intersection:
            raise ValueError(
                f"Один и тот же класс одновременно в preserve и rewrite: {sorted(intersection)}"
            )

        return ClassPolicy(preserve=preserve, rewrite=rewrite)

    def decide(self, class_name: str) -> str:
        cname = class_name.lower()
        if cname in self.preserve:
            return "preserve"
        if cname in self.rewrite:
            return "rewrite"
        return "rewrite"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO preprocessing for renders before Stable Diffusion.")
    parser.add_argument("--input", type=Path, required=True, help="Папка с рендерами")
    parser.add_argument("--output", type=Path, required=True, help="Куда сохранять результат")
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8x-seg.pt",
        help="YOLO Segmentation model path/name (например yolov8x-seg.pt)",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        required=True,
        help="JSON файл с классами preserve/rewrite",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="", help="Device, например cpu/cuda:0")
    parser.add_argument(
        "--rewrite-blur",
        type=int,
        default=21,
        help="Размер blur ядра для rewrite зон на sd_base.png",
    )
    parser.add_argument(
        "--min-mask-area",
        type=int,
        default=200,
        help="Минимальная площадь маски в пикселях, меньше игнорируется",
    )
    return parser.parse_args()


def iter_images(path: Path) -> Iterable[Path]:
    for p in sorted(path.iterdir()):
        if p.suffix.lower() in SUPPORTED_EXTENSIONS and p.is_file():
            yield p


def ensure_odd(k: int) -> int:
    if k < 3:
        return 3
    return k if k % 2 == 1 else k + 1


def save_mask(mask: np.ndarray, dst: Path) -> None:
    cv2.imwrite(str(dst), (mask.astype(np.uint8) * 255))


def compose_sd_base(image_bgr: np.ndarray, rewrite_mask: np.ndarray, blur_ksize: int) -> np.ndarray:
    blur_ksize = ensure_odd(blur_ksize)
    blurred = cv2.GaussianBlur(image_bgr, (blur_ksize, blur_ksize), 0)
    out = image_bgr.copy()
    out[rewrite_mask] = blurred[rewrite_mask]
    return out


def process_image(
    model: YOLO,
    image_path: Path,
    out_dir: Path,
    policy: ClassPolicy,
    conf: float,
    device: str,
    blur_ksize: int,
    min_mask_area: int,
) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Не удалось прочитать {image_path}")

    result = model.predict(source=image, conf=conf, device=device or None, verbose=False)[0]
    h, w = image.shape[:2]

    preserve_mask = np.zeros((h, w), dtype=bool)
    rewrite_mask = np.zeros((h, w), dtype=bool)

    class_counter: Dict[str, int] = {}

    if result.masks is not None and result.boxes is not None:
        cls_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
        masks = result.masks.data.detach().cpu().numpy()
        names = result.names

        for cls_id, mask_arr in zip(cls_ids, masks):
            class_name = str(names[int(cls_id)])
            class_counter[class_name] = class_counter.get(class_name, 0) + 1

            mask = mask_arr > 0.5
            if int(mask.sum()) < min_mask_area:
                continue

            decision = policy.decide(class_name)
            if decision == "preserve":
                preserve_mask |= mask
            else:
                rewrite_mask |= mask

    # safety: preserve has priority in overlaps
    rewrite_mask &= ~preserve_mask

    stem = image_path.stem
    img_out_dir = out_dir / stem
    img_out_dir.mkdir(parents=True, exist_ok=True)

    preserve_mask_path = img_out_dir / "preserve_mask.png"
    rewrite_mask_path = img_out_dir / "rewrite_mask.png"
    inpaint_mask_path = img_out_dir / "inpaint_mask.png"
    sd_base_path = img_out_dir / "sd_base.png"
    prompt_hints_path = img_out_dir / "prompt_hints.txt"

    save_mask(preserve_mask, preserve_mask_path)
    save_mask(rewrite_mask, rewrite_mask_path)
    save_mask(rewrite_mask, inpaint_mask_path)

    sd_base = compose_sd_base(image, rewrite_mask, blur_ksize)
    cv2.imwrite(str(sd_base_path), sd_base)

    with prompt_hints_path.open("w", encoding="utf-8") as f:
        if not class_counter:
            f.write("No objects detected by YOLO.\n")
            f.write("Tip: lower --conf or use a more suitable segmentation model.\n")
        else:
            f.write("Detected objects:\n")
            for name, count in sorted(class_counter.items(), key=lambda x: (-x[1], x[0])):
                label = policy.decide(name)
                f.write(f"- {name}: {count} ({label})\n")


def main() -> None:
    args = parse_args()

    if not args.input.exists() or not args.input.is_dir():
        raise FileNotFoundError(f"Папка input не найдена: {args.input}")

    args.output.mkdir(parents=True, exist_ok=True)

    policy = ClassPolicy.from_json(args.policy)
    model = YOLO(args.model)

    images = list(iter_images(args.input))
    if not images:
        raise RuntimeError(f"В папке {args.input} нет изображений")

    for image_path in images:
        process_image(
            model=model,
            image_path=image_path,
            out_dir=args.output,
            policy=policy,
            conf=args.conf,
            device=args.device,
            blur_ksize=args.rewrite_blur,
            min_mask_area=args.min_mask_area,
        )
        print(f"[OK] {image_path.name}")

    print(f"Готово. Результаты в {args.output}")


if __name__ == "__main__":
    main()
