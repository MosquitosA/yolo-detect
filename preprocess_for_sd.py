#!/usr/bin/env python3
"""YOLO-based preprocessor for 3ds Max renders before Stable Diffusion processing."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Set

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
        return ClassPolicy.from_lists(raw.get("preserve", []), raw.get("rewrite", []))

    @staticmethod
    def from_lists(preserve: List[str], rewrite: List[str]) -> "ClassPolicy":
        preserve_set = {str(v).strip().lower() for v in preserve if str(v).strip()}
        rewrite_set = {str(v).strip().lower() for v in rewrite if str(v).strip()}

        intersection = preserve_set & rewrite_set
        if intersection:
            raise ValueError(
                f"Один и тот же класс одновременно в preserve и rewrite: {sorted(intersection)}"
            )

        return ClassPolicy(preserve=preserve_set, rewrite=rewrite_set)

    def decide(self, class_name: str, default_group: str = "rewrite") -> str:
        cname = class_name.lower()
        if cname in self.preserve:
            return "preserve"
        if cname in self.rewrite:
            return "rewrite"
        return default_group


def parse_class_list(raw: str) -> List[str]:
    if not raw.strip():
        return []
    normalized = raw.replace("\n", ",").replace(";", ",")
    return [v.strip() for v in normalized.split(",") if v.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO preprocessing for renders before Stable Diffusion."
    )
    parser.add_argument("--input", type=Path, required=True, help="Папка с рендерами")
    parser.add_argument("--output", type=Path, required=True, help="Куда сохранять результат")
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8x-seg.pt",
        help="YOLO Segmentation model path/name (например yolov8x-seg.pt)",
    )
    parser.add_argument("--policy", type=Path, required=True, help="JSON файл с классами")
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
        help="Минимальная площадь маски в пикселях",
    )
    parser.add_argument("--mask-threshold", type=float, default=0.5, help="Порог бинаризации mask")
    parser.add_argument(
        "--default-group",
        type=str,
        choices=["preserve", "rewrite"],
        default="rewrite",
        help="Куда относить классы, которых нет в policy",
    )
    parser.add_argument("--dilate", type=int, default=0, help="Дилатация маски (итерации)")
    parser.add_argument("--erode", type=int, default=0, help="Эрозия маски (итерации)")
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


def apply_morphology(mask: np.ndarray, dilate_iter: int = 0, erode_iter: int = 0) -> np.ndarray:
    mask_uint8 = mask.astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    if dilate_iter > 0:
        mask_uint8 = cv2.dilate(mask_uint8, kernel, iterations=dilate_iter)
    if erode_iter > 0:
        mask_uint8 = cv2.erode(mask_uint8, kernel, iterations=erode_iter)
    return mask_uint8 > 127


def mask_overlay(image_bgr: np.ndarray, preserve_mask: np.ndarray, rewrite_mask: np.ndarray) -> np.ndarray:
    overlay = image_bgr.copy()
    green = np.zeros_like(image_bgr)
    green[:, :, 1] = 255
    red = np.zeros_like(image_bgr)
    red[:, :, 2] = 255
    overlay[preserve_mask] = cv2.addWeighted(overlay[preserve_mask], 0.5, green[preserve_mask], 0.5, 0)
    overlay[rewrite_mask] = cv2.addWeighted(overlay[rewrite_mask], 0.5, red[rewrite_mask], 0.5, 0)
    return overlay


def to_image_mask(mask_arr: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """Convert a model mask to the same size as source image.

    Some YOLO segmentation models can return mask tensors in network resolution
    (e.g. 640x640/512x640) instead of original image shape. We explicitly resize
    with nearest interpolation to keep masks binary-compatible.
    """
    target_h, target_w = image_shape
    if mask_arr.shape[:2] == (target_h, target_w):
        return mask_arr

    resized = cv2.resize(mask_arr.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return resized


def process_image_array(
    model: YOLO,
    image_bgr: np.ndarray,
    policy: ClassPolicy,
    conf: float,
    device: str,
    blur_ksize: int,
    min_mask_area: int,
    mask_threshold: float = 0.5,
    default_group: str = "rewrite",
    dilate_iter: int = 0,
    erode_iter: int = 0,
) -> Dict[str, object]:
    result = model.predict(source=image_bgr, conf=conf, device=device or None, verbose=False)[0]
    h, w = image_bgr.shape[:2]

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
            mask_resized = to_image_mask(mask_arr, (h, w))
            mask = mask_resized > mask_threshold
            if int(mask.sum()) < min_mask_area:
                continue

            decision = policy.decide(class_name, default_group=default_group)
            if decision == "preserve":
                preserve_mask |= mask
            else:
                rewrite_mask |= mask

    rewrite_mask &= ~preserve_mask

    preserve_mask = apply_morphology(preserve_mask, dilate_iter=dilate_iter, erode_iter=erode_iter)
    rewrite_mask = apply_morphology(rewrite_mask, dilate_iter=dilate_iter, erode_iter=erode_iter)
    rewrite_mask &= ~preserve_mask

    sd_base = compose_sd_base(image_bgr, rewrite_mask, blur_ksize)
    overlay = mask_overlay(image_bgr, preserve_mask, rewrite_mask)

    return {
        "preserve_mask": preserve_mask,
        "rewrite_mask": rewrite_mask,
        "inpaint_mask": rewrite_mask.copy(),
        "sd_base": sd_base,
        "overlay": overlay,
        "classes": class_counter,
    }


def write_outputs(output_dir: Path, stem: str, processed: Dict[str, object]) -> Path:
    img_out_dir = output_dir / stem
    img_out_dir.mkdir(parents=True, exist_ok=True)

    save_mask(processed["preserve_mask"], img_out_dir / "preserve_mask.png")
    save_mask(processed["rewrite_mask"], img_out_dir / "rewrite_mask.png")
    save_mask(processed["inpaint_mask"], img_out_dir / "inpaint_mask.png")
    cv2.imwrite(str(img_out_dir / "sd_base.png"), processed["sd_base"])
    cv2.imwrite(str(img_out_dir / "overlay.png"), processed["overlay"])

    with (img_out_dir / "prompt_hints.txt").open("w", encoding="utf-8") as f:
        classes = processed["classes"]
        if not classes:
            f.write("No objects detected by YOLO.\n")
            f.write("Tip: lower --conf or use a more suitable segmentation model.\n")
        else:
            f.write("Detected objects:\n")
            for name, count in sorted(classes.items(), key=lambda x: (-x[1], x[0])):
                f.write(f"- {name}: {count}\n")
    return img_out_dir


def process_image_file(
    model: YOLO,
    image_path: Path,
    out_dir: Path,
    policy: ClassPolicy,
    conf: float,
    device: str,
    blur_ksize: int,
    min_mask_area: int,
    mask_threshold: float,
    default_group: str,
    dilate_iter: int,
    erode_iter: int,
) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Не удалось прочитать {image_path}")

    processed = process_image_array(
        model=model,
        image_bgr=image,
        policy=policy,
        conf=conf,
        device=device,
        blur_ksize=blur_ksize,
        min_mask_area=min_mask_area,
        mask_threshold=mask_threshold,
        default_group=default_group,
        dilate_iter=dilate_iter,
        erode_iter=erode_iter,
    )
    write_outputs(out_dir, image_path.stem, processed)


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
        process_image_file(
            model=model,
            image_path=image_path,
            out_dir=args.output,
            policy=policy,
            conf=args.conf,
            device=args.device,
            blur_ksize=args.rewrite_blur,
            min_mask_area=args.min_mask_area,
            mask_threshold=args.mask_threshold,
            default_group=args.default_group,
            dilate_iter=args.dilate,
            erode_iter=args.erode,
        )
        print(f"[OK] {image_path.name}")

    print(f"Готово. Результаты в {args.output}")


if __name__ == "__main__":
    main()
