#!/usr/bin/env python3
"""Gradio UI for YOLO render preprocessing before Stable Diffusion."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, Tuple

import cv2
import gradio as gr
from ultralytics import YOLO

from preprocess_for_sd import ClassPolicy, parse_class_list, process_image_array, write_outputs

_model_cache: Dict[str, YOLO] = {}


def get_model(model_name: str) -> YOLO:
    model_name = model_name.strip()
    if not model_name:
        raise gr.Error("Укажи название или путь к модели YOLO-seg, например yolov8x-seg.pt")
    if model_name not in _model_cache:
        _model_cache[model_name] = YOLO(model_name)
    return _model_cache[model_name]


def to_rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def run_single(
    image,
    model_name: str,
    preserve_classes: str,
    rewrite_classes: str,
    conf: float,
    mask_threshold: float,
    min_mask_area: int,
    blur_ksize: int,
    default_group: str,
    dilate_iter: int,
    erode_iter: int,
    device: str,
    save_outputs: bool,
) -> Tuple:
    if image is None:
        raise gr.Error("Загрузи изображение")

    policy = ClassPolicy.from_lists(parse_class_list(preserve_classes), parse_class_list(rewrite_classes))
    model = get_model(model_name)

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    processed = process_image_array(
        model=model,
        image_bgr=image_bgr,
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

    preserve_mask = (processed["preserve_mask"].astype("uint8") * 255)
    rewrite_mask = (processed["rewrite_mask"].astype("uint8") * 255)
    inpaint_mask = (processed["inpaint_mask"].astype("uint8") * 255)
    overlay = to_rgb(processed["overlay"])
    sd_base = to_rgb(processed["sd_base"])

    classes = processed["classes"]
    if classes:
        report = "\n".join([f"- {k}: {v}" for k, v in sorted(classes.items(), key=lambda x: (-x[1], x[0]))])
    else:
        report = "No detections"

    output_path = ""
    if save_outputs:
        out_root = Path("outputs_gradio")
        out_root.mkdir(exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix="preview_", suffix=".png", delete=False) as tmp:
            temp_name = Path(tmp.name).stem
        saved = write_outputs(out_root, temp_name, processed)
        output_path = str(saved.resolve())

    return overlay, preserve_mask, rewrite_mask, inpaint_mask, sd_base, report, output_path


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="YOLO Render Preprocessor for SD") as demo:
        gr.Markdown(
            "# YOLO Preprocessor (3ds Max → Stable Diffusion)\n"
            "Загрузи рендер, задай классы и получи маски + подготовленную базу для inpaint/img2img."
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(label="Input render", type="numpy")
                model_name = gr.Textbox(value="yolov8x-seg.pt", label="YOLO-seg model")
                device = gr.Textbox(value="", label="Device (cpu, cuda:0, пусто=auto)")
                default_group = gr.Radio(["rewrite", "preserve"], value="rewrite", label="Default group")
                preserve_classes = gr.Textbox(
                    label="Preserve classes (comma/newline separated)",
                    lines=4,
                    value="car,truck,bus,motorcycle,bicycle,person,chair,couch,tv,laptop",
                )
                rewrite_classes = gr.Textbox(
                    label="Rewrite classes (comma/newline separated)",
                    lines=4,
                    value="sky,potted plant,dining table",
                )

                conf = gr.Slider(0.01, 0.95, value=0.25, step=0.01, label="Confidence")
                mask_threshold = gr.Slider(0.05, 0.95, value=0.5, step=0.01, label="Mask threshold")
                min_mask_area = gr.Slider(0, 5000, value=200, step=10, label="Min mask area (px)")
                blur_ksize = gr.Slider(3, 101, value=21, step=2, label="Rewrite blur kernel")
                dilate_iter = gr.Slider(0, 10, value=0, step=1, label="Mask dilate iterations")
                erode_iter = gr.Slider(0, 10, value=0, step=1, label="Mask erode iterations")
                save_outputs = gr.Checkbox(value=False, label="Save outputs to outputs_gradio/")
                run_btn = gr.Button("Run preprocessing", variant="primary")

            with gr.Column(scale=1):
                overlay_out = gr.Image(label="Overlay (green=preserve, red=rewrite)")
                preserve_out = gr.Image(label="Preserve mask")
                rewrite_out = gr.Image(label="Rewrite mask")
                inpaint_out = gr.Image(label="Inpaint mask")
                sd_base_out = gr.Image(label="SD base")
                classes_out = gr.Textbox(label="Detected classes", lines=8)
                save_path_out = gr.Textbox(label="Saved folder")

        run_btn.click(
            fn=run_single,
            inputs=[
                image_input,
                model_name,
                preserve_classes,
                rewrite_classes,
                conf,
                mask_threshold,
                min_mask_area,
                blur_ksize,
                default_group,
                dilate_iter,
                erode_iter,
                device,
                save_outputs,
            ],
            outputs=[
                overlay_out,
                preserve_out,
                rewrite_out,
                inpaint_out,
                sd_base_out,
                classes_out,
                save_path_out,
            ],
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7860)
