# YOLO preprocessor for 3ds Max renders -> Stable Diffusion

Скрипт `preprocess_for_sd.py` и UI `gradio_app.py` делят объекты на 2 группы:

- **preserve** — объекты, где важно сохранить геометрию/четкость форм.
- **rewrite** — объекты, которые можно сильнее перерисовать в SD.

## Установка

```bash
pip install ultralytics opencv-python numpy gradio
```

## CLI режим

### Подготовка политики классов

Сделай JSON файл по примеру `class_policy.example.json`:

```json
{
  "preserve": ["car", "person", "chair"],
  "rewrite": ["sky", "potted plant"]
}
```

Если класс не указан ни в одном списке, по умолчанию он идет в `rewrite` (можно поменять `--default-group preserve`).

### Запуск

```bash
python preprocess_for_sd.py \
  --input ./renders \
  --output ./prepared \
  --policy ./class_policy.example.json \
  --model yolov8x-seg.pt \
  --conf 0.25 \
  --mask-threshold 0.5 \
  --rewrite-blur 21 \
  --min-mask-area 200 \
  --dilate 0 \
  --erode 0
```

Дополнительные полезные настройки:

- `--device` (`cpu`, `cuda:0`, или пусто для auto)
- `--default-group` (`rewrite`/`preserve`)
- `--mask-threshold` (порог бинаризации сегментации)
- `--dilate` / `--erode` (морфология масок)

## Gradio интерфейс (рекомендуется)

Запуск:

```bash
python gradio_app.py
```

После запуска открой `http://localhost:7860`.

В интерфейсе можно:

- загрузить рендер и сразу увидеть overlay/маски/SD-base,
- менять модель YOLO-seg,
- на лету редактировать `preserve` и `rewrite` классы,
- настраивать `confidence`, `mask threshold`, `min mask area`, blur,
- применять dilate/erode к маскам,
- менять default group для неизвестных классов,
- выбрать устройство (`cpu/cuda`) и сохранить все артефакты в `outputs_gradio/`.

## Что генерируется

Для каждого входного файла (CLI) создаётся папка `output/<имя_файла>/`:

- `preserve_mask.png` — белая маска объектов, которые нужно сохранить.
- `rewrite_mask.png` — белая маска объектов, которые можно перерисовать.
- `inpaint_mask.png` — копия `rewrite_mask.png`.
- `sd_base.png` — картинка с замыленными rewrite-областями.
- `overlay.png` — визуализация (green=preserve, red=rewrite).
- `prompt_hints.txt` — список найденных классов.
