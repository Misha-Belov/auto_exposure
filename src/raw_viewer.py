#!/usr/bin/env python3

import json
import re
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from picamera2 import Picamera2


# ============================================================
# Настройки
# ============================================================

PREFERRED_SIZE = (2028, 1520)
PREFERRED_BIT_DEPTH = 12

PANEL_W = 640
PANEL_H = 480

WINDOW_NAME = "RAW Bayer Viewer"

# Гистограмма и статистика каналов довольно тяжёлые.
# 1 = считать каждый кадр.
# 3 = считать каждый третий кадр.
ANALYSIS_EVERY_N_FRAMES = 3

# Дать автоэкспозиции установиться, после чего зафиксировать
# ExposureTime и AnalogueGain.
FREEZE_EXPOSURE_AFTER_START = True
AE_WARMUP_SECONDS = 1.5

OUTPUT_DIRECTORY = Path.home() / "camera-lab" / "captures"


# ============================================================
# Bayer
# ============================================================

BAYER_SITES = {
    "RGGB": {
        "R":  (0, 0),
        "G1": (0, 1),
        "G2": (1, 0),
        "B":  (1, 1),
    },

    "BGGR": {
        "B":  (0, 0),
        "G1": (0, 1),
        "G2": (1, 0),
        "R":  (1, 1),
    },

    "GRBG": {
        "G1": (0, 0),
        "R":  (0, 1),
        "B":  (1, 0),
        "G2": (1, 1),
    },

    "GBRG": {
        "G1": (0, 0),
        "B":  (0, 1),
        "R":  (1, 0),
        "G2": (1, 1),
    },
}


# Используем именно четырёхбуквенные обозначения OpenCV,
# чтобы не путаться с историческими COLOR_BayerBG2BGR aliases.
BAYER_TO_OPENCV = {
    "RGGB": cv2.COLOR_BayerRGGB2BGR,
    "BGGR": cv2.COLOR_BayerBGGR2BGR,
    "GRBG": cv2.COLOR_BayerGRBG2BGR,
    "GBRG": cv2.COLOR_BayerGBRG2BGR,
}


# Цвета текста OpenCV: BGR
TEXT_COLOURS = {
    "R":  (60, 60, 255),
    "G1": (60, 255, 60),
    "G2": (60, 190, 60),
    "B":  (255, 100, 60),
}


# ============================================================
# Глобальная выбранная мышкой точка
# ============================================================

selected_sensor_x = 0
selected_sensor_y = 0

sensor_width = 0
sensor_height = 0


# ============================================================
# Вспомогательные функции
# ============================================================

def nothing(_):
    pass


def select_sensor_mode(picam2):
    """
    Ищем точно 2028x1520 / 12 bit.
    """

    for mode in picam2.sensor_modes:
        if (
            tuple(mode["size"]) == PREFERRED_SIZE
            and int(mode["bit_depth"]) == PREFERRED_BIT_DEPTH
        ):
            return mode

    print("\nДоступные режимы сенсора:")

    for i, mode in enumerate(picam2.sensor_modes):
        print(
            f"{i}: "
            f"{mode['size']} "
            f"{mode['bit_depth']} bit "
            f"{mode['fps']:.2f} fps "
            f"unpacked={mode.get('unpacked')}"
        )

    raise RuntimeError(
        f"Не найден режим "
        f"{PREFERRED_SIZE[0]}x{PREFERRED_SIZE[1]} "
        f"{PREFERRED_BIT_DEPTH}-bit"
    )


def detect_bayer_pattern(raw_format):
    """
    Например:
        SBGGR16 -> BGGR
        SRGGB16 -> RGGB
    """

    match = re.search(
        r"(RGGB|BGGR|GRBG|GBRG)",
        str(raw_format)
    )

    if match is None:
        raise RuntimeError(
            f"Не могу определить Bayer pattern "
            f"из формата {raw_format!r}"
        )

    return match.group(1)


def unpack_picamera_raw(
    raw_buffer,
    width,
    height,
    bit_depth,
):
    """
    Преобразует буфер Picamera2:

        uint8 bytes
                ↓
        uint16 container
                ↓
        активные width пикселей
                ↓
        native 12-bit / 10-bit значения

    Для Raspberry Pi 5 RAW находится в старших
    битах 16-битного слова.
    """

    if raw_buffer.dtype == np.uint8:

        # На всякий случай гарантируем contiguous memory.
        raw_buffer = np.ascontiguousarray(raw_buffer)

        # ВАЖНО:
        # не astype(np.uint16),
        # а именно view().
        raw_words = raw_buffer.view("<u2")

    elif raw_buffer.dtype == np.uint16:

        raw_words = raw_buffer

    else:
        raise RuntimeError(
            f"Неожиданный dtype RAW: {raw_buffer.dtype}"
        )

    if raw_words.shape[0] < height:
        raise RuntimeError(
            "RAW-буфер меньше ожидаемой высоты"
        )

    if raw_words.shape[1] < width:
        raise RuntimeError(
            "RAW-буфер меньше ожидаемой ширины"
        )

    # Удаляем padding после активных пикселей.
    raw16 = raw_words[:height, :width]

    shift = 16 - bit_depth

    # Настоящие значения фотосенсора.
    raw_native = raw16 >> shift

    return raw16, raw_native


def split_bayer(raw, pattern):
    """
    Возвращает views в исходный RAW.
    """

    sites = BAYER_SITES[pattern]

    result = {}

    for name, (dy, dx) in sites.items():
        result[name] = raw[dy::2, dx::2]

    return result


def sample_bayer_cell(raw, pattern, x, y):
    """
    Получаем четыре значения одной Bayer-ячейки 2x2.

    Координаты приводим к верхнему левому углу Bayer cell.
    """

    x = int(np.clip(x, 0, raw.shape[1] - 2))
    y = int(np.clip(y, 0, raw.shape[0] - 2))

    x &= ~1
    y &= ~1

    values = {}

    for name, (dy, dx) in BAYER_SITES[pattern].items():
        values[name] = int(raw[y + dy, x + dx])

    return x, y, values


# ============================================================
# RAW preview
# ============================================================

def create_raw_panel(
    raw_native,
    maximum_code,
    display_gain,
    sample_x,
    sample_y,
):
    """
    RAW показываем как grayscale БЕЗ демозаики.

    Важно:
    display_gain влияет только на визуализацию.
    Сам raw_native не изменяется.
    """

    alpha = (
        255.0
        / maximum_code
        * display_gain
    )

    raw8 = cv2.convertScaleAbs(
        raw_native,
        alpha=alpha,
        beta=0,
    )

    panel = cv2.cvtColor(
        raw8,
        cv2.COLOR_GRAY2BGR,
    )

    panel = cv2.resize(
        panel,
        (PANEL_W, PANEL_H),
        interpolation=cv2.INTER_AREA,
    )

    # Позиция выбранной точки в preview.
    px = round(sample_x / raw_native.shape[1] * PANEL_W)
    py = round(sample_y / raw_native.shape[0] * PANEL_H)

    cv2.drawMarker(
        panel,
        (px, py),
        (0, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=22,
        thickness=1,
    )

    draw_title(
        panel,
        "RAW Bayer - no demosaic"
    )

    return panel


# ============================================================
# Demosaic
# ============================================================

def create_demosaic_panel(
    raw_native,
    pattern,
    maximum_code,
    display_gain,
):
    """
    OpenCV demosaic.

    Исходный raw_native остаётся неизменным.
    """

    conversion_code = BAYER_TO_OPENCV[pattern]

    # OpenCV поддерживает Bayer для uint16.
    bgr16 = cv2.cvtColor(
        raw_native,
        conversion_code,
    )

    alpha = (
        255.0
        / maximum_code
        * display_gain
    )

    bgr8 = cv2.convertScaleAbs(
        bgr16,
        alpha=alpha,
        beta=0,
    )

    panel = cv2.resize(
        bgr8,
        (PANEL_W, PANEL_H),
        interpolation=cv2.INTER_AREA,
    )

    draw_title(
        panel,
        "Demosaic - display only"
    )

    return panel, bgr8


# ============================================================
# Histogram
# ============================================================

def calculate_histogram(raw_native, maximum_code):
    """
    Полная гистограмма:
        для 12 bit -> ровно 4096 bins.
    """

    return np.bincount(
        raw_native.reshape(-1),
        minlength=maximum_code + 1,
    )


def create_histogram_panel(
    histogram,
    maximum_code,
    raw_mean,
):
    panel = np.zeros(
        (PANEL_H, PANEL_W, 3),
        dtype=np.uint8,
    )

    panel[:] = (20, 20, 20)

    draw_title(
        panel,
        f"RAW histogram 0..{maximum_code}"
    )

    left = 55
    right = 20
    top = 55
    bottom = 45

    chart_width = PANEL_W - left - right
    chart_height = PANEL_H - top - bottom

    x0 = left
    y0 = top

    x1 = left + chart_width
    y1 = top + chart_height

    cv2.rectangle(
        panel,
        (x0, y0),
        (x1, y1),
        (90, 90, 90),
        1,
    )

    # 4096 bins слишком много для 565 пикселей экрана.
    # Объединяем соседние bins.
    edges = np.linspace(
        0,
        maximum_code + 1,
        chart_width + 1,
        dtype=np.int32,
    )

    reduced = np.zeros(
        chart_width,
        dtype=np.float64,
    )

    for x in range(chart_width):
        start = edges[x]
        end = edges[x + 1]

        if end > start:
            reduced[x] = histogram[start:end].sum()

    # Логарифмическая вертикальная шкала:
    # иначе пик гистограммы скрывает слабые детали.
    log_hist = np.log1p(reduced)

    peak = log_hist.max()

    if peak > 0:
        normalized = log_hist / peak
    else:
        normalized = log_hist

    ys = (
        y1
        - normalized * (chart_height - 2)
    ).astype(np.int32)

    xs = (
        np.arange(chart_width)
        + x0
    ).astype(np.int32)

    points = np.column_stack(
        (xs, ys)
    ).reshape((-1, 1, 2))

    cv2.polylines(
        panel,
        [points],
        False,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )

    # Среднее значение.
    mean_x = int(
        x0
        + raw_mean / maximum_code * chart_width
    )

    cv2.line(
        panel,
        (mean_x, y0),
        (mean_x, y1),
        (0, 255, 255),
        1,
    )

    # X labels
    ticks = [
        0,
        maximum_code // 4,
        maximum_code // 2,
        maximum_code * 3 // 4,
        maximum_code,
    ]

    for value in ticks:
        x = int(
            x0
            + value / maximum_code * chart_width
        )

        cv2.line(
            panel,
            (x, y1),
            (x, y1 + 5),
            (130, 130, 130),
            1,
        )

        text = str(value)

        cv2.putText(
            panel,
            text,
            (x - 15, y1 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (190, 190, 190),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        panel,
        f"mean = {raw_mean:.1f}",
        (65, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        panel,
        "Y scale: log(count)",
        (PANEL_W - 170, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (170, 170, 170),
        1,
        cv2.LINE_AA,
    )

    return panel


# ============================================================
# Statistics
# ============================================================

def calculate_channel_statistics(channels):
    result = {}

    for name, channel in channels.items():
        result[name] = {
            "mean": float(np.mean(channel)),
            "min": int(np.min(channel)),
            "max": int(np.max(channel)),
        }

    return result


def create_stats_panel(
    pattern,
    bit_depth,
    maximum_code,
    statistics,
    cell_x,
    cell_y,
    cell_values,
    metadata,
    fps,
    capture_ms,
    analysis_ms,
):
    panel = np.zeros(
        (PANEL_H, PANEL_W, 3),
        dtype=np.uint8,
    )

    panel[:] = (25, 25, 25)

    draw_title(
        panel,
        "R / G1 / G2 / B sensor values"
    )

    y = 55

    cv2.putText(
        panel,
        (
            f"Bayer: {pattern}    "
            f"bit depth: {bit_depth}    "
            f"range: 0..{maximum_code}"
        ),
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )

    y += 30

    cv2.putText(
        panel,
        "channel       mean        min       max      selected",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (170, 170, 170),
        1,
        cv2.LINE_AA,
    )

    y += 27

    for name in ("R", "G1", "G2", "B"):

        stat = statistics[name]

        text = (
            f"{name:<4}       "
            f"{stat['mean']:8.1f}   "
            f"{stat['min']:5d}   "
            f"{stat['max']:5d}      "
            f"{cell_values[name]:5d}"
        )

        cv2.putText(
            panel,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            TEXT_COLOURS[name],
            1,
            cv2.LINE_AA,
        )

        y += 27

    y += 8

    cv2.line(
        panel,
        (20, y),
        (PANEL_W - 20, y),
        (80, 80, 80),
        1,
    )

    y += 27

    cv2.putText(
        panel,
        (
            f"Selected Bayer cell: "
            f"x={cell_x}, y={cell_y}"
        ),
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    y += 32

    exposure = metadata.get(
        "ExposureTime",
        "?"
    )

    analogue_gain = metadata.get(
        "AnalogueGain",
        "?"
    )

    digital_gain = metadata.get(
        "DigitalGain",
        "?"
    )

    frame_duration = metadata.get(
        "FrameDuration",
        "?"
    )

    black_levels = metadata.get(
        "SensorBlackLevels",
        "?"
    )

    lines = [
        f"ExposureTime: {exposure} us",
        f"AnalogueGain: {analogue_gain}",
        f"DigitalGain: {digital_gain}",
        f"FrameDuration: {frame_duration} us",
        f"SensorBlackLevels: {black_levels}",
        "",
        f"Viewer FPS: {fps:.1f}",
        f"Capture: {capture_ms:.2f} ms",
        f"Analysis+demosaic: {analysis_ms:.2f} ms",
    ]

    for line in lines:

        if not line:
            y += 10
            continue

        cv2.putText(
            panel,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )

        y += 24

    return panel


# ============================================================
# GUI
# ============================================================

def draw_title(image, text):
    cv2.rectangle(
        image,
        (0, 0),
        (image.shape[1], 31),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        image,
        text,
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )


def mouse_callback(event, x, y, flags, userdata):
    global selected_sensor_x
    global selected_sensor_y

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # RAW находится в верхнем левом panel.
    if 0 <= x < PANEL_W and 0 <= y < PANEL_H:

        selected_sensor_x = int(
            x / PANEL_W * sensor_width
        )

        selected_sensor_y = int(
            y / PANEL_H * sensor_height
        )

        selected_sensor_x = int(
            np.clip(
                selected_sensor_x,
                0,
                sensor_width - 2,
            )
        )

        selected_sensor_y = int(
            np.clip(
                selected_sensor_y,
                0,
                sensor_height - 2,
            )
        )


def compose_dashboard(
    raw_panel,
    demosaic_panel,
    histogram_panel,
    stats_panel,
):
    top = np.hstack(
        (
            raw_panel,
            demosaic_panel,
        )
    )

    bottom = np.hstack(
        (
            histogram_panel,
            stats_panel,
        )
    )

    return np.vstack(
        (
            top,
            bottom,
        )
    )


# ============================================================
# Save
# ============================================================

def save_capture(
    raw16,
    raw_native,
    demosaic8,
    metadata,
    pattern,
    bit_depth,
):
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    prefix = OUTPUT_DIRECTORY / timestamp

    np.save(
        f"{prefix}_raw16.npy",
        raw16,
    )

    np.save(
        f"{prefix}_raw_native.npy",
        raw_native,
    )

    raw_native.tofile(
        f"{prefix}_raw_native.bin"
    )

    cv2.imwrite(
        f"{prefix}_demosaic.png",
        demosaic8,
    )

    info = {
        "bayer_pattern": pattern,
        "bit_depth": bit_depth,
        "shape": list(raw_native.shape),
        "metadata": metadata,
    }

    with open(
        f"{prefix}_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            info,
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    print("\nСохранён кадр:")
    print(prefix)


# ============================================================
# Main
# ============================================================

def main():
    global selected_sensor_x
    global selected_sensor_y
    global sensor_width
    global sensor_height

    picam2 = Picamera2()

    mode = select_sensor_mode(picam2)

    sensor_width, sensor_height = mode["size"]
    bit_depth = int(mode["bit_depth"])
    maximum_code = (1 << bit_depth) - 1

    selected_sensor_x = sensor_width // 2
    selected_sensor_y = sensor_height // 2

    print("\nВыбран режим:")

    print(
        f"  size:       "
        f"{sensor_width}x{sensor_height}"
    )

    print(
        f"  bit depth:  "
        f"{bit_depth}"
    )

    print(
        f"  fps:        "
        f"{mode['fps']:.2f}"
    )

    print(
        f"  unpacked:   "
        f"{mode['unpacked']}"
    )

    config = picam2.create_video_configuration(

        # Маленький ISP-stream нужен Picamera2,
        # но мы его не используем для анализа.
        main={
            "size": (320, 240),
            "format": "RGB888",
        },

        raw={
            "size": mode["size"],
            "format": mode["unpacked"],
        },

        sensor={
            "output_size": mode["size"],
            "bit_depth": bit_depth,
        },

        buffer_count=4,
        display=None,
    )

    picam2.configure(config)

    actual_config = picam2.camera_configuration()

    raw_config = actual_config["raw"]

    raw_format = str(
        raw_config["format"]
    )

    raw_width, raw_height = raw_config["size"]

    pattern = detect_bayer_pattern(
        raw_format
    )

    print("\nФактический RAW:")

    print(
        f"  format: {raw_format}"
    )

    print(
        f"  size:   "
        f"{raw_width}x{raw_height}"
    )

    print(
        f"  stride: "
        f"{raw_config['stride']}"
    )

    print(
        f"  Bayer:  {pattern}"
    )

    if "PISP_COMP" in raw_format:
        raise RuntimeError(
            "Получен PISP compressed RAW. "
            "Для viewer нужен unpacked RAW."
        )

    picam2.start()

    # --------------------------------------------------------
    # Auto exposure warmup
    # --------------------------------------------------------

    if FREEZE_EXPOSURE_AFTER_START:

        print(
            "\nЖдём стабилизации автоэкспозиции..."
        )

        time.sleep(
            AE_WARMUP_SECONDS
        )

        metadata = picam2.capture_metadata()

        exposure = int(
            metadata.get(
                "ExposureTime",
                10000,
            )
        )

        analogue_gain = float(
            metadata.get(
                "AnalogueGain",
                1.0,
            )
        )

        picam2.set_controls(
            {
                "AeEnable": False,
                "ExposureTime": exposure,
                "AnalogueGain": analogue_gain,
            }
        )

        print(
            f"Exposure зафиксирован: "
            f"{exposure} us"
        )

        print(
            f"AnalogueGain зафиксирован: "
            f"{analogue_gain:.3f}"
        )

    # --------------------------------------------------------
    # Window
    # --------------------------------------------------------

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        WINDOW_NAME,
        PANEL_W * 2,
        PANEL_H * 2,
    )

    cv2.setMouseCallback(
        WINDOW_NAME,
        mouse_callback,
    )

    # Только визуализация.
    # RAW-данные этим коэффициентом НЕ изменяются.
    cv2.createTrackbar(
        "Display gain x100",
        WINDOW_NAME,
        100,
        800,
        nothing,
    )

    print("\nУправление:")
    print(
        "  click на RAW -> выбрать Bayer cell"
    )
    print(
        "  Display gain -> только яркость preview"
    )
    print(
        "  S -> сохранить RAW"
    )
    print(
        "  Q / Esc -> выход"
    )

    # --------------------------------------------------------
    # Cache тяжёлых вычислений
    # --------------------------------------------------------

    cached_histogram = np.zeros(
        maximum_code + 1,
        dtype=np.int64,
    )

    cached_statistics = {
        name: {
            "mean": 0.0,
            "min": 0,
            "max": 0,
        }
        for name in ("R", "G1", "G2", "B")
    }

    raw_mean = 0.0

    frame_number = 0

    fps_counter = 0
    fps = 0.0
    fps_start = time.perf_counter()

    last_analysis_ms = 0.0

    try:

        while True:

            # ====================================================
            # Capture
            # ====================================================

            capture_start = time.perf_counter()

            with picam2.captured_request() as request:

                raw_buffer = request.make_array(
                    "raw"
                )

                metadata = request.get_metadata()

            capture_ms = (
                time.perf_counter()
                - capture_start
            ) * 1000.0

            raw16, raw_native = unpack_picamera_raw(
                raw_buffer,
                raw_width,
                raw_height,
                bit_depth,
            )

            # ====================================================
            # Display gain
            # ====================================================

            display_gain = max(
                cv2.getTrackbarPos(
                    "Display gain x100",
                    WINDOW_NAME,
                )
                / 100.0,
                0.01,
            )

            # ====================================================
            # Demosaic
            # ====================================================

            analysis_start = time.perf_counter()

            demosaic_panel, demosaic8 = (
                create_demosaic_panel(
                    raw_native,
                    pattern,
                    maximum_code,
                    display_gain,
                )
            )

            # ====================================================
            # Statistics + histogram
            # ====================================================

            if (
                frame_number
                % ANALYSIS_EVERY_N_FRAMES
                == 0
            ):

                channels = split_bayer(
                    raw_native,
                    pattern,
                )

                cached_statistics = (
                    calculate_channel_statistics(
                        channels
                    )
                )

                cached_histogram = (
                    calculate_histogram(
                        raw_native,
                        maximum_code,
                    )
                )

                raw_mean = float(
                    np.mean(raw_native)
                )

            # ====================================================
            # Selected Bayer cell
            # ====================================================

            cell_x, cell_y, cell_values = (
                sample_bayer_cell(
                    raw_native,
                    pattern,
                    selected_sensor_x,
                    selected_sensor_y,
                )
            )

            # ====================================================
            # Panels
            # ====================================================

            raw_panel = create_raw_panel(
                raw_native,
                maximum_code,
                display_gain,
                cell_x,
                cell_y,
            )

            histogram_panel = (
                create_histogram_panel(
                    cached_histogram,
                    maximum_code,
                    raw_mean,
                )
            )

            last_analysis_ms = (
                time.perf_counter()
                - analysis_start
            ) * 1000.0

            # ====================================================
            # FPS
            # ====================================================

            fps_counter += 1

            now = time.perf_counter()

            elapsed = (
                now - fps_start
            )

            if elapsed >= 1.0:

                fps = (
                    fps_counter
                    / elapsed
                )

                fps_counter = 0
                fps_start = now

            # ====================================================
            # Statistics panel
            # ====================================================

            stats_panel = create_stats_panel(
                pattern=pattern,
                bit_depth=bit_depth,
                maximum_code=maximum_code,
                statistics=cached_statistics,
                cell_x=cell_x,
                cell_y=cell_y,
                cell_values=cell_values,
                metadata=metadata,
                fps=fps,
                capture_ms=capture_ms,
                analysis_ms=last_analysis_ms,
            )

            dashboard = compose_dashboard(
                raw_panel,
                demosaic_panel,
                histogram_panel,
                stats_panel,
            )

            # Footer / help
            cv2.putText(
                dashboard,
                (
                    "Click RAW: sample pixel | "
                    "S: save | Q/Esc: quit | "
                    "Display gain changes preview only"
                ),
                (10, dashboard.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(
                WINDOW_NAME,
                dashboard,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key in (
                ord("q"),
                ord("Q"),
                27,
            ):
                break

            if key in (
                ord("s"),
                ord("S"),
            ):

                save_capture(
                    raw16=raw16.copy(),
                    raw_native=raw_native.copy(),
                    demosaic8=demosaic8,
                    metadata=metadata,
                    pattern=pattern,
                    bit_depth=bit_depth,
                )

            frame_number += 1

    finally:

        picam2.stop()
        picam2.close()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
