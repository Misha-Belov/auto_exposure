#!/usr/bin/env python3

"""
Пример запуска:

    python3 src/raw_hist_processor_from_recordings.py
    python3 src/raw_hist_processor_from_recordings.py --recording capture_20260807_194408
    python3 src/raw_hist_processor_from_recordings.py --recording capture_20260807_194408 --loop
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# CAMERA SETTINGS
# ============================================================

PREFERRED_SIZE = (2028, 1520)
PREFERRED_BIT_DEPTH = 12

WINDOW_NAME = "RAW Histogram Processor (recordings)"

# Размер каждого видео на экране.
PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 480

# Высота гистограммы.
HISTOGRAM_HEIGHT = 300

# Для записей с диска экспозиция уже зафиксирована.
FREEZE_EXPOSURE = False
AE_WARMUP_SECONDS = 1.5


# ============================================================
# DATASET SETTINGS
# ============================================================

DEFAULT_DATA_ROOT = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "raw-recordings"
)


# ============================================================
# USER PIXEL PROCESSING
# ============================================================

def process_raw(
    raw: np.ndarray,
    histogram: np.ndarray,
    max_value: int,
) -> np.ndarray:
    """
    ГЛАВНАЯ ФУНКЦИЯ БУДУЩЕГО АЛГОРИТМА.

    raw:
        исходный RAW Bayer;
        shape = (1520, 2028);
        dtype = uint16;
        значения = 0...4095 для 12-bit.

    histogram:
        histogram[i] = количество пикселей
        со значением i.

        Для 12 bit:
            len(histogram) == 4096

    max_value:
        4095 для 12-bit.

    Возвращаем:
        новый RAW-массив такого же размера.

    ---------------------------------------------------------
    ПОКА НИКАКОЙ ОБРАБОТКИ НЕТ.
    ---------------------------------------------------------
    """

    processed = raw.copy()

    # ========================================================
    # ЗДЕСЬ БУДЕТ ВАША ЛОГИКА
    # ========================================================
    #
    # Например:
    #
    # mean = np.average(
    #     np.arange(len(histogram)),
    #     weights=histogram,
    # )
    #
    # if mean < 1000:
    #     processed = processed * 1.5
    #
    #
    # Или:
    #
    # processed[processed < 300] = 0
    #
    #
    # Или LUT:
    #
    # lut = np.arange(max_value + 1, dtype=np.uint16)
    # processed = lut[raw]
    #
    #
    # Или преобразование на основе CDF гистограммы.
    #
    # ========================================================

    return processed


# ============================================================
# RECORDING DISCOVERY
# ============================================================

def find_recording_dirs(data_root: Path) -> list[Path]:
    """
    Ищем директории с записью RAW в data/raw-recordings/.
    """

    if not data_root.exists():
        raise FileNotFoundError(
            f"Путь с записями не найден: {data_root}"
        )

    return sorted(
        path
        for path in data_root.iterdir()
        if path.is_dir()
    )


def select_recording(
    recordings: list[Path],
    requested_name: str | None,
) -> Path:
    """
    Выбираем нужную запись из доступных директорий.
    """

    if requested_name is None:
        if not recordings:
            raise RuntimeError(
                "Нет доступных записей в data/raw-recordings/"
            )

        return recordings[0]

    for recording in recordings:
        if recording.name == requested_name:
            return recording

    raise FileNotFoundError(
        f"Запись не найдена: {requested_name}"
    )


def load_recording(recording_dir: Path) -> tuple[np.ndarray, dict]:
    """
    Загружаем массив frames.npy и metadata из session.json.
    """

    frames_path = recording_dir / "frames.npy"
    session_path = recording_dir / "session.json"

    if not frames_path.exists():
        raise FileNotFoundError(
            f"Не найден файл: {frames_path}"
        )

    frames = np.load(
        frames_path,
        allow_pickle=False,
    )

    if session_path.exists():
        with open(session_path, "r", encoding="utf-8") as file:
            session = json.load(file)
    else:
        session = {}

    return frames, session


# ============================================================
# RAW -> DISPLAY
# ============================================================

def raw_to_preview(
    raw: np.ndarray,
    max_value: int,
    display_gain: float,
) -> np.ndarray:
    """
    Только преобразование для монитора.

    RAW остаётся неизменным.

    0...4095
       ↓
    0...255
    """

    scale = (
        255.0
        / max_value
        * display_gain
    )

    image8 = cv2.convertScaleAbs(
        raw,
        alpha=scale,
        beta=0,
    )

    image = cv2.cvtColor(
        image8,
        cv2.COLOR_GRAY2BGR,
    )

    image = cv2.resize(
        image,
        (
            PREVIEW_WIDTH,
            PREVIEW_HEIGHT,
        ),
        interpolation=cv2.INTER_AREA,
    )

    return image


# ============================================================
# HISTOGRAM
# ============================================================

def calculate_histogram(
    raw: np.ndarray,
    max_value: int,
) -> np.ndarray:
    """
    Полная гистограмма RAW.

    Для 12-bit получается 4096 значений.
    """

    histogram = np.bincount(
        raw.reshape(-1),
        minlength=max_value + 1,
    )

    return histogram


# ============================================================
# HISTOGRAM STATISTICS
# ============================================================

def histogram_statistics(
    histogram: np.ndarray,
) -> dict:
    """
    Некоторые характеристики гистограммы.

    Они пригодятся будущему алгоритму.
    """

    x = np.arange(
        len(histogram),
        dtype=np.float64,
    )

    count = histogram.sum()

    if count == 0:
        return {
            "mean": 0,
            "p01": 0,
            "p50": 0,
            "p99": 0,
        }

    mean = float(
        np.sum(x * histogram) / count
    )

    cdf = np.cumsum(histogram)

    def percentile(p: float) -> int:
        threshold = count * p
        return int(
            np.searchsorted(
                cdf,
                threshold,
            )
        )

    return {
        "mean": mean,
        "p01": percentile(0.01),
        "p50": percentile(0.50),
        "p99": percentile(0.99),
    }


def find_balance_point(
    histogram: np.ndarray,
    max_value: int,
) -> int:
    """
    Находим точку, где площадь гистограммы слева и справа
    от неё примерно одинаковая.
    """

    count = int(histogram.sum())

    if count <= 0:
        return max_value // 2

    cdf = np.cumsum(histogram.astype(np.float64))
    midpoint = count / 2.0
    balance_point = int(np.searchsorted(cdf, midpoint))

    return int(np.clip(balance_point, 0, max_value))


def estimate_auto_gain(
    histogram: np.ndarray,
    max_value: int,
    current_gain: float,
) -> float:
    """
    Подстраиваем gain так, чтобы балансная точка гистограммы
    стремилась к середине доступного диапазона.
    """

    if current_gain <= 0:
        current_gain = 1.0

    balance_point = find_balance_point(histogram, max_value)
    target_point = max_value / 2.0

    if balance_point <= 0:
        return float(np.clip(current_gain, 0.01, 8.0))

    desired_gain = current_gain * (target_point / balance_point)
    smoothed_gain = current_gain * 0.7 + desired_gain * 0.3

    return float(np.clip(smoothed_gain, 0.01, 8.0))


# ============================================================
# HISTOGRAM DRAWING
# ============================================================

def draw_histogram(
    histogram: np.ndarray,
    max_value: int,
    statistics: dict,
    width: int,
    height: int,
    label: str = "RAW histogram",
) -> np.ndarray:

    canvas = np.zeros(
        (
            height,
            width,
            3,
        ),
        dtype=np.uint8,
    )

    canvas[:] = (
        20,
        20,
        20,
    )

    left = 60
    right = 25
    top = 45
    bottom = 45

    graph_width = (
        width
        - left
        - right
    )

    graph_height = (
        height
        - top
        - bottom
    )

    # --------------------------------------------------------
    # Сжимаем 4096 bins до ширины экрана.
    # --------------------------------------------------------

    edges = np.linspace(
        0,
        max_value + 1,
        graph_width + 1,
        dtype=np.int32,
    )

    reduced = np.zeros(
        graph_width,
        dtype=np.float64,
    )

    for i in range(graph_width):
        start = edges[i]
        end = edges[i + 1]
        reduced[i] = histogram[start:end].sum()

    # --------------------------------------------------------
    # Логарифмическая Y шкала.
    # --------------------------------------------------------

    reduced = np.log1p(reduced)

    maximum = reduced.max()

    if maximum > 0:
        reduced /= maximum

    y_values = (
        top
        + graph_height
        - reduced * graph_height
    ).astype(np.int32)

    x_values = (
        np.arange(graph_width)
        + left
    )

    points = np.column_stack(
        (
            x_values,
            y_values,
        )
    ).astype(np.int32)

    cv2.polylines(
        canvas,
        [points.reshape(-1, 1, 2)],
        False,
        (
            220,
            220,
            220,
        ),
        1,
        cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # Оси
    # --------------------------------------------------------

    cv2.line(
        canvas,
        (
            left,
            top + graph_height,
        ),
        (
            left + graph_width,
            top + graph_height,
        ),
        (
            100,
            100,
            100,
        ),
        1,
    )

    cv2.line(
        canvas,
        (
            left,
            top,
        ),
        (
            left,
            top + graph_height,
        ),
        (
            100,
            100,
            100,
        ),
        1,
    )

    # --------------------------------------------------------
    # Метки X
    # --------------------------------------------------------

    values = [
        0,
        max_value // 4,
        max_value // 2,
        3 * max_value // 4,
        max_value,
    ]

    for value in values:
        x = int(
            left
            + (
                value
                / max_value
            )
            * graph_width
        )

        cv2.line(
            canvas,
            (
                x,
                top + graph_height,
            ),
            (
                x,
                top + graph_height + 5,
            ),
            (
                150,
                150,
                150,
            ),
            1,
        )

        cv2.putText(
            canvas,
            str(value),
            (
                x - 18,
                top + graph_height + 25,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (
                200,
                200,
                200,
            ),
            1,
            cv2.LINE_AA,
        )

    # --------------------------------------------------------
    # Percentile markers
    # --------------------------------------------------------

    percentile_data = [
        (
            statistics["p01"],
            "P1",
        ),
        (
            statistics["p50"],
            "P50",
        ),
        (
            statistics["p99"],
            "P99",
        ),
    ]

    for value, name in percentile_data:
        x = int(
            left
            + value
            / max_value
            * graph_width
        )

        cv2.line(
            canvas,
            (
                x,
                top,
            ),
            (
                x,
                top + graph_height,
            ),
            (
                0,
                180,
                255,
            ),
            1,
        )

        cv2.putText(
            canvas,
            name,
            (
                x + 2,
                top + 15,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (
                0,
                180,
                255,
            ),
            1,
            cv2.LINE_AA,
        )

    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    text = (
        f"{label} 0..{max_value}     "
        f"mean={statistics['mean']:.1f}     "
        f"P1={statistics['p01']}     "
        f"P50={statistics['p50']}     "
        f"P99={statistics['p99']}"
    )

    cv2.putText(
        canvas,
        text,
        (
            15,
            27,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (
            230,
            230,
            230,
        ),
        1,
        cv2.LINE_AA,
    )

    return canvas


# ============================================================
# TITLE
# ============================================================

def add_title(
    image: np.ndarray,
    text: str,
) -> None:

    cv2.rectangle(
        image,
        (
            0,
            0,
        ),
        (
            image.shape[1],
            32,
        ),
        (
            0,
            0,
            0,
        ),
        -1,
    )

    cv2.putText(
        image,
        text,
        (
            10,
            23,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (
            255,
            255,
            255,
        ),
        1,
        cv2.LINE_AA,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Просмотр RAW-истории из data/raw-recordings/"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Папка с директориями записей",
    )
    parser.add_argument(
        "--recording",
        default=None,
        help="Имя поддиректории, например capture_20260807_194408",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Повторять воспроизведение с начала",
    )
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    recordings = find_recording_dirs(data_root)
    recording_dir = select_recording(recordings, args.recording)

    frames, session = load_recording(recording_dir)

    sensor = session.get("sensor", {})
    width = int(sensor.get("size", [PREFERRED_SIZE[0]])[0])
    height = int(sensor.get("size", [PREFERRED_SIZE[1]])[1])
    bit_depth = int(sensor.get("bit_depth", PREFERRED_BIT_DEPTH))

    max_value = (1 << bit_depth) - 1

    print("\nLoaded recording:")
    print(f"directory: {recording_dir}")
    print(f"frames shape: {frames.shape}")
    print(f"size: {width}x{height}")
    print(f"bit depth: {bit_depth}")
    print(f"max RAW value: {max_value}")

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL,
    )

    dashboard_width = (
        PREVIEW_WIDTH * 2
    )

    dashboard_height = (
        PREVIEW_HEIGHT
        + HISTOGRAM_HEIGHT
    )

    cv2.resizeWindow(
        WINDOW_NAME,
        dashboard_width,
        dashboard_height,
    )

    print("\nControls:")
    print("Q / Esc = quit")
    print("Automatic gain adapts from histogram balance")

    frame_index = 0
    auto_gain = 1.0

    while True:
        raw = frames[frame_index]

        histogram = calculate_histogram(
            raw,
            max_value,
        )

        statistics = histogram_statistics(histogram)

        processed_raw = process_raw(
            raw=raw,
            histogram=histogram,
            max_value=max_value,
        )

        processed_raw = np.clip(
            processed_raw,
            0,
            max_value,
        ).astype(
            np.uint16,
            copy=False,
        )

        processed_for_hist = np.clip(
            processed_raw.astype(np.float32) * auto_gain,
            0,
            max_value,
        ).astype(np.uint16, copy=False)

        histogram_processed = calculate_histogram(
            processed_for_hist,
            max_value,
        )

        auto_gain = estimate_auto_gain(
            histogram_processed,
            max_value,
            auto_gain,
        )

        processed_for_hist = np.clip(
            processed_raw.astype(np.float32) * auto_gain,
            0,
            max_value,
        ).astype(np.uint16, copy=False)

        histogram_processed = calculate_histogram(
            processed_for_hist,
            max_value,
        )

        balance_point = find_balance_point(
            histogram_processed,
            max_value,
        )

        raw_preview = raw_to_preview(
            raw,
            max_value,
            1.0,
        )

        processed_preview = raw_to_preview(
            processed_for_hist,
            max_value,
            1.0,
        )

        add_title(
            raw_preview,
            "Original RAW Bayer",
        )

        add_title(
            processed_preview,
            "Processed RAW Bayer",
        )

        # Draw two histograms: original and processed, side by side
        histogram_image_raw = draw_histogram(
            histogram,
            max_value,
            statistics,
            PREVIEW_WIDTH,
            HISTOGRAM_HEIGHT,
            label="Original RAW histogram",
        )

        statistics_processed = histogram_statistics(
            histogram_processed
        )

        histogram_image_processed = draw_histogram(
            histogram_processed,
            max_value,
            statistics_processed,
            PREVIEW_WIDTH,
            HISTOGRAM_HEIGHT,
            label="Processed RAW histogram",
        )

        histogram_image = np.hstack(
            (
                histogram_image_raw,
                histogram_image_processed,
            )
        )

        info = (
            f"frame {frame_index}/{frames.shape[0] - 1} | "
            f"recording {recording_dir.name} | "
            f"gain={auto_gain:.2f} | "
            f"balance={balance_point} | "
            f"mean={statistics['mean']:.1f} | "
            f"P50={statistics['p50']}"
        )

        cv2.putText(
            raw_preview,
            info,
            (
                10,
                PREVIEW_HEIGHT - 15,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (
                0,
                255,
                255,
            ),
            1,
            cv2.LINE_AA,
        )

        video_row = np.hstack(
            (
                raw_preview,
                processed_preview,
            )
        )

        dashboard = np.vstack(
            (
                video_row,
                histogram_image,
            )
        )

        cv2.imshow(
            WINDOW_NAME,
            dashboard,
        )

        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord("q"), ord("Q")):
            break

        if key == ord("n"):
            frame_index = min(frame_index + 1, frames.shape[0] - 1)
        elif key == ord("p"):
            frame_index = max(frame_index - 1, 0)
        else:
            frame_index += 1
            if frame_index >= frames.shape[0]:
                if args.loop:
                    frame_index = 0
                else:
                    frame_index = frames.shape[0] - 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
