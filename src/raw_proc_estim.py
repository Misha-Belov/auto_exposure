#!/usr/bin/env python3
"""Оценка яркости RAW-записи по медиане гистограммы.

Exposure = K * ln(median / (max_value - median))
Clipping = K * ln((1 + balance) / (1 - balance))
balance = (вес белых - вес чёрных пикселей) / суммарный вес

Гистограмма учитывает положение: центр имеет вес 1, края — EDGE_WEIGHT.
Обе оценки используют эту взвешенную гистограмму.
Отрицательное значение — темно, положительное — светло.
Перед анализом вычитается SensorBlackLevels из metadata.jsonl.
Полезный диапазон RAW нормируется к 0..max_value.
Пиксели со значениями 0 и max_value исключаются из расчёта медианы.
Если других пикселей нет, оценка не определена (N/A).

Запуск: python3 src/raw_proc_estim.py --coefficient 1 --loop
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

if __package__:
    from . import raw_proc_f as recording
    from .raw_calibration import correct_black_level, load_black_levels
else:
    import raw_proc_f as recording
    from raw_calibration import correct_black_level, load_black_levels


WINDOW_NAME = "RAW Exposure Estimator"
EXPOSURE_COEFFICIENT = 100.0
CLIPPING_COEFFICIENT = 100.0
EDGE_WEIGHT = 0.1  # Вес границы относительно центра; 1.0 отключает приоритет.


def create_center_weights(
    shape: tuple[int, int],
    edge_weight: float = EDGE_WEIGHT,
) -> np.ndarray:
    """Плавно уменьшаем вес от центра к границам кадра.

    x и y нормированы к -1..1:
        weight = edge_weight + (1 - edge_weight) * (1 - x²) * (1 - y²)
    """
    if not 0 < edge_weight <= 1:
        raise ValueError("Вес края должен быть больше 0 и не больше 1")

    height, width = shape
    x = np.linspace(-1.0, 1.0, width) if width > 1 else np.zeros(1)
    y = np.linspace(-1.0, 1.0, height) if height > 1 else np.zeros(1)
    center_priority = np.outer(1.0 - y**2, 1.0 - x**2)

    return edge_weight + (1.0 - edge_weight) * center_priority


def calculate_weighted_histogram(
    raw: np.ndarray,
    max_value: int,
    weights: np.ndarray,
) -> np.ndarray:
    """Вместо количества пикселей суммируем их пространственные веса."""
    if raw.shape != weights.shape:
        raise ValueError("Размер карты весов должен совпадать с размером RAW")

    return np.bincount(
        raw.ravel(),
        weights=weights.ravel(),
        minlength=max_value + 1,
    )


def find_unclipped_median(
    histogram: np.ndarray,
    max_value: int,
) -> float:
    """Находим медиану по весам значений 1..max_value-1."""
    valid_histogram = histogram[1:max_value]
    count = valid_histogram.sum()

    if count == 0:
        return float("nan")

    cumulative = np.cumsum(valid_histogram)
    median_index = np.searchsorted(cumulative, count / 2.0)

    # Срез начинается со значения RAW=1, поэтому возвращаем смещение.
    return float(median_index + 1)


def estimate_exposure(
    histogram: np.ndarray,
    max_value: int,
    coefficient: float = EXPOSURE_COEFFICIENT,
) -> float:
    """Переводим медиану без клиппинга в оценку яркости без сглаживания."""
    median = find_unclipped_median(histogram, max_value)
    ratio = median / (max_value - median)
    exposure = coefficient * np.log(ratio)

    return float(exposure)


def estimate_clipping(
    histogram: np.ndarray,
    max_value: int,
    coefficient: float = CLIPPING_COEFFICIENT,
) -> float:
    """Оценка по разности взвешенных долей клиппинга.

    Чёрный кадр: -inf; белый: +inf; равные доли клиппинга: 0.
    При отсутствии клиппинга результат также равен 0.
    """
    total = float(histogram.sum())

    if total == 0:
        return float("nan")

    black_weight = float(histogram[0])
    white_weight = float(histogram[max_value])

    if black_weight == total:
        return -float("inf")

    if white_weight == total:
        return float("inf")

    balance = (white_weight - black_weight) / total
    ratio = (1.0 + balance) / (1.0 - balance)

    return float(coefficient * np.log(ratio))


def create_dashboard(
    raw: np.ndarray,
    histogram: np.ndarray,
    max_value: int,
    exposure: float,
    clipping: float,
    frame_index: int,
    frame_count: int,
) -> np.ndarray:
    """Показываем RAW после коррекции уровня чёрного и его гистограмму."""
    preview = recording.raw_to_preview(
        raw,
        max_value,
        display_gain=1.0,
    )

    exposure_text = f"{exposure:+.3f}" if np.isfinite(exposure) else "N/A"
    preview_title = (
        f"Exposure: {exposure_text} | "
        f"Frame {frame_index + 1}/{frame_count}"
    )
    recording.add_title(preview, preview_title)

    statistics = recording.histogram_statistics(histogram)
    graph = recording.draw_histogram(
        histogram,
        max_value,
        statistics,
        recording.PREVIEW_WIDTH,
        recording.HISTOGRAM_HEIGHT,
    )

    median = find_unclipped_median(histogram, max_value)
    median_text = f"{median:g}" if np.isfinite(median) else "N/A"
    clipping_text = "N/A" if np.isnan(clipping) else f"{clipping:+.3f}"
    graph_title = (
        f"Clipping: {clipping_text} | "
        f"Weighted P50={median_text}"
    )
    recording.add_title(graph, graph_title)

    return np.vstack((preview, graph))


def main() -> None:
    # Параметры запуска.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=recording.DEFAULT_DATA_ROOT,
    )
    parser.add_argument(
        "--recording",
        help="Имя директории записи",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Повторять запись",
    )
    parser.add_argument(
        "--coefficient",
        type=float,
        default=CLIPPING_COEFFICIENT,
        help="Положительный масштаб K",
    )
    args = parser.parse_args()

    if not np.isfinite(args.coefficient) or args.coefficient <= 0:
        parser.error("--coefficient должен быть конечным и положительным")

    # Загрузка RAW-записи.
    data_root = args.data_root.resolve()
    directories = recording.find_recording_dirs(data_root)
    directory = recording.select_recording(directories, args.recording)
    frames, session = recording.load_recording(directory)

    if frames.ndim != 3 or 0 in frames.shape:
        raise ValueError("Нужен непустой массив (кадры, высота, ширина)")

    if not np.issubdtype(frames.dtype, np.unsignedinteger):
        raise ValueError("RAW должен содержать беззнаковые целые числа")

    sensor = session.get("sensor", {})
    bit_depth = int(sensor.get("bit_depth", recording.PREFERRED_BIT_DEPTH))

    if not 1 <= bit_depth <= 16:
        raise ValueError("Поддерживается RAW от 1 до 16 бит")

    max_value = (1 << bit_depth) - 1
    frame_count = len(frames)
    frame_index = 0
    black_levels = load_black_levels(directory, bit_depth, frame_count)
    raw_format = sensor.get("raw_format", "")
    weights = create_center_weights(frames.shape[1:])

    print(f"Recording: {directory}")
    print(f"Coefficient: {args.coefficient}")
    print(f"Histogram weights: center=1, edge={EDGE_WEIGHT}")
    print(f"Native black levels, first frame: {black_levels[0]}")
    print("Negative = dark; positive = bright")
    print("Q/Esc: quit; N/P: next/previous frame")

    # Обработка и показ кадров.
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            raw = correct_black_level(
                frames[frame_index],
                max_value,
                black_levels[frame_index],
                raw_format,
            )
            histogram = calculate_weighted_histogram(
                raw,
                max_value,
                weights,
            )
            exposure = estimate_exposure(
                histogram,
                max_value,
                args.coefficient,
            )

            clipping = estimate_clipping(
                histogram,
                max_value,
                args.coefficient,
            )

            dashboard = create_dashboard(
                raw,
                histogram,
                max_value,
                exposure,
                clipping,
                frame_index,
                frame_count,
            )
            cv2.imshow(WINDOW_NAME, dashboard)

            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q"), ord("Q")):
                break

            if key in (ord("p"), ord("P")):
                frame_index = max(frame_index - 1, 0)
            elif key in (ord("n"), ord("N")):
                frame_index = min(frame_index + 1, frame_count - 1)
            elif frame_index + 1 < frame_count:
                frame_index += 1
            elif args.loop:
                frame_index = 0
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
