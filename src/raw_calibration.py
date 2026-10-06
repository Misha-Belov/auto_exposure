"""Коррекция уровня чёрного для сохранённого RAW из Picamera2."""

import json
from pathlib import Path

import numpy as np


# SensorBlackLevels перечислены в порядке R, Gr, Gb, B.
BAYER_CHANNELS = {
    "RGGB": ((0, 1), (2, 3)),
    "BGGR": ((3, 2), (1, 0)),
    "GRBG": ((1, 0), (3, 2)),
    "GBRG": ((2, 3), (0, 1)),
}


def load_black_levels(
    directory: Path,
    bit_depth: int,
    frame_count: int,
) -> np.ndarray:
    """Читаем уровни каждого кадра и переводим из 16 бит в разрядность RAW."""
    levels = np.full((frame_count, 4), np.nan, dtype=np.float64)

    with (directory / "metadata.jsonl").open(encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            index = record["frame_index"]
            black = record["metadata"].get("SensorBlackLevels")

            if black is None or len(black) != 4:
                raise ValueError(f"Нет SensorBlackLevels для кадра {index}")
            if not 0 <= index < frame_count:
                raise ValueError(f"Неверный индекс кадра в metadata.jsonl: {index}")
            if np.isfinite(levels[index]).all():
                raise ValueError(f"Повторный индекс кадра в metadata.jsonl: {index}")

            levels[index] = black

    # libcamera сообщает уровень в 16-битной шкале независимо от RAW.
    # Например, 4096 / 16 = 256 для 12 бит.
    levels /= 1 << (16 - bit_depth)
    max_value = (1 << bit_depth) - 1

    if not np.isfinite(levels).all():
        raise ValueError("В metadata.jsonl не хватает уровней чёрного для кадров")
    if np.any(levels < 0) or np.any(levels >= max_value):
        raise ValueError("Уровень чёрного выходит за допустимый диапазон RAW")

    return levels


def correct_black_level(
    raw: np.ndarray,
    max_value: int,
    black_levels: np.ndarray,
    raw_format: str,
) -> np.ndarray:
    """Переводим [black, max] в [0, max], сохраняя исходный массив.

    Формула для каждого Bayer-канала:
        corrected = clip((raw - black) * max / (max - black), 0, max)
    """
    # Вычитаем во float: uint16 иначе переполнится при raw < black.
    corrected = raw.astype(np.float64)

    if np.all(black_levels == black_levels[0]):
        black = black_levels[0]
        corrected -= black
        corrected *= max_value / (max_value - black)
    else:
        pattern = next(
            (name for name in BAYER_CHANNELS if name in raw_format),
            None,
        )
        if pattern is None:
            raise ValueError(f"Неизвестный Bayer-формат: {raw_format!r}")

        for row in range(2):
            for column in range(2):
                channel = BAYER_CHANNELS[pattern][row][column]
                black = black_levels[channel]
                pixels = corrected[row::2, column::2]
                pixels -= black
                pixels *= max_value / (max_value - black)

    np.clip(corrected, 0, max_value, out=corrected)
    np.rint(corrected, out=corrected)
    return corrected.astype(np.uint16)
