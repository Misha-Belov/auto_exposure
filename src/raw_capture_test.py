#!/usr/bin/env python3

import numpy as np
from picamera2 import Picamera2


PREFERRED_SIZE = (2028, 1520)
PREFERRED_BIT_DEPTH = 12


def select_mode(picam2):
    for mode in picam2.sensor_modes:
        if (
            mode["size"] == PREFERRED_SIZE
            and mode["bit_depth"] == PREFERRED_BIT_DEPTH
        ):
            return mode

    raise RuntimeError(
        f"Не найден режим "
        f"{PREFERRED_SIZE[0]}x{PREFERRED_SIZE[1]} "
        f"{PREFERRED_BIT_DEPTH} bit"
    )


def main():
    picam2 = Picamera2()

    mode = select_mode(picam2)

    print("Выбранный режим сенсора:")
    print(mode)

    config = picam2.create_video_configuration(
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
            "bit_depth": mode["bit_depth"],
        },

        buffer_count=4,
        display=None,
    )

    picam2.configure(config)

    actual_config = picam2.camera_configuration()

    print("\nФактически применённая конфигурация:")
    print(actual_config)

    raw_config = actual_config["raw"]

    width, height = raw_config["size"]
    stride = raw_config["stride"]
    raw_format = raw_config["format"]

    bit_depth = actual_config["sensor"]["bit_depth"]

    print("\nПараметры RAW:")
    print(f"format    = {raw_format}")
    print(f"size      = {width} x {height}")
    print(f"stride    = {stride} bytes")
    print(f"bit depth = {bit_depth}")

    if "PISP_COMP" in raw_format:
        raise RuntimeError(
            "Получен сжатый PISP RAW. "
            "Для прямой работы с пикселями нужен "
            "uncompressed Sxxxx16."
        )

    picam2.start()

    try:
        # Picamera2 возвращает RAW-буфер как байты.
        raw8 = picam2.capture_array("raw")

    finally:
        picam2.stop()
        picam2.close()

    print("\n=== Буфер, который вернул Picamera2 ===")
    print(f"shape = {raw8.shape}")
    print(f"dtype = {raw8.dtype}")
    print(f"min   = {raw8.min()}")
    print(f"max   = {raw8.max()}")

    # ВАЖНО:
    # Не astype(np.uint16)!
    # Нужно именно переинтерпретировать пары байтов
    # как одно 16-битное слово.
    raw_words = raw8.view(np.uint16)

    print("\n=== После view(np.uint16) ===")
    print(f"shape = {raw_words.shape}")
    print(f"dtype = {raw_words.dtype}")

    # Строка имеет padding.
    #
    # stride = 4096 bytes
    # 4096 / 2 = 2048 uint16
    #
    # но активных пикселей только 2028.
    raw16 = raw_words[:height, :width].copy()

    print("\n=== Активная область RAW ===")
    print(f"shape = {raw16.shape}")
    print(f"dtype = {raw16.dtype}")
    print(f"min   = {raw16.min()}")
    print(f"max   = {raw16.max()}")

    # На Raspberry Pi 5 unpacked RAW left-aligned.
    #
    # Для 12 bit:
    #
    # [12 значащих бит][0000]
    #
    # Поэтому возвращаем исходные 0..4095.
    shift = 16 - bit_depth

    raw_native = raw16 >> shift

    print("\n=== Исходные значения сенсора ===")
    print(f"bit depth = {bit_depth}")
    print(f"shift     = {shift}")
    print(f"shape     = {raw_native.shape}")
    print(f"dtype     = {raw_native.dtype}")
    print(f"min       = {raw_native.min()}")
    print(f"max       = {raw_native.max()}")
    print(
        f"theoretical max = {(1 << bit_depth) - 1}"
    )

    # Для 12-bit Pi 5 младшие 4 бита raw16
    # в нормальной ситуации должны быть нулями.
    low_mask = (1 << shift) - 1

    print(
        "\nМаксимум среди padding-битов:",
        np.max(raw16 & low_mask)
    )

    np.save(
        "raw_frame_pi5_16bit.npy",
        raw16
    )

    np.save(
        "raw_frame_native.npy",
        raw_native
    )

    # Также сохраним бинарные значения сенсора
    # без stride/padding.
    raw_native.tofile(
        "raw_frame_native.raw"
    )

    print("\nСохранено:")
    print("  raw_frame_pi5_16bit.npy")
    print("  raw_frame_native.npy")
    print("  raw_frame_native.raw")


if __name__ == "__main__":
    main()#!/usr/bin/env python3
import numpy as np
from picamera2 import Picamera2


PREFERRED_SIZE = (2028, 1520)
PREFERRED_BIT_DEPTH = 12


def select_mode(picam2: Picamera2) -> dict:
    for mode in picam2.sensor_modes:
        if (
            mode["size"] == PREFERRED_SIZE
            and mode["bit_depth"] == PREFERRED_BIT_DEPTH
        ):
            return mode

    print("Предпочтительный режим не найден.")
    print("Будет выбран первый доступный режим.")
    return picam2.sensor_modes[0]


def main() -> None:
    picam2 = Picamera2()
    mode = select_mode(picam2)

    print("Выбран режим:")
    print(mode)

    config = picam2.create_video_configuration(
        # Main-поток Picamera2 создаёт всегда.
        # Здесь он маленький и фактически не используется.
        main={
            "size": (320, 240),
            "format": "RGB888",
        },

        # Критически важно использовать mode["unpacked"].
        raw={
            "size": mode["size"],
            "format": mode["unpacked"],
        },

        sensor={
            "output_size": mode["size"],
            "bit_depth": mode["bit_depth"],
        },

        buffer_count=4,
        display=None,
    )

    picam2.configure(config)

    actual_config = picam2.camera_configuration()

    print
