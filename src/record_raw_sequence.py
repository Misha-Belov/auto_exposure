#!/usr/bin/env python3

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from picamera2 import Picamera2


# ============================================================
# НАСТРОЙКИ ЭКСПЕРИМЕНТА
# ============================================================

PREFERRED_SIZE = (2028, 1520)
PREFERRED_BIT_DEPTH = 12

# Продолжительность записываемого фрагмента.
DURATION_SECONDS = 5.0

# Для начала рекомендую 30 fps.
# У вашего режима IMX477 максимальная частота выше,
# но объём RAW получается очень большим.
TARGET_FPS = 30.0


# ============================================================
# ЭКСПОЗИЦИЯ
# ============================================================

# Вариант 1:
# None -> дать AE подобрать значения, потом зафиксировать.
#
# Вариант 2:
# указать конкретные значения вручную.
#
# Например:
# MANUAL_EXPOSURE_US = 10000
# MANUAL_ANALOGUE_GAIN = 1.0

MANUAL_EXPOSURE_US = None
MANUAL_ANALOGUE_GAIN = None

AE_WARMUP_SECONDS = 2.0

# После изменения параметров пропускаем несколько кадров,
# чтобы новые controls гарантированно вступили в силу.
CONTROL_SETTLE_FRAMES = 4


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_ROOT = Path.home() / "camera-lab" / "raw-recordings"

WRITE_SHA256 = True


# ============================================================
# JSON HELPERS
# ============================================================

def json_safe(value):
    """
    Приводит metadata Picamera2/libcamera
    к JSON-совместимому виду.
    """

    if value is None:
        return None

    if isinstance(value, (
        str,
        int,
        float,
        bool,
    )):
        return value

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(item)
            for item in value
        ]

    # Transform, ColorSpace, enums libcamera и т.п.
    return str(value)


# ============================================================
# CAMERA MODE
# ============================================================

def select_sensor_mode(picam2):
    for mode in picam2.sensor_modes:

        if (
            tuple(mode["size"]) == PREFERRED_SIZE
            and int(mode["bit_depth"]) == PREFERRED_BIT_DEPTH
        ):
            return mode

    print("\nДоступные sensor modes:\n")

    for index, mode in enumerate(picam2.sensor_modes):
        print(
            f"[{index}] "
            f"{mode['size']} "
            f"{mode['bit_depth']} bit "
            f"{mode['fps']:.2f} fps "
            f"unpacked={mode.get('unpacked')}"
        )

    raise RuntimeError(
        "Требуемый режим сенсора не найден"
    )


# ============================================================
# RAW BUFFER
# ============================================================

def raw_buffer_to_words(raw_buffer):
    """
    Picamera2 для SBGGR16 может вернуть:

        dtype = uint8
        shape = (height, stride_bytes)

    Пары байтов на самом деле являются uint16.
    """

    if raw_buffer.dtype == np.uint8:

        raw_buffer = np.ascontiguousarray(
            raw_buffer
        )

        return raw_buffer.view("<u2")

    if raw_buffer.dtype == np.uint16:

        return raw_buffer

    raise RuntimeError(
        f"Неожиданный RAW dtype: {raw_buffer.dtype}"
    )


# ============================================================
# RAM CHECK
# ============================================================

def available_memory_bytes():
    try:
        with open(
            "/proc/meminfo",
            "r",
            encoding="utf-8",
        ) as file:

            for line in file:

                if line.startswith("MemAvailable:"):

                    value_kib = int(
                        line.split()[1]
                    )

                    return (
                        value_kib * 1024
                    )

    except OSError:
        pass

    return None


def human_bytes(value):
    units = [
        "B",
        "KiB",
        "MiB",
        "GiB",
        "TiB",
    ]

    value = float(value)

    for unit in units:

        if value < 1024:
            return f"{value:.2f} {unit}"

        value /= 1024

    return f"{value:.2f} PiB"


# ============================================================
# CHECKSUM
# ============================================================

def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:

        while True:

            block = file.read(
                8 * 1024 * 1024
            )

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


# ============================================================
# MAIN
# ============================================================

def main():
    picam2 = Picamera2()

    mode = select_sensor_mode(picam2)

    width, height = mode["size"]

    bit_depth = int(
        mode["bit_depth"]
    )

    maximum_code = (
        1 << bit_depth
    ) - 1

    max_sensor_fps = float(
        mode["fps"]
    )

    if TARGET_FPS > max_sensor_fps:

        raise RuntimeError(
            f"TARGET_FPS={TARGET_FPS} превышает "
            f"максимум режима {max_sensor_fps:.2f}"
        )

    frame_duration_us = round(
        1_000_000 / TARGET_FPS
    )

    frame_count = round(
        DURATION_SECONDS * TARGET_FPS
    )

    bytes_per_frame = (
        width
        * height
        * np.dtype("<u2").itemsize
    )

    frames_bytes = (
        bytes_per_frame
        * frame_count
    )

    print("\n================================")
    print("RAW RECORDING")
    print("================================")

    print(
        f"Sensor:      "
        f"{width}x{height}"
    )

    print(
        f"Bit depth:   "
        f"{bit_depth}"
    )

    print(
        f"Range:       "
        f"0...{maximum_code}"
    )

    print(
        f"FPS:         "
        f"{TARGET_FPS}"
    )

    print(
        f"Duration:    "
        f"{DURATION_SECONDS} s"
    )

    print(
        f"Frames:      "
        f"{frame_count}"
    )

    print(
        f"RAW/frame:   "
        f"{human_bytes(bytes_per_frame)}"
    )

    print(
        f"RAM needed:  "
        f"{human_bytes(frames_bytes)}"
    )

    # --------------------------------------------------------
    # Проверяем RAM
    # --------------------------------------------------------

    available = available_memory_bytes()

    if available is not None:

        print(
            f"RAM free:    "
            f"{human_bytes(available)}"
        )

        # Не разрешаем съесть почти всю память системы.
        if frames_bytes > available * 0.70:

            raise RuntimeError(
                "\nДля записи недостаточно свободной RAM.\n"
                "Уменьшите DURATION_SECONDS или TARGET_FPS.\n"
                f"Нужно примерно {human_bytes(frames_bytes)}, "
                f"доступно {human_bytes(available)}."
            )

    # --------------------------------------------------------
    # Создаём большой массив заранее.
    #
    # Это важно:
    # во время записи не делаем append и не меняем размер.
    # --------------------------------------------------------

    print("\nВыделяем RAM...")

    frames = np.empty(
        (
            frame_count,
            height,
            width,
        ),
        dtype="<u2",
    )

    sensor_timestamps = np.zeros(
        frame_count,
        dtype=np.int64,
    )

    host_timestamps = np.zeros(
        frame_count,
        dtype=np.int64,
    )

    metadata_records = []

    # --------------------------------------------------------
    # Camera config
    # --------------------------------------------------------

    config = picam2.create_video_configuration(

        # Picamera2 нужен main stream.
        # Нам он практически не нужен,
        # поэтому делаем маленьким.
        main={
            "size": (
                320,
                240,
            ),
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

        controls={
            "FrameDurationLimits": (
                frame_duration_us,
                frame_duration_us,
            ),
        },

        buffer_count=6,

        display=None,
    )

    picam2.configure(
        config
    )

    actual_config = (
        picam2.camera_configuration()
    )

    raw_config = actual_config["raw"]

    raw_width, raw_height = (
        raw_config["size"]
    )

    raw_format = str(
        raw_config["format"]
    )

    stride = int(
        raw_config["stride"]
    )

    print("\nActual RAW configuration:")

    print(
        f"format: {raw_format}"
    )

    print(
        f"size:   "
        f"{raw_width}x{raw_height}"
    )

    print(
        f"stride: {stride}"
    )

    if "PISP_COMP" in raw_format:

        raise RuntimeError(
            "Получен compressed PISP RAW. "
            "Нужен unpacked RAW."
        )

    shift = (
        16 - bit_depth
    )

    # --------------------------------------------------------
    # Start camera
    # --------------------------------------------------------

    picam2.start()

    try:

        # ====================================================
        # EXPOSURE
        # ====================================================

        if (
            MANUAL_EXPOSURE_US is None
            or MANUAL_ANALOGUE_GAIN is None
        ):

            print(
                "\nAuto exposure warmup..."
            )

            time.sleep(
                AE_WARMUP_SECONDS
            )

            auto_metadata = (
                picam2.capture_metadata()
            )

            exposure_us = int(
                auto_metadata.get(
                    "ExposureTime",
                    10000,
                )
            )

            analogue_gain = float(
                auto_metadata.get(
                    "AnalogueGain",
                    1.0,
                )
            )

            print(
                f"AE selected ExposureTime: "
                f"{exposure_us} us"
            )

            print(
                f"AE selected AnalogueGain: "
                f"{analogue_gain}"
            )

        else:

            exposure_us = int(
                MANUAL_EXPOSURE_US
            )

            analogue_gain = float(
                MANUAL_ANALOGUE_GAIN
            )

        # Ограничиваем exposure продолжительностью кадра.
        #
        # Оставляем небольшой запас.
        maximum_reasonable_exposure = (
            frame_duration_us - 100
        )

        if exposure_us > maximum_reasonable_exposure:

            print(
                "\nWARNING:"
            )

            print(
                f"ExposureTime {exposure_us} us "
                f"слишком велик для {TARGET_FPS} fps."
            )

            exposure_us = (
                maximum_reasonable_exposure
            )

            print(
                f"Используем "
                f"{exposure_us} us"
            )

        # ----------------------------------------------------
        # Freeze camera controls
        # ----------------------------------------------------

        picam2.set_controls(
            {
                "AeEnable": False,
                "ExposureTime": exposure_us,
                "AnalogueGain": analogue_gain,
                "FrameDurationLimits": (
                    frame_duration_us,
                    frame_duration_us,
                ),
            }
        )

        print("\nFixed controls:")

        print(
            f"ExposureTime:       "
            f"{exposure_us} us"
        )

        print(
            f"AnalogueGain:       "
            f"{analogue_gain}"
        )

        print(
            f"FrameDuration:      "
            f"{frame_duration_us} us"
        )

        # Даём controls примениться.
        for _ in range(
            CONTROL_SETTLE_FRAMES
        ):
            picam2.capture_metadata()

        # ====================================================
        # RECORD
        # ====================================================

        print("\n================================")
        print("RECORDING...")
        print("================================\n")

        recording_start = (
            time.perf_counter()
        )

        for index in range(
            frame_count
        ):

            with picam2.captured_request() as request:

                # Важно делать преобразование,
                # пока request ещё удерживает buffer.
                raw_buffer = request.make_array(
                    "raw"
                )

                raw_words = (
                    raw_buffer_to_words(
                        raw_buffer
                    )
                )

                active_raw16 = raw_words[
                    :raw_height,
                    :raw_width
                ]

                # Pi 5:
                #
                # 12-bit SBGGR16:
                #
                # xxxxxxxxxxxx0000
                #
                # Переводим без потерь в 0...4095.
                np.right_shift(
                    active_raw16,
                    shift,
                    out=frames[index],
                )

                metadata = (
                    request.get_metadata()
                )

                host_timestamp_ns = (
                    time.monotonic_ns()
                )

            sensor_timestamp_ns = int(
                metadata.get(
                    "SensorTimestamp",
                    0,
                )
            )

            sensor_timestamps[index] = (
                sensor_timestamp_ns
            )

            host_timestamps[index] = (
                host_timestamp_ns
            )

            metadata_records.append(
                {
                    "frame_index": index,
                    "sensor_timestamp_ns":
                        sensor_timestamp_ns,
                    "host_monotonic_ns":
                        host_timestamp_ns,
                    "metadata":
                        json_safe(metadata),
                }
            )

            if (
                index == 0
                or (index + 1) % 30 == 0
                or index + 1 == frame_count
            ):

                print(
                    f"\rCaptured "
                    f"{index + 1}/{frame_count}",
                    end="",
                    flush=True,
                )

        recording_elapsed = (
            time.perf_counter()
            - recording_start
        )

        print()

    finally:

        picam2.stop()
        picam2.close()

    actual_fps = (
        frame_count
        / recording_elapsed
    )

    print(
        f"\nCapture finished in "
        f"{recording_elapsed:.3f} s"
    )

    print(
        f"Measured application FPS: "
        f"{actual_fps:.3f}"
    )

    # ========================================================
    # Проверка RAW
    # ========================================================

    print("\nRAW statistics:")

    print(
        f"dtype: "
        f"{frames.dtype}"
    )

    print(
        f"shape: "
        f"{frames.shape}"
    )

    print(
        f"minimum: "
        f"{frames.min()}"
    )

    print(
        f"maximum: "
        f"{frames.max()}"
    )

    print(
        f"mean: "
        f"{frames.mean():.2f}"
    )

    if frames.max() > maximum_code:

        raise RuntimeError(
            "RAW содержит значения "
            "выше допустимого диапазона."
        )

    # ========================================================
    # Создаём output directory
    # ========================================================

    session_name = (
        "capture_"
        + datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    output_directory = (
        OUTPUT_ROOT
        / session_name
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    print(
        f"\nSaving to:\n"
        f"{output_directory}"
    )

    # ========================================================
    # frames.npy
    # ========================================================

    frames_path = (
        output_directory
        / "frames.npy"
    )

    print(
        "\nSaving frames.npy..."
    )

    np.save(
        frames_path,
        frames,
        allow_pickle=False,
    )

    # ========================================================
    # timestamps
    # ========================================================

    np.save(
        output_directory
        / "sensor_timestamps_ns.npy",
        sensor_timestamps,
        allow_pickle=False,
    )

    np.save(
        output_directory
        / "host_timestamps_ns.npy",
        host_timestamps,
        allow_pickle=False,
    )

    # ========================================================
    # metadata.jsonl
    # ========================================================

    metadata_path = (
        output_directory
        / "metadata.jsonl"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:

        for record in metadata_records:

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
            )

            file.write("\n")

    # ========================================================
    # session.json
    # ========================================================

    session = {
        "version": 1,

        "sensor": {
            "size": [
                width,
                height,
            ],

            "bit_depth":
                bit_depth,

            "maximum_code":
                maximum_code,

            "raw_format":
                raw_format,

            "stride_bytes":
                stride,

            "shift_from_pi5_container":
                shift,
        },

        "recording": {
            "requested_duration_seconds":
                DURATION_SECONDS,

            "requested_fps":
                TARGET_FPS,

            "frame_count":
                frame_count,

            "actual_capture_duration_seconds":
                recording_elapsed,

            "application_fps":
                actual_fps,
        },

        "controls": {
            "AeEnable":
                False,

            "ExposureTime":
                exposure_us,

            "AnalogueGain":
                analogue_gain,

            "FrameDurationLimits": [
                frame_duration_us,
                frame_duration_us,
            ],
        },

        "numpy": {
            "frames_shape":
                list(frames.shape),

            "frames_dtype":
                str(frames.dtype),

            "value_range": [
                0,
                maximum_code,
            ],
        },

        "picamera2_configuration":
            json_safe(actual_config),
    }

    with open(
        output_directory
        / "session.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            session,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # SHA256
    # ========================================================

    if WRITE_SHA256:

        print("\nCalculating SHA256...")

        files_to_hash = [
            "frames.npy",
            "sensor_timestamps_ns.npy",
            "host_timestamps_ns.npy",
            "metadata.jsonl",
            "session.json",
        ]

        with open(
            output_directory
            / "sha256.txt",
            "w",
            encoding="utf-8",
        ) as checksum_file:

            for filename in files_to_hash:

                path = (
                    output_directory
                    / filename
                )

                digest = (
                    sha256_file(path)
                )

                checksum_file.write(
                    f"{digest}  {filename}\n"
                )

                print(
                    f"{digest}  {filename}"
                )

    # ========================================================
    # FINAL
    # ========================================================

    print("\n================================")
    print("DONE")
    print("================================")

    print(
        f"\nRecording:\n"
        f"{output_directory}"
    )

    print(
        f"\nframes.npy:\n"
        f"{human_bytes(frames_path.stat().st_size)}"
    )

    print(
        "\nДля загрузки:"
    )

    print(
        "frames = np.load("
        f"'{frames_path}', "
        "mmap_mode='r')"
    )


if __name__ == "__main__":
    main()
