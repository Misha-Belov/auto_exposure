#!/usr/bin/env python3

import time

import cv2
import numpy as np
from picamera2 import Picamera2


# ============================================================
# CAMERA SETTINGS
# ============================================================

PREFERRED_SIZE = (2028, 1520)
PREFERRED_BIT_DEPTH = 12

WINDOW_NAME = "RAW Histogram Processor"

# Размер каждого видео на экране.
PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 480

# Высота гистограммы.
HISTOGRAM_HEIGHT = 300

# Зафиксировать автоматическую экспозицию после прогрева.
FREEZE_EXPOSURE = True
AE_WARMUP_SECONDS = 1.5


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
    #     weights=histogram
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
# CAMERA MODE
# ============================================================

def select_sensor_mode(picam2: Picamera2) -> dict:
    """
    Находим режим 2028x1520 / 12 bit.
    """

    for mode in picam2.sensor_modes:

        if (
            tuple(mode["size"]) == PREFERRED_SIZE
            and int(mode["bit_depth"]) == PREFERRED_BIT_DEPTH
        ):
            return mode

    print("Доступные режимы:")

    for index, mode in enumerate(picam2.sensor_modes):

        print(
            index,
            mode["size"],
            mode["bit_depth"],
            mode["fps"],
            mode.get("unpacked"),
        )

    raise RuntimeError(
        "Не найден требуемый режим сенсора"
    )


# ============================================================
# UNPACK RASPBERRY PI 5 RAW
# ============================================================

def unpack_raw(
    raw_buffer: np.ndarray,
    width: int,
    height: int,
    bit_depth: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Picamera2 может вернуть SBGGR16 как uint8 buffer.

    Например:

        shape = (1520, 4096)
        dtype = uint8

    Фактически каждые два байта являются одним uint16.

    Raspberry Pi 5 хранит 12-bit RAW в старших битах:

        xxxxxxxxxxxx0000

    Поэтому:

        raw12 = raw16 >> 4
    """

    if raw_buffer.dtype == np.uint8:

        raw_buffer = np.ascontiguousarray(
            raw_buffer
        )

        raw16_with_padding = raw_buffer.view(
            "<u2"
        )

    elif raw_buffer.dtype == np.uint16:

        raw16_with_padding = raw_buffer

    else:

        raise RuntimeError(
            f"Неизвестный RAW dtype: {raw_buffer.dtype}"
        )

    # Убираем padding в конце строк.
    raw16 = raw16_with_padding[
        :height,
        :width
    ]

    shift = 16 - bit_depth

    # Получаем настоящий диапазон сенсора:
    #
    # 12-bit → 0...4095
    raw_native = raw16 >> shift

    return raw16, raw_native


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
# HISTOGRAM DRAWING
# ============================================================

def draw_histogram(
    histogram: np.ndarray,
    max_value: int,
    statistics: dict,
    width: int,
    height: int,
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

        reduced[i] = histogram[
            start:end
        ].sum()

    # --------------------------------------------------------
    # Логарифмическая Y шкала.
    # --------------------------------------------------------

    reduced = np.log1p(
        reduced
    )

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
    ).astype(
        np.int32
    )

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
        f"RAW histogram 0..{max_value}     "
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

def main():

    picam2 = Picamera2()

    mode = select_sensor_mode(
        picam2
    )

    width, height = mode["size"]

    bit_depth = int(
        mode["bit_depth"]
    )

    max_value = (
        1 << bit_depth
    ) - 1

    print(
        "\nSelected sensor mode:"
    )

    print(
        f"size: {width}x{height}"
    )

    print(
        f"bit depth: {bit_depth}"
    )

    print(
        f"max RAW value: {max_value}"
    )

    print(
        f"fps: {mode['fps']:.2f}"
    )

    print(
        f"unpacked: {mode['unpacked']}"
    )

    # --------------------------------------------------------
    # Camera configuration
    # --------------------------------------------------------

    config = picam2.create_video_configuration(

        # Picamera2 требует основной stream.
        # Мы его не используем.
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

        buffer_count=4,

        display=None,
    )

    picam2.configure(
        config
    )

    actual_config = (
        picam2.camera_configuration()
    )

    raw_config = actual_config[
        "raw"
    ]

    raw_width, raw_height = (
        raw_config["size"]
    )

    print(
        "\nActual RAW configuration:"
    )

    print(
        raw_config
    )

    if "PISP_COMP" in str(
        raw_config["format"]
    ):
        raise RuntimeError(
            "Получен compressed PISP RAW."
        )

    # --------------------------------------------------------
    # Start
    # --------------------------------------------------------

    picam2.start()

    # --------------------------------------------------------
    # Freeze exposure
    # --------------------------------------------------------

    if FREEZE_EXPOSURE:

        print(
            "\nWaiting for auto exposure..."
        )

        time.sleep(
            AE_WARMUP_SECONDS
        )

        metadata = (
            picam2.capture_metadata()
        )

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
            f"ExposureTime = "
            f"{exposure} us"
        )

        print(
            f"AnalogueGain = "
            f"{analogue_gain:.3f}"
        )

    # --------------------------------------------------------
    # GUI
    # --------------------------------------------------------

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

    cv2.createTrackbar(
        "Display gain x100",
        WINDOW_NAME,
        100,
        800,
        lambda x: None,
    )

    print(
        "\nControls:"
    )

    print(
        "Q / Esc = quit"
    )

    print(
        "Display gain changes "
        "only visualization"
    )

    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    frame_counter = 0

    fps = 0.0

    fps_start = (
        time.perf_counter()
    )

    try:

        while True:

            loop_start = (
                time.perf_counter()
            )

            # =================================================
            # CAPTURE RAW
            # =================================================

            with picam2.captured_request() as request:

                raw_buffer = (
                    request.make_array(
                        "raw"
                    )
                )

                metadata = (
                    request.get_metadata()
                )

            raw16, raw = unpack_raw(
                raw_buffer,
                raw_width,
                raw_height,
                bit_depth,
            )

            # =================================================
            # HISTOGRAM
            # =================================================

            histogram = (
                calculate_histogram(
                    raw,
                    max_value,
                )
            )

            statistics = (
                histogram_statistics(
                    histogram
                )
            )

            # =================================================
            # USER PROCESSING
            # =================================================

            processed_raw = process_raw(
                raw=raw,
                histogram=histogram,
                max_value=max_value,
            )

            # Важная защита:
            # пользовательская функция должна вернуть
            # допустимые 12-bit значения.
            processed_raw = np.clip(
                processed_raw,
                0,
                max_value,
            ).astype(
                np.uint16,
                copy=False,
            )

            # =================================================
            # VISUALIZATION
            # =================================================

            display_gain = max(
                cv2.getTrackbarPos(
                    "Display gain x100",
                    WINDOW_NAME,
                )
                / 100.0,
                0.01,
            )

            raw_preview = raw_to_preview(
                raw,
                max_value,
                display_gain,
            )

            processed_preview = (
                raw_to_preview(
                    processed_raw,
                    max_value,
                    display_gain,
                )
            )

            add_title(
                raw_preview,
                "Original RAW Bayer",
            )

            add_title(
                processed_preview,
                "Processed RAW Bayer",
            )

            histogram_image = (
                draw_histogram(
                    histogram,
                    max_value,
                    statistics,
                    dashboard_width,
                    HISTOGRAM_HEIGHT,
                )
            )

            # =================================================
            # FPS
            # =================================================

            frame_counter += 1

            now = time.perf_counter()

            elapsed = (
                now - fps_start
            )

            if elapsed >= 1.0:

                fps = (
                    frame_counter
                    / elapsed
                )

                frame_counter = 0

                fps_start = now

            loop_ms = (
                time.perf_counter()
                - loop_start
            ) * 1000.0

            # -------------------------------------------------
            # Overlay information
            # -------------------------------------------------

            info = (
                f"FPS {fps:.1f} | "
                f"frame {loop_ms:.1f} ms | "
                f"Exp {metadata.get('ExposureTime', '?')} us | "
                f"Gain {metadata.get('AnalogueGain', '?')}"
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

            # =================================================
            # DASHBOARD
            # =================================================

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

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key in (
                27,
                ord("q"),
                ord("Q"),
            ):
                break

    finally:

        picam2.stop()

        picam2.close()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
