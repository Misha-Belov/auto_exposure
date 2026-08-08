#!/usr/bin/env python3

from picamera2 import Picamera2


def main() -> None:
    picam2 = Picamera2()

    print("Доступные режимы сенсора:\n")

    for index, mode in enumerate(picam2.sensor_modes):
        print(
            f"[{index}] "
            f"size={mode['size']}, "
            f"depth={mode['bit_depth']} бит, "
            f"fps={mode['fps']:.2f}, "
            f"packed={mode['format']}, "
            f"unpacked={mode['unpacked']}, "
            f"crop={mode['crop_limits']}"
        )

    picam2.close()


if __name__ == "__main__":
    main()
