#!/usr/bin/env python3

from pprint import pprint
from picamera2 import Picamera2


def main() -> None:
    picam2 = Picamera2()

    print("\n=== Свойства камеры ===")
    pprint(picam2.camera_properties)

    print("\n=== Режимы сенсора ===")
    pprint(picam2.sensor_modes)

    config = picam2.create_video_configuration(
        main={"format": "RGB888", "size": (1280, 720)}
    )
    picam2.configure(config)

    print("\n=== Доступные параметры управления ===")
    for name, limits in sorted(picam2.camera_controls.items()):
        print(f"{name}: {limits}")

    picam2.close()


if __name__ == "__main__":
    main()
