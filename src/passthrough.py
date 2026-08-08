#!/usr/bin/env python3

import time

import cv2
from picamera2 import Picamera2


WIDTH = 1280
HEIGHT = 720
WINDOW_NAME = "Camera passthrough"


def main() -> None:
    picam2 = Picamera2()

    config = picam2.create_video_configuration(
        main={
            "format": "RGB888",
            "size": (WIDTH, HEIGHT),
        },
        buffer_count=4,
    )

    picam2.configure(config)
    picam2.start()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    frame_count = 0
    fps = 0.0
    measurement_start = time.monotonic()

    try:
        while True:
            frame = picam2.capture_array("main")

            frame_count += 1
            now = time.monotonic()
            elapsed = now - measurement_start

            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                measurement_start = now

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    finally:
        picam2.stop()
        picam2.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
