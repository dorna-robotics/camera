"""
Grab one frame from the Hikrobot GigE camera (PoE) and save it.

Headless — saves hik.png next to the script instead of imshow, so it
works over SSH. Same get_all() 9-tuple as the D405/uEye: depth/ir
slots are None on this color-only device.
"""
import cv2

from camera import HikRobot

CAMERA_IP = "10.0.1.50"


def main():
    cam = HikRobot()
    if not cam.connect(ip=CAMERA_IP, raise_on_fail=True):
        raise SystemExit("connect failed")

    print("connected:", cam.serial_number, "at", cam.ip)
    print("stream_actual:", cam.stream_actual)
    print("exposure (us):", cam.get_exposure())

    # Same 9-tuple shape as the D405 — depth/ir slots come back None.
    _, _, _, _, _, color_img, depth_int, _, ts = cam.get_all()
    print("frame:", color_img.shape, "timestamp:", ts)
    print("K:", cam.get_K(), f"({cam.intr_source})")

    cv2.imwrite("hik.png", color_img)
    print("saved hik.png")

    cam.close()


if __name__ == "__main__":
    main()
