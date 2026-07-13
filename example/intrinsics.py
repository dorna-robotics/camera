"""
Print the camera matrix K and distortion coefficients D for the
connected D405 at the default stream configuration.

Intrinsics are a physical property of the lens + sensor and depend on
the stream resolution — running this again after connect(stream=...)
with a different width/height will print different numbers.
"""
import numpy as np

from camera import Camera


def main():
    cam = Camera()
    if not cam.connect():
        raise SystemExit("connect failed")

    # One get_all() call pulls the current intrinsics off the live frame
    # profile. depth_int is the returned rs.intrinsics object.
    _, _, _, _, _, _, depth_int, _, _ = cam.get_all()

    K = cam.camera_matrix(depth_int)
    D = cam.dist_coeffs(depth_int)

    print(f"stream:  {depth_int.width}x{depth_int.height}")
    print(f"model:   {depth_int.model}")
    print("K =")
    print(np.array2string(K, precision=4, suppress_small=True))
    print("D =")
    print(np.array2string(D, precision=6, suppress_small=True))

    cam.close()


if __name__ == "__main__":
    main()
