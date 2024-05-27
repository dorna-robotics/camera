# Camera 
This is the Python API repository that provides helper methods to use [D4XX Intel® RealSense™][realsense] camera. The API relies on [pyrealsense2][pyrealsense] library to get both RGB and depth frame data. In addition to that, the API offers methods to sync your RealSense camera with [Dorna 2 Robotic arm][dorna].


## Installation
Notice that the program has been tested only on Python 3.7+.

### 1- RealSense SDK
Download and install the [Intel® RealSense™ SDK 2.0](https://www.intelrealsense.com/sdk-2/) on the host computer.

### 2- Install camera library
Use `git clone` to download the `camera` repository, or simply download the [zip file](https://github.com/dorna-robotics/camera/archive/master.zip), and unzip the file.  
```bash
git clone https://github.com/dorna-robotics/camera.git
```
Next, go to the downloaded directory, where the `requirements.txt` file is located, and run:
```bash
# install requirements
pip install -r requirements.txt
```
Finally
```bash
pip install . --upgrade --force-reinstall
```

## Example usage
Use the `camera` module to connect to your camera and get the useful data (RGB, depth, etc.).

In this example, we import the `camera` module and show depth image.
``` python
from camera import Camera
import cv2


# create camera module
camera = Camera()

# connect to the camera 
camera.connect(filter={})

# display depth image in a while loop, until press "q"
while True:
    # get the camera data
    depth_frame, ir_frame, color_frame, depth_img, ir_img, color_img, depth_int, frames, timestamp = camera.get_all()
    
    # display depth image
    cv2.imshow("depth_img", depth_img)
    #cv2.imshow("color_img", color_img)

    # get xyz in camera frame of the center pixel in the image
    height, width = depth_img.shape
    xyz, sample = camera.xyz((width/2, height/2), depth_frame, depth_int)
    print("xyz: ", xyz)
    
    # exit if "q" is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# always close the camera connection, once your application is over
camera.close()
```  

## Methods
The `Camera` class is used to interface with a RealSense camera, providing methods for connecting to the camera, capturing frames, applying filters, and performing various operations on the captured data.

#### `connect(self, serial_number="", mode="rgbd", preset_path=None, filter={"spatial":[2, 0.5, 20], "temporal":[0.1, 40], "hole_filling":1})`
Connects to a RealSense device and configures the pipeline.

- **Parameters:**
  - `serial_number` (default=""): Serial number of the device to connect to.
  - `mode` (default="rgbd"): Mode of the camera, either `"rgbd"` or `"motion"` (used to capture the gyro data).
  - `preset_path` (default=None): Path to the preset configuration file.
  - `filter` (default={"spatial":[2, 0.5, 20], "temporal":[0.1, 40], "hole_filling":1}): Dictionary of filter settings. Apply post-processing filter to the depth data (for more information check [this link](https://dev.intelrealsense.com/docs/post-processing-filters)).

- **Returns:** `True` if the connection is successful.

#### `close(self)`
Stops the camera pipeline.

- **Returns:** `True` if the pipeline is stopped successfully.
> 🚨 **Notice:** It is necessary to always close the camera connection once your application is over.  

> 🚨 **Notice:** Multiple users can't access the camera at the same time (multiple connection is not supported).


#### `all_device(self)`
Lists all connected RealSense devices.

- **Returns:** List of dictionaries containing device information (`name` and `serial_number`).

#### `get_all(self, align_to=rs.stream.color)`
Captures all frames and images.

- **Parameters:**
  - `align_to` (default=rs.stream.color): Stream to align the depth frame to.

- **Returns:** Tuple containing `depth_frame`, `ir_frame`, `color_frame`, `depth_img`, `ir_img`, `color_img`, `depth_int`, `frames`, and `timestamp`.


#### `camera_matrix(self, depth_int, ratio=1)`
Returns the camera matrix based on the provided depth intrinsic.

- **Parameters:**
  - `depth_int`: Depth intrinsic object.
  - `ratio` (default=1): Scaling ratio for the matrix.

- **Returns:** `np.array` representing the camera matrix.


### `xyz(self, pxl, depth_frame, depth_int, wnd=(0,0), z_gt=(10, 2000))`

Convert camera pixel coordinates to its associated XYZ in the camera coordinate system.

- **Parameters:**
  - `pxl`: Tuple, Pixel coordinates to convert to XYZ. The pixel coordinate is given in `(pxl_x, pxl_y)` format where `pxl_x` is the pixel value in the width direction and `pxl_y` is the pixel value in the height direction, where `(0, 0)` is the top left corner of the image.
  - `depth_frame`: Depth frame from the camera.
  - `depth_int`: Intrinsic of the depth camera.
  - `wnd`: Tuple, Window size of the pixels for averaging (default: (0,0)).
  - `z_gt`: Tuple, ground truth range for valid depth values (default: (10, 2000)). If the minimum and maximum of `z_gt` is identical (`z_gt[0] == z_gt[1]`) then the method ignores the depth frame data and uses `z_gt` to estimate the XYZ.

- **Returns:**
  - `xyz`: Numpy array, XYZ value in camera coordinate, corresponding to the input pixel. If `xyz == np.array([0, 0, 0])` then it means that the `xyz` value is not valid. 
  - `sample`: List of tuples, Sample pixel coordinates and corresponding XYZ values used in the estimation.


### `xyz_estimate(self, pxl, pxl_ref, xyz_ref, method="plane")`

Find the XYZ coordinates of a given pixel with respect to a reference frame known to the user.

- **Parameters:**
  - `pxl`: Tuple, Pixel coordinates of the point to estimate XYZ for.
  - `pxl_ref`: List of tuples, Pixel coordinates of reference points in the reference frame.
  - `xyz_ref`: List of tuples, Corresponding XYZ coordinates of the reference points.
  - `method`: String, Method for estimating the XYZ coordinates. Options are `"plane"` (default, which is useful when the `xyz_ref` points are lying on a flat surface) or `"idw"` (stands for `inverse distance weighting`, which is a method used for spatial interpolation, where the value of a point is estimated based on the values of nearby known points, with weights assigned based on the inverse of their distances to the point being estimated).

- **Returns:**
  - `xyz`: Numpy array, Estimated XYZ coordinates of the input pixel.


#### `dist_coeffs(self, depth_int)`
Returns the distortion coefficients of the camera.

- **Parameters:**
  - `depth_int`: Depth intrinsic object.

- **Returns:** `np.array` containing the distortion coefficients.

#### `motion_rec(self)`
> 🚨 **Notice:** Only available on certain models that supports gyro data.

> 🚨 **Notice:** Only available when the `mode` parameter in the `connection` method is set to `"motion"`.

Starts recording motion data.

#### `motion_stop(self)`
> 🚨 **Notice:** Only available on certain models that supports gyro data.

> 🚨 **Notice:** Only available when the `mode` parameter in the `connection` method is set to `"motion"`.

Stops recording motion data and retrieves the recorded data.

- **Returns:** Tuple containing lists of accelerometer data (`accel`) and gyroscope data (`gyro`).

[dorna]: https://dorna.ai
[realsense]: https://www.intelrealsense.com
[pyrealsense]: https://github.com/IntelRealSense/librealsense