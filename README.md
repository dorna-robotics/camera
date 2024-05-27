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

### `connect(self, serial_number="", mode="rgbd", preset_path=None, filter={"spatial":[2, 0.5, 20], "temporal":[0.1, 40], "hole_filling":1})`
Connects to a RealSense device and configures the pipeline.

- **Parameters:**
  - `serial_number` (default=""): Serial number of the device to connect to.
  - `mode` (default="rgbd"): Mode of the camera, either `"rgbd"` or `"motion"` (used to capture the gyro data).
  - `preset_path` (default=None): Path to the preset configuration file.
  - `filter` (default={"spatial":[2, 0.5, 20], "temporal":[0.1, 40], "hole_filling":1}): Dictionary of filter settings. Apply post-processing filter to the depth data. For more information check [this link](https://dev.intelrealsense.com/docs/post-processing-filters)

- **Returns:** `True` if the connection is successful.

### `close(self)`
Stops the camera pipeline.

- **Returns:** `True` if the pipeline is stopped successfully.
> 🚨 **Notice:** It is necessary to always close the camera connection once your application is over. 
> 🚨 **Notice:** Multiple users can't access the camera at the same time (multiple connection is not supported).


### `all_device(self)`
Lists all connected RealSense devices.

- **Returns:** List of dictionaries containing device information (`name` and `serial_number`).

### `get_all(self, align_to=rs.stream.color)`
Captures all frames and images.

- **Parameters:**
  - `align_to` (default=rs.stream.color): Stream to align the depth frame to.

- **Returns:** Tuple containing `depth_frame`, `ir_frame`, `color_frame`, `depth_img`, `ir_img`, `color_img`, `depth_int`, `frames`, and `timestamp`.


### `camera_matrix(self, depth_int, ratio=1)`
Returns the camera matrix based on the provided depth intrinsic.

- **Parameters:**
  - `depth_int`: Depth intrinsic object.
  - `ratio` (default=1): Scaling ratio for the matrix.

- **Returns:** `np.array` representing the camera matrix.

### `xyz(self, pixel, depth_frame, depth_intrinsics)`
Calculates the real-world 3D coordinates from the pixel coordinates and depth frame, in camera coordinate system.

- **Parameters:**
  - `pixel`: A tuple of pixel coordinates `(x, y)`. `x` is the value in the width direction, and `y` is the value in the height direction.
  - `depth_frame`: The depth frame from which to extract the depth value.
  - `depth_intrinsics`: Depth intrinsic object.

- **Returns:** A tuple containing the 3D coordinates `(x, y, z)` in millimeter (`mm`) and the depth value at the given pixel.

### `dist_coeffs(self, depth_int)`
Returns the distortion coefficients of the camera.

- **Parameters:**
  - `depth_int`: Depth intrinsic object.

- **Returns:** `np.array` containing the distortion coefficients.

### `motion_rec(self)`
> 🚨 **Notice:** Only available on certain models that supports gyro data.
> 🚨 **Notice:** Only available when the `mode` parameter in the `connection` method is set to `"motion"`.

Starts recording motion data.

### `motion_stop(self)`
> 🚨 **Notice:** Only available on certain models that supports gyro data.
> 🚨 **Notice:** Only available when the `mode` parameter in the `connection` method is set to `"motion"`.

Stops recording motion data and retrieves the recorded data.

- **Returns:** Tuple containing lists of accelerometer data (`accel`) and gyroscope data (`gyro`).




## Methods
## 
### Config
The `config.json` file contains all the necessary configuration parameters for the camera. Read the `config.json` file as a Python dictionary and initiate the `camera` object. You can setup the path to the camera preset here as well. Go over the file to figure out different parameters.

### Methods
Here is the list of methods available for the `camera` object.

#### `.on()`
This method loads all the presets in the camera, enables the infrared, depth and RGB channels and starts the camera pipeline. 
> Call this method to initiate the camera object and before getting any data from the camera. 

#### `.off()`  
This method stops the camera pipeline. 
> Call this method when you don't need the camera object anymore.

#### `.get_all(save = False, align_to = rs.stream.color)`
This method returns frame, image and depth intrinsic data. 
```python
depth_frame, ir_frame, color_frame, depth_img, ir_img, color_img, depth_int = camera.get_all()
```

#### `.center(pxl, depth_frame, depth_int, r = 10, l = 20)`
Returns the XYZ Cartesian coordinate of the pixel point `pxl`, given the depth frame and depth intrinsics data, in respect to the camera coordinate system. If the XYZ data is not valid for `pxl`. Then the method searches over the pixels inside the circle with radius `r`  and centered around `pxl`, finds the first closest `l` pixels with valid XYZ and average over their XYZs to estimate the XYZ for `pxl`. 

#### `.length(pxl0, pxl1, depth_frame, depth_int, l =1000, start = 30)`
Returns the euclidean distance between the two pixel points `pxl0` and `pxl1` in the real world, given the depth frame and depth intrinsic data.

[dorna]: https://dorna.ai
[realsense]: https://www.intelrealsense.com
[pyrealsense]: https://github.com/IntelRealSense/librealsense