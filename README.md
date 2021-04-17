# Camera 
This is the Python API repository that provides helper methods to use [D4XX Intel® RealSense™][realsense] camera. The API relies on [pyrealsense2][pyrealsense] library to get both RGB and depth frame data. In addition to that, the API offers methods to sync your RealSense camera with [Dorna 2 Robotic arm][dorna].


## Installation
Notice that the program has been tested only on Python 3.7+.

### Download
First, use `git clone` to download the repository:  
```bash
git clone https://github.com/dorna-robotics/camera.git
```
Or simply download the [zip file](https://github.com/dorna-robotics/camera/archive/master.zip), and uncompress the file.  

### Install
Next, go to the downloaded directory, where the `setup.py` file is located, and run:
```bash
python setup.py install --force
```

## Depth camera
Use the `camera` module to connect to your camera and get the useful data (RGB, depth, etc.).

### Example
In this example, we import the `camera` module and show depth and color data.
``` python
import json
import cv2
from camera import camera

with open("config.json") as json_file:
    arg = json.load(json_file)
camera = camera(arg)
camera.on()

# get 5 frames of depth and RGB data
for i in range(5):
    depth_frame, ir_frame, color_frame, depth_img, ir_img, color_img, depth_int = camera.get_all()
    
    print(np.asanyarray(depth_frame.get_data()).shape)   
    cv2.imshow("color",color_img) 
    cv2.imshow("depth",depth_img)
    cv2.waitKey(0)             
    
camera.off()
```  
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
This method returns frame, image and depth intrinsics data. 
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