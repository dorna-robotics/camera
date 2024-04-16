import time
import pyrealsense2 as rs
import numpy as np
import math
import cv2
import json
import os
"""
Euclidean dimension is in mm
"""

class Helper(object):
    """docstring for helper"""
    def __init__(self):
        super(Helper, self).__init__()

        self.filter = None

    """
    convert camera pixel to xyz
    Use linear model to estimate the center of the
    linear model: xyz = T*pixel + B => change it to a overdetermined model Ax = b and find x ()
    wnd : the window size of the pixels
    """
    def xyz(self, pxl, depth_frame, depth_int, wnd = (0,0), z_min = 10, z_max = 2000): 
        if type(self.filter) == dict and "decimate" in self.filter:
            pxl = [x/self.filter["decimate"] for x in pxl]

        sample_pxl = []
        sample_xyz = []
        xyz = np.array([0. for i in range(3)])

        # number of row and column
        dim = np.asanyarray(depth_frame.get_data()).shape

        # make a copy and pixel to int
        pxl_org = np.array(pxl)
        pxl = np.array(pxl).astype(int)

        # lattice
        lattice = np.array([[x,y] for x in range(math.floor(-wnd[0]/2), math.floor(wnd[0]/2)+1) for y in range(math.floor(-wnd[1]/2), math.floor(wnd[1]/2)+1)])
        for l in lattice:
            try:
                pxl_new = pxl + l 
                # make sure pixels are not out of range
                if any([pxl_new[0] < 0, pxl_new[0] >= dim[1], pxl_new[1] < 0, pxl_new[1] >= dim[0]]):
                    continue

                # calculate xyz
                xyz_new = rs.rs2_deproject_pixel_to_point(depth_int, pxl_new.tolist(), depth_frame.get_distance(pxl_new[0], pxl_new[1]))
                xyz_new  = 1000 * np.array(xyz_new) # in mm  

                if xyz_new[2] > z_max or xyz_new[2] < z_min:
                    continue

                sample_pxl.append(pxl_new)
                sample_xyz.append(xyz_new)

            except Exception as ex:
                print("error: ", ex)
                pass

        if list(wnd) == [0, 0] and len(sample_xyz) == 1: 
                xyz = sample_xyz[0]

        # there are enough data to run the over determined system
        elif len(sample_pxl):
            A = np.zeros(9)
            b = np.zeros(1)
            for p in sample_pxl:
                t0 = np.hstack((p, [0, 0], [0, 0], [1, 0, 0]))
                t1 = np.hstack(([0, 0],p, [0, 0], [0, 1, 0]))
                t2 = np.hstack(([0, 0], [0, 0],p, [0, 0, 1]))  
                A = np.vstack((A, t0, t1, t2))
            A = np.delete(A, (0), axis=0)               
            
            for xyz in sample_xyz:
                b = np.vstack((b, xyz[0], xyz[1], xyz[2]))
            b = np.delete(b, (0), axis=0)   

            # Ax = b
            x = np.linalg.lstsq(A,b, rcond=None)[0] # computing the numpy solution

            T = np.hstack((x[0:2], x[2:4], x[4:6]))
            B = np.transpose(x[6:9])

            xyz = np.matrix(pxl_org) * T + B
            xyz = np.asarray(xyz).reshape(-1)

        return xyz, [v for v in zip(sample_pxl, sample_xyz)]


    """
    give two pixels
    find the distance between them
    notice that the depth data is usually not valid around  edges
    We start from the two edges and find the distance between the two valid pixels on the line connecting the two edges.

    """
    def length(self, pxl0, pxl1, depth_frame, depth_int, l =1000, start = 30):
        pxl0 = np.array(pxl0)
        pxl1 = np.array(pxl1)

        xyz0 = np.array([0. for i in range(3)])
        xyz1 = np.array([0. for i in range(3)])

        # xyz0        
        for t0 in range(start, l):   
            p = pxl0 + t0/l * (pxl1 - pxl0)

            xyz_t = self.xyz(p, depth_frame, depth_int)
            if sum(xyz_t == [0, 0, 0]) < len(xyz_t):
                xyz0 = xyz_t
                break 

        # xyz0        
        for t1 in range(start, l):   
            p = pxl1 + t1/l * (pxl0 - pxl1)

            xyz_t = self.xyz(p, depth_frame, depth_int)
            if sum(xyz_t == [0, 0, 0]) < len(xyz_t):
                xyz1 = xyz_t
                break 

        distance = 0
        try:
            #distance = np.linalg.norm(xyz0 - xyz1) * l / (l-t0-t1)
            distance = np.linalg.norm(xyz0[0:2] - xyz1[0:2]) * l / (l-t0-t1) 
        except Exception as ex:
            pass

        return distance

    def draw_axis(self, img, corner_pixel, axis_pixel):
        # bgr color
        img = cv2.line(img, tuple(np.array(corner_pixel).astype(int)), tuple(np.array(axis_pixel[0]).astype(int)), (0,0,255), 1)
        img = cv2.line(img, tuple(np.array(corner_pixel).astype(int)), tuple(np.array(axis_pixel[1]).astype(int)), (0,255,0), 1)
        img = cv2.line(img, tuple(np.array(corner_pixel).astype(int)), tuple(np.array(axis_pixel[2]).astype(int)), (255,0,0), 1)
        return img

    def mouse_click(self, event, x, y, flags, param):
        if not event == cv2.EVENT_LBUTTONDOWN:
            return None
        print("clicked: ", (int(x), int(y)))


class Camera(Helper):
    """docstring for ClassName"""
    def __init__(self):
        super(Camera, self).__init__()


    def camera_matrix(self, depth_int):
        ratio = 1
        if type(self.filter) == dict and "decimate" in self.filter:
            ratio = self.filter["decimate"]

        return np.array([[ratio*depth_int.fx,   0.        , ratio*depth_int.ppx],
                                [  0.        , ratio*depth_int.fy, ratio*depth_int.ppy],
                                [  0.        ,   0.        ,   1.        ]])

    def dist_coeffs(self, depth_int):
        return np.array(depth_int.coeffs)


    def frame(self, align_to=rs.stream.color, time_out=10):
        # Get frameset of ir and depth
        frames = self.pipeline.wait_for_frames(1000 * time_out)

        # Create an align object
        align = rs.align(align_to)

        # Align the depth frame to ir frame
        aligned_frames = align.process(frames)
        
        # frames
        depth_frame = frames.get_depth_frame()
        ir_frame = frames.get_infrared_frame()
        color_frame = frames.get_color_frame()
        # filters
        if self.decimate:
            depth_frame = self.decimate.process(depth_frame)
        if self.depth_to_disparity:
            depth_frame = self.depth_to_disparity.process(depth_frame)
        if self.spatial:
            depth_frame = self.spatial.process(depth_frame)
        if self.temporal:
            depth_frame = self.temporal.process(depth_frame)
        if self.disparity_to_depth:
            depth_frame = self.disparity_to_depth.process(depth_frame)
        if self.hole_filling:
            depth_frame = self.hole_filling.process(depth_frame)

        depth_frame = depth_frame.as_depth_frame()
        # Get aligned frames
        return (
            depth_frame,
            ir_frame,
            color_frame, 
            frames)

    def all_device(self):
        # Create a context object
        ctx = rs.context()

        # Get a list of all connected devices
        devices = ctx.query_devices()

        return [{"all": device, "name": device.get_info(rs.camera_info.name), "serial_number": device.get_info(rs.camera_info.serial_number)} for device in devices]


    def connect(self, serial_number="", preset_path=None, filter={"decimate":2, "spatial":[2, 0.5, 20], "temporal":[0.4, 20], "hole_filling":1}) :
        # filter
        self.filter = filter

        # Create a pipeline
        self.pipeline = rs.pipeline()

        #Create a config and configure the pipeline to stream
        config = rs.config()
        
        # serial number and name
        all_device = self.all_device()
        if serial_number:
            config.enable_device(serial_number)
            for device in all_device:
                if device['serial_number'] == serial_number:
                    name = device['name']
                    break
        else:
            serial_number = all_device[0]["serial_number"]
            name = all_device[0]["name"]
        
        config.enable_device(serial_number)

        # preset
        # assign preset if not exists
        if not preset_path:
            # Get the directory of the current module
            module_dir = os.path.dirname(__file__)

            # Construct the path to the JSON file
            preset_path = os.path.join(module_dir, 'preset', name.replace(" ", "_")+".json")

        # json string
        self.preset = json.load(open(preset_path))
        self.preset_string = json.dumps(self.preset)

        # stream
        config.enable_stream(rs.stream.depth, int(self.preset["viewer"]["stream-width"]), int(self.preset["viewer"]["stream-height"]), rs.format.z16, int(self.preset["viewer"]["stream-fps"]))
        config.enable_stream(rs.stream.infrared,1,  int(self.preset["viewer"]["stream-width"]), int(self.preset["viewer"]["stream-height"]), rs.format.y8, int(self.preset["viewer"]["stream-fps"]))
        config.enable_stream(rs.stream.color, int(self.preset["viewer"]["stream-width"]), int(self.preset["viewer"]["stream-height"]), rs.format.bgr8, int(self.preset["viewer"]["stream-fps"]))
        
        profile = self.pipeline.start(config)

        # apply advanced mode
        device = profile.get_device()
        self.advnc_mode = rs.rs400_advanced_mode(device)
        self.advnc_mode.load_json(self.preset_string)

        # decimate
        if type(self.filter) == dict and "decimate" in self.filter and self.filter["decimate"] != None:
            self.decimate = rs.decimation_filter()
            self.decimate.set_option(rs.option.filter_magnitude, self.filter["decimate"])
            # depth_to_disparity
            self.depth_to_disparity = rs.disparity_transform(True)
            self.disparity_to_depth = rs.disparity_transform(False)
        else:
            self.decimate = None
            self.depth_to_disparity = None
            self.disparity_to_depth = None

        # spatial
        if type(self.filter) == dict and "spatial" in self.filter and self.filter["spatial"] != None:
            self.spatial = rs.spatial_filter()
            self.spatial.set_option(rs.option.filter_magnitude, self.filter["spatial"][0])
            self.spatial.set_option(rs.option.filter_smooth_alpha, self.filter["spatial"][1])
            self.spatial.set_option(rs.option.filter_smooth_delta, self.filter["spatial"][2])
        else:
            self.spatial = None
        
        # temporal
        if type(self.filter) == dict and "temporal" in self.filter and self.filter["temporal"] != None: 
            self.temporal = rs.temporal_filter()
            self.temporal.set_option(rs.option.filter_smooth_alpha, self.filter["temporal"][0])
            self.temporal.set_option(rs.option.filter_smooth_delta, self.filter["temporal"][1])
        else:
            self.temporal = None

        # hole_filling
        if type(self.filter) == dict and "hole_filling" in self.filter and self.filter["hole_filling"] != None:
            self.hole_filling = rs.hole_filling_filter()
            self.hole_filling.set_option(rs.option.holes_fill, self.filter["hole_filling"])
        else:
            self.hole_filling = None

        # global time and auto exposure
        sensor_dep = device.first_depth_sensor()
        #rs.option.global_time_enabled
        sensor_dep.set_option(rs.option.global_time_enabled, 1) # time
        #sensor_dep.set_option(rs.option.enable_auto_exposure, 1) # auto expose

    def close(self):
        try:
            self.pipeline.stop()
        except:
            pass


    """
    get frame, image and depth intrinsic data
    """
    def get_all(self, align_to="color", alpha=0.03):
        # Create an align object
        align_to = getattr(rs.stream, align_to)        
        depth_frame, ir_frame, color_frame, frames = self.frame(align_to)

        depth_img = np.asanyarray(depth_frame.get_data())
        depth_img = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=alpha), cv2.COLORMAP_JET)
        ir_img = np.asanyarray(ir_frame.get_data())
        color_img = np.asanyarray(color_frame.get_data())

        depth_int = rs.video_stream_profile(depth_frame.profile).get_intrinsics()

        return depth_frame, ir_frame, color_frame, depth_img, ir_img, color_img, depth_int, frames, frames.get_timestamp()/1000


if __name__ == '__main__':    
    camera = Camera()
    camera.connect()
    
    while True:
        start = time.time()
        depth_frame, ir_frame, color_frame, depth_img, ir_img, color_img, depth_int, _, timestamp = camera.get_all()
        print(time.time()-start)
        cv2.imshow("img",depth_img)

        xyz, sample = camera.xyz((720, 360), depth_frame, depth_int)
        if cv2.waitKey(1) == ord('q'):
            break
           
    camera.close()