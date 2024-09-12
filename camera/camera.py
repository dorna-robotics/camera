import importlib.resources
import time
import pyrealsense2 as rs
import numpy as np
import cv2
import json
import threading
import queue
"""
Euclidean dimension is in mm
"""

class Helper(object):
    """docstring for helper"""
    def __init__(self):
        super(Helper, self).__init__()

        self.filter = None

    """
    find the xyz of a given pixel with respect to a frame known to the user
    method: bilinear, plane, idw
    """
    def xyz_estimate(self, pxl, pxl_ref, xyz_ref, method="plane"):
        # init
        xyz = np.array([0. for i in range(3)])
        
        if method == "plane":
            # adjust the inputs
            pxl_ref = np.array(pxl_ref)
            xyz_ref = np.array(xyz_ref)

            # Calculate the centroid of the points
            centroid = np.mean(xyz_ref, axis=0)

            # Subtract the centroid from the points
            centered_xyz_ref = xyz_ref - centroid

            # Compute the SVD
            _, _, vh = np.linalg.svd(centered_xyz_ref)

            # The normal of the plane is the last row of vh
            normal = vh[-1, :]

            # The plane equation is normal[0] * (x - cx) + normal[1] * (y - cy) + normal[2] * (z - cz) = 0
            # Rearrange to normal[0] * x + normal[1] * y + normal[2] * z = d
            d = -normal.dot(centroid)

            # Find the transformation from 2D pixel to 3D coordinates
            # This typically requires camera intrinsics (camera matrix)
            # Here we assume a direct linear relationship for simplicity
            # Use the known points to establish the relationship

            # Example: linear relationship (this is a simplification)
            # For accurate transformation, use camera calibration techniques
            A = np.hstack([pxl_ref, np.ones((pxl_ref.shape[0], 1))])
            B = xyz_ref
            coeff, _, _, _ = np.linalg.lstsq(A, B, rcond=None)

            # Apply the transformation to the new pixel
            new_pixel_homogeneous = np.array([pxl[0], pxl[1], 1])
            estimated_xyz = new_pixel_homogeneous.dot(coeff)

            # Use the plane equation to adjust the estimated z-coordinate
            # Plane equation: normal[0]*x + normal[1]*y + normal[2]*z + d = 0
            # Solve for z: z = -(normal[0]*x + normal[1]*y + d) / normal[2]
            estimated_xyz[2] = -(normal[0]*estimated_xyz[0] + normal[1]*estimated_xyz[1] + d) / normal[2]
            xyz = np.array(estimated_xyz)

        elif method == "idw":
            # Initialize sums for weights and weighted coordinates
            weight_sum = 0
            weighted_x_sum = 0
            weighted_y_sum = 0
            weighted_z_sum = 0

            # Small value to prevent division by zero
            epsilon = 1e-6

            # Calculate weights and weighted sums
            for (u, v), (x, y, z) in zip(pxl_ref, xyz_ref):
                distance = np.sqrt((u - pxl[0])**2 + (v - pxl[1])**2)
                if distance < epsilon:
                    # If the random pixel is exactly at a known point, return its coordinates
                    xyz = np.array([x, y, z])
                    break
                weight = 1 / (distance**power + epsilon)
                weight_sum += weight
                weighted_x_sum += x * weight
                weighted_y_sum += y * weight
                weighted_z_sum += z * weight

            # Compute the estimated coordinates using weighted averages
            xyz = np.array([weighted_x_sum / weight_sum,
                            weighted_y_sum / weight_sum,
                            weighted_z_sum / weight_sum,
                ])

        return xyz


    """
    convert camera pixel to xyz
    Use linear model to estimate the center of the
    linear model: xyz = T*pixel + B => change it to a over determined model Ax = b and find x ()
    wnd : the window size of the pixels
    """
    def xyz(self, pxl, depth_frame, depth_int, wnd = (0,0), z_gt=(10, 10000)): 
        sample_pxl = []
        sample_xyz = []
        xyz = np.array([0. for i in range(3)])
        try:
            if type(self.filter) == dict and "decimate" in self.filter:
                pxl = [x/self.filter["decimate"] for x in pxl]


            # ground truth is given
            if z_gt[0] == z_gt[1]:
                xyz = z_gt[0]*np.array([(pxl[0]-depth_int.ppx)/depth_int.fx, (pxl[1]-depth_int.ppy)/depth_int.fy, 1]) 
                return xyz, [v for v in zip(sample_pxl, sample_xyz)]

            # number of row and column
            dim = np.asanyarray(depth_frame.get_data()).shape

            # make a copy and pixel to int
            pxl_org = np.array(pxl)
            pxl = np.array(pxl).astype(int)

            # lattice
            lattice = np.array([[x,y] for x in range(int(np.floor(-wnd[0]/2)), int(np.floor(wnd[0]/2))+1) for y in range(int(np.floor(-wnd[1]/2)), int(np.floor(wnd[1]/2))+1)])
            for l in lattice:
                try:
                    pxl_new = pxl + l 
                    # make sure pixels are not out of range
                    if any([pxl_new[0] < 0, pxl_new[0] >= dim[1], pxl_new[1] < 0, pxl_new[1] >= dim[0]]):
                        continue

                    # calculate xyz
                    xyz_new = rs.rs2_deproject_pixel_to_point(depth_int, pxl_new.tolist(), depth_frame.get_distance(pxl_new[0], pxl_new[1]))
                    xyz_new  = 1000 * np.array(xyz_new) # in mm  

                    if xyz_new[2] > z_gt[1] or xyz_new[2] < z_gt[0]:
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
        except:
            pass
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

            xyz_t,_ = self.xyz(p, depth_frame, depth_int)
            if sum(xyz_t == [0, 0, 0]) < len(xyz_t):
                xyz0 = xyz_t
                break 

        # xyz0        
        for t1 in range(start, l):   
            p = pxl1 + t1/l * (pxl0 - pxl1)

            xyz_t,_ = self.xyz(p, depth_frame, depth_int)
            if sum(xyz_t == [0, 0, 0]) < len(xyz_t):
                xyz1 = xyz_t
                break 

        distance = 0
        try:
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

        # all devices
        if not hasattr(rs, '_all_device'):
            # Create a context object
            ctx = rs.context()
    
            # Get a list of all connected devices
            devices = ctx.query_devices()
            
            rs._all_device = list([{"name": device.get_info(rs.camera_info.name), "serial_number": device.get_info(rs.camera_info.serial_number), "obj": device} for device in devices])


    def camera_matrix(self, depth_int, ratio=1):
        if ratio is None and type(self.filter) == dict and "decimate" in self.filter:
            ratio = self.filter["decimate"]

        return np.array([[ratio*depth_int.fx,   0.        , ratio*depth_int.ppx],
                                [  0.        , ratio*depth_int.fy, ratio*depth_int.ppy],
                                [  0.        ,   0.        ,   1.        ]])


    def dist_coeffs(self, depth_int):
        return np.array(depth_int.coeffs)


    def frame(self, align_to, time_out=10):
        # Get frameset of ir and depth
        frames = self.pipeline.wait_for_frames(1000 * time_out)

        # Create an align object
        align = rs.align(align_to)

        # Align the depth frame to ir frame
        aligned_frames = align.process(frames)
        
        # frames
        depth_frame = aligned_frames.get_depth_frame()
        ir_frame = aligned_frames.get_infrared_frame()
        color_frame = aligned_frames.get_color_frame()
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
        return depth_frame, ir_frame, color_frame, frames

    
    def all_device(self):
        return list(rs._all_device)

    #filter={"spatial":[2, 0.5, 20], "temporal":[0.1, 40], "hole_filling":1}
    def connect(self, serial_number="", mode="bgrd", preset_path=None, filter={}, exposure=None):
        # filter
        self.filter = filter

        # Create a pipeline
        self.pipeline = rs.pipeline()

        #Create a config and configure the pipeline to stream
        config = rs.config()
        
        # serial number and name
        if serial_number:
            config.enable_device(serial_number)
            
            for device in self.all_device():
                if device['serial_number'] == serial_number:
                    name = device['name']
                    break
            
        else:
            serial_number = self.all_device()[0]["serial_number"]
            name = self.all_device()[0]["name"]
            config.enable_device(serial_number)

        # preset
        # assign preset if not exists
        if not preset_path:
            with importlib.resources.path("camera.preset", name.replace(" ", "_")+".json") as config_file:
                with open(config_file, 'r') as file:
                    self.preset = json.load(file)
        else:
            self.preset = json.load(open(preset_path))

        # json string
        self.preset_string = json.dumps(self.preset)

        # stream
        if mode == "motion":
            config.enable_stream(rs.stream.accel)
            config.enable_stream(rs.stream.gyro)
            profile = self.pipeline.start(config)
        else:
            config.enable_stream(rs.stream.depth, int(self.preset["viewer"]["stream-width"]), int(self.preset["viewer"]["stream-height"]), rs.format.z16, int(self.preset["viewer"]["stream-fps"]))
            config.enable_stream(rs.stream.infrared, 1, int(self.preset["viewer"]["stream-width"]), int(self.preset["viewer"]["stream-height"]), rs.format.y8, int(self.preset["viewer"]["stream-fps"]))
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
            self.sensor_dep = device.first_depth_sensor()
            #rs.option.global_time_enabled
            self.sensor_dep.set_option(rs.option.global_time_enabled, 1) # time
            
            if exposure:
                self.sensor_dep.set_option(rs.option.exposure, min(165000, max(1, exposure)))

            return True


    def get_exposure(self):
        return self.sensor_dep.get_option(rs.option.exposure)


    def set_exposure(self, exposure):
        self.sensor_dep.set_option(rs.option.exposure, min(165000, max(1, exposure)))
        return self.get_exposure()


    def close(self):
        try:
            self.pipeline.stop()
        except:
            pass
        return True


    """
    get frame, image and depth intrinsic data
    """
    def get_all(self, align_to=rs.stream.color, alpha=0.03):
        # Create an align object
        depth_frame, ir_frame, color_frame, frames = self.frame(align_to)

        depth_img = np.asanyarray(depth_frame.get_data())
        depth_img = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=alpha), cv2.COLORMAP_JET)
        ir_img = np.asanyarray(ir_frame.get_data())
        color_img = np.asanyarray(color_frame.get_data())
        depth_int = rs.video_stream_profile(color_frame.profile).get_intrinsics()

        return depth_frame, ir_frame, color_frame, depth_img, ir_img, color_img, depth_int, frames, frames.get_timestamp()/1000


    def motion_thread_func(self):
        while self.motion_rec_start:
            # Wait for the next set of frames
            frames = self.pipeline.wait_for_frames()
            try:
                # grab data
                t = time.time()
                accel = frames[0].as_motion_frame().get_motion_data()
                gyro = frames[1].as_motion_frame().get_motion_data()

                # add data
                self.motion_queue.put([[accel.x, accel.y, accel.z, t], [gyro.x, gyro.y, gyro.z, t]])
            except Exception as ex:
                pass
            time.sleep(0.001)


    def motion_rec(self):
        # fresh start
        self.motion_stop()

        # new motion queue
        self.motion_queue = queue.Queue()

        # start the loop
        self.motion_rec_start = True

        # start the thread
        self.motion_thread = threading.Thread(target=self.motion_thread_func)
        self.motion_thread.start()
    
    def motion_stop(self):
        # init gyro and accel
        accel = []
        gyro = []

        # close the loop
        self.motion_rec_start = False

        #close the thread
        try:
            self.motion_thread.join()
        except Exception as ex:
            pass

        # get all the data
        try:
            while not self.motion_queue.empty():
                data = self.motion_queue.get()
                accel.append(data[0])
                gyro.append(data[1])
        except:
            pass

        return accel, gyro