import time
import pyrealsense2 as rs
import numpy as np
import cv2
import threading
import queue
import random
import subprocess
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
    find the xyz of a given pixel with respect to a frame known to the user
    method: bilinear, plane, idw
    """
    def xyz_estimate(self, pxl, pxl_ref, xyz_ref, method="plane", power=2):
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


    def pixel(self, xyz, depth_int):
        """
        intr: rs.intrinsics from depth_profile.get_intrinsics()
        point3d: (X,Y,Z) in the same units (meters) intr expects.
        """
        X, Y, Z = xyz
        if Z == 0:
            return 0, 0
        
        # normalized coordinates
        x = X / Z
        y = Y / Z

        # distortion
        k1, k2, p1, p2, k3 = depth_int.coeffs
        r2 = x*x + y*y
        x_dist = x*(1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2) \
                + 2*p1*x*y + p2*(r2 + 2*x*x)
        y_dist = y*(1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2) \
                + 2*p2*x*y + p1*(r2 + 2*y*y)

        u = depth_int.fx * x_dist + depth_int.ppx
        v = depth_int.fy * y_dist + depth_int.ppy
        return u, v


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
            
            rs._all_device = list([{
                    "name": device.get_info(rs.camera_info.name), 
                    "serial_number": device.get_info(rs.camera_info.serial_number), 
                    "usb_type": device.get_info(rs.camera_info.usb_type_descriptor), 
                    "usb_port": device.get_info(rs.camera_info.physical_port) , 
                    "obj": device
                } for device in devices])


    def _refresh_devices(self):
        """Refresh cached device list (important after reset / replug)."""
        ctx = rs.context()
        devices = ctx.query_devices()
        rs._all_device = list([{
            "name": device.get_info(rs.camera_info.name),
            "serial_number": device.get_info(rs.camera_info.serial_number),
            "usb_type": device.get_info(rs.camera_info.usb_type_descriptor),
            "usb_port": device.get_info(rs.camera_info.physical_port),
            "obj": device
        } for device in devices])
        return rs._all_device

    def _wait_for_device(self, serial_number: str, timeout: float = 10.0, sleep: float = 0.25) -> bool:
        """Wait until a device with given serial appears."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                devs = self._refresh_devices()
                if any(d["serial_number"] == serial_number for d in devs):
                    return True
            except Exception:
                pass
            time.sleep(sleep)
        return False

    def _stop_pipeline_quiet(self):
        try:
            if getattr(self, "pipeline", None) is not None:
                self.pipeline.stop()
        except Exception:
            pass


    def _sh(self, cmd, timeout=5):
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, timeout=timeout)
        return p.returncode, p.stdout

    def _hw_reset(self):
        # firmware reset by serial (only affects this camera)
        ctx = rs.context()
        for dev in ctx.query_devices():
            if dev.get_info(rs.camera_info.serial_number) == self.serial_number:
                dev.hardware_reset()
                return True
        return False

    def _usb_unbind_bind(self):
        # Linux only
        if os.name != "posix":
            return False
        
        # linux USB reset by topology path like '4-1-2' (requires sudo)
        if not getattr(self, "usb_port", None):
            return False
        port = self.usb_port
        rc, out = self._sh(["bash", "-lc", f"echo '{port}' | sudo tee /sys/bus/usb/drivers/usb/unbind"], timeout=5)
        time.sleep(2)
        rc2, out2 = self._sh(["bash", "-lc", f"echo '{port}' | sudo tee /sys/bus/usb/drivers/usb/bind"], timeout=5)
        time.sleep(3)
        return (rc == 0 and rc2 == 0)

    def _recover(self):
        # stop pipeline first
        try:
            self.pipeline.stop()
        except:
            pass

        # 1) try RealSense firmware reset
        try:
            self._hw_reset()
        except:
            pass
        time.sleep(5)

        # reconnect (no recursion: start=False)
        try:
            self.connect(
                serial_number=self.serial_number,
                mode=self.mode,
                filter=self.filter,
                exposure=self.exposure_set,
                stream=self.stream,
                K=self.K,
                D=self.D,
                native_res=self.native_res,
                start=False
            )
            return True
        except:
            pass

        # 2) escalate: USB unbind/bind then reconnect
        try:
            if self._usb_unbind_bind():
                self.connect(
                    serial_number=self.serial_number,
                    mode=self.mode,
                    filter=self.filter,
                    exposure=self.exposure_set,
                    stream=self.stream,
                    K=self.K,
                    D=self.D,
                    native_res=self.native_res,
                    start=False
                )
                return True
        except:
            pass

        return False



    def camera_matrix(self, depth_int, ratio=1):
        if ratio is None and type(self.filter) == dict and "decimate" in self.filter:
            ratio = self.filter["decimate"]

        return np.array([[ratio*depth_int.fx,   0.        , ratio*depth_int.ppx],
                                [  0.        , ratio*depth_int.fy, ratio*depth_int.ppy],
                                [  0.        ,   0.        ,   1.        ]])


    def dist_coeffs(self, depth_int):
        return np.array(depth_int.coeffs)


    def frame(self, align_to, time_out=5):
        # self-healing wait_for_frames
        try:
            frames = self.pipeline.wait_for_frames(1000 * time_out)
        except RuntimeError:
            if not self._recover():
                raise
            # retry once after recover
            print("Recovery successful, retrying frame capture...")
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

        return depth_frame, ir_frame, color_frame, frames

    
    def all_device(self):
        return list(rs._all_device)


    # filter={"spatial":[2, 0.5, 20], "temporal":[0.1, 40], "hole_filling":1}
    def connect(
        self,
        serial_number="",
        mode="bgrd",
        filter={},
        exposure=None,
        stream={"width":848, "height":480, "fps":15},
        K=None,
        D=None,
        native_res=None,
        start=True,
        # production controls:
        max_tries=3,
        recover_on_fail=True,
        raise_on_fail=False,
        device_wait_sec=10.0
    ):
        """
        Production-safe connect:
        - returns True on success
        - returns False on failure (no exception) unless raise_on_fail=True
        - on failure, attempts: hw reset -> usb unbind/bind (Linux) -> retry
        """

        # store params for recovery and later use
        self.filter = filter
        self.serial_number = serial_number  # may be filled after selection below
        self.mode = mode
        self.stream = stream
        self.exposure_set = exposure
        self.K = K
        self.D = D
        self.native_res = native_res
        self.start = start

        # always stop any previous pipeline quietly
        self._stop_pipeline_quiet()

        # refresh device list (important if previously disconnected/replugged)
        try:
            self._refresh_devices()
        except Exception:
            pass

        # choose serial if not provided
        if not serial_number:
            devs = self.all_device()
            if not devs:
                if raise_on_fail:
                    raise RuntimeError("No RealSense devices found.")
                return False
            serial_number = random.choice(devs)["serial_number"]
        self.serial_number = serial_number

        # wait for device to be present (helps after resets)
        if not self._wait_for_device(serial_number, timeout=device_wait_sec):
            if raise_on_fail:
                raise RuntimeError(f"RealSense device not present (serial={serial_number}).")
            return False

        # resolve usb_port fresh (can change after replug)
        self.usb_port = None
        try:
            for d in self.all_device():
                if d["serial_number"] == serial_number:
                    self.usb_port = d.get("usb_port", None)
                    break
        except Exception:
            self.usb_port = None

        # prepare filter objects (safe defaults)
        # (these are set again after pipeline start too; keeping this light)
        self.decimate = None
        self.depth_to_disparity = None
        self.spatial = None
        self.temporal = None
        self.disparity_to_depth = None
        self.hole_filling = None

        def _start_pipeline_once():
            # Create a pipeline and config each attempt (important!)
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial_number)

            if mode == "motion":
                config.enable_stream(rs.stream.accel)
                config.enable_stream(rs.stream.gyro)
                profile = self.pipeline.start(config)
                return profile

            # normal streams
            try:
                config.enable_stream(rs.stream.depth, stream["width"], stream["height"], rs.format.z16, stream["fps"])
                config.enable_stream(rs.stream.infrared, 1, stream["width"], stream["height"], rs.format.y8, stream["fps"])
                config.enable_stream(rs.stream.color, stream["width"], stream["height"], rs.format.bgr8, stream["fps"])
                profile = self.pipeline.start(config)
            except Exception:
                # fallback
                config = rs.config()
                config.enable_device(serial_number)
                config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
                config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 15)
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
                profile = self.pipeline.start(config)

            # apply advanced mode + sensor config
            device = profile.get_device()
            try:
                self.advnc_mode = rs.rs400_advanced_mode(device)
            except Exception:
                self.advnc_mode = None

            # build post-start filters (your original behavior)
            if isinstance(self.filter, dict) and self.filter.get("decimate", None) is not None:
                self.decimate = rs.decimation_filter()
                self.decimate.set_option(rs.option.filter_magnitude, self.filter["decimate"])
                self.depth_to_disparity = rs.disparity_transform(True)
                self.disparity_to_depth = rs.disparity_transform(False)

            if isinstance(self.filter, dict) and self.filter.get("spatial", None) is not None:
                self.spatial = rs.spatial_filter()
                self.spatial.set_option(rs.option.filter_magnitude, self.filter["spatial"][0])
                self.spatial.set_option(rs.option.filter_smooth_alpha, self.filter["spatial"][1])
                self.spatial.set_option(rs.option.filter_smooth_delta, self.filter["spatial"][2])

            if isinstance(self.filter, dict) and self.filter.get("temporal", None) is not None:
                self.temporal = rs.temporal_filter()
                self.temporal.set_option(rs.option.filter_smooth_alpha, self.filter["temporal"][0])
                self.temporal.set_option(rs.option.filter_smooth_delta, self.filter["temporal"][1])

            if isinstance(self.filter, dict) and self.filter.get("hole_filling", None) is not None:
                self.hole_filling = rs.hole_filling_filter()
                self.hole_filling.set_option(rs.option.holes_fill, self.filter["hole_filling"])

            # global time and exposure
            try:
                self.sensor_dep = device.first_depth_sensor()
                self.sensor_dep.set_option(rs.option.global_time_enabled, 1)
                if exposure:
                    self.set_exposure(exposure)
            except Exception:
                pass

            # optional override intrinsics
            self.intr = None
            if K is not None and D is not None:
                K_ = np.array(K)
                D_ = np.array(D)
                sx = 1.0
                sy = 1.0
                if native_res is not None:
                    sx = stream["width"] / native_res[0]
                    sy = stream["height"] / native_res[1]

                intr = rs.intrinsics()
                intr.width  = stream["width"]
                intr.height = stream["height"]
                intr.ppx    = float(K_[0, 2]) * sx
                intr.ppy    = float(K_[1, 2]) * sy
                intr.fx     = float(K_[0, 0]) * sx
                intr.fy     = float(K_[1, 1]) * sy
                intr.model  = rs.distortion.brown_conrady
                intr.coeffs = [float(D_[0]), float(D_[1]), float(D_[2]), float(D_[3]), float(D_[4])]
                self.intr = intr

            return profile

        last_ex = None

        for attempt in range(1, max_tries + 1):
            try:
                # ensure device present each attempt (after resets it can disappear briefly)
                if not self._wait_for_device(serial_number, timeout=device_wait_sec):
                    raise RuntimeError(f"Device disappeared (serial={serial_number}).")

                profile = _start_pipeline_once()

                if start:
                    # warm “sanity check” frame read (ensures streams are alive)
                    self.get_all()
                return True

            except Exception as ex:
                last_ex = ex
                # stop anything partially started
                self._stop_pipeline_quiet()

                if not recover_on_fail or attempt >= max_tries:
                    break

                # Recovery ladder (precise, deterministic):
                # attempt 1 failure -> hw reset
                # attempt 2 failure -> usb unbind/bind (Linux)
                try:
                    if attempt == 1:
                        # 1) firmware reset
                        try:
                            self._hw_reset()
                        except Exception:
                            pass
                        time.sleep(5.0)
                    else:
                        # 2) USB reset (Linux only; may require sudoers/udev)
                        try:
                            self._refresh_devices()
                            # re-resolve usb_port just in case
                            self.usb_port = None
                            for d in self.all_device():
                                if d["serial_number"] == serial_number:
                                    self.usb_port = d.get("usb_port", None)
                                    break
                        except Exception:
                            pass

                        try:
                            self._usb_unbind_bind()
                        except Exception:
                            pass
                        time.sleep(3.0)

                    # refresh after recovery
                    try:
                        self._refresh_devices()
                    except Exception:
                        pass

                except Exception:
                    # never crash recovery path
                    pass

        # failure path
        if raise_on_fail:
            raise RuntimeError(f"Failed to connect RealSense (serial={serial_number}). Last error: {last_ex}")
        return False



    #filter={"spatial":[2, 0.5, 20], "temporal":[0.1, 40], "hole_filling":1}
    def connect_old(self, serial_number="", mode="bgrd", filter={}, exposure=None, stream={"width":848, "height":480, "fps":15}, K=None, D=None, native_res=None, start=True):
        # filter
        self.filter = filter
        # --- store params for recovery ---
        self.serial_number = serial_number  # will be filled after selection below
        self.mode = mode
        self.stream = stream
        self.exposure_set = exposure
        self.K = K
        self.D = D
        self.native_res = native_res
        self.start = start

        # --- filters ---
        if "decimate" in self.filter:
            self.decimate = rs.decimation_filter(self.filter["decimate"])
        if "depth_to_disparity" in self.filter:
            self.depth_to_disparity = rs.disparity_transform(self.filter["depth_to_disparity"])
        if "spatial" in self.filter:
            self.spatial = rs.spatial_filter(self.filter["spatial"][0], self.filter["spatial"][1], self.filter["spatial"][2])
        if "temporal" in self.filter:
            self.temporal = rs.temporal_filter(self.filter["temporal"][0], self.filter["temporal"][1])
        if "disparity_to_depth" in self.filter:
            self.disparity_to_depth = rs.disparity_transform(self.filter["disparity_to_depth"])
        if "hole_filling" in self.filter:
            self.hole_filling = rs.hole_filling_filter(self.filter["hole_filling"])

        # Create a pipeline
        self.pipeline = rs.pipeline()
        
        #Create a config and configure the pipeline to stream
        config = rs.config()

        # serial number
        if not serial_number:
            serial_number = random.choice(self.all_device())["serial_number"]
        config.enable_device(serial_number)
        self.serial_number = serial_number
        
        # usb port
        self.usb_port = None
        for d in self.all_device():
            if d["serial_number"] == serial_number:
                self.usb_port = d.get("usb_port", None)
                break

        # stream
        if mode == "motion":
            config.enable_stream(rs.stream.accel)
            config.enable_stream(rs.stream.gyro)
            profile = self.pipeline.start(config)
        else:
            try:
                config.enable_stream(rs.stream.depth, stream["width"], stream["height"], rs.format.z16, stream["fps"])
                config.enable_stream(rs.stream.infrared, 1, stream["width"], stream["height"], rs.format.y8, stream["fps"])
                config.enable_stream(rs.stream.color, stream["width"], stream["height"], rs.format.bgr8, stream["fps"])
                profile = self.pipeline.start(config)                
            except:
                config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
                config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 15)
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
                profile = self.pipeline.start(config)
                
            # apply advanced mode
            device = profile.get_device()
            self.advnc_mode = rs.rs400_advanced_mode(device)

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
                self.set_exposure(exposure)

            # K and D
            self.intr = None
            if K is not None and D is not None:
                K = np.array(K)
                D = np.array(D)
                sx = 1
                sy = 1
                if native_res is not None:
                    sx = stream["width"] / native_res[0]
                    sy = stream["height"] / native_res[1]
                # Create a new rs.intrinsics object
                self.intr = rs.intrinsics()
                self.intr.width  = stream["width"]
                self.intr.height = stream["height"]
                self.intr.ppx    = float(K[0, 2]) * sx  # cx
                self.intr.ppy    = float(K[1, 2]) * sy  # cy
                self.intr.fx     = float(K[0, 0]) * sx  # fx
                self.intr.fy     = float(K[1, 1]) * sy  # fy

                # Use Brown-Conrady distortion model (OpenCV's standard k1,k2,p1,p2,k3)
                self.intr.model  = rs.distortion.brown_conrady

                # Assign the 5 distortion coefficients
                self.intr.coeffs = [float(D[0]), float(D[1]), float(D[2]),
                                    float(D[3]), float(D[4])]

            if start:
                self.get_all()
            return True

    def warmup(self, sec=600, sleep=0.1):
        n = int(sec / sleep)
        for _ in range(n):
            self.get_all()
            time.sleep(sleep)


    def get_temp(self):
        return self.sensor_dep.get_option(rs.option.asic_temperature)


    def get_exposure(self):
        return self.sensor_dep.get_option(rs.option.exposure)


    def set_exposure(self, exposure):
        self.auto_exposure(False)
        self.sensor_dep.set_option(rs.option.exposure, min(165000, max(1, exposure)))
        return self.get_exposure()

    def auto_exposure(self, enable=True):
        self.sensor_dep.set_option(rs.option.enable_auto_exposure, 1 if enable else 0)
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
    def get_all(self, align_to=rs.stream.infrared, alpha=0.03): # rs.stream.color
        # Create an align object
        depth_frame, ir_frame, color_frame, frames = self.frame(align_to)

        depth_img = np.asanyarray(depth_frame.get_data())
        depth_img = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=alpha), cv2.COLORMAP_JET)
        ir_img = np.asanyarray(ir_frame.get_data())
        color_img = np.asanyarray(color_frame.get_data())
        if self.intr is not None:
            depth_int = self.intr
        else:
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