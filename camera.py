import threading
import time
import pickle

import pyrealsense2 as rs
import numpy as np

import cv2
import json

"""
detect 6 points with the camera.
find those points in the robot head coordinate system

give these 12 points as an input (12 points where each point has 3 element x, y, z)
find T and B such: P_h = P_c * T + B 
"""
class camera_robot_sync(object):
    """docstring for camera_robot_sync"""
    def __init__(self, arg):
        super(camera_robot_sync, self).__init__()

    def camera_to_head(self, H, C):
        H0 = np.matrix(H[0: 3])
        H1 = np.matrix(H[3:])
        
        C0 = np.matrix(C[0: 3])
        C1 = np.matrix(C[3:])

        T = np.linalg.inv(C0-C1) * (H0-H1)
        B = np.array(H[0]) - np.array(C[0])* T

        return T, B  

class helper(object):
    """docstring for helper"""
    def __init__(self):
        super(helper, self).__init__()

    """
    convert camera pixel to xyz
    """
    def xyz(self, pxl, depth_frame, depth_int, wnd = (0,0)): 
        # make pixel to int
        pxl = np.array(pxl)
        pxl = pxl.astype(int)
        # lattice
        lattice = np.array([[x,y] for x in range(-wnd[0], wnd[0]+1) for y in range(wnd[1]+1)])
        avg = np.array([0. for i in range(3)])
        
        i = 0
        for l in lattice:
            try:
                pxl_p = pxl+l
                pxl_n = pxl-l
                xyz_p = rs.rs2_deproject_pixel_to_point(depth_int, pxl_p.tolist(), depth_frame.get_distance(pxl_p[0], pxl_p[1]))
                xyz_p  = np.array(xyz_p)
                xyz_n = rs.rs2_deproject_pixel_to_point(depth_int, pxl_n.tolist(), depth_frame.get_distance(pxl_n[0], pxl_n[1])) 
                xyz_n  = np.array(xyz_n)
                avg += xyz_p + xyz_n
                i += 1 
            except:
                pass

        if i == 0:
            return [None for i in range(3)]
        
        #return (1000*avg/(2*i)).tolist()
        return 1000*avg/(2*i)


    def chess_corner(self, color_img, ptrn):
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1, 0.001)
        #criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # bgr to gray
        gray_img = cv2.cvtColor(color_img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray_img, ptrn,None)

        # If found, add object points, image points (after refining them)
        corners2 = []
        if ret == True:
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                
        return ret , corners2


    """
    sync the camera with a chess board
    """
    def camera_to_chess(self, depth_frame, depth_int, color_img, ptrn):
        # initialize transformation matrix, initial point
        o_p_point = np.array([None, None, None])
        t_e_e_p = np.matrix([[None for i in range(3)] for j in range(3)])

        # chess detection
        ret, corners = self.chess_corner(color_image, ptrn)        
        if ret:            
            # form it in more readable way
            corners = np.ravel(corners).reshape(-1,2)
            corners = corners.tolist()

            # bases 
            # e_1_p: x direction 
            x_vector = []
            for i in range(ptrn[1]):
                pixel = [corners[i*ptrn[0]], corners[(i+1)*ptrn[0] - 1]]
                point = [self.xyz(p, depth_frame, depth_int) for p in pixel]
                x_vector.append(point[-1] - point[0])
            x_vector = np.mean(x_vector, axis=0)
            e_1_p = x_vector / np.linalg.norm(x_vector)
            
            # e_2_p: y direction 
            y_vector = []
            for i in range(ptrn[0]):
                #pixel = [corners[i*chess[1]], corners[(i+1)*chess[1] - 1]]
                pixel = [corners[ptrn[1]*ptrn[0]-1-i], corners[i]]
                point = [self.xyz(p, depth_frame, depth_int) for p in pixel]
                y_vector.append(point[-1] - point[0])
            y_vector = np.mean(y_vector, axis=0)
            e_2_p = y_vector / np.linalg.norm(y_vector)

            # e_3_p: z direction
            z_vector = np.cross(x_vector, y_vector)
            e_3_p = z_vector / np.linalg.norm(z_vector)

            # O_p
            o_p_pixel = corners[0]
            o_p_point = self.xyz(o_p_pixel, depth_frame, depth_int)
            #o_p_point = np.array(o_p_point)

            x_p_point = o_p_point + e_1_p
            x_p_pixel = rs.rs2_project_point_to_pixel(depth_int, x_p_point.tolist())

            y_p_point = o_p_point + e_2_p
            y_p_pixel = rs.rs2_project_point_to_pixel(depth_int, y_p_point.tolist())

            z_p_point = o_p_point + e_3_p
            z_p_pixel = rs.rs2_project_point_to_pixel(depth_int, z_p_point.tolist())

            # transform matrix T(e', e) = inverse(T(e, e'))
            t_e_p_e = np.matrix([e_1_p, e_2_p, e_3_p]).transpose()
            t_e_e_p = np.linalg.inv(t_e_p_e)

        return [o_p_point, t_e_e_p]


    def draw_axis(self, img, corner_pixel, axis_pixel):
        # bgr color
        img = cv2.line(img, tuple(np.array(corner_pixel).astype(int)), tuple(np.array(axis_pixel[0]).astype(int)), (0,0,255), 1)
        img = cv2.line(img, tuple(np.array(corner_pixel).astype(int)), tuple(np.array(axis_pixel[1]).astype(int)), (0,255,0), 1)
        img = cv2.line(img, tuple(np.array(corner_pixel).astype(int)), tuple(np.array(axis_pixel[2]).astype(int)), (255,0,0), 1)
        return img


class camera(helper):
    """docstring for ClassName"""
    def __init__(self, arg):
        super(camera, self).__init__()
        # args
        self.arg = arg

    def frame(self, align_to = rs.stream.color):
        # Get frameset of ir and depth
        frames = self.pipeline.wait_for_frames(1000 * self.arg["time_out"])
        
        # Create an align object
        align = rs.align(align_to)

        # Align the depth frame to ir frame
        aligned_frames = align.process(frames)
        
        # Get aligned frames
        return (
            aligned_frames.get_depth_frame(),
            aligned_frames.get_infrared_frame(),
            aligned_frames.get_color_frame(),
            )

    def on(self):
        # Create a pipeline
        self.pipeline = rs.pipeline()

        #Create a config and configure the pipeline to stream
        config = rs.config()
        config.enable_stream(rs.stream.depth, self.arg["width"], self.arg["height"], rs.format.z16, self.arg["fps"])
        config.enable_stream(rs.stream.infrared,1,  self.arg["width"], self.arg["height"], rs.format.y8, self.arg["fps"])
        config.enable_stream(rs.stream.color, self.arg["width"], self.arg["height"], rs.format.rgb8, self.arg["fps"])
        profile = self.pipeline.start(config)

        # apply advanced mode
        dev = profile.get_device()
        advnc_mode = rs.rs400_advanced_mode(dev)
        json_obj = json.load(open(self.arg["preset_path"]))
        json_string = str(json_obj).replace("'", '\"')
        advnc_mode.load_json(json_string)

        # decimate
        decimate = rs.decimation_filter()
        decimate.set_option(rs.option.filter_magnitude, 2 ** self.arg["decimate_scale"])

    def off(self):
        self.pipeline.stop()


if __name__ == '__main__':
    import json
    
    
    with open("config.json") as json_file:
        arg = json.load(json_file)
    camera = camera(arg)
    camera.on()

    for i in range(5):
        depth_frame, ir_frame, color_frame = camera.frame()

        depth_img = np.asanyarray(depth_frame.get_data())
        depth_img = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)
        ir_img = np.asanyarray(ir_frame.get_data())
        color_img = np.asanyarray(color_frame.get_data())

        #depth_int = rs.video_stream_profile(color_frame.profile).get_intrinsics()        
        depth_int = rs.video_stream_profile(depth_frame.profile).get_intrinsics()        

        x,y,z = camera.xyz((1280/2, 720/2), depth_frame, depth_int)
        print("xyz: ", x, y, z)
        cv2.imshow("color",color_img)
        cv2.imshow("depth",depth_img)
        cv2.waitKey(0)         

    camera.off()