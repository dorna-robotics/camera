import time
import pyrealsense2 as rs
import numpy as np
import math

import cv2
import json

class helper(object):
    """docstring for helper"""
    def __init__(self):
        super(helper, self).__init__()

    """
    convert camera pixel to xyz
    """
    def xyz(self, pxl, depth_frame, depth_int, wnd = (0,0), z_min = 10, z_max = 1000): 
        dim = np.asanyarray(depth_frame.get_data()).shape

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
                # make sure pixeles are not out of range
                if any([any([p[0] <0, p[0] >= dim[1], p[1] < 0, p[1] >= dim[0]]) for p in [pxl_p, pxl_n]]):
                    continue

                xyz_p = rs.rs2_deproject_pixel_to_point(depth_int, pxl_p.tolist(), depth_frame.get_distance(pxl_p[0], pxl_p[1]))
                xyz_p  = 1000*np.array(xyz_p)  
                xyz_n = rs.rs2_deproject_pixel_to_point(depth_int, pxl_n.tolist(), depth_frame.get_distance(pxl_n[0], pxl_n[1])) 
                xyz_n  = 1000*np.array(xyz_n)

                valid_point = [xyz_p[2] < z_max and xyz_p[2] > z_min, xyz_n[2] < z_max and xyz_n[2] > z_min]  
                """
                avg += valid_point[0]*xyz_p + valid_point[1]*xyz_n
                i += sum(valid_point)
                """
                if sum(valid_point) == 2:
                    avg += xyz_p + xyz_n
                    i += 2                  
            except Exception as ex:
                print(ex)
                pass

        if i == 0:
            return avg
        
        return avg/i


    def chess_corner(self, color_img, ptrn):
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1, 0.001)

        # bgr to gray
        gray_img = cv2.cvtColor(color_img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray_img, ptrn,None)
        color_img = cv2.drawChessboardCorners(color_img, ptrn, corners,ret)

        # If found, add object points, image points (after refining them)
        corners2 = []
        if ret == True:
            corners2 = cv2.cornerSubPix(gray_img,corners,(11,11),(-1,-1),criteria)
                
        return ret , corners2


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
        config.enable_stream(rs.stream.color, self.arg["width"], self.arg["height"], rs.format.bgr8, self.arg["fps"])
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


class dcamera(camera):
    """docstring for dcamera"""
    def __init__(self, arg):
        super(dcamera, self).__init__(arg) 

    """
    get frame, image and depth intrinsics data
    """
    def get_all(self, save = False, align_to = rs.stream.color):
        depth_frame, ir_frame, color_frame = self.frame(align_to)

        depth_img = np.asanyarray(depth_frame.get_data())
        depth_img = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)
        ir_img = np.asanyarray(ir_frame.get_data())
        color_img = np.asanyarray(color_frame.get_data())

        depth_int = rs.video_stream_profile(depth_frame.profile).get_intrinsics()        
        if save:
            try:
                cv2.imwrite("img/"+str(int(time.time()))+".jpg", color_img) 
            except:
                pass
        return depth_frame, ir_frame, color_frame, depth_img, ir_img, color_img, depth_int
    
    """
    give two pixels
    find the distance between them
    notice that the depth data is usually not valid around  edges
    We start from the two edges and find the distance between the two valid pixels on the line connecting the two edges.

    """
    def length(self, pxl0, pxl1, depth_frame, depth_int, l =1000, start = 30):
        pxl0 = np.array(pxl0)
        pxl1 = np.array(pxl1)

        print("pxl0, pxl1: ", pxl0, pxl1)

        xyz0 = np.array([0. for i in range(3)])
        xyz1 = np.array([0. for i in range(3)])

        # xyz0        
        for t0 in range(start, l):   
            p = pxl0 + t0/l * (pxl1 - pxl0)

            xyz_t = self.xyz(p, depth_frame, depth_int)
            if sum(xyz_t == [0, 0, 0]) < len(xyz_t):
                xyz0 = xyz_t
                print("pixel0: ", p, t0, xyz0)
                break 

        # xyz0        
        for t1 in range(start, l):   
            p = pxl1 + t1/l * (pxl0 - pxl1)

            xyz_t = self.xyz(p, depth_frame, depth_int)
            if sum(xyz_t == [0, 0, 0]) < len(xyz_t):
                xyz1 = xyz_t
                print("pixel1: ", p, t1, xyz1)
                break 

        distance = 0
        try:
            #distance = np.linalg.norm(xyz0 - xyz1) * l / (l-t0-t1)
            distance = np.linalg.norm(xyz0[0:2] - xyz1[0:2]) * l / (l-t0-t1) 
        except Exception as ex:
            pass

        return distance

    """
    give the center of a circle in pixel, and the radius, find the xyz
    if depth is not valid, use linear model to estimate the center of the
    linear model: xyz = T*pixel + B => change it to a overdetermined model Ax = b and find x ()
    l : max number of valid pixels to estimable xyz
    r: length in pixel
    """
    def circle_center(self, pxl, depth_frame, depth_int, r = 10, l = 20):
        dim = np.asanyarray(depth_frame.get_data()).shape        
        xyz = self.xyz(pxl, depth_frame, depth_int, (int(min(r*math.sqrt(2)/2, 5)),int(min(r*math.sqrt(2)/2, 5))))

        # depth data is not valid
        if sum(xyz == [0, 0, 0]) == len(xyz):
            """
            increase r_c until we find a valid depth region otherwise return 0
            """
            
            sample_pxl = []
            sample_xyz = []
            r_c = 1
            i = 0
            while r_c <= r and len(sample_pxl) <= l:

                # update theta          
                theta = i/r
                # current pixel
                pxl_c = (int(r_c *math.cos(theta))+ pxl[0], int(r_c *math.sin(theta))+ pxl[1])
                
                """
                # make sure pixel are not out of range
                if any([pxl_c[0] <0, pxl_c[0] >= dim[0], pxl_c[1] < 0, pxl_c[1] >= dim[1]]):
                    pass               
                """
                if pxl_c not in sample_pxl: # make sure it does not exist in sample_pixels
                    xyz_c = self.xyz(pxl_c, depth_frame, depth_int) # find xyz
                    if sum(xyz_c == [0, 0, 0]) != len(xyz_c): # make sure its a valid xyz
                        #print("sampleing xyz: ",r_c,  pxl_c, xyz_c)
                        sample_pxl.append(pxl_c)
                        sample_xyz.append(xyz_c)

                # update i
                i += 1
                if i >= 2*r_c*math.pi:
                    i = 0
                    r_c += 1
                
            # there are enogh data to run the over determined system    
            if len(sample_pxl) >= 5:
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

                xyz = np.matrix(pxl) * T + B
                xyz = np.asarray(xyz).reshape(-1)

        return xyz

if __name__ == '__main__':
    import json
    
    
    with open("config.json") as json_file:
        arg = json.load(json_file)
    camera = dcamera(arg)
    camera.on()

    for i in range(4):
        depth_frame, ir_frame, color_frame, depth_img, ir_img, color_img, depth_int = camera.get_all()
        
        print(np.asanyarray(depth_frame.get_data()).shape)   
        cv2.imshow("color",color_img) 
        cv2.imshow("depth",depth_img)
        cv2.waitKey(0)             
        
    camera.off()