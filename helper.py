"""
robot toolhead length:
raw_toolhead: 46.0502 mm toolhead: 2*(0.625 in) + 0.14 in
total length: 81.3562 mm


theta_2 position from o_p (in e_p coordinate system)
[-25, 15, 70 + 81.3562]

robot coordinate vs e_p coordinate
         x                y
robot y__|            e_p |__x                 


camera_to_robot:
find a point from camera prespective
map it to the e_p space
add it with -[-25, 15, 70 + 81.3562]
convert (x, y, z) to (y, -x, z)

"""
import numpy as np
import cv2
import glob
import pyrealsense2 as rs



# pixel to camera coordinate system
def base_change(pixel, t_e_e_p, o_p, aligned_depth_frame, depth_intrinsics, bias_matrix):
    #bias_matrix = np.matrix([0.203625, -0.0103, 0.003275]).T
    bias_matrix = np.matrix([0, 0, 0]).T
    p_e = dc_pixel_to_point(pixel, aligned_depth_frame, depth_intrinsics)
    
    p_e_p = (np.dot(t_e_e_p, np.matrix(p_e - o_p).T) + bias_matrix).T
    p_e_p = np.array(p_e_p)
    return p_e_p[0] 

def camera_to_robot(point):
    #return point
    #return [point[0] - 95 - 6, point[1] - 15 - 6, point[2]]
    #return [point[0] - 95 - 6, point[1] - 15 - 6, point[2] - 70]
    # 1569401229.p data
    return [point[0] -60, point[1] - 10, point[2] -56]
    #return [point[1] - 15, -25 - point[0], point[2] - 70 - 81.3562] 

def point_adjust(point, w, h):
    point[0] = point[0]%w
    point[1] = point[1]%h
    return point

# depth camera pixel to point
def dc_pixel_to_point(pixel, d_frame, depth_intrinsics):
    return np.array(
        rs.rs2_deproject_pixel_to_point(depth_intrinsics, pixel, d_frame.get_distance(int(pixel[0]), int(pixel[1])))
        )

def distance_3d(point_0, point_1):
    return sum([(point_0[i] - point_1[i])**2 for i in range(min(len(point_0), len(point_1)))])**0.5


if __name__ == '__main__':
    
    img = cv2.imread("chessboard_3.jpg")
    ptrn = (9,6)
    ret, corners = chess_corner(img, ptrn)

    print(corners)
    # draw
    img = cv2.drawChessboardCorners(img, ptrn, corners,ret)
    img = cv2.resize(img, (500, 500))   
    cv2.imshow('img',img)
    cv2.waitKey(0)
