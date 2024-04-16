import numpy as np
import cv2
import json

"""
Camera is connected to the robot toolhead,
Find T and B such that
if rel = 0: camera is connected to a fixed position
    xyz_r = xyz_c * T + B
if rel = 1: camera is connected to the robot toolhead 
    xyz_h = xyz_c * T + B 
    once we find xyz_h

"""
def sync(camera_object, robot_object, rel = 0):
    camera_xyz = []
    robot_head_xyz = []
    T = False
    B = False
    
    stop = False
    while True:
        key = input("Take a photo from a 4*4 chess and start training? (y/n)")
        if key == "n":
            break

        # img data
        depth_frame, ir_frame, color_frame, depth_img, ir_img, color_img, depth_int = camera_object.get_all()

        # find corners and reshape them
        ret, corners = camera_object.chess_corner(color_img, (3, 3))
        corners = np.ravel(corners).reshape(-1,2)
        corners = corners.tolist() 
        corner_index = [0, 2, 8, 6]
        if ret:
            # for each corner print xyz
            i = 0
            T_camera = []
            for j in corner_index:
                xyz = camera_object.xyz(corners[j], depth_frame, depth_int)
                T_camera.append(xyz)
                print("camera point "+ str(i) + " : " + str(xyz))
                cv2.putText(color_img, str(i), (int(corners[j][0]), int(corners[j][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                i += 1
        else:
            print("No chess board was detected. Try again...")
            continue
        
        cv2.imshow("color",color_img)
        cv2.waitKey(0)

        # make sure camera data is useful
        key = input("Use camera data? (y/n)")
        if key == "n":
            continue

        # touch the camera xyz point
        T_robot = []
        for i in range(len(corner_index)):
            key = input("Robot is touching? (y/n)")
            if key == "n":
                continue
            
            # get current robot xyz
            sys = dict(robot_object.sys)
            xyz = [sys[x] for x in ["x", "y", "z"]]
            T_robot.append(xyz)
            print("head point "+ str(i) + " : " + str(xyz)) 

        # make sure we touched all the points
        if len(T_robot) != len(T_camera):
            continue    
        else:
            # add the templates to head and camera
            for i in range(len(T_robot)):
                camera_xyz.append(T_camera[i])
                robot_xyz.append(T_robot[i])
        
        T, B = solving_T_B(camera_xyz, robot_xyz)
        print("T: ", T)
        print("B: ", B)
    
    return [T, B]

"""
detect 4 points with the camera.
find these 4 points in the robot coordinate system

give these 8 points as an input where each point has 3 element x, y, z.
find T and B such: P_r = P_c * T + B 
"""
"""
camera is in fixed position compare to the robot base
xyz_robot = xyz_camera *T + B
xyz_robot and xyz_camera is 1*3 matrix
"""
def camera_robot_base_fixed(camera_object, robot_object):
    camera_xyz = []
    robot_xyz = []
    T = False
    B = False
    
    stop = False
    while True:
        key = input("Take a photo from a 4*4 chess and start training? (y/n)")
        if key == "n":
            break

        # img data
        depth_frame, ir_frame, color_frame, depth_img, ir_img, color_img, depth_int = camera_object.get_all()

        # find corners and reshape them
        ret, corners = camera_object.chess_corner(color_img, (3, 3))
        corners = np.ravel(corners).reshape(-1,2)
        corners = corners.tolist() 
        corner_index = [0,1, 2, 3, 4, 5, 6, 7, 8]
        if ret:
            # for each corner print xyz
            i = 0
            T_camera = []
            for j in corner_index:
                xyz = camera_object.xyz(corners[j], depth_frame, depth_int)
                T_camera.append(xyz)
                print("camera point "+ str(i) + " : " + str(xyz))
                cv2.putText(color_img, str(i), (int(corners[j][0]), int(corners[j][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                i += 1
        else:
            print("No chess board was detected. Try again...")
            continue
        
        cv2.imshow("color",color_img)
        cv2.waitKey(0)

        # make sure camera data is useful
        key = input("Use camera data? (y/n)")
        if key == "n":
            continue

        # touch the camera xyz point
        T_robot = []
        for i in range(len(corner_index)):
            key = input("Robot is touching? (y/n)")
            if key == "n":
                continue
            
            # get current robot xyz
            sys = dict(robot_object.sys)
            xyz = [sys[x] for x in ["x", "y", "z"]]
            T_robot.append(xyz)
            print("head point "+ str(i) + " : " + str(xyz)) 

        # make sure we touched all the points
        if len(T_robot) != len(T_camera):
            continue    
        else:
            # add the templates to head and camera
            for i in range(len(T_robot)):
                camera_xyz.append(T_camera[i])
                robot_xyz.append(T_robot[i])
        
        T, B = solving_T_B(camera_xyz, robot_xyz)
        print("T: ", T)
        print("B: ", B)
    
    return [T, B]




def solving_T_B(camera_xyz, robot_xyz):
    A = np.zeros(12)
    b = np.zeros(1)
    for xyz in camera_xyz:
        t0 = np.hstack((xyz, [0, 0, 0], [0, 0, 0], [1, 0, 0]))
        t1 = np.hstack(([0, 0, 0],xyz, [0, 0, 0], [0, 1, 0]))
        t2 = np.hstack(([0, 0, 0], [0, 0, 0],xyz, [0, 0, 1]))  
        A = np.vstack((A, t0, t1, t2))
    A = np.delete(A, (0), axis=0)    
    
    for xyz in robot_xyz:
        b = np.vstack((b, xyz[0], xyz[1], xyz[2]))
    b = np.delete(b, (0), axis=0)   

    # Ax = b
    x = np.linalg.lstsq(A,b, rcond=None)[0] # computing the numpy solution
    
    T = np.hstack((x[0:3], x[3:6], x[6:9]))
    B = np.transpose(x[9:12])
    return [T, B ]       



if __name__ == '__main__':
    import json
    from camera import camera
    from dorna2 import dorna
    
    # camera object
    with open("config.json") as json_file:
        arg = json.load(json_file)
    camera = camera(arg)
    camera.on()

    # robot object and tool length is 60.58
    ip, port = "192.168.254.23", 443
    robot = dorna()
    #robot.connect(ip, port)
    #robot.play({"cmd":"toollength","id":10, "toollength":60.58})
    #robot.wait(id=10, stat=2)

    # sync
    T, B = camera_robot_base_fixed(camera, robot)


    camera.off()
    robot.close()

"""
[[ 1.00971726  0.00405274  0.00299815]
 [ 0.01345001 -1.00985097  0.00201243]
 [-0.07225321 -0.03746375  0.09055587]]
[[369.61834328  26.00728476 -42.64390858]]

T:  [[ 1.00362522e+00 -6.08170296e-03  2.14515751e-02]
 [-1.16490803e-02 -1.01397851e+00  2.47533302e-04]
 [ 2.87219177e-01 -1.45933692e-01  1.07591355e-01]]
B:  [[142.69219896  81.35848784 -47.9392697 ]]
"""