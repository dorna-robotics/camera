import json
from camera import dcamera
from dorna2 import dorna
import numpy as np
import cv2
import time
import math

def follow_chess():
    """
    T = np.matrix([[ 1.00971726, 0.00405274, 0.00299815],
                 [ 0.01345001, -1.00985097, 0.00201243],
                 [-0.07225321, -0.03746375, 0.09055587]])
    B = np.matrix([[369.61834328, 26.00728476, -42.64390858]])
    """
    T = np.matrix([[ 1.00362522e+00, -6.08170296e-03,  2.14515751e-02],
                     [-1.16490803e-02, -1.01397851e+00,  2.47533302e-04],
                     [ 2.87219177e-01, -1.45933692e-01,  1.07591355e-01]])
    B = np.matrix([[142.69219896,  81.35848784, -47.9392697 ]] )   
    
    # camera object
    with open("config.json") as json_file:
        arg = json.load(json_file)
    camera = dcamera(arg)
    camera.on()

    # robot object and tool length is 60.58
    ip, port = "192.168.254.23", 443
    robot = dorna()
    robot.connect(ip, port)

    stop = False
    while not stop:
        # img data
        depth_frame, ir_frame, color_frame, depth_img, ir_img, color_img, depth_int = camera.get_all()

        # find corners and reshape them
        ret, corners = camera.chess_corner(color_img, (3, 3))
        corners = np.ravel(corners).reshape(-1,2)
        corners = corners.tolist() 
        corner_index = [0]
        if ret: 
            i = 0
            for j in corner_index:
                xyz = camera.xyz(corners[j], depth_frame, depth_int)
                xyz_robot = xyz * T + B
                #xyz_robot = xyz_robot[0]
                cv2.putText(color_img, str(i), (int(corners[j][0]), int(corners[j][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow("color",color_img)
                cv2.waitKey(0)
                i = i+1
                print(xyz)
                print(xyz * T)
                print(xyz_robot)

                #print({"x": xyz_robot[0,0], "y": xyz_robot[0,1], "z": xyz_robot[0,2]+30})
                robot.play(cmd = "lmove", rel =0, x = xyz_robot[0,0], y = xyz_robot[0,1], z = xyz_robot[0,2]+5, a = -90, vel = 200)
                robot.play(cmd = "sleep", time = 5)
                robot.play(cmd = "lmove", rel = 0, x = 336.77, y= 227.855, z= 261.62, a = -66.888, id = 10)
                robot.wait(id = 10, stat = 2)
        else:
            print("No chess board was detected. Try again...")
            continue
        # make sure camera data is useful
        key = input("repeat? (y/n)")
        if key == "n":
            stop = True


    camera.off()
    robot.close()

def find_circle(src):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)    
    gray = cv2.medianBlur(gray, 5)
    
    
    rows = gray.shape[0]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8,
                               param1=100, param2=30,
                               minRadius=10, maxRadius=100)
    
    
    circle = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(src, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(src, center, radius, (255, 0, 255), 3)
            circle.append([i[0], i[1],i[2]])
    
    cv2.imshow("detected circles", src)
    cv2.waitKey(1)
    
    return circle

def distance(x,y):
    print(x,y)
    s = [(x[i]-y[i]) for i in range(len(x))]
    print(s)
    s = sum([z**2 for z in s])
    print(s)
    return math.sqrt(s)

def follow_coin():
    # min_height = 570
    min_height = 10
    min_distance = 60
    num_seg = 10
    T = np.matrix([[ 1.00362522e+00, -6.08170296e-03,  2.14515751e-02],
                     [-1.16490803e-02, -1.01397851e+00,  2.47533302e-04],
                     [ 2.87219177e-01, -1.45933692e-01,  1.07591355e-01]])
    B = np.matrix([[142.69219896,  81.35848784, -47.9392697 ]] )   
    
    # camera object
    with open("config.json") as json_file:
        arg = json.load(json_file)
    camera = dcamera(arg)
    camera.on()

    # robot object and tool length is 60.58
    ip, port = "192.168.254.23", 443
    #robot = dorna()
    #robot.connect(ip, port)

    stop = False
    point_list = []
    step_list = []
    # get current robot xyz
    #sys = dict(robot.sys)
    sys = {"x": 0, "y": 0, "z": 0}
    xyz_final = np.array([sys[x] for x in ["x", "y", "z"]])
    circle = False
    circle_prev = False
    j = 0
    i = 1
    while not stop:
        # take img
        depth_frame, ir_frame, color_frame, depth_img, ir_img, color_img, depth_int = camera.get_all()

        # find corners and reshape them
        circle = find_circle(color_img)
        if len(circle) == 0:
            continue

        s = []
        # check the min height
        for c in circle:
            xyz = camera.xyz(c[0:2], depth_frame, depth_int)
            if xyz[2] > min_height:
                s.append(c)
                break
        if not len(s):
            continue


        circle = list(s)   
        # circle position
        if circle_prev:
            print("###")
            #circle_distance = distance(circle[0][0:2], circle_prev[0][0:2])
            circle_distance = np.linalg.norm(np.array(circle[0][0:2], dtype=np.float64) - np.array(circle_prev[0][0:2], dtype=np.float64) )
            if circle_distance < 30:
                continue
            else:
                print(circle_distance, circle, circle_prev)
        # update circle_prev
        circle_prev = list(circle)
                
        # go toward circle
        #xyz = camera.xyz(circle[0][0:2], depth_frame, depth_int)
        #xyz_circle_robot = xyz * T + B
        #xyz_circle_robot = np.array(xyz_circle_robot).flatten()
        #cmd = {"cmd": "lmove", "rel": 0, "x": xyz_circle_robot[0], "y": xyz_circle_robot[1], "cont": 1, "corner": 80, "id": i+1}
        #robot.play(**cmd)
        #print(cmd)        
        i += 1

    camera.off()
    #robot.close()


def follow_coin_copy():
    min_height = 570
    min_distance = 60
    num_seg = 10
    T = np.matrix([[ 1.00362522e+00, -6.08170296e-03,  2.14515751e-02],
                     [-1.16490803e-02, -1.01397851e+00,  2.47533302e-04],
                     [ 2.87219177e-01, -1.45933692e-01,  1.07591355e-01]])
    B = np.matrix([[142.69219896,  81.35848784, -47.9392697 ]] )   
    
    # camera object
    with open("config.json") as json_file:
        arg = json.load(json_file)
    camera = dcamera(arg)
    camera.on()

    # robot object and tool length is 60.58
    ip, port = "192.168.254.23", 443
    robot = dorna()
    robot.connect(ip, port)

    stop = False
    point_list = []
    step_list = []
    # get current robot xyz
    sys = dict(robot.sys)
    xyz_final = np.array([sys[x] for x in ["x", "y", "z"]])
    circle = False
    circle_prev = False
    j = 0
    while not stop:
        # take img
        depth_frame, ir_frame, color_frame, depth_img, ir_img, color_img, depth_int = camera.get_all()

        # find corners and reshape them
        circle = find_circle(color_img)

        # number of circles and their size
        if len(circle) == 0:
            continue

        s = []
        for c in circle:
            if c[2] >18:
                xyz = camera.xyz(c[0:2], depth_frame, depth_int)
                if xyz[2] > min_height:
                    s.append(c)
                    break
        if not len(s):
            continue


        circle = s   
        # circle position
        if circle_prev:
            if np.linalg.norm(np.array(circle[0][0:2]) - np.array(circle_prev[0][0:2])) < 20:
                continue
        
        # update circle_prev
        circle_prev = list(circle)
        
        
        # go toward circle
        #xyz = camera.xyz(circle[0][0:2], depth_frame, depth_int)
        xyz_circle_robot = xyz * T + B
        xyz_circle_robot = np.array(xyz_circle_robot).flatten()
        distance = np.floor(np.linalg.norm(xyz_final-xyz_circle_robot)) # floor distance
        if distance < min_distance:
            cmd = {"cmd": "lmove", "rel": 0, "x": xyz_circle_robot[0], "y": xyz_circle_robot[1], "id": 101, "cont":0, "corner": 80}
            robot.play(**cmd)
            print(json.dumps(cmd))
            xyz_final = np.array(xyz_circle_robot)
        else:
            end = xyz_final + (min_distance/distance) * (xyz_circle_robot - xyz_final) 
            pnts = []
            for i in range(num_seg):
                end = xyz_final + ((min_distance*(i+1))/(num_seg*distance)) * (xyz_circle_robot - xyz_final)
                pnts.append([end[0], end[1], 100 + i])
                cmd = {"cmd": "lmove", "rel": 0, "x": end[0], "y": end[1], "id": 100+i, "cont": 1, "corner": 80}
                robot.play(**cmd)
                print(json.dumps(cmd))
            xyz_final = np.array(end)
        while True:
            time.sleep(0.005)
            try:
                sys = dict(robot.sys)
                if (sys["id"] >=101 and sys["stat"] == 2) or sys["stat"] < 0:
                    break
            except Exception as ex:
                pass

        j += 1
    camera.off()
    robot.close()

if __name__ == '__main__':
    follow_coin()  
