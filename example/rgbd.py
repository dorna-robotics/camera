from camera import Camera
import cv2


# create camera module
camera = Camera()

# connect to the camera 
camera.connect(filter={}, exposure=50000, gain=30)

# display depth image in a while loop, until press "q"
while True:
    # get the camera data
    depth_frame, ir_frame, color_frame, depth_img, ir_img, color_img, depth_int, frames, timestamp = camera.get_all()
    
    # display depth image
    #cv2.imshow("depth_img", depth_img)
    cv2.imshow("color_img", color_img)

    # get xyz in camera frame of the center pixel in the image
    height, width, _ = depth_img.shape
    xyz, sample = camera.xyz((width/2, height/2), depth_frame, depth_int)
    print("xyz: ", xyz)
    
    # exit if "q" is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# always close the camera connection, once your application is over
camera.close()