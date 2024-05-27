import matplotlib.pyplot as plt
from camera import Camera
import numpy as np


# create camera object
camera = Camera()

# connect to the camera, and set the mode to motion
camera.connect(mode="motion")

# start recording the motion data
camera.motion_rec()
print("start")

# wait for 1.5 seconds
time.sleep(1.5)

# stop recording and retrieve the motion data
accel, gyro = camera.motion_stop()
print("end")

# close the camera connection
camera.close()

################
# display data #
################
# Example list of points
points = gyro

# Extract x, y, z, and t from the points
x = [point[0] for point in points]
y = [point[1] for point in points]
z = [point[2] for point in points]
t = [point[3] for point in points]

# Calculate the norm of the vector [x, y, z] for each point
signal = [np.linalg.norm([x[i], y[i], z[i]]) for i in range(len(points))]

# Compute Fourier transformation
fourier_transform = np.fft.fft(signal)
frequencies = np.fft.fftfreq(len(signal), t[1] - t[0])  # Frequency values

plt.subplot(2, 1, 1)
plt.plot(t, signal, "-")
plt.title('Original Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
plt.plot(frequencies, np.abs(fourier_transform), "-")
plt.title('Fourier Transformation')
plt.xlabel('Frequency')
plt.ylabel('Magnitude')

plt.show()