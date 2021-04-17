"""
classic detection algorithm from cv2
"""
import numpy as np
import cv2
from camera import dcamera
import json
import math

from scipy.ndimage import label

def template_detection_2(im):
	import random as rng
	rng.seed(12345)	
	#hsv
	hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
	th, bw = cv2.threshold(hsv[:, :, 2], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

	# distance transformation
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
	morph = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
	dist = cv2.distanceTransform(morph, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

	# normalization
	cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)

	# make border
	borderSize = 50
	distborder = cv2.copyMakeBorder(dist, borderSize, borderSize, borderSize, borderSize, 
	                                cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)

	# kernel size
	gap = 20                                
	kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*(borderSize-gap)+1, 2*(borderSize-gap)+1))
	kernel2 = cv2.copyMakeBorder(kernel2, gap, gap, gap, gap, 
	                                cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)
	
	distTempl = cv2.distanceTransform(kernel2, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
	# normalization
	cv2.normalize(distTempl, distTempl, 0, 1.0, cv2.NORM_MINMAX)

	nxcor = cv2.matchTemplate(distborder, distTempl, cv2.TM_CCOEFF_NORMED)

	#cv2.normalize(nxcor, nxcor, 0, 1.0, cv2.NORM_MINMAX)

	# Threshold to obtain the peaks
	# This will be the markers for the foreground objects
	_, nxcor = cv2.threshold(nxcor, 0.4, 1.0, cv2.THRESH_BINARY)
	# Dilate a bit the dist image
	kernel1 = np.ones((3,3), dtype=np.uint8)
	nxcor = cv2.dilate(nxcor, kernel1)

	
	# Create the CV_8U version of the distance image
	# It is needed for findContours()
	dist_8u = nxcor.astype('uint8')
	# Find total markers
	contours, _ = cv2.findContours(dist_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# Create the marker image for the watershed algorithm
	markers = np.zeros(nxcor.shape, dtype=np.uint8)
	# Draw the foreground markers
	for i in range(len(contours)):
	    cv2.drawContours(markers, contours, i, (i+1), -1)
	# Draw the background marker
	cv2.circle(markers, (5,5), 3, (255,255,255), -1)

	# Perform the watershed algorithm
	cv2.watershed(im, np.int32(markers))
	#mark = np.zeros(markers.shape, dtype=np.uint8)
	mark = markers.astype('uint8')
	mark = cv2.bitwise_not(mark)


	# Generate random colors
	colors = []
	for contour in contours:
	    colors.append((rng.randint(0,256), rng.randint(0,256), rng.randint(0,256)))
	# Create the result image
	dst = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)
	# Fill labeled objects with random colors
	for i in range(markers.shape[0]):
	    for j in range(markers.shape[1]):
	        index = markers[i,j]
	        if index > 0 and index <= len(contours):
	            im[i,j,:] = colors[index-1]

	# Visualize the final image
	return im

def contour(im):
	#hsv
	hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
	th, bw = cv2.threshold(hsv[:, :, 2], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

	# distance transformation
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
	morph = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
	dist = cv2.distanceTransform(morph, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

	# make border
	borderSize = 50
	distborder = cv2.copyMakeBorder(dist, borderSize, borderSize, borderSize, borderSize, 
	                                cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)

	# kernel size
	gap = 10                                
	kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*(borderSize-gap)+1, 2*(borderSize-gap)+1))
	kernel2 = cv2.copyMakeBorder(kernel2, gap, gap, gap, gap, 
	                                cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)
	
	distTempl = cv2.distanceTransform(kernel2, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

	nxcor = cv2.matchTemplate(distborder, distTempl, cv2.TM_CCOEFF_NORMED)	

	mn, mx, _, _ = cv2.minMaxLoc(nxcor)
	th, peaks = cv2.threshold(nxcor, mx*0.5, 255, cv2.THRESH_BINARY)
	peaks8u = cv2.convertScaleAbs(peaks)
	
	contours, hierarchy = cv2.findContours(peaks8u, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

	peaks8u = cv2.convertScaleAbs(peaks)    # to use as mask
	for i in range(len(contours)):
		x, y, w, h = cv2.boundingRect(contours[i])
		_, mx, _, mxloc = cv2.minMaxLoc(dist[y:y+h, x:x+w], peaks8u[y:y+h, x:x+w])
		cv2.circle(im, (int(mxloc[0]+x), int(mxloc[1]+y)), int(mx), (255, 0, 0), 2)
		cv2.drawContours(im, contours, i, (0, 0, 255), 2)
	return im

def canny(im):
	kernel = np.ones((4,4),np.uint8)
	
	# edge or threshold
	edges = cv2.Canny(im,60,200)
	edges = cv2.dilate(edges,kernel,iterations = 2)

	# find contours
	contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	# find rotated rectangle
	for cnt in contours:
	    area = cv2.contourArea(cnt) 
	    # area filter
	    if area > 1000 and area <20000:
	        rect = cv2.minAreaRect(cnt)
	        # box ratio filter
	        if rect[1][0]/rect[1][1] < 2 and rect[1][0]/rect[1][1] > 0.5:
	            """
	            box = cv2.boxPoints(rect)
	            box = np.int0(box)
	            im = cv2.drawContours(im,[box],0,(0,255,0) ,1)
	            """
	            ellipse = cv2.fitEllipse(cnt)
	            cv2.ellipse(im,ellipse,(0,255,0),2)	            
	return im


def watershed(img):

	# Pre-processing.
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
	_, img_bin = cv2.threshold(img_gray, 0, 255,
			cv2.THRESH_OTSU)
	img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN,
			np.ones((3, 3), dtype=int))

	border = cv2.dilate(img_bin, None, iterations=10)
	border = border - cv2.erode(border, None)

	dt = cv2.distanceTransform(img_bin, 2, 3)
	dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(np.uint8)
	_, dt = cv2.threshold(dt, 180, 255, cv2.THRESH_BINARY)
	lbl, ncc = label(dt)
	lbl = lbl * (255 / (ncc + 1))
	# Completing the markers now. 
	lbl[border == 255] = 255

	lbl = lbl.astype(np.int32)
	cv2.watershed(img, lbl)

	lbl[lbl == -1] = 0
	lbl = lbl.astype(np.uint8)
	_, lbl = cv2.threshold(lbl, 170, 255, cv2.THRESH_BINARY)
	result = 255 - lbl

	# contours
	cnts, _ = cv2.findContours(lbl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	#cv2.drawContours(img, contours, -1, (0,255,0), 3)
	for cnt in cnts:
		rect = cv2.minAreaRect(cnt)
		box = cv2.boxPoints(rect)
		box = np.int0(box)
		
		mx = max(rect[1])
		mn = min(rect[1])
		
		if mx > 200 or mn ==0 or mx/mn > 1.8 or mx/mn < 1.2:
			continue  
		
		print(mx)
		#print(ratio)
		# draw axes
		center = (int(rect[0][0]), int(rect[0][1]))
		
		rotation = rect[2]
		if rotation in [0, 90, 180, 270, 360, -90, -180, -270]:
			continue

		rotation = math.radians(rotation)


		length = 20
		px = (int(center[0]+ length * math.cos(rotation)), int(center[1] + length * math.sin(rotation)))
		py = (int(center[0]- length * math.sin(rotation)), int(center[1] + length * math.cos(rotation)))
		cv2.line(img,center,px,(0,0,255),5)
		cv2.line(img,center,py,(0,0,255),5)

		# draw box
		cv2.drawContours(img,[box],0,(0,255,0),2)

	"""
	result[result != 255] = 0
	result = cv2.dilate(result, None)
	img[result == 255] = (0, 0, 255)        
	"""

	return img, img_bin


def find_by_collor(image):
	import cv2
	import numpy as np
	from imutils import contours


	original = image.copy()
	image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	mask = np.zeros(image.shape, dtype=np.uint8)

	colors = {
		'gray': ([76, 0, 41], [179, 255, 70]),        # Gray
		'blue': ([69, 120, 100], [179, 255, 255]),    # Blue
		'yellow': ([21, 110, 117], [45, 255, 255]),   # Yellow
		'orange': ([0, 110, 125], [17, 255, 255])     # Orange
	}

	# Color threshold to find the squares
	open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
	close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
	for color, (lower, upper) in colors.items():
	    lower = np.array(lower, dtype=np.uint8)
	    upper = np.array(upper, dtype=np.uint8)
	    color_mask = cv2.inRange(image, lower, upper)
	    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
	    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, close_kernel, iterations=5)

	    color_mask = cv2.merge([color_mask, color_mask, color_mask])
	    mask = cv2.bitwise_or(mask, color_mask)

	gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
	cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if len(cnts) == 2 else cnts[1]
	# Sort all contours from top-to-bottom or bottom-to-top
	(cnts, _) = contours.sort_contours(cnts, method="top-to-bottom")

	# Take each row of 3 and sort from left-to-right or right-to-left
	cube_rows = []
	row = []
	for (i, c) in enumerate(cnts, 1):
	    row.append(c)
	    if i % 3 == 0:  
	        (cnts, _) = contours.sort_contours(row, method="left-to-right")
	        cube_rows.append(cnts)
	        row = []

	# Draw text
	number = 0
	for row in cube_rows:
	    for c in row:
	        x,y,w,h = cv2.boundingRect(c)
	        cv2.rectangle(original, (x, y), (x + w, y + h), (36,255,12), 2)

	        cv2.putText(original, "#{}".format(number + 1), (x,y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
	        number += 1

	cv2.imshow('mask', mask)
	cv2.imwrite('mask.png', mask)
	cv2.imshow('original', original)
	cv2.waitKey()	

if __name__ == '__main__':
	import json


	with open("config.json") as json_file:
		arg = json.load(json_file)
	camera = dcamera(arg)
	camera.on()


	while True:
		try:
			depth_frame, ir_frame, color_frame, depth_img, ir_img, color_img, depth_int = camera.get_all()

			# watershed detection
			img, _ = watershed(color_img.copy())
			cv2.imshow("color",img)
			cv2.waitKey(1)
		except KeyboardInterrupt:
			break

	camera.off()