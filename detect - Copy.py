import time
import cv2
import mrcnn.model as modellib
import balloon
import os
import sys
import random
import colorsys

import numpy as np
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import IPython.display
import math

class picker(object):
	"""docstring for ClassName"""
	def __init__(self, img, cnts, 
				radius_gap=40, 
				circle_defect = 0.7, 
				area = 625, 
				open_arc = [200, 60]):
		super(picker, self).__init__()
		self.img = img
		self.cnts = cnts

		# region parameters
		self.radius_gap = radius_gap # the amount we add to the radisu of the circle around contour
		self.circle_defect = circle_defect # 
		self.area = area # the minimum area of the candidate mushroom
		self.open_arc = open_arc # the minimum free arc for the candidate mushroom

		# [radius, (center_x, center_y)]
		self.circle = [[] for i in range(len(self.cnts))]

		# [[[star_0, length_0], [star_1, length_1], ...], [], ...]
		self.bad_region = [[[] for j in range(len(self.cnts)) ] for i in range(len(self.cnts))]
		self.good_region = [[] for i in range(len(self.cnts))]
		
		"""
		[start_degree, type]
		type: 0 if the region toward the open_arc[0] is free
		type 1: if the region toward open_arc[1] is free 
		"""
		self.candidate_region = [[] for i in range(len(self.cnts))]

	"""
	return circle, bad and good region
	"""
	def region(self):
		"""
		circle
		"""
		for i in range(len(self.cnts)):
			(x,y),r = cv2.minEnclosingCircle(self.cnts[i])
			c = (int(x),int(y))
			r = int(r)
			cv2.circle(self.img, c,r+self.radius_gap,(0,255,0),1)
			self.circle[i] = [r, c]
		"""
		bad region
		"""
		for i in range(len(self.cnts)):
			for j in range(len(self.cnts)):
				if i < j:
					r_1 = self.circle[i][0] + self.radius_gap
					r_2 = self.circle[j][0] + self.radius_gap

					d = math.hypot((self.circle[i][1][0]- self.circle[j][1][0]), (self.circle[i][1][1]- self.circle[j][1][1]))
					
					# check the intersect
					if d < r_1 + r_2: 
						theta_1 = math.acos((r_1**2 + d**2 - r_2**2)/(2*r_1*d))
						theta_2 = math.acos((r_2**2 + d**2 - r_1**2)/(2*r_2*d))

						# theta varies from -pi to pi
						theta = math.atan2((self.circle[j][1][1]- self.circle[i][1][1]), (self.circle[j][1][0]- self.circle[i][1][0]))
						
						# update bad_region
						self.bad_region[i][j] = [int(math.degrees(theta-theta_1))%360,  min(int(2*math.degrees(theta_1)), 360)]
						self.bad_region[j][i] = [int(math.degrees(math.pi+theta-theta_2))%360, min(int(2*math.degrees(theta_2)), 360)]

						# draw bad region
						cv2.line(self.img, self.circle[i][1],self.circle[j][1],(0,0,255),1)
						cv2.ellipse(self.img, self.circle[i][1], (r_1, r_1),0, self.bad_region[i][j][0],sum(self.bad_region[i][j]), (0,0,255), 3, 8,0)
						cv2.ellipse(self.img, self.circle[j][1], (r_2, r_2),0, self.bad_region[j][i][0],sum(self.bad_region[j][i]), (0,0,255), 3, 8,0)

		"""
		good region
		"""
		h, w, _ = self.img.shape
		for i in range(len(self.cnts)):
			# candidate
			candidate = False
			_area = cv2.contourArea(self.cnts[i])
			r = self.circle[i][0] + self.radius_gap
			x,y = self.circle[i][1] 
			
			# make sure the candidate looks like circle
			if x >=r and y >= r and x<= w-r and y <= h-r: 
				if _area >= max(self.area, self.circle_defect*math.pi*(r-self.radius_gap)**2):
					candidate= True

			# make sure the contour is a candidate
			if not candidate:
				continue 
			
			# break long intervals
			b_r = []
			for b in self.bad_region[i]:
				if b:		
					if sum(b) >= 360: # (b_0, 359) (0, sum(b)%360)
						b_r.append([b[0], 359-b[0]])
						b_r.append([0, sum(b)%360])
					else:
						b_r.append(b)

			# sort b_r
			b_r = np.array(b_r)
			if b_r.any():
				b_r = b_r[b_r[:, 0].argsort()] 
			b_r = b_r.tolist()

			# remove redundant intervals
			j = 0
			while j < len(b_r)-1:
				if sum(b_r[j+1]) < sum(b_r[j]): 
					b_r.pop(j+1) # this is redundant
				elif b_r[j+1][0] < sum(b_r[j]):
					b_r[j] = [b_r[j][0], sum(b_r[j+1]) - b_r[j][0]] # merge the two
					b_r.pop(j+1)
				else:
					j += 1	
			
			# g_r 
			if b_r:
				g_r = []
				start = 0
				end = 360
				for b in b_r:
					g_r.append([start, b[0]-start])
					start = sum(b)
				g_r.append([sum(b), 360-sum(b)])

				# connect [a 360-a] and [0, b]
				if len(g_r) > 1 and g_r[0][0] == 0 and sum(g_r[-1]) == 360:
					g_r[-1] = [g_r[-1][0], g_r[-1][1] + g_r[0][1]]
					g_r.pop(0)

			else:
				g_r = [[0, 360]]

			# update pick region
			self.good_region[i] = g_r

			print(self.good_region[i])
			#plot good region
			for g in self.good_region[i]:		
				cv2.ellipse(self.img,self.circle[i][1], (self.circle[i][0] + self.radius_gap, self.circle[i][0] + self.radius_gap),0,g[0],sum(g), (255,0,0), 3, 8,0)


	def candidate(self):
		for i in range(len(self.cnts)):
			# search over good region
			for g in self.good_region[i]:
				if g[1] > self.open_arc[0]:
					# [radius, (center), (angel_start, angel_length), type]
					self.candidate_region[i] = [self.circle[i][0], self.circle[i][1], g, 0]
					theta = math.radians(g[0] + g[1]/2 - 90) # start of the line and goes 180 
					x = int(self.circle[i][0]*math.cos(theta))
					y = int(self.circle[i][0]*math.sin(theta))
					font = cv2.FONT_HERSHEY_SIMPLEX
					cv2.putText(self.img, str(i), self.circle[i][1], font, 1, (0, 255, 0), 2, cv2.LINE_AA)
					cv2.line(self.img, (self.circle[i][1][0] + x, self.circle[i][1][1] + y),(self.circle[i][1][0] - x, self.circle[i][1][1] - y),(0,255,255),1)
					break



def random_colors(N, bright=True):
	"""
	Generate random colors.
	To get visually distinct colors, generate them in HSV space then
	convert to RGB.
	"""
	brightness = 1.0 if bright else 0.7
	hsv = [(i / N, 1, brightness) for i in range(N)]
	colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
	random.shuffle(colors)
	return colors

def display_instances(image, boxes, masks, class_ids, class_names,
						scores=None, title="",
						figsize=(16, 16), ax=None,
						show_mask=True, show_bbox=True,
						colors=None, captions=None):
	"""
	boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
	masks: [height, width, num_instances]
	class_ids: [num_instances]
	class_names: list of class names of the dataset
	scores: (optional) confidence scores for each box
	title: (optional) Figure title
	show_mask, show_bbox: To show masks and bounding boxes or not
	figsize: (optional) the size of the image
	colors: (optional) An array or colors to use with each object
	captions: (optional) A list of strings to use as captions for each object
	"""
	# Number of instances
	circle = []
	contour = []
	N = boxes.shape[0]
	if not N:
		print("\n*** No instances to display *** \n")
	else:
		assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]
	
	if show_mask:
		masked_image = image.astype(np.uint32).copy()
		colors = colors or random_colors(N)
	
	for i in range(N):
		# Mask
		mask = masks[:, :, i].astype(np.uint8)
		if show_mask:
			masked_image = apply_mask(masked_image, mask, colors[i])

		# find contours
		contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		cv2.drawContours(image, contours, -1, (0,255,0), 1)
		cnt = contours[0]
		contour.append(contours[0])

	# picker class
	gap = 15
	area = 400
	defect = 0.7
	open_arc = 200 # if there 
	free_arc = 90
	test = picker(image, contour)
	test.region()
	test.candidate()
	return image


if __name__ == '__main__':
	"""
	camera
	"""
	import json
	from camera import camera
	with open("config.json") as json_file:
		arg = json.load(json_file)
	camera = camera(arg)
	camera.on()

	"""
	detection model
	"""
	model_dir = "C://Users//hossein//Desktop//"
	weight_path = "C://Users//hossein//Desktop//Dorna_AI//mask_rcnn//Mask_RCNN//logs//mushroom20191222T2338//mask_rcnn_mushroom_0030.h5"
	# configuration model
	config = balloon.BalloonConfig()

	# Create model object in inference mode.
	model = modellib.MaskRCNN(mode="inference", model_dir=model_dir, config=config)

	# Load weights trained on MS-COCO
	model.load_weights(weight_path, by_name=True)

	class_names = ["BG", "mushroom"]


	for i in range(5):
		depth_frame, ir_frame, color_frame = camera.frame()

		#depth_img = np.asanyarray(depth_frame.get_data())
		#depth_img = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)
		#ir_img = np.asanyarray(ir_frame.get_data())
		img = np.asanyarray(color_frame.get_data())
		# Run detection
		results = model.detect([img], verbose=0)
		# Visualize results
		r = results[0]

		img = display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], show_mask=False, show_bbox=False)
		cv2.imshow("mushroom detection ",img)
		cv2.waitKey(0) 