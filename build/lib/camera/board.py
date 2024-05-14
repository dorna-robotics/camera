import cv2

class Aruco(object):
	"""docstring for aruco"""
	def __init__(self, dict_type = "DICT_6X6_250"):
		super(Aruco, self).__init__()
		self.dict_type = getattr(cv2.aruco, dict_type)

	def detect(self, img):
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		dictionary = cv2.aruco.getPredefinedDictionary(self.dict_type)
		prms =  cv2.aruco.DetectorParameters()
		detector = cv2.aruco.ArucoDetector(dictionary, prms)

		aruco_corner, aruco_id, aruco_reject = detector.detectMarkers(img_gray)
		return aruco_corner, aruco_id, aruco_reject, img_gray


class Charuco(Aruco):
	"""docstring for charuco"""
	def __init__(self, sqr_x=5, sqr_y=5, square_length=0.04, marker_length=0.02):
		super(Charuco, self).__init__()
		#dictionary = cv2.aruco.Dictionary_get(self.dict_type)
		dictionary = cv2.aruco.getPredefinedDictionary(self.dict_type)
		self.board = cv2.aruco.CharucoBoard((sqr_x, sqr_y), square_length, marker_length, dictionary)

	"""
	save charuco board, and save the result in a file
	"""
	def save(self, board_path="board.png", width=1000, height=1000, margin=0):
		cv2.imwrite(board_path, self.board.generateImage((width,height), margin, margin))

	"""
	detect charuco chess board corners
	"""
	def corner(self, img):
		# init
		response = 0
		charuco_corner = []
		charuco_id = []
		
		# aruco markers
		aruco_corner, aruco_id, aruco_reject, img_gray = self.detect(img)
		
		# Get charuco corners and ids from detected aruco markers
		if aruco_corner:
			response, charuco_corner, charuco_id = cv2.aruco.interpolateCornersCharuco(
				markerCorners=aruco_corner,
				markerIds=aruco_id,
				image=img_gray,
				board=self.board)
			
			if response:
				# refine
				charuco_corner = cv2.cornerSubPix(img_gray,
					charuco_corner,
					(11,11),
					(-1,-1),
					(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1, 0.001))
				
				cv2.aruco.drawDetectedCornersCharuco(img, charuco_corner, charuco_id, cornerColor=(0, 255, 0))
				
		return response, charuco_corner, charuco_id, img_gray

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

def main_charuco():
	test = Charuco(8, 8, 12, 8)
	test.save()
	"""
	img = cv2.imread("board.png")
	response, corner, ids, img_gray = test.corner(img)
	print(ids)
	cv2.imshow("charuco", img)
	cv2.waitKey(0)
	"""

def main_aruco():
	arc = Aruco()
	for i in range(3):
		img = cv2.aruco.drawMarker(cv2.aruco.getPredefinedDictionary(arc.dict_type), i, 500)
		cv2.imwrite("arc_"+str(i)+".png", img)

if __name__ == '__main__':
	main_aruco()