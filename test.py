import cv2
import numpy as np
import torch
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

video_capture = cv2.VideoCapture(0)
background = None
backgrounds = []
while True:
	ret, frame = video_capture.read()
	results = model(frame[:, :, ::-1])
	# print(results)
	results.save("test")
	# break
	# results.save("results.png")
	# results.save("results.png")

	# if len(backgrounds) < 5:
	# 	backgrounds += [frame]
	# 	print(len(backgrounds))
	# elif len(backgrounds) == 5:
	# 	background = backgrounds[0]
	# 	background = cv2.addWeighted(background, 0.2**(1/8.0), backgrounds[1], 0.2**(1/8.0), 0)
	# 	background = cv2.addWeighted(background, 0.2**(1/8.0), backgrounds[2], 0.2**(1/4.0), 0)
	# 	background = cv2.addWeighted(background, 0.2**(1/4.0), backgrounds[3], 0.2**(1/2.0), 0)
	# 	background = cv2.addWeighted(background, 0.2**(1/2.0), backgrounds[4], 0.2,          10)
	# 	backgrounds = range(6)
	# 	print("here")
	# else:
	# 	less_background = frame - background
	# 	hsv = cv2.cvtColor(less_background, cv2.COLOR_BGR2HSV)
	# 	h, s, v = cv2.split(hsv)

	# 	v[v > 0] += 1
	# 	v[v <= 0] = 0

	# 	final_hsv = cv2.merge((h, s, v))

	# 	less_background = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
		
	# 	# ret, thresh = cv2.threshold(less_background, 127, 255, 0)
	# 	imgray = cv2.cvtColor(less_background,cv2.COLOR_BGR2GRAY)
	# 	thresh, less_background = cv2.threshold(imgray, 100, 255, cv2.THRESH_BINARY)
	# 	# kernel = np.ones((5,5),np.uint8)
	# 	# dilation = cv2.dilate(less_background,kernel,iterations = 1)

	# 	kernel = np.ones((10,10),np.uint8)
	# 	dilation = cv2.erode(less_background, kernel, iterations=3)

	# 	_, contours, _ = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	# 	target_contours = [x for x in contours if cv2.contourArea(x) > 10000 and cv2.contourArea(x) < 500000]			
	# 	cv2.drawContours(frame, target_contours, -1, (0,0,255), 6)

	# 	convex_contours = [cv2.convexHull(x) for x in target_contours]
	# 	cv2.drawContours(frame, convex_contours, -1, (0,255,0), 6)

	# 	bounding_boxes = [cv2.boundingRect(x) for x in target_contours]
	# 	for box in bounding_boxes:
	# 		x,y,w,h = box
	# 		cv2.rectangle(frame,(x,y),(x+w,y+h), (255,0,0), 6)
	
	result = cv2.imread("test/image0.jpg")
	try:
		cv2.imshow("a", result)
	except Exception as e:
		print(e)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break