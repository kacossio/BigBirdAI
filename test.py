import cv2
import numpy as np
import torch
import time
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

video_capture = cv2.VideoCapture(0)
background = None
backgrounds = []
t_start = time.time()
fs = 0 # frames

labels = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]

while True:

	# read a frame
	ret, frame = video_capture.read()
	# do detection on that frame
	results = model(frame)
	all_results = results.xyxy[0].tolist()

	# for every frame...
	for x1,y1,x2,y2,conf,lab in all_results:
		# get the class name
		class_of_obj = labels[int(lab)]

		if class_of_obj == "bird":
			frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 3)
			# @Kevin, do whatever stuff u want here, cropping can be done using x1,x2,y1,y2. For now I just have it display the image
	fs += 1
	print("FPS:", fs / (time.time() - t_start))


	try:
		cv2.imshow("a", frame)
	except Exception as e:
		print(e)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break