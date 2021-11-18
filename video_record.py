import numpy as np
import cv2
from pylepton import Lepton
import capture_image as cap
import time

image=np.zeros((64,64,3))
dim = (640,640)
counter=0
index=9
width=64
height= 64

writer= cv2.VideoWriter('images/basicvideo.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 8, (width,height))

while True:
	image=cap.capture()
	writer.write(image)
	
	image = cv2.resize(image, None, fx=5, fy=5, interpolation = cv2.INTER_CUBIC)	
	cv2.imshow('Example - Show image in window',image)
	
	detect =cv2.waitKey(1)& 0xFF
	if detect == ord('q'):
		break 
cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows() # destroys the window showing image
writer.release()
	
