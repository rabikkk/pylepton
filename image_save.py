import numpy as np
import cv2
from pylepton import Lepton
import capture_image as cap
import time
	# ~ print("Measuring observed offsets")
	# ~ print("    Camera should be against uniform temperature surface")
mean_offset = np.zeros((60,80))
image=np.zeros((64,64,3))
dim = (640,640)
counter=np.ones((6,1))
index=9
image1=np.zeros((64,64,3))

while True:
	detect =cv2.waitKey(1)& 0xFF
	start_time = time.time() # start time of the loop
	image=cap.capture()		
		
	# ~ if detect== ord('1'): #take a screenshot if '1' is pressed
	cv2.imwrite('images/'+str(index)+'.step/index'+str(index)+str(int(counter[0]))+'.bmp',image) #save screenshot as index.bmp
	counter[0]+=1
	# ~ if detect  == ord('2'): #take a screenshot if '2' is pressed
	# ~ cv2.imwrite('images/'+str(index)+'.step/open'+str(index)+str(int(counter[1]))+'.bmp',image) #sa222ve screenshot as open.bmp
	# ~ counter[1]+=1
	# ~ if detect == ord('3'): #take a screenshot if '3' is pressed
	# ~ cv2.imwrite('images/'+str(index)+'.step/close'+str(index)+str(int(counter[2]))+'.bmp',image) #save screenshot as close.bmp
	# ~ counter[2]+=1
	# ~ if detect == ord('4'): #take a screenshot if '4' is pressed
	# ~ cv2.imwrite('images/'+str(index)+'.step/last'+str(index)+str(int(counter[3]))+'.bmp',image) #save screenshot as last.bmp
	# ~ counter[3]+=1
	# ~ if detect == ord('5'): #take a screenshot if '5' is pressed
	# ~ cv2.imwrite('images/'+str(index)+'.step/dynamic'+str(index)+str(int(counter[4]))+'.bmp',image) #save screenshot as last.bmp
	# ~ counter[4]+=1
	#show image
	#cv2.putText(image,str(1.0 / (time.time() - start_time)),(10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),2,cv2.LINE_AA) 
	image = cv2.resize(image, None, fx=5, fy=5, interpolation = cv2.INTER_CUBIC)
		
	cv2.imshow('Example - Show image in window',image)
	
	if detect == ord('q'):
		break 
cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows() # destroys the window showing image
	
