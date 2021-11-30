import numpy as np
import cv2
from pylepton import Lepton
import time
#import capture_image as cap
import ctypes
import compare as cap
import capture_image as cap
ffc = ctypes.cdll.LoadLibrary("/home/pi/Documents/pylepton/leptonSDKEmb32PUB/libLepton_SDK.so")
image=np.zeros((64,64))


while (1):
	image=cap.step()
	try:
		detect=cv2.waitKey(1) & 0xFF	
		print(image.shape)
			#show image
		cv2.imshow('Example - Show image in window',image)	
	except AttributeError:
		pass
	if  detect== ord('q'):
		break 
cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows() # destroys the window showing image
	
	# ~ print(image)
