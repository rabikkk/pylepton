import numpy as np
import cv2
from pylepton import Lepton
import time
# ~ import ctypes
# ~ so_file = "/home/pi/Documents/pylepton/leptonSDKEmb32PUB/ffc.so"
# ~ ffc_enable=ctypes.CDLL(so_file)
dim = (800,600)#new dimensions

with Lepton() as l:
	while True:
		start_time = time.time() # start time of the loop
		image,_ = l.capture()#capture image
		cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX) # extend contrast
		image=np.uint8(image)#changing type for image
		
	# resize image 80,60 to 160,120
		image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)	
		
		
		dst = cv2.blur(image, (5000,5000)) 
		avg_hist = image.mean()
		ffc = (image/dst)*avg_hist
		
		
		

		'''
		#size_learn
		dimensions = image.shape
		print(dimensions)
		'''
		
		#RGB image
		# ~ image = cv2.applyColorMap(image, cv2.COLORMAP_JET)

	
		#show image
		# ~ plt.imshow(np.squeeze(image),cmap="gray", vmin=0, vmax=4096)
		# ~ plt.show()
		cv2.imshow('Example - Show image in window',ffc)
		print(image)	
		#closed
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break 
			
	cv2.waitKey(0) # waits until a key is pressed
	cv2.destroyAllWindows() # destroys the window showing image
	
