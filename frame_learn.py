import numpy as np
import cv2
from pylepton import Lepton
import time

from skimage import data
from skimage.filters import threshold_multiotsu

dim = (160,120)#new dimensions

with Lepton() as l:
	while True:
		start_time = time.time() # start time of the loop
		image,_ = l.capture()#capture image
		cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX) # extend contrast
		image=np.uint8(image)#changing type for image
		# ~ print(image)
		
		#size_learn
		#dimensions = image.shape
		# ~ print(dimensions)
		# resize image
		# ~ image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
		
		# ~ image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
		# Applying multi-Otsu threshold for the default value, generating
		# three classes.
		
		thresholds = threshold_multiotsu(image)
		
		# Using the threshold values, we generate the three regions.
		regions = np.digitize(image, bins=thresholds)
		regions = np.digitize(image, bins=thresholds)
		cv2.normalize(regions,regions, 0,255, cv2.NORM_MINMAX) # extend contrast
		image=np.uint8(regions)
		ret, thresh1 = cv2.threshold(image, 254, 255, cv2.THRESH_BINARY)
		# ~ image=image/255.0
	

		# ~ for i in range(len(image)):
			# ~ for k in range(len(image[i])):
				# ~ if image[i][k]!=255:
					# ~ image[i][k]=0
					
					
		  
		# resize image 80,60 to 160,120
		# ~ image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
		'''
		#size_learn
		dimensions = image.shape
		print(dimensions)
		'''
		
		#RGB image
		# ~ image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
		
		
		# FPS = 1 / time to process loop
		print("FPS: ", 1.0 / (time.time() - start_time)) 
	
		#show image
		cv2.imshow('Example - Show image in window',thresh1)
			
		#closed
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break 
			
	cv2.waitKey(0) # waits until a key is pressed
	cv2.destroyAllWindows() # destroys the window showing image
	
