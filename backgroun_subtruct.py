import numpy as np
import cv2
from pylepton import Lepton

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

dim = (160,120)

with Lepton() as l:
	while True:
		image,_ = l.capture()
		cv2.normalize(image, image, 0, 65535, cv2.NORM_MINMAX) # extend contrast
		np.right_shift(image, 8, image) # fit data into 8 bits
		image=np.uint8(image)
		  
		# resize image
		image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
		fgmask = fgbg.apply(image)
		fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)	
		#show image
		cv2.imshow('Example - Show image in window',fgmask)	
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break 
	cv2.waitKey(0) # waits until a key is pressed
	cv2.destroyAllWindows() # destroys the window showing image
	
