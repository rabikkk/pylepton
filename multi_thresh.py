import numpy as np
import cv2
from pylepton import Lepton

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from skimage import data
from skimage.filters import threshold_multiotsu

dim = (160,120)
# Setting the font size for all plots.
matplotlib.rcParams['font.size'] = 9

with Lepton() as l:
	while True:
		image,_ = l.capture()
		cv2.normalize(image, image, 0, 65535, cv2.NORM_MINMAX) # extend contrast
		np.right_shift(image, 8, image) # fit data into 8 bits
		image=np.uint8(image)
		
		# resize image
		image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
		
		
		# Applying multi-Otsu threshold for the default value, generating
		# three classes.
		
		thresholds = threshold_multiotsu(image)
		
		# Using the threshold values, we generate the three regions.
		regions = np.digitize(image, bins=thresholds)
		
		fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3.5))
		
		# Plotting the original image.
		ax[0].imshow(image, cmap='gray')
		ax[0].set_title('Original')
		ax[0].axis('off')
		
		# Plotting the histogram and the two thresholds obtained from
		# multi-Otsu.
		ax[1].hist(image.ravel(), bins=255)
		ax[1].set_title('Histogram')
		for thresh in thresholds:
		    ax[1].axvline(thresh, color='r')
		
		# Plotting the Multi Otsu result.
		ax[2].imshow(regions, cmap='jet')
		ax[2].set_title('Multi-Otsu result')
		ax[2].axis('off')
		
		plt.subplots_adjust()
		
		plt.show()		
		  

		
		
		#show image
		cv2.imshow('Example - Show image in window',image)	
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break 
	cv2.waitKey(0) # waits until a key is pressed
	cv2.destroyAllWindows() # destroys the window showing image
	
