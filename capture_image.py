import numpy as np
import cv2
from pylepton import Lepton
import time
from skimage import data
from skimage.filters import threshold_multiotsu
from skimage.metrics import structural_similarity as ssim
import ctypes
ffc = ctypes.cdll.LoadLibrary("/home/pi/Documents/pylepton/leptonSDKEmb32PUB/libLepton_SDK.so")
ffc_counter=0
dim = (640,480)#new dimensions


imageA=np.zeros((64,64,3))
imageB=np.zeros((64,64,3))
image=np.zeros((64,64,3))
x=0
start_time = time.time() # start time of the loop
def step():
	while True:
		global x,imageA,imageB
		if x%2==0:
			imageA=capture()
		else:
			imageB=capture()
		x+=1
		if x>=2:
			s = ssim(imageA, imageB)
			if s==1.0:
				pass
			else:
				break
	return cv2.cvtColor(imageA,cv2.COLOR_GRAY2RGB)
				
		

def capture():
	global ffc_counter
	with Lepton() as l:
		image,_ = l.capture()
		if ffc_counter==0:
			ffc.lepton_perform_ffc()
			ffc_counter+=1
			print("perform ffc")

		cv2.normalize(image,image, 0,255, cv2.NORM_MINMAX) # extend contrast

		image=np.uint8(image)
		
		# Applying multi-Otsu threshold for the default value, generating
		# three classes.
		
		thresholds = threshold_multiotsu(image)
		
		# Using the threshold values, we generate the three regions.
		regions = np.digitize(image, bins=thresholds)
		#print(thresholds)
		cv2.normalize(regions,regions, 0,255, cv2.NORM_MINMAX) # extend contrast
		image=np.uint8(regions)
		
	
		ret,image = cv2.threshold(image, thresholds[0], 255, cv2.THRESH_BINARY)
		image = cv2.resize(image, None, fx=64/80, fy=64/60, interpolation = cv2.INTER_CUBIC)	
	return image
