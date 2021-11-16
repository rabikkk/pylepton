import numpy as np
import cv2
from pylepton import Lepton
import skvideo.io
from skimage import data
from skimage.filters import threshold_multiotsu
	# ~ print("Measuring observed offsets")
	# ~ print("    Camera should be against uniform temperature surface")
mean_offset = np.zeros((60,80))
k=0
img_array = []	
font = cv2.FONT_HERSHEY_SIMPLEX

open_image = cv2.imread('open.png',0)
open_image1 = cv2.imread('open1.png',0)
open_image2 = cv2.imread('open2.png',0)
open_image3 = cv2.imread('open3.png',0)

index_image= cv2.imread('index.png',0)
index_image1= cv2.imread('index1.png',0)
index_image2= cv2.imread('index2.png',0)
close_image= cv2.imread('close.png',0)
main_image= cv2.imread('main1.png',0)

w, h = open_image.shape[::-1]
w1, h1 = close_image.shape[::-1]
dim = (800,600)
i=0


with Lepton() as l:
	while True:

		image,_ = l.capture()

		# ~ if np.max(image)<6850:
			# ~ image=mean_offset
		cv2.normalize(image,image, 0,255, cv2.NORM_MINMAX) # extend contrast


		image=np.uint8(image)
		i+=1
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
		cv2.normalize(regions,regions, 0,255, cv2.NORM_MINMAX) # extend contrast
		image=np.uint8(regions)
		
	

		print(np.max(image))
		for i in range(len(image)):
			for k in range(len(image[i])):
				if image[i][k]!=255:
					image[i][k]=0

			
		open_res = cv2.matchTemplate(image,open_image,cv2.TM_CCOEFF_NORMED)
		open_res1 = cv2.matchTemplate(image,open_image1,cv2.TM_CCOEFF_NORMED)
		open_res2 = cv2.matchTemplate(image,open_image2,cv2.TM_CCOEFF_NORMED)
		open_res3 = cv2.matchTemplate(image,open_image3,cv2.TM_CCOEFF_NORMED)
		
		threshold = 0.8
		
		loc =np.where( open_res1>= threshold )or np.where( open_res2>= threshold )or np.where( open_res3>= threshold )
		res1 = cv2.matchTemplate(image,close_image,cv2.TM_CCOEFF_NORMED)
		loc1 = np.where( res1 >= threshold)
		index_res = cv2.matchTemplate(image,index_image,cv2.TM_CCOEFF_NORMED)
		index_res1 = cv2.matchTemplate(image,index_image1,cv2.TM_CCOEFF_NORMED)
		index_res2 = cv2.matchTemplate(image,index_image2,cv2.TM_CCOEFF_NORMED)
		loc2 =np.where( index_res2 >= threshold)
		res3 = cv2.matchTemplate(image,main_image,cv2.TM_CCOEFF_NORMED)
		loc3 = np.where( res3 >= threshold)
		image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
		image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
		
		for pt in zip(*loc[::-1]):
			cv2.putText(image, 'Open', (10,450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
		for pt in zip(*loc1[::-1]):
			cv2.putText(image, 'Close', (10,450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
		for pt in zip(*loc2[::-1]):
			cv2.putText(image, 'Index', (10,450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
		for pt in zip(*loc3[::-1]):
			cv2.putText(image, 'Main', (10,450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
		# ~ for pt in zip(*loc1[::-1]):
			# ~ cv2.rectangle(image, pt, (pt[0] + w1, pt[1] + h1), (0,255,0), 2)
		# ~ image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
		# ~ print(image)
		# resize image
		# ~ //image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
		img_array.append(image)#inserting image on array
		#show image
		cv2.imshow('Example - Show image in window',image)	
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
skvideo.io.vwrite("video8.mp4", img_array)#record video
cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows() # destroys the window showing image
	
