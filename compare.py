import numpy as np
import cv2
from pylepton import Lepton
import capture_image as cap
import time
from skimage.metrics import structural_similarity as ssim
	# ~ print("Measuring observed offsets")
	# ~ print("    Camera should be against uniform temperature surface")
imageA=np.zeros((64,64,3))
imageB=np.zeros((64,64,3))
image=np.zeros((64,64,3))
x=0	
start_time = time.time() # start time of the loop
def capture():
	#while True:
		global x,imageA,imageB,start_time
		if x%2==0:
			imageA=cap.capture()
			imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
		else:
			imageB=cap.capture()
			imageB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
		x+=1
		if x>=2:
			s = ssim(imageA, imageB)
			if s==1.0:
				pass
			else:
				image=cv2.cvtColor(imageA,cv2.COLOR_GRAY2RGB)
				# ~ # FPS = 1 / time to process loop
				# ~ print("FPS: ", 1.0 / (time.time() - start_time))
				# ~ start_time=time.time() 
				#print(imageA.shape)
				return image
				#print(1.0 / (time.time() - start_time))
				#start_time = time.time() # start time of the loop
				
# ~ while True:
	# ~ image=capture()
	# ~ try:

		# ~ print("checked for shape".format(image.shape))
		# ~ cv2.imshow('Example - Show image in window',image)	
	# ~ except AttributeError:
		# ~ pass
	# ~ if cv2.waitKey(1) & 0xFF == ord('q'):
			# ~ break 
		# ~ #print("shape not found")
    # ~ #code to move to next frame
# ~ #	print(image.shape)
# ~ cv2.waitKey(0) # waits until a key is pressed
# ~ cv2.destroyAllWindows() # destroys the window showing image			
