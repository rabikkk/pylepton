# pylepton Library

# FLIR_LEPTON_Raspberry_Pi

## Kullanılan Önemli siteler

Görüntü alma --> [pylepton](https://github.com/groupgets/pylepton.git)

##  Görüntü Alma Kod içerikleri

-- [frame_learn.py]() :Alınan görüntünün frame hızını vermektedir.

-- [background_subtruct.py](): Opencv ile arka plan çıkartılması sağlanmıştır.

-- [capture_image.py]() :Görüntünün alınması kütüphane haline getirilmiştir. 

multithresh uygulanmıştır!!

```
import numpy as np
import cv2
from pylepton import Lepton
import time
import capture_image as cap

image=np.zeros((640,480,3))

while (1):
	image=cap.capture()
	print(image.shape)
		#show image
	cv2.imshow('Example - Show image in window',image)	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break 
cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows() # destroys the window showing image
```

-- [deneme.py]() : kütüphane denemesidir.

--[image_save.py]() : görüntüleri kaydetme için yazılmıştır.

-- [multi_thresh.py]() :multi thresh uygulanmıştır.

# NOT!!!

## perform ffc'nin c'den python'a entegre edilmesi:


`cd pylepton/leptonSDKEmb32PUB`

Lepton_I2C.c ve Lepton_I2C.h dosyalarını bu klasörün içine aktarınız.

`gcc -Wall -fPIC -c Lepton.I2C.c LEPTON_SDK.c LEPTON_SYS.c LEPTON_I2C_Protocol. c LEPTON_I2C_Service.c LEPTON_OEM.c
raspi_I2C.c crc16fast.c LEPTON_VID.c`

`gcc -shared -Wl,-soname,libLepton_SDK.so.1 -o libLepton_SDK.so Lepton_I2C.o LEPTON_I2C_Protocol.o LEPTON_I2C_Service.o LEPTON_OEM.o raspi_I2C.o LEPTON_SYS.o crc16fast.o LEPTON_VID.o LEPTON_SDK.o`

daha sonra ffc çağırdığınız dosyanın içine bu kod parçacığını ekleyin.

```
import ctypes
ffc = ctypes.cdll.LoadLibrary("/home/pi/Documents/pylepton/leptonSDKEmb32PUB/libLepton_SDK.so")
```
```
if ffc_counter==0:
	ffc.lepton_perform_ffc()
	ffc_counter+=1
	print("perform ffc")
```
[KAYNAK](https://groups.google.com/g/flir-lepton/c/i8rq6g7wZuQ)
## Deep learning Model Kodları

--[label.txt](): classların adının yazıldığı dosyalardır.

```
open
close
last

```
--[detect.tftlite](): Bilgisayarda öğretilen  modelin tflite dönüşmüş halidir.

--[detect.py](): modelin çalıştırıldığ koddur.

## Gereksinimler:
İndirilmesi gerekenler:

```
pip3 install opencv-python 
sudo apt-get install libcblas-dev
sudo apt-get install libhdf5-dev
sudo apt-get install libhdf5-serial-dev
sudo apt-get install libatlas-base-dev
sudo apt-get install libjasper-dev 
sudo apt-get install libqtgui4 
sudo apt-get install libqt4-testv
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install python3-tflite-runtime

```
## Alınan Hata:
Yardım alınan [Repo'da](https://github.com/nicknochnack/TFODRPi) bu kısımlar sırayla 0-1-2-3 diye gitmektedir fakat benim oluşturduğum [detect.tftlite]() dosyası bu sıralamayla datayı oluşturmamıştır. 

```
def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  # ~ image.astype(int)
  # Get all output details
  boxes = get_output_tensor(interpreter, 1)
  classes = get_output_tensor(interpreter, 3)
  scores = get_output_tensor(interpreter, 0)
  count = np.int(get_output_tensor(interpreter, 2))

```
## KAYNAKLAR
[Tensorflow Object Detection Walkthrough with Raspberry Pi](https://github.com/nicknochnack/TFODRPi)

[Youtube kanalı-Tensorflow Object Detection in 5 Hours with Python | Full Course with 3 Projects](https://www.youtube.com/watch?v=yqkISICHH-U)

--> [Object Detection Training — Preparing your custom dataset](https://medium.com/deepquestai/object-detection-training-preparing-your-custom-dataset-6248679f0d1d)

[record to video on Raspberry_pi with python](http://www.learningaboutelectronics.com/Articles/How-to-record-video-Python-OpenCV.php)
    
    
