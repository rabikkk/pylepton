import re
import cv2
import tensorflow as tf
import numpy as np
# ~ from capture_image import capture
from compare import capture 
import time
from PIL import Image
from matplotlib import pyplot as plt

def preprocess_image(frame):
  """Preprocess the input image to feed to the TFLite model"""
  # ~ frame1 = tf.image.convert_image_dtype(frame, tf.uint8)
  frame1= tf.image.convert_image_dtype(frame,dtype=tf.uint8)
  # ~ frame = tf.image.decode_png(frame, channels=3, dtype=tf.uint8)  
  original_image = frame1
  resized_img = frame1[tf.newaxis, :]
  return resized_img, original_image

def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""
  # Feed the input image to the model
  set_input_tensor(interpreter, image)
  interpreter.invoke()

  # Get all outputs from the model
  boxes = get_output_tensor(interpreter, 1)
  classes = get_output_tensor(interpreter, 3)
  scores = get_output_tensor(interpreter, 0)
  count = int(get_output_tensor(interpreter, 2))

  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {
        'bounding_box': boxes[i],
        'class_id': classes[i],
        'score': scores[i]
      }
      results.append(result)
  return results

def load_labels(path='labels.txt'):
  """Loads the labels file. Supports files with or without index numbers."""
  with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    labels = {}
    for row_number, content in enumerate(lines):
      pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
      if len(pair) == 2 and pair[0].strip().isdigit():
        labels[int(pair[0])] = pair[1].strip()
      else:
        labels[row_number] = pair[0].strip()
  return labels

def set_input_tensor(interpreter, image):
  """Set the input tensor."""
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image

def get_output_tensor(interpreter, index):
  """Retur the output tensor at the given index."""
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor

def run_odt_and_draw_results(frame,labels, interpreter, threshold):
  """Run object detection on the input image and draw the detection results"""
  # Load the input image and preprocess it
  preprocessed_image, original_image = preprocess_image(frame)

  # Run object detection on the input image
  results = detect_objects(interpreter, preprocessed_image, threshold=threshold)

  # Plot the detection results on the input image
  original_image_np = original_image.numpy().astype(np.uint8)
  for obj in results:
    # Convert the object bounding box from relative coordinates to absolute
    # coordinates based on the original image resolution
    ymin, xmin, ymax, xmax = obj['bounding_box']
    xmin = int(xmin * original_image_np.shape[1])
    xmax = int(xmax * original_image_np.shape[1])
    ymin = int(ymin * original_image_np.shape[0])
    ymax = int(ymax * original_image_np.shape[0])
    # Draw the bounding box and label on the image
    cv2.rectangle(original_image_np, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
    # Make adjustments to make the label visible for all objects
    y = ymin - 15 if ymin - 15 > 15 else ymin + 15

    # Find the class index of the current object
    class_id = int(obj['class_id'])
    print(labels[int(obj['class_id'])])
    
  original_uint8 = original_image_np.astype(np.uint8)
  return original_uint8

image=np.zeros((64,64,3))
labels = load_labels()
# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path='/home/pi/Documents/pylepton/models/ssdmobnet/model1.tflite')
interpreter.allocate_tensors()
threshold=0.7
start_time = time.time() # start time of the loop
shape=0
while True:
  image=capture()
  try:
    # ~ print(image.shape)
    if (image.shape)!=None:
      detection_result_image = run_odt_and_draw_results(image,labels, interpreter,threshold)
    # FPS = 1 / time to process loop
    # ~ print("FPS: ", 1.0 / (time.time() - start_time))
    # ~ start_time=time.time() 
      cv2.imshow('Pi Feed', detection_result_image)
  
  except AttributeError:
    pass
    
  if cv2.waitKey(1) & 0xFF == ord('q'):
			break 

cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows() # destroys the window showing image		
