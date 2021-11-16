import re
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import tensorflow as tf
import numpy as np
# ~ from capture_image import capture
import time
from compare import capture 
path='alex.txt'
model_path="models/lenet/model.tflite"

def load_labels(path):
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
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image

def classify_image(interpreter, image, top_k=1):
  """Returns a sorted array of classification results."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = np.squeeze(interpreter.get_tensor(output_details['index']))

  # If the model is quantized (uint8 data), then dequantize the results
  if output_details['dtype'] == np.uint8:
    scale, zero_point = output_details['quantization']
    output = scale * (output - zero_point)

  ordered = np.argpartition(-output, top_k)
  return [(i, output[i]) for i in ordered[:top_k]]

def main(image):
  labels = load_labels(path)
  interpreter = tf.lite.Interpreter(model_path)
  interpreter.allocate_tensors()
  _, height, width, _ = interpreter.get_input_details()[0]['shape']
  results = classify_image(interpreter, image)
  label_id, prob = results[0]
  cv2.imshow('Pi Feed',image)

  if cv2.waitKey(10) & 0xFF ==ord('q'):
    cap.release()
    cv2.destroyAllWindows()


labels = load_labels(path)
interpreter = tf.lite.Interpreter(model_path)
interpreter.allocate_tensors()
_, height, width, _ = interpreter.get_input_details()[0]['shape']
while True:
  start_time = time.time() # start time of the loop
  image=capture()
  try:
    if image.shape!=None:
      results = classify_image(interpreter, image)
      label_id, prob = results[0]
      cv2.imshow('Pi Feed',image)
      if prob>0.6: 
        print( '%s %.2f\n' % (labels[label_id], prob))
  except AttributeError:
    pass
    
  if cv2.waitKey(1) & 0xFF == ord('q'):
			break 

cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows() # destroys the window showing image		
