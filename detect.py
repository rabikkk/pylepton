import re
import cv2
import tensorflow as tf
import numpy as np
from capture_image import step
import time
from PIL import Image
from matplotlib import pyplot as plt

def preprocess_image(frame):
  """Preprocess the input image to feed to the TFLite model"""
  frame = tf.image.convert_image_dtype(frame, tf.uint8)
  original_image = frame
  resized_img = frame[tf.newaxis, :]
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

new='rabia'
old='rabia'

def run_odt_and_draw_results(frame,labels, interpreter, threshold):
  global new,old

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
    new=int(obj['class_id'])
    if new!=old:
      old=new
      print(labels[int(obj['class_id'])])
    
  original_uint8 = original_image_np.astype(np.uint8)
  return original_uint8

def main():
    img1 = cv2.imread('black.bmp')#black image
    labels = load_labels()
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path='/home/pi/Desktop/pylepton/models/ssdmobnet/detect1.tflite')
    interpreter.allocate_tensors()
    threshold=0.7
    while True:
        frame = step()
        psnr = cv2.PSNR(img1,frame)
        if psnr>5.0:
        # Run inference and draw detection result on the local copy of the original file
          frame = run_odt_and_draw_results(frame,labels, interpreter,threshold)
        cv2.imshow('Pi Feed', frame)

        if cv2.waitKey(10) & 0xFF ==ord('q'):
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

