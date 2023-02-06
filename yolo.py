import os
import time
import math
import urllib

import pandas as pd
import numpy as np
import cv2

def download_model_weights():
    download_url = "https://pjreddie.com/media/files/yolov3.weights"
    
    print("downloading model weights...")
    opener = urllib.request.URLopener()
    opener.retrieve(download_url, "yolo-coco/yolov3.weights")
    print("model download is complete.")

    return



def get_predictions(raw_image):
    
    yolo_dir = "pretrain_yolo"
    confidence_lvl = 0.4
    treshold = 0.55
    
    #load the COCO class labels our YOLO model was trained on
    labelsPath = os.path.sep.join([yolo_dir, "coco.names"])
    labels = open(labelsPath).read().strip().split("\n")

    
    #creating colour list to represent each class label
    np.random.seed(250)
    colours = np.random.randint(0,255, size= (len(labels),3),dtype="uint8")
    
    
    # download model weights if not already downloaded
    model_found = 0
    files = os.listdir("yolo-coco")
    
    if "yolov3.weights" in files:
        model_found = 1

    if model_found == 0:
        download_model_weights()
    
    
    #derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([yolo_dir, "yolov3.weights"])
    configPath = os.path.sep.join([yolo_dir, "yolov3.cfg"])
    
    
    # load our YOLO object detector pretrained model trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO model from disk..")
    yolo_model = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    
    
    # load input image and grab its spatial dimensions
    nparr = np.frombuffer(raw_image, np.uint8)
    
    
    # decode image
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    (H, W) = image.shape[:2]
    
    
    # determine only the *output* layer names that we need from YOLO
    ln = yolo_model.getLayerNames()
    ln = [ln[i - 1] for i in yolo_model.getUnconnectedOutLayers()]
    
    
    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255, (416, 416), swapRB=True, crop=False)
    yolo_model.setInput(blob)
    start = time.time()
    layerOutputs = yolo_model.forward(ln)
    end = time.time()
    
    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))
    
    # initialize our lists of detected bounding boxes, confidences, and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []
    
    
    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected probability is greater than the minimum probability
            if confidence > confidence_lvl:
                # scale the bounding box coordinates back relative to the size of the image
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
                
    
    # apply non-maxima suppression to surpress weak overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_lvl, treshold)

    predictions = []
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # append prediction box coordinates, box display colors, labels and probabilities
            predictions.append({
                "boxes": boxes[i], 
                "color": [int(c) for c in colours[classIDs[i]]], 
                "label": labels[classIDs[i]], 
                "confidence": confidences[i]
            })
            
    return predictions
   
    