#!/usr/bin/env python
# coding: utf-8

# In[237]:


import cv2
import matplotlib.pyplot as plt
import sort
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import sort
from utils import *
"""
This is a YOLO based detection algorithms for tracking pedestrians designed for further simulation purposes.
we incorporated in the code image grid composition before inference in order to give accurate bounding boxes.
in the command line :
python detect.py --video_path {video_path} --crop_image {crop_image}
the code outputs a csv file with the detected xy coordinates, and video with the corresponding tracked bounding boxes

"""


class Object_Detector():
  """
  YOLO-v3 based object detector. This YOLO-v3 is pretrained on MS-COCO dataset.
  """
  def __init__(self,classes_path='./data/coco.names',cropper=False):
    self.network = cv2.dnn.readNet('./yolo/yolov3.weights', './yolo/yolov3.cfg')
    #self.image_path=image_path
    with open(classes_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    f.close()
    self.classes=classes
    self.cropper=cropper


  def crop_image(self):
    img = plt.imread(self.image_path)
    if self.cropper :
        height = img.shape[0]
        width = img.shape[1]
        # Cut the image in half
        width_cutoff = width // 2
        height_cutoff=height//2
        left1 = img[height_cutoff:, :width_cutoff]
        right1 = img[:height_cutoff, width_cutoff:]
        right2 = img[:height_cutoff, :width_cutoff]
        left2 = img[height_cutoff:, width_cutoff:]
        return left1,right1,right2,left2
    else:
        return img

  def get_yolo_output(self,input_image):
        image=input_image
        self.network.setInput(cv2.dnn.blobFromImage(image, 0.00392, (416,416), (0,0,0), True, crop=False))
        layer_names = self.network.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.network.getUnconnectedOutLayers()]
        outs = self.network.forward(output_layers)
        class_ids = []
        confidences = []
        boxes = []
        Width = image.shape[1]
        Height = image.shape[0]
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.1:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.1)
        ## class retrieval
        kept_boxes=[]
        for i in indices:
            #i = i[0]
            box = boxes[i]
            if class_ids[i]==0:
                kept_boxes.append(box)
                label = text = "{}: {:.4f}".format(self.classes[class_ids[i]], confidences[i])
                cv2.rectangle(image, (round(box[0]),round(box[1])), (round(box[0]+box[2]),round(box[1]+box[3])), (0, 0, 0), 2)
                cv2.putText(image, label, (round(box[0])-10,round(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                box.append(round(confidences[i],2))
                kept_boxes.append(box)
        return kept_boxes,image


  def deep_detect(self,show_results=False):
    outputs=list(self.crop_image())
    image_datas=[]
    boxes=[]
    for element in outputs:
        kept_boxes,image=self.get_yolo_output(element)
        boxes.append(kept_boxes)
        image_datas.append(image)
    if show_results:
        if self.cropper :
            f, axarr = plt.subplots(2,2)
            axarr[0,0].imshow(image_datas[0])
            axarr[0,1].imshow(image_datas[1])
            axarr[1,0].imshow(image_datas[2])
            axarr[1,1].imshow(image_datas[3])
        else:
            plt.imshow(image)
    return list(np.concatenate(boxes))

  def track(self,video_filename):
      try:
          tracks = {}
          person_tracker = sort.Sort()
          #video_filename='./data/Building_evacutaion.mp4'
          cap = cv2.VideoCapture(video_filename)
          fourcc = cv2.VideoWriter_fourcc(*'DIVX')
          video_name='./data/output_video.avi'
          height, width, layers = (720, 1280, 3)
          video = cv2.VideoWriter(video_name, fourcc, 20, (width, height))
          i=0
          while cap.isOpened():
              i=i+1
              ret, frame = cap.read()
              #img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
              img=frame
              kept_boxes,image=self.get_yolo_output(img)
                #plt.imsave('./data/detections/frame{}.png'.format(i),image)
              video.write(image)
              detections=[]
              for detection in kept_boxes:
                  cx,cy,w,h,score= detection
                  x1, y1, x2, y2 = get_corner_coordinates([cx, cy, w, h])
                  detection=[x1, y1, x2, y2, score]
                  detections.append(detection)
              tracked_persons=person_tracker.update(np.array(detections))
              for x1, y1, x2, y2, personid in tracked_persons:
                  center_pos = (int((x1 + x2)/2), int(y1 + y2)/2,i)
                  tracks[int(personid)] = tracks.get(personid, []) + [center_pos]
              if i==1000000:
                  break

          print('video processed')
          cv2.destroyAllWindows()
          video.release()  # releasing the video generated
          return tracks
      except Exception as e :
        print('video processed return {}'.format(e))
        return tracks

  def extract_xy_coordinates(self,frame=1):
    boxes=self.detect(show_results=True)
    tracker=pd.DataFrame(columns=['Frame','x_coordinate','y_coordinate'])
    for box in boxes:
        pedestrian_x=(box[1]-box[0])/2
        pedestrian_y=(box[3]-box[2])/2
        tracker.loc[len(tracker)]=[frame,pedestrian_x,pedestrian_y]
    return tracker

def compute_bounding_box_surfaces(self,box):
    return abs((box[2]-box[0]) * (box[3]-box[1]))

def zoom_at(self,img, zoom=2, angle=0, coord=None):

    cy, cx = [ i/2 for i in img.shape[:-1] ] if coord is None else coord[::-1]
    rot_mat = cv2.getRotationMatrix2D((cx,cy), angle, zoom)
    result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result




if __name__ == '__main__':
    parser=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--image_path', type=str, default='./jhu_crowd_v2.0/test/images/0186.jpg' , help='image path')
    parser.add_argument('--video_path', type=str, default='./data/Building_evacutaion.mp4' , help='video path')
    parser.add_argument('--crop_image', type=bool, default=True, help='perform image cropping before running inference')
    parser.add_argument('--output_path', type=str, default='./data/output_data.csv' , help='')
    args = parser.parse_args()
    tracker=Object_Detector()
    boxes=tracker.track(args.video_path)
    #x_y_coordinates=tracker.extract_xy_coordinates()
    #x_y_coordinates.to_csv(args.output_path)
    print('saved coordinates at {}'.format(args.output_path))
