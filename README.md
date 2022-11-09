# Pedestrian Detection and Tracking

This is a  pedestrian tracker from a side-view camera using YOLO based detection algorithms for tracking pedestrians designed for further simulation purposes.

the code outputs a csv file with the detected xy coordinates, and video with the corresponding tracked bounding boxes

- For object detection in each frame, pretrained YOLO-v3(on MS-COCO dataset) is used. The number of detections are reduced to keep only person class and filter out others.
- For tracking the detected objects, a kalman filter based SORT(Simple Online and Realtime Tracking) algorithm is used https://arxiv.org/abs/1602.00763

<b>Code Walkthrough</b>
- detect.py  ----> main file for the deep_track and track functions
we incorporated in the code image grid composition before inference in order to give accurate bounding boxes.
- sort.py   -----> SORT algorithm implementation
- utils.py  -------> utility methods


<b>How to run the code</b>
```
python detect.py --video_path {video_path} --crop_image {crop_image}
```

This would return the list of tracks. In order to see the whole trajectory.

# References
- Simple Online and Realtime Tracking https://arxiv.org/abs/1602.00763
- SORT implementation https://github.com/abewley/sort
- YOlOv3 An Incremental Improvement Paper: https://arxiv.org/abs/1804.02767
