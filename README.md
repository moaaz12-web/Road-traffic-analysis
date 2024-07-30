# Real time traffic analysis using YOLOv8, SORT, and OpenCV

## Features
1. Segments the road in video into 5 lanes, draw long polygons around it.
2. Keep track of vehicles currently inside the lanes, and overlays it on screen using "Inside lane {lane_name"" placeholder.
3. Keeps track of all vehices that have passed the lanes, and overlays it on screen using "Lane {lane_name) count" placeholder.
4. Highlights a lane in red if there are more than 4 vehicles currently inside it, indicating a traffic jam.

 
 ## Project demo
 Link: https://drive.google.com/file/d/1oc1PhSfpF4nEw7n3KRkCodfULVT5STnt/view?usp=drive_link

## File information
1. The `Create_zones.py` file is used to manually annotate your input video, where you can create the lane polygons by yourself.
2. The `SORT.py` file is imported from SORT repository to directly use the SORT tracker for tracking the vehicles in real time.
3. The `main.py` is the main file for all logic, where we load the video, process it every frame, apply SORT tracker and track the vehicles across all frames, keep track of all vehicles in every lane and also highlight them in red if there are more than 4 vehicles inside a lane, indicating a traffic jam.
4. 
