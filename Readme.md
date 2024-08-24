# Road traffic analysis
Performs real-time vehicle detection and tracking, counting the number of vehicles within each lane as well as those that have crossed. Highlights a lane in red if more than four vehicles are present, indicating a potential traffic jam. 

Demo is at: https://drive.google.com/file/d/1LIhAlqQKaiyiuird2FRmFz16_61pYS5m/view?usp=sharing

## File information
1. The file `lane_analysis_realtime.py` contains code to test the video on your laptop in real time by showing it in a window, instead of saving it. The file uses SORT as the tracker.

2. The file `lane_analysis_save.py` contains code to save the video on your laptop to a specified path, without showing it in a window. The file uses SORT as the tracker.

3. The file `lane_analysis_deepsort.py` contains code to test the video on your laptop in real time by showing it in a window, instead of saving it. The file uses DEEPSORT as the tracker.

4. The file `lane_analysis_deeposort_save.py` contains code to save the video on your laptop to a specified path, without showing it in a window. The file uses DEEPSORT as the tracker.

5. The `utils.py` file contains helper functions that are used throughout the codebase.

6. The `requirements.txt` file is for installing the necessary requirements and dependencies.

7. The `sort.py` file contains the official code for the SORT tracker, taken from its github repo.

8. The `Create Zones.py` file is used for manually drawing lanes onto the video. So, if you have your own video and you want to test this project on it, you can manually draw the lanes, and then replace the coordinates in all of the files in point 1 to 4.


## How to run:
1. First create a virtual environment (if you want to, as it takes care of any conflicting dependencies).

2. Then, install all the libraries using `pip install -r requirements.txt`.

3. Then, make sure you place the `Lanes Anaysis.mp4' video file in the current directory where all these files are. Link to downloading the file is: https://drive.google.com/file/d/1kgCsFfN-R3iMYcIe4qWubx3dlcDsXtwu/view?usp=sharing 

4. After that, run any of the files mentioned from steps 1 to 4 based on whether you want to save the video, view it in real time, use SORT tracker, or use DeepSORT tracker.