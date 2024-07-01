import numpy as np
import cv2
import time
from ultralytics import YOLO
from sort import Sort
from collections import defaultdict
import math

# Constants
MODEL_PATH = 'yolov8n.pt'
VIDEO_PATH = '/content/drive/MyDrive/client_vid_small_tricky.mp4'
OUTPUT_PATH = '/content/drive/MyDrive/output_v1_SORT_v1.mp4'
CLASSES_PATH = 'classes.txt'
RUN_DURATION = 5 * 60  # 5 minutes in seconds
BUFFER_TIME = 120  # 120 seconds buffer time before considering a vehicle parked
TIME_LIMIT = 15 * 60  # 20 minutes time limit
FONT = cv2.FONT_HERSHEY_COMPLEX
FONT_SCALE = 0.7
THICKNESS = 2
LEFT_STATS_FONT_SCALE = 0.9

# Load the YOLO model
model = YOLO(MODEL_PATH)

# Initialize tracker
tracker = Sort(max_age=300)

# Parking polygon coordinates
parking_polygon = np.array([[279, 227], [101, 289], [918, 513], [1056, 418], [289, 227], [279, 227]])

# Tracking dictionaries
vehicle_timers = {}
entry_timestamps = {}
exit_timestamps = {}
counters = {"total_vehicles": 0, "vehicles_within_20": 0, "vehicles_exceed_20": 0}
vehicle_buffer = {}
lost_vehicles = defaultdict(dict)  # To store lost vehicles with their details
bbox_history = defaultdict(list)  # To store bbox history for prediction
status = {}  # To store the status of each vehicle
history_limit = 100  # Number of past positions to use for prediction

def predict_bbox(history):
    if not history:
        return None  # No data to predict
    return history[-1]  # Return the latest bounding box

# Valid classes for detection
valid_classes = ['car', 'truck', 'bus']

def highlight_parking_polygon(frame):
    cv2.polylines(frame, [parking_polygon], isClosed=True, color=(255, 0, 0), thickness=2)
    return frame

def draw_text(frame, text, position, bg_color):
    text_size, _ = cv2.getTextSize(text, FONT, LEFT_STATS_FONT_SCALE, THICKNESS)
    text_x = min(position[0], frame.shape[1] - text_size[0] - 10)
    bottom_left_corner = (text_x, position[1])
    top_left_corner = (text_x, position[1] - text_size[1] - 10)
    cv2.rectangle(frame, top_left_corner, (bottom_left_corner[0] + text_size[0] + 10, bottom_left_corner[1] + 10), bg_color, -1)
    cv2.putText(frame, text, bottom_left_corner, FONT, LEFT_STATS_FONT_SCALE, (0, 0, 0), THICKNESS)

def display_timer(frame, vehicle_id, entry_time, exit_time, x1, y1, color):
    elapsed_time = time.time() - entry_time if exit_time is None else exit_time - entry_time
    exit_time_text = "None" if exit_time is None else time.strftime("%H:%M:%S", time.localtime(exit_time))
    minutes, seconds = divmod(elapsed_time, 60)
    timer_text = f"{int(minutes)}:{int(seconds):02d} sec"
    entry_time_text = time.strftime("%H:%M:%S", time.localtime(entry_time))
    color = (0, 255, 0) if elapsed_time <= TIME_LIMIT else (0, 0, 255)  # Green if within 15 mins, Red if exceeded

    vertical_offset = 20  # Adjust this value for more spacing
    cv2.putText(frame, f"ID: {vehicle_id}", (x1, y1 - vertical_offset * 3), FONT, FONT_SCALE, color, THICKNESS)
    cv2.putText(frame, f"Entry: {entry_time_text}", (x1, y1 - vertical_offset * 2), FONT, FONT_SCALE, color, THICKNESS)
    cv2.putText(frame, f"Exit: {exit_time_text}", (x1, y1 - vertical_offset), FONT, FONT_SCALE, color, THICKNESS)
    cv2.putText(frame, f"Time: {timer_text}", (x1, y1), FONT, FONT_SCALE, color, THICKNESS)

    return frame


def manage_timers():
    current_time = time.time()
    for vehicle_id, start_time in list(vehicle_timers.items()):
        if exit_timestamps[vehicle_id] is None and current_time - start_time > TIME_LIMIT:
            counters["vehicles_exceed_20"] += 1
            exit_timestamps[vehicle_id] = current_time
            status[vehicle_id] = 'DONE'
            del vehicle_timers[vehicle_id]

def display_counters(frame):
    draw_text(frame, f"Total Vehicles: {counters['total_vehicles']}", (10, 350), (222, 255, 176))
    draw_text(frame, f"Within 20 mins: {counters['vehicles_within_20']}", (10, 450), (222, 255, 176))
    draw_text(frame, f"Exceeded 20 mins: {counters['vehicles_exceed_20']}", (10, 550), (222, 255, 176))
    return frame

def find_closest_bbox(pred_bbox, current_bboxes):
    closest_bbox = None
    min_distance = float('inf')
    for bbox in current_bboxes:
        distance = np.sum(np.abs(np.array(pred_bbox[:4]) - np.array(bbox[:4])))
        if distance < min_distance:
            min_distance = distance
            closest_bbox = bbox
    return closest_bbox

# Load class names
with open(CLASSES_PATH, 'r') as f:
    classnames = f.read().splitlines()

# Start processing video
cap = cv2.VideoCapture(VIDEO_PATH)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

start_time = time.time()
# Main processing loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret or time.time() - start_time > RUN_DURATION:
        break

    results = model(frame)
    current_detections = np.empty((0, 5))

    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            confidence = box.conf[0]
            class_detect = box.cls[0]
            class_detect = int(class_detect)
            class_detect = classnames[class_detect]
            conf = math.ceil(confidence * 100)

            if class_detect in valid_classes:
                detections = np.array([x1, y1, x2, y2, conf])
                current_detections = np.vstack([current_detections, detections])

    tracked_objects = tracker.update(np.array(current_detections))

    current_vehicle_ids = set()

    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = map(int, obj)
        current_vehicle_ids.add(obj_id)
        bbox_history[obj_id].append((x1, y1, x2, y2))

        if len(bbox_history[obj_id]) > history_limit:
            bbox_history[obj_id].pop(0)

        # Update the vehicle timers and positions based on re-identified vehicles
        if obj_id in lost_vehicles:
            if obj_id in vehicle_timers:
                entry_timestamps[obj_id] = lost_vehicles[obj_id]['entry']
                exit_timestamps[obj_id] = lost_vehicles[obj_id]['exit']
            lost_vehicles.pop(obj_id)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)

        if cv2.pointPolygonTest(parking_polygon, (center_x, center_y), False) >= 0:
            if obj_id not in vehicle_timers and obj_id not in vehicle_buffer:
                closest_black_bbox = None
                min_distance = float('inf')
                for lost_vehicle_id, lost_vehicle_data in list(lost_vehicles.items()):
                    predicted_bbox = predict_bbox(bbox_history[lost_vehicle_id])
                    if predicted_bbox:
                        distance = np.sum(np.abs(np.array(predicted_bbox[:4]) - np.array([x1, y1, x2, y2])))
                        if distance < min_distance:
                            min_distance = distance
                            closest_black_bbox = lost_vehicle_id

                if closest_black_bbox is not None and min_distance < 500:  # Threshold for proximity
                    # Transfer information from the black bounding box to the current one
                    entry_timestamps[obj_id] = lost_vehicles[closest_black_bbox]['entry']
                    exit_timestamps[obj_id] = lost_vehicles[closest_black_bbox]['exit']
                    vehicle_timers[obj_id] = lost_vehicles[closest_black_bbox]['entry']
                    bbox_history[obj_id].extend(bbox_history[closest_black_bbox])

                    del lost_vehicles[closest_black_bbox]
                else:
                    vehicle_buffer[obj_id] = time.time()
            elif obj_id in vehicle_buffer and time.time() - vehicle_buffer[obj_id] > BUFFER_TIME:
                vehicle_timers[obj_id] = vehicle_buffer[obj_id]
                entry_timestamps[obj_id] = vehicle_buffer[obj_id]
                exit_timestamps[obj_id] = None
                counters["total_vehicles"] += 1
                del vehicle_buffer[obj_id]

            if obj_id in vehicle_timers:
                frame = display_timer(frame, obj_id, vehicle_timers[obj_id], exit_timestamps[obj_id], x1, y1, (0,0,255))

        else:
            if obj_id in vehicle_timers and exit_timestamps[obj_id] is None:
                elapsed_time = time.time() - vehicle_timers[obj_id]
                if elapsed_time <= TIME_LIMIT:
                    counters["vehicles_within_20"] += 1
                else:
                    counters["vehicles_exceed_20"] += 1
                exit_timestamps[obj_id] = time.time()
                status[obj_id] = 'DONE'
                del vehicle_timers[obj_id]
            elif obj_id in vehicle_buffer:
                del vehicle_buffer[obj_id]

    for vehicle_id in list(vehicle_timers.keys()):
        if vehicle_id not in current_vehicle_ids and exit_timestamps[vehicle_id] is None:
            if status.get(vehicle_id) != 'DONE':
                predicted_bbox = predict_bbox(bbox_history[vehicle_id])
                # if predicted_bbox:
                #     x1, y1, x2, y2 = predicted_bbox
                #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
                #     center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                #     cv2.circle(frame, (center_x, center_y), 5, (255, 255, 255), -1)
                #     if cv2.pointPolygonTest(parking_polygon, (center_x, center_y), False) >= 0:
                #         frame = display_timer(frame, vehicle_id, vehicle_timers[vehicle_id], exit_timestamps[vehicle_id], x1, y1, (0,255,0))
                lost_vehicles[vehicle_id] = {
                    'entry': vehicle_timers[vehicle_id],
                    'exit': exit_timestamps[vehicle_id],
                    'bbox': bbox_history[vehicle_id]
                }
                status[vehicle_id] = 'PREDICTED'

    frame = highlight_parking_polygon(frame)
    manage_timers()
    frame = display_counters(frame)
    out.write(frame)

out.release()
cap.release()
cv2.destroyAllWindows()
