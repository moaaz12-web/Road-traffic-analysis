from utils import draw_text, detect_traffic_jam, check_line_crossing, calculate_intersection_percentage, highlight_traffic_jams
import cv2
import math
import numpy as np
from ultralytics import YOLO
from sort import *
import time

# Video path and YOLO model
video_path = './Lanes Anaysis.mp4'
cap = cv2.VideoCapture(video_path)
model = YOLO('yolov8n.pt')

# Lane coordinates
road_zoneA = np.array([[516, 140], [561, 140], [115, 440], [107, 331], [508, 150], [516, 140]], np.int32)
road_zoneB = np.array([[571, 142], [604, 143], [162, 651], [131, 456], [559, 152], [571, 142]], np.int32)
road_zoneC = np.array([[608, 144], [643, 143], [627, 682], [171, 662], [600, 155], [608, 144]], np.int32)
road_zoneD = np.array([[680, 145], [727, 143], [1065, 691], [689, 693], [681, 158], [680, 145]], np.int32)
road_zoneE = np.array([[737, 146], [794, 144], [1271, 610], [1074, 690], [747, 159], [737, 146]], np.int32)

# Lane lines
zoneA_Line = np.array([road_zoneA[2], road_zoneA[3]]).reshape(-1)
zoneB_Line = np.array([road_zoneB[2], road_zoneB[3]]).reshape(-1)
zoneC_Line = np.array([road_zoneC[2], road_zoneC[3]]).reshape(-1)
zoneD_Line = np.array([road_zoneD[2], road_zoneD[3]]).reshape(-1)
zoneE_Line = np.array([road_zoneE[2], road_zoneE[3]]).reshape(-1)

# Define colors
color_zoneA = (100, 90, 255)
color_zoneB = (90, 255, 90)
color_zoneC = (255, 90, 90)
color_zoneD = (155, 190, 90)
color_zoneE = (25, 30, 240)

color_line = (0, 0, 0)
color_text = (255, 255, 255)
color_rectangle = (0, 0, 0)
color_jam_area = (0, 0, 255)

# Initialize tracker and counters
tracker = Sort()
zoneAcounter = set()
zoneBcounter = set()
zoneCcounter = set()
zoneDcounter = set()
zoneEcounter = set()

# IDs of vehicles currently inside each zone
vehicles_in_zoneA = set()
vehicles_in_zoneB = set()
vehicles_in_zoneC = set()
vehicles_in_zoneD = set()
vehicles_in_zoneE = set()

# Previous positions of tracked vehicles
previous_positions = {}

# Timer to control traffic jam detection
last_jam_check = time.time()
jam_check_interval = 3  # seconds

# States to keep track of jams
jam_in_zoneA = False
jam_in_zoneB = False
jam_in_zoneC = False
jam_in_zoneD = False
jam_in_zoneE = False

highlight_until = 0

# Counters for the number of detected vehicles
total_detected_vehicles = 0
seen_vehicle_ids = set()

# Start timer
# start_time = time.time()
# run_duration = 120 * 60  # 2 hours in seconds

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('./SORT_final_inference.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # # Check if the run duration has exceeded
    # if time.time() - start_time > run_duration:
    #     print("2 hours have elapsed. Stopping the video processing.")
    #     break

    results = model(frame)
    current_detections = np.empty([0, 5])

    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            confidence = box.conf[0]
            class_detect = box.cls[0]
            class_detect = int(class_detect)
            class_detect = classnames[class_detect]
            conf = math.ceil(confidence * 100)

            cv2.putText(frame, f'{class_detect}', (x1 + 8, y1 - 12 if y1 - 12 > 0 else y1 + 14), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_rectangle, 2)

            if class_detect in ['car', 'truck', 'bus','motorbike'] and conf > 30:
                detections = np.array([x1, y1, x2, y2, conf])
                current_detections = np.vstack([current_detections, detections])

    cv2.polylines(frame, [road_zoneA], isClosed=True, color=color_zoneA, thickness=5)
    cv2.polylines(frame, [road_zoneB], isClosed=True, color=color_zoneB, thickness=5)
    cv2.polylines(frame, [road_zoneC], isClosed=True, color=color_zoneC, thickness=5)
    cv2.polylines(frame, [road_zoneD], isClosed=True, color=color_zoneD, thickness=5)
    cv2.polylines(frame, [road_zoneE], isClosed=True, color=color_zoneE, thickness=5)

    cv2.line(frame, (zoneA_Line[0], zoneA_Line[1]), (zoneA_Line[2], zoneA_Line[3]), color_line, 2)
    cv2.line(frame, (zoneB_Line[0], zoneB_Line[1]), (zoneB_Line[2], zoneB_Line[3]), color_line, 2)
    cv2.line(frame, (zoneC_Line[0], zoneC_Line[1]), (zoneC_Line[2], zoneC_Line[3]), color_line, 2)
    cv2.line(frame, (zoneD_Line[0], zoneD_Line[1]), (zoneD_Line[2], zoneD_Line[3]), color_line, 2)
    cv2.line(frame, (zoneE_Line[0], zoneE_Line[1]), (zoneE_Line[2], zoneE_Line[3]), color_line, 2)

    track_results = tracker.update(current_detections)
    current_vehicles_in_zoneA = set()
    current_vehicles_in_zoneB = set()
    current_vehicles_in_zoneC = set()
    current_vehicles_in_zoneD = set()
    current_vehicles_in_zoneE = set()

    parked_vehicles_in_zoneA = []
    parked_vehicles_in_zoneB = []
    parked_vehicles_in_zoneC = []
    parked_vehicles_in_zoneD = []
    parked_vehicles_in_zoneE = []

    for result in track_results:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2

        right_cx = x2
        right_cy = y2

        mid_cx = x1 + w // 2  # Middle of the width
        mid_cy = y1 + h // 2  # Middle of the height

        # Coordinates to highlight
        bottom_middle = (mid_cx, y2)  # Bottom middle of the bounding box
        right_middle = (x2, mid_cy)   # Right middle of the bounding box
        top_right = (x2, y1)          # Top-right corner of the bounding box

        # Draw circles at the specified points
        cv2.circle(frame, bottom_middle, 2, (0, 0, 255), -1)  # Red
        cv2.circle(frame, right_middle, 2, (255, 0, 0), -1)   # Blue
        cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)       # Green
        cv2.circle(frame, top_right, 2, (190, 100, 100), -1)  # Gray-blue
        cv2.circle(frame, (right_cx, cy), 2, (0, 255, 255), -1)  # Yellow
        cv2.circle(frame, (mid_cx, cy), 2, (255, 255, 0), -1)    # Cyan

        # Check if the vehicle is inside road_zoneA, road_zoneB, or road_zoneC
        inside_zoneA = (cv2.pointPolygonTest(road_zoneA, (mid_cx, cy), False) >= 0)
        inside_zoneB = cv2.pointPolygonTest(road_zoneB, bottom_middle, False) >= 0
        inside_zoneC = (
            cv2.pointPolygonTest(road_zoneC, bottom_middle, False) >= 0 or
            cv2.pointPolygonTest(road_zoneC, right_middle, False) >= 0 or
            cv2.pointPolygonTest(road_zoneC, (cx, cy), False) >= 0 or
            cv2.pointPolygonTest(road_zoneC, top_right, False) >= 0 or
            cv2.pointPolygonTest(road_zoneC, (right_cx, cy), False) >= 0 or
            cv2.pointPolygonTest(road_zoneC, (top_right), False) >= 0 or
            cv2.pointPolygonTest(road_zoneC, (mid_cx, cy), False) >= 0
        )

        inside_zoneD =(
                cv2.pointPolygonTest(road_zoneD, ((mid_cx, cy)), False) >= 0

        )

        inside_zoneE = (
                cv2.pointPolygonTest(road_zoneE, ((mid_cx, cy)), False) >= 0
        )

        int_pctA = calculate_intersection_percentage([x1, y1, x2, y2], road_zoneA)
        int_pctB = calculate_intersection_percentage([x1, y1, x2, y2], road_zoneB)
        int_pctC = calculate_intersection_percentage([x1, y1, x2, y2], road_zoneC)
        int_pctD = calculate_intersection_percentage([x1, y1, x2, y2], road_zoneD)
        int_pctE = calculate_intersection_percentage([x1, y1, x2, y2], road_zoneE)


        if id in previous_positions:
            was_inside_zoneA = previous_positions[id]['inside_zoneA']
            was_inside_zoneB = previous_positions[id]['inside_zoneB']
            was_inside_zoneC = previous_positions[id]['inside_zoneC']
            was_inside_zoneD = previous_positions[id]['inside_zoneD']
            was_inside_zoneE = previous_positions[id]['inside_zoneE']


            # Check for line crossing
            prev_position = previous_positions[id]['position']
            current_position = bottom_middle
            if was_inside_zoneA and check_line_crossing(current_position, zoneA_Line):
                zoneAcounter.add(id)
            if was_inside_zoneB and check_line_crossing(current_position, zoneB_Line):
                zoneBcounter.add(id)
            if was_inside_zoneC and check_line_crossing(current_position, zoneC_Line):
                zoneCcounter.add(id)
            if was_inside_zoneD and check_line_crossing(current_position, zoneD_Line):
                zoneDcounter.add(id)
            if was_inside_zoneE and check_line_crossing(current_position, zoneE_Line):
                zoneEcounter.add(id)

            previous_positions[id]['moved'] = previous_positions[id]['position'] != (right_cx, right_cy)
            previous_positions[id]['inside_zoneA'] = inside_zoneA
            previous_positions[id]['inside_zoneB'] = inside_zoneB
            previous_positions[id]['inside_zoneC'] = inside_zoneC
            previous_positions[id]['inside_zoneD'] = inside_zoneD
            previous_positions[id]['inside_zoneE'] = inside_zoneE
            previous_positions[id]['inside_zone'] = inside_zoneA or inside_zoneB or inside_zoneC or inside_zoneD or inside_zoneE

            previous_positions[id]['position'] = (right_cx, right_cy)
        else:
            previous_positions[id] = {'position': (right_cx, right_cy), 'moved': False, 'inside_zoneA': inside_zoneA, 'inside_zoneB': inside_zoneB, 'inside_zoneC': inside_zoneC, 'inside_zoneD': inside_zoneD, 'inside_zoneE': inside_zoneE,
            'inside_zone': inside_zoneA or inside_zoneB or inside_zoneC or inside_zoneD or inside_zoneE}
            if id not in seen_vehicle_ids:
                total_detected_vehicles += 1
                seen_vehicle_ids.add(id)

        if inside_zoneA and int_pctA > 0:
            current_vehicles_in_zoneA.add(id)
            if not previous_positions[id]['moved']:
                parked_vehicles_in_zoneA.append((right_cx, right_cy))

        if inside_zoneB and int_pctB> 0:
            current_vehicles_in_zoneB.add(id)
            if not previous_positions[id]['moved']:
                parked_vehicles_in_zoneB.append((right_cx, right_cy))

        if inside_zoneC and int_pctC > 0:
            current_vehicles_in_zoneC.add(id)
            if not previous_positions[id]['moved']:
                parked_vehicles_in_zoneC.append((right_cx, right_cy))

        if inside_zoneD and int_pctD > 0:
            current_vehicles_in_zoneD.add(id)
            if not previous_positions[id]['moved']:
                parked_vehicles_in_zoneD.append((right_cx, right_cy))

        if inside_zoneE and int_pctE > 0:
            current_vehicles_in_zoneE.add(id)
            if not previous_positions[id]['moved']:
                parked_vehicles_in_zoneE.append((right_cx, right_cy))

    vehicles_in_zoneA = current_vehicles_in_zoneA
    vehicles_in_zoneB = current_vehicles_in_zoneB
    vehicles_in_zoneC = current_vehicles_in_zoneC
    vehicles_in_zoneD = current_vehicles_in_zoneD
    vehicles_in_zoneE = current_vehicles_in_zoneE

    draw_text(frame, f'Lane A: {len(zoneAcounter)}', [9, 70 + 15 * 0], (255,255,51))  # Yellow for Lanes
    draw_text(frame, f'In Lane A: {len(vehicles_in_zoneA)}', [9, 110 + 15 * 1], (66, 255, 252))  # Light blue for In Lane
    draw_text(frame, f'Lane B: {len(zoneBcounter)}', [9, 150 + 15 * 2], (255,255,51))  # Yellow for Lanes
    draw_text(frame, f'In Lane B: {len(vehicles_in_zoneB)}', [9, 190 + 15 * 3], (66, 255, 252))  # Light blue for In Lane
    draw_text(frame, f'Lane C: {len(zoneCcounter)}', [9, 230 + 15 * 4], (255,255,51))  # Yellow for Lanes
    draw_text(frame, f'In Lane C: {len(vehicles_in_zoneC)}',[9, 270 + 15 * 5], (66, 255, 252))  # Light blue for In Lane
    draw_text(frame, f'Lane D: {len(zoneDcounter)}', [9, 310 + 15 * 6], (255,255,51))  # Yellow for Lanes
    draw_text(frame, f'In Lane D: {len(vehicles_in_zoneD)}', [9, 350 + 15 * 7], (66, 255, 252))  # Light blue for In Lane
    draw_text(frame, f'Lane E: {len(zoneEcounter)}', [9, 390 + 15 * 8], (255,255,51))  # Yellow for Lanes
    draw_text(frame, f'In Lane E: {len(vehicles_in_zoneE)}', [9, 430 + 15 * 9], (66, 255, 252))  # Light blue for In Lane

    # Highlight traffic jams for each lane
    jam_in_zoneA = detect_traffic_jam(vehicles_in_zoneA, previous_positions)
    jam_in_zoneB = detect_traffic_jam(vehicles_in_zoneB, previous_positions)
    jam_in_zoneC = detect_traffic_jam(vehicles_in_zoneC, previous_positions)
    jam_in_zoneD = detect_traffic_jam(vehicles_in_zoneD, previous_positions)
    jam_in_zoneE = detect_traffic_jam(vehicles_in_zoneE, previous_positions)

    # Define the zones and their current traffic jam status
    jam_zones = {
        "zoneA": jam_in_zoneA,
        "zoneB": jam_in_zoneB,
        "zoneC": jam_in_zoneC,
        "zoneD": jam_in_zoneD,
        "zoneE": jam_in_zoneE
    }

    # Define the road zones with their polygonal areas
    road_zones = {
        "zoneA": road_zoneA,
        "zoneB": road_zoneB,
        "zoneC": road_zoneC,
        "zoneD": road_zoneD,
        "zoneE": road_zoneE
    }

    # Call the function to highlight traffic jams
    frame = highlight_traffic_jams(frame, jam_zones, road_zones, color_jam_area)

    # Count current detected vehicles
    current_detected_vehicles = len(track_results)

    draw_text(frame, f'Frame count: {current_detected_vehicles}', [9, 470 + 15 * 10], (66, 98, 255))
    draw_text(frame, f'Total: {total_detected_vehicles}', [9, 510+ 15 * 11], (66, 98, 255))

    out.write(frame)


out.release()
cap.release()
cv2.destroyAllWindows()