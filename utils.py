import cv2
import math
import numpy as np

def draw_text(frame, text, position, bg_color):
    font = cv2.FONT_HERSHEY_COMPLEX
    font_scale = 0.6
    thickness = 1
    color = (0, 0, 0)
    # bg_color=(0, 255, 255)
    # bg_color=(222, 255, 176)

    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = min(position[0], frame.shape[1] - text_size[0] - 10)
    bottom_left_corner = (text_x, position[1])
    top_left_corner = (text_x, position[1] - text_size[1] - 10)
    
    # Draw the rectangle (background)
    cv2.rectangle(frame, top_left_corner, (bottom_left_corner[0] + text_size[0] + 10, bottom_left_corner[1] + 10), bg_color, -1)
    
    # Draw the text on top of the rectangle
    cv2.putText(frame, text, bottom_left_corner, font, font_scale, color , thickness)

def detect_traffic_jam(vehicles_in_zone, positions):
    # Check if there are more than 3 vehicles in the zone
    if len(vehicles_in_zone) > 3:
        vehicles_list = list(vehicles_in_zone)
        # Iterate over all pairs of vehicles
        for i in range(len(vehicles_list)):
            for j in range(i + 1, len(vehicles_list)):
                id1, id2 = vehicles_list[i], vehicles_list[j]
                pos1, pos2 = positions[id1]['position'], positions[id2]['position']
                # Calculate the Euclidean distance between the two vehicles
                distance = math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
                print(f"Distance between vehicle {id1} and {id2}: {distance:.2f} pixels")
                # If any pair of vehicles is farther than 150 pixels, return False
                if distance >= 550:
                    return False
        # If all pairs are within 500 pixels, return True
        return True
    return False


def highlight_traffic_jams(frame, jam_zones, road_zones, color_jam_area):
    """
    Highlights traffic jams in different zones on a frame.

    Parameters:
    - frame: The current video frame.
    - jam_zones: A dictionary with zone identifiers as keys and boolean values indicating if a jam is detected.
    - road_zones: A dictionary with zone identifiers as keys and their corresponding polygonal area.
    - color_jam_area: The color to use for highlighting traffic jams.

    Returns:
    - The modified frame with traffic jams highlighted.
    """
    for zone, is_jammed in jam_zones.items():
        if is_jammed:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [road_zones[zone]], color_jam_area)
            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
    return frame

def check_line_crossing(current_position, line):
    line_start, line_end = (line[0], line[1]), (line[2], line[3])
    if current_position[1] > line_start[1] or current_position[1] > line_end[1]:
        return True
    return False


def get_video_properties(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps else 0

    cap.release()

    return {
        'width': width,
        'height': height,
        'fps': fps,
        'frame_count': frame_count,
        'duration': duration
    }


def calculate_intersection_percentage(bbox, zone):
        # Create mask for the zone
        mask_shape = (zone[:, 1].max() + 1, zone[:, 0].max() + 1)
        mask = np.zeros(mask_shape, dtype=np.uint8)
        cv2.fillPoly(mask, [zone], 1)

        # Create mask for the bounding box
        bbox_mask = np.zeros(mask_shape, dtype=np.uint8)
        x_min, y_min, x_max, y_max = bbox
        bbox_mask[y_min:y_max, x_min:x_max] = 1

        # Calculate intersection
        intersection = np.logical_and(mask, bbox_mask).sum()

        # Calculate bbox area
        bbox_area = (x_max - x_min) * (y_max - y_min)

        return (intersection / bbox_area) * 100 if bbox_area > 0 else 0


