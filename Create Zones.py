# import cv2
# import numpy as np

# # Global variables
# polygon_points = []

# # Read your video file
# video_path = r'D:\\FIVERR WORK\\LANE YOLO\\vid.mp4'
# cap = cv2.VideoCapture(video_path)


# # Callback function for mouse events
# def mouse_callback(event, x, y, flags, param):
#     global polygon_points
#     if event == cv2.EVENT_LBUTTONDOWN:
#         polygon_points.append((x, y))
#         print(f"Point Added: (X: {x}, Y: {y})")



# while True:
#     # Capture the first frame
#     ret, frame = cap.read()
#     frame = cv2.resize(frame, (1920, 1080))
#     # Create a window and set the mouse callback
#     cv2.namedWindow('Frame')
#     cv2.setMouseCallback('Frame', mouse_callback)

#     # Draw the polygon on the frame
#     if len(polygon_points) > 1:
#         cv2.polylines(frame, [np.array(polygon_points)], isClosed=False, color=(0, 255, 0), thickness=2)

#     cv2.imshow('Frame', frame)

#     # Press 'Esc' to exit
#     key = cv2.waitKey(0)
#     if key == 27:
#         break

# cv2.destroyAllWindows()
# cap.release()

# # Print the polygon points
# print("Polygon Points:")
# for point in polygon_points:
#     print(f"X: {point[0]}, Y: {point[1]}")


# import cv2
# import numpy as np

# # Global variables
# polygon_points = []

# # Read your video file
# video_path = r'D:\\FIVERR WORK\\LANE YOLO\\Lanes Anaysis.mp4'
# cap = cv2.VideoCapture(video_path)

# # Callback function for mouse events
# def mouse_callback(event, x, y, flags, param):
#     global polygon_points
#     if event == cv2.EVENT_LBUTTONDOWN:
#         polygon_points.append((x, y))
#         print(f"Point Added: (X: {x}, Y: {y})")

# # Capture the first frame
# ret, frame = cap.read()
# frame = cv2.resize(frame, (1920, 1080))

# while True:
#     # Create a window and set the mouse callback
#     cv2.namedWindow('Frame')
#     cv2.setMouseCallback('Frame', mouse_callback)

#     # Copy the original frame for drawing
#     frame_copy = frame.copy()

#     # Draw the polygon and the points on the frame
#     if len(polygon_points) > 1:
#         cv2.polylines(frame_copy, [np.array(polygon_points)], isClosed=False, color=(0, 255, 0), thickness=2)

#     # Draw points and their coordinates
#     for point in polygon_points:
#         cv2.circle(frame_copy, point, 5, (0, 0, 255), -1)
#         cv2.putText(frame_copy, f"({point[0]}, {point[1]})", (point[0] + 10, point[1] - 10), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

#     cv2.imshow('Frame', frame_copy)

#     # Press 'Esc' to exit
#     key = cv2.waitKey(1)
#     if key == 27:
#         break

# cv2.destroyAllWindows()
# cap.release()

# # Print the polygon points
# print("Polygon Points:")
# for point in polygon_points:
#     print(f"X: {point[0]}, Y: {point[1]}")


import cv2
import numpy as np

# Global variables
polygon_points = []
all_polygons = []
dragging_point_index = None

# Read your video file
video_path = './Lanes Anaysis.mp4'
cap = cv2.VideoCapture(video_path)

# Callback function for mouse events
def mouse_callback(event, x, y, flags, param):
    global polygon_points, all_polygons, dragging_point_index

    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if we clicked near an existing point
        for i, point in enumerate(polygon_points):
            if abs(point[0] - x) < 10 and abs(point[1] - y) < 10:
                dragging_point_index = i
                break
        else:
            # Add new point if no point is being dragged
            polygon_points.append((x, y))
            print(f"Point Added: (X: {x}, Y: {y})")
            if len(polygon_points) == 5:
                polygon_points.append(polygon_points[0])  # Close the polygon
                all_polygons.append(polygon_points.copy())
                polygon_points = []  # Reset for the next polygon
                print("Polygon Completed")

    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging_point_index is not None:
            # Update the position of the dragging point
            polygon_points[dragging_point_index] = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        dragging_point_index = None

# Capture the first frame
ret, frame = cap.read()
# frame = cv2.resize(frame, (1920, 1080))

# Create a window and set the mouse callback
cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', mouse_callback)

while True:
    # Copy the original frame for drawing
    frame_copy = frame.copy()

    # Draw the polygons on the frame
    for poly in all_polygons:
        cv2.polylines(frame_copy, [np.array(poly)], isClosed=True, color=(0, 255, 0), thickness=2)

    # Draw the current points and lines
    if len(polygon_points) > 0:
        cv2.polylines(frame_copy, [np.array(polygon_points)], isClosed=False, color=(0, 255, 0), thickness=2)
        for point in polygon_points:
            cv2.circle(frame_copy, point, 5, (0, 0, 255), -1)
            cv2.putText(frame_copy, f"({point[0]}, {point[1]})", (point[0] + 10, point[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('Frame', frame_copy)

    # Press 'Esc' to exit
    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()
cap.release()

# Print the polygon points
print("All Polygons Points:")
for poly in all_polygons:
    formatted_poly = [[point[0], point[1]] for point in poly]
    print(formatted_poly)
