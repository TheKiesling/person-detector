import cv2
import mediapipe as mp
import numpy as np
import yolov5

class PoseDetectionService2:
    def __init__(self,
                 yolo_model_path='yolov5s.pt',
                 yolo_device='cpu',
                 detection_threshold=0.6,
                 padding=25,
                 static_image_mode=False,
                 model_complexity=1,
                 enable_segmentation=False,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        """
        Initializes the pose detection service that combines YOLOv5 for person detection
        and Mediapipe for pose detection.
        
        Args:
            yolo_model_path (str): Path to the YOLOv5 model.
            yolo_device: Device for YOLO (0 for GPU or "cpu" for CPU).
            detection_threshold (float): Minimum threshold to consider a detection.
            padding (int): Number of pixels to expand the detected region.
            static_image_mode (bool): Static image mode for Mediapipe.
            model_complexity (int): Complexity of the Mediapipe pose model.
            enable_segmentation (bool): Enables segmentation in Mediapipe if required.
            min_detection_confidence (float): Confidence threshold for pose detection.
            min_tracking_confidence (float): Confidence threshold for pose tracking.
        """
        self.detection_threshold = detection_threshold
        self.padding = padding
        
        self.yolo_model = yolov5.YOLOv5(yolo_model_path, yolo_device, load_on_init=True)
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.pose_connections = self.mp_pose.POSE_CONNECTIONS

    def detect_pose(self, frame):
        """
        Processes the complete frame: first runs YOLO to detect people and then,
        for each detected person, performs pose detection using Mediapipe.
        
        Args:
            frame (numpy.ndarray): Complete BGR image.
            
        Returns:
            list: List of dictionaries, each with the keys "bbox" and "landmarks".
                  "bbox" is the bounding box (x1, y1, x2, y2) and "landmarks" is a list
                  of points (x, y, z, visibility) adjusted to the original coordinates.
        """
        pose_results = []
        
        # Detect people
        yolo_out = self.yolo_model.predict(frame, size=640, augment=False)
        detections = yolo_out.pred[0]
        
        img_height, img_width, _ = frame.shape

        for detection in detections:
            xmin, ymin, xmax, ymax, conf, class_id = detection
            if conf < self.detection_threshold:
                continue
            if int(class_id) != 0: # Persona
                continue

            # Convert coordinates to integers and add padding
            x1 = max(0, int(xmin) - self.padding)
            y1 = max(0, int(ymin) - self.padding)
            x2 = min(img_width, int(xmax) + self.padding)
            y2 = min(img_height, int(ymax) + self.padding)
            
            # Extract the region of interest (ROI) of the person
            person_roi = frame[y1:y2, x1:x2]
            if person_roi.size == 0:
                continue

            # Process the ROI with mediapipe
            roi_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
            results = self.pose.process(roi_rgb)

            # Adjust the landmarks coordinates of the ROI to the image
            if results.pose_landmarks:
                adjusted_landmarks = []
                roi_height, roi_width, _ = person_roi.shape
                for lm in results.pose_landmarks.landmark:
                    adj_x = x1 + int(lm.x * roi_width)
                    adj_y = y1 + int(lm.y * roi_height)
                    adjusted_landmarks.append((adj_x, adj_y, lm.z, lm.visibility))
                detection_result = {
                    "bbox": (x1, y1, x2, y2),
                    "landmarks": adjusted_landmarks
                }
            else:
                detection_result = {
                    "bbox": (x1, y1, x2, y2),
                    "landmarks": None
                }
            pose_results.append(detection_result)

        return pose_results

    def draw_pose(self, frame, pose_results):
        """
        Draws the bounding boxes on the frame and, if available, the landmarks and
        the pose connections.
        
        Args:
            frame (numpy.ndarray): Original image in BGR.
            pose_results (list): List of detection results returned by detect_pose.
            
        Returns:
            numpy.ndarray: Image with the overlays.
        """
        for res in pose_results:
            x1, y1, x2, y2 = res["bbox"]
            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Skeleton
            if res["landmarks"]:

                for point in res["landmarks"]:
                    x, y, _, _ = point
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

                for connection in self.pose_connections:
                    start_idx, end_idx = connection
                    if start_idx < len(res["landmarks"]) and end_idx < len(res["landmarks"]):
                        pt1 = res["landmarks"][start_idx]
                        pt2 = res["landmarks"][end_idx]
                        pt1_coords = (pt1[0], pt1[1])
                        pt2_coords = (pt2[0], pt2[1])
                        cv2.line(frame, pt1_coords, pt2_coords, (0, 255, 0), 2)
        return frame

    def draw_pose_on_black(self, frame, pose_results):
        """
        Draws the detections on a black background, keeping the same dimensions
        as the original frame.
        
        Args:
            frame (numpy.ndarray): Original image (used to get dimensions).
            pose_results (list): Detection results.
            
        Returns:
            numpy.ndarray: Black image with the poses drawn.
        """
        black_frame = np.zeros_like(frame)
        for res in pose_results:
            x1, y1, x2, y2 = res["bbox"]
            cv2.rectangle(black_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            if res["landmarks"]:
                for point in res["landmarks"]:
                    x, y, _, _ = point
                    cv2.circle(black_frame, (x, y), 3, (0, 255, 0), -1)
                for connection in self.pose_connections:
                    start_idx, end_idx = connection
                    if start_idx < len(res["landmarks"]) and end_idx < len(res["landmarks"]):
                        pt1 = res["landmarks"][start_idx]
                        pt2 = res["landmarks"][end_idx]
                        pt1_coords = (pt1[0], pt1[1])
                        pt2_coords = (pt2[0], pt2[1])
                        cv2.line(black_frame, pt1_coords, pt2_coords, (0, 255, 0), 2)
        return black_frame
