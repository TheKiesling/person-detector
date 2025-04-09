import mediapipe as mp
import cv2
import numpy as np


class PoseDetectionService:
    def __init__(self, static_image_mode=False, model_complexity=1,
                 enable_segmentation=False, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        """
        Initializes the PoseDetectionService with the given parameters.
        
        Args:
            static_image_mode (bool): If True, treat input images as a batch
            of static images.
            model_complexity (int): Complexity of the pose landmark model.
            enable_segmentation (bool): If True, enables segmentation mask
            prediction.
            min_detection_confidence (float): Minimum confidence value for
            person detection.
            min_tracking_confidence (float): Minimum confidence value for pose
            landmarks tracking.
        """
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

    def detect_pose(self, frame_bgr):
        """
        Detects pose landmarks in the given BGR frame.
        
        Args:
            frame_bgr (numpy.ndarray): The input image in BGR format.
            
        Returns:
            results: The results of pose detection which includes landmarks.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        return results

    def draw_pose(self, frame_bgr, results):
        """
        Draws pose landmarks on the given BGR frame.
        
        Args:
            frame_bgr (numpy.ndarray): The input image in BGR format.
            results: The results of pose detection which includes landmarks.
            
        Returns:
            numpy.ndarray: The image with drawn landmarks.
        """
        if results.pose_landmarks:
            self.mp_draw.draw_landmarks(
                frame_bgr,
                results.pose_landmarks,
                self.pose_connections
            )
        return frame_bgr

    def draw_pose_on_black(self, frame_bgr, results):
        """
        Draws pose landmarks on a black frame.
        
        Args:
            frame_bgr (numpy.ndarray): The input image in BGR format.
            results: The results of pose detection which includes landmarks.
            
        Returns:
            numpy.ndarray: The black frame with drawn landmarks.
        """
        height, width, _ = frame_bgr.shape

        black_frame = np.zeros((height, width, 3), dtype=np.uint8)

        if results.pose_landmarks:
            self.mp_draw.draw_landmarks(
                black_frame,
                results.pose_landmarks,
                self.pose_connections
            )
        return black_frame
