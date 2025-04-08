import cv2
import mediapipe as mp


class HandDetectionService:
    def __init__(self,
                 max_num_hands,
                 detection_confidence,
                 tracking_confidence):
        """
        Initializes the HandDetectionService which uses MediaPipe to detect
        hands in images.
        
        Args:
            max_num_hands (int): Maximum number of hands to detect.
            detection_confidence (float): Minimum confidence value for hand
            detection to be considered successful.
            tracking_confidence (float): Minimum confidence value for hand
            landmarks to be considered tracked successfully.
        """
        
        self.mp_hands = mp.solutions.hands
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        
        self.mp_draw = mp.solutions.drawing_utils

    def detect_hands(self, frame_bgr):
        """
        Detects hands in the given BGR frame.
        
        Args:
            frame_bgr (numpy.ndarray): The input image in BGR format.
            
        Returns:
            results: The results of hand detection which includes landmarks and
            handedness.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        return results

    def draw_landmarks(self, frame_bgr, results):
        """
        Draws hand landmarks on the given BGR frame.
        
        Args:
            frame_bgr (numpy.ndarray): The input image in BGR format.
            results: The results of hand detection which includes landmarks and
            handedness.
            
        Returns:
            numpy.ndarray: The image with drawn landmarks.
        """
        if results and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame_bgr,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
        return frame_bgr
