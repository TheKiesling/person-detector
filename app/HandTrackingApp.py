import cv2

from services.Camera import CameraService
from services.HandDetector import HandDetectionService


class HandTrackingApp:
    def __init__(self,
                 camera_index,
                 max_num_hands,
                 detection_confidence,
                 tracking_confidence):
        """
        Initializes the HandTrackingApp with camera and hand detection
        services.
        
        Args:
            camera_index (int): Index of the camera to use.
            max_num_hands (int): Maximum number of hands to detect.
            detection_confidence (float): Confidence threshold for hand
            detection.
            tracking_confidence (float): Confidence threshold for hand
            tracking.
        """
        self.camera_service = CameraService(camera_index)
        
        self.hand_detector = HandDetectionService(
            max_num_hands=max_num_hands,
            detection_confidence=detection_confidence,
            tracking_confidence=tracking_confidence
        )

    def run(self):
        """
        Starts the hand tracking application.
        
        This method initializes the camera, captures frames, detects hands,
        draws landmarks, and displays the output in a window. The application
        runs until the user presses the 'Esc' key.
        """
        self.camera_service.start_camera()
        while True:
            frame = self.camera_service.get_frame()
            results = self.hand_detector.detect_hands(frame)
            frame = self.hand_detector.draw_landmarks(frame, results)
            
            cv2.imshow("", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.camera_service.release()
