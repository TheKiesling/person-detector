import cv2


class CameraService:
    def __init__(self, camera_index=0):
        """
        Initializes the camera service.
        
        Args:
            camera_index (int): The index of the camera to use. Default is 0.
        """
        self.camera_index = camera_index
        self.cap = None

    def start_camera(self):
        """
        Starts the camera.

        Raises:
            RuntimeError: If the camera cannot be opened.
        """
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("Camera could not be opened.")

    def get_frame(self):
        """
        Captures a frame from the camera.
        
        Raises:
            Exception: If the camera fails to capture an image.
        
        Returns:
            tuple: A tuple containing a boolean indicating
            success and the captured frame.
        """
        ret, frame = self.cap.read()
        if not ret:
            raise Exception("Failed to capture image from camera.")
        return frame

    def release(self):
        """
        Releases the camera resource.
        """
        self.cap.release()
        cv2.destroyAllWindows()
