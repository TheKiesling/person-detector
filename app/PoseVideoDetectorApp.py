import cv2
import numpy as np

from services.PoseDetector2 import PoseDetectionService2


class PoseVideoDetectorApp:
    def __init__(self, input_source, output_pose, output_skeleton, output_width=None, output_height=None):
        """
        Initializes the PoseVideoDetectorApp with input source and output
        files.
        
        Args:
            input_source (str): Path to the input video file or camera index.
            output_pose (str): Path to save the output video with pose
            detection.
            output_skeleton (str): Path to save the output video with skeleton
            detection.
            output_width (int, optional): Desired output width. If None, uses input width.
            output_height (int, optional): Desired output height. If None, uses input height.
        """
        self.input_source = input_source
        self.output_pose = output_pose
        self.output_skeleton = output_skeleton
        self.output_width = output_width
        self.output_height = output_height
        self.pose_service = PoseDetectionService2()

        self.cap = cv2.VideoCapture(self.input_source)

        if not self.cap.isOpened():
            raise Exception("Failed to capture video from source")

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        # If output dimensions are not specified, use input dimensions
        if self.output_width is None:
            self.output_width = self.frame_width
        if self.output_height is None:
            self.output_height = self.frame_height

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer_pose = cv2.VideoWriter(
            self.output_pose,
            fourcc,
            self.fps,
            (self.output_width, self.output_height)
        )

        self.writer_skeleton = cv2.VideoWriter(
            self.output_skeleton,
            fourcc,
            self.fps,
            (self.output_width, self.output_height)
        )

    def _resize_with_padding(self, frame):
        """
        Resizes the frame to the output dimensions while maintaining aspect ratio
        by adding black padding if necessary.
        
        Args:
            frame (numpy.ndarray): Input frame to resize
            
        Returns:
            numpy.ndarray: Resized frame with padding if needed
        """
        # Calculate aspect ratios
        input_ratio = self.frame_width / self.frame_height
        output_ratio = self.output_width / self.output_height
        
        if input_ratio > output_ratio:
            # Input is wider than output
            new_width = self.output_width
            new_height = int(self.output_width / input_ratio)
            padding_top = (self.output_height - new_height) // 2
            padding_bottom = self.output_height - new_height - padding_top
            padding_left = 0
            padding_right = 0
        else:
            # Input is taller than output
            new_height = self.output_height
            new_width = int(self.output_height * input_ratio)
            padding_left = (self.output_width - new_width) // 2
            padding_right = self.output_width - new_width - padding_left
            padding_top = 0
            padding_bottom = 0
            
        # Resize the frame
        resized = cv2.resize(frame, (new_width, new_height))
        
        # Add padding
        padded = cv2.copyMakeBorder(
            resized,
            padding_top,
            padding_bottom,
            padding_left,
            padding_right,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )
        
        return padded

    def run(self):
        """
        Main loop to read frames from the video source, detect poses, and save
        the output.
        """
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            results = self.pose_service.detect_pose(frame)

            frame_with_pose = frame.copy()
            frame_with_pose = self.pose_service.draw_pose(frame_with_pose,
                                                          results)

            skeleton_frame = self.pose_service.draw_pose_on_black(frame,
                                                                  results)

            # Resize frames with padding if needed
            if (self.output_width != self.frame_width or 
                self.output_height != self.frame_height):
                frame_with_pose = self._resize_with_padding(frame_with_pose)
                skeleton_frame = self._resize_with_padding(skeleton_frame)

            self.writer_pose.write(frame_with_pose)
            self.writer_skeleton.write(skeleton_frame)

            cv2.imshow("Pose en video original", frame_with_pose)
            cv2.imshow("Pose en fondo negro", skeleton_frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.cleanup()

    def cleanup(self):
        """
        Releases the video capture and writer objects and closes all
        OpenCV windows.
        """
        self.cap.release()
        self.writer_pose.release()
        self.writer_skeleton.release()
        cv2.destroyAllWindows()
