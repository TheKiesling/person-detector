import cv2

from services.PoseDetector import PoseDetectionService


class PoseVideoDetectorApp:
    def __init__(self, input_source, output_pose, output_skeleton):
        """
        Initializes the PoseVideoDetectorApp with input source and output
        files.
        
        Args:
            input_source (str): Path to the input video file or camera index.
            output_pose (str): Path to save the output video with pose
            detection.
            output_skeleton (str): Path to save the output video with skeleton
            detection.
        """
        self.input_source = input_source
        self.output_pose = output_pose
        self.output_skeleton = output_skeleton
        self.pose_service = PoseDetectionService()

        self.cap = cv2.VideoCapture(self.input_source)

        if not self.cap.isOpened():
            raise Exception("Failed to capture video from source")

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer_pose = cv2.VideoWriter(
            self.output_pose,
            fourcc,
            self.fps,
            (self.frame_width, self.frame_height)
        )

        self.writer_skeleton = cv2.VideoWriter(
            self.output_skeleton,
            fourcc,
            self.fps,
            (self.frame_width, self.frame_height)
        )

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
