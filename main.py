import sys

from app.HandTrackingApp import HandTrackingApp
from app.PoseVideoDetectorApp import PoseVideoDetectorApp


def main():
    app = None
    if sys.argv[1] == "hand_tracking":
        app = HandTrackingApp(
            camera_index=0,
            max_num_hands=10,
            detection_confidence=0.5,
            tracking_confidence=0.5
        )
        
    elif sys.argv[1] == "pose_video" and len(sys.argv) > 2:
        app = PoseVideoDetectorApp(
            input_source=sys.argv[2],
            output_pose='output/pose.mp4',
            output_skeleton='output/skeleton.mp4',
            output_width=750,
            output_height=650
        )
    
    app.run()
    

if __name__ == "__main__":
    main()
