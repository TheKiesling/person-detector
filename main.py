from app.HandTrackingApp import HandTrackingApp


def main():
    app = HandTrackingApp(camera_index=0,
                          max_num_hands=10,
                          detection_confidence=0.5,
                          tracking_confidence=0.5)
    
    app.run()


if __name__ == "__main__":
    main()
