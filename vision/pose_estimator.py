import numpy as np

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: mediapipe not installed. Pose estimation disabled.")


class PoseEstimator:
    """
    Extracts 33 body keypoints from video frames using MediaPipe Pose.

    Privacy: Only skeletal data (joint coordinates) is retained.
    No facial features, biometric signatures, or raw image data
    ever leaves this processing step.

    Runs efficiently on edge hardware (Jetson Nano) thanks to
    MediaPipe's lightweight model_complexity=0 setting.
    """

    def __init__(self):
        if MEDIAPIPE_AVAILABLE:
            mp_pose = mp.solutions.pose
            self.pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        else:
            self.pose = None

    def extract_keypoints(self, frame) -> np.ndarray | None:
        """
        Returns a (33, 3) array of [x, y, visibility] for each landmark,
        or None if no person is detected.

        Landmarks follow the MediaPipe Pose index convention:
          0  = nose, 11 = left shoulder, 12 = right shoulder,
          15 = left wrist, 16 = right wrist, 23 = left hip, 24 = right hip
        """
        if not MEDIAPIPE_AVAILABLE or self.pose is None:
            return None

        import cv2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if results.pose_landmarks is None:
            return None

        keypoints = np.array([
            [lm.x, lm.y, lm.visibility]
            for lm in results.pose_landmarks.landmark
        ])
        return keypoints

    def close(self):
        if self.pose is not None:
            self.pose.close()
