import numpy as np
from vision.pose_estimator import PoseEstimator


class VisualTensionScorer:
    """
    Computes a visual tension score [0.0, 1.0] from body pose keypoints.

    Three cues are combined:
      1. Arm raise — wrists above shoulders signals agitation or aggression.
      2. Movement speed — rapid keypoint displacement between frames.
      3. Torso lean — lateral imbalance indicates a confrontational stance.

    Falls back to a neutral baseline (0.3) if MediaPipe is unavailable.
    """

    BASELINE = 0.3

    def __init__(self):
        self.pose_estimator = PoseEstimator()
        self._prev_keypoints: dict[int, np.ndarray] = {}

    def score(self, frame, person_id: int = 0) -> float:
        """
        Returns tension score for a cropped person frame.

        Args:
            frame:      BGR image crop of the tracked person.
            person_id:  Stable track ID used to compute movement deltas.
        """
        keypoints = self.pose_estimator.extract_keypoints(frame)
        if keypoints is None:
            return self.BASELINE

        scores: list[float] = []

        # 1. Arm raise (wrists above shoulders → agitation)
        left_wrist_y = keypoints[15, 1]
        right_wrist_y = keypoints[16, 1]
        left_shoulder_y = keypoints[11, 1]
        right_shoulder_y = keypoints[12, 1]
        arm_raise = (
            float(left_wrist_y < left_shoulder_y) +
            float(right_wrist_y < right_shoulder_y)
        ) / 2
        scores.append(arm_raise * 0.6)

        # 2. Movement speed (displacement of keypoints from previous frame)
        if person_id in self._prev_keypoints:
            delta = np.linalg.norm(
                keypoints[:, :2] - self._prev_keypoints[person_id][:, :2]
            )
            movement_score = float(np.clip(delta * 5, 0, 1))
            scores.append(movement_score)

        # 3. Torso lean / lateral asymmetry
        hip_center_x = (keypoints[23, 0] + keypoints[24, 0]) / 2
        shoulder_center_x = (keypoints[11, 0] + keypoints[12, 0]) / 2
        lean = float(np.clip(abs(hip_center_x - shoulder_center_x) * 3, 0, 1))
        scores.append(lean)

        self._prev_keypoints[person_id] = keypoints
        return float(np.clip(np.mean(scores), 0.0, 1.0))

    def close(self):
        self.pose_estimator.close()
