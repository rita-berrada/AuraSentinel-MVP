import numpy as np
from vision.pose_estimator import PoseEstimator


class VisualTensionScorer:
    """
    Computes a visual tension score [0.0, 1.0] per tracked person.

    Primary signals (always available from YOLOv8 bounding boxes):
      1. Centroid velocity  — fast movement → agitation
      2. Box size change    — sudden area increase → rushing / confrontation
      3. Aspect ratio shift — wide/flat box → falling or crouching

    Secondary signals (used when MediaPipe Pose is available):
      4. Arm raise   — wrists above shoulders
      5. Torso lean  — lateral asymmetry

    Scores are smoothed across frames with an exponential moving average
    so a single noisy frame doesn't spike a false alert.
    """

    ALERT_THRESHOLD = 0.6
    SMOOTHING = 0.4  # EMA factor — higher = reacts faster to changes

    def __init__(self):
        self.pose_estimator = PoseEstimator()
        self._prev_box: dict[int, tuple] = {}
        self._prev_keypoints: dict[int, np.ndarray] = {}
        self._smoothed: dict[int, float] = {}

    def score(self, frame, person_id: int = 0, box: tuple | None = None) -> float:
        """
        Returns smoothed tension score for a tracked person.

        Args:
            frame:      BGR image crop of the bounding box region.
            person_id:  Stable YOLOv8 track ID.
            box:        (x1, y1, x2, y2) in pixel coordinates.
        """
        raw = self._compute_raw(frame, person_id, box)

        # Exponential moving average to smooth out per-frame noise
        prev = self._smoothed.get(person_id, raw)
        smoothed = self.SMOOTHING * raw + (1 - self.SMOOTHING) * prev
        self._smoothed[person_id] = smoothed
        return float(np.clip(smoothed, 0.0, 1.0))

    def _compute_raw(self, frame, person_id: int, box: tuple | None) -> float:
        scores = []

        # ── Box-based signals (always available) ──────────────────────────
        if box is not None:
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            area = w * h
            ratio = w / h if h > 0 else 1.0

            if person_id in self._prev_box:
                pcx, pcy, parea, pratio = self._prev_box[person_id]

                # 1. Centroid velocity (normalised: ~40px/frame = max tension)
                velocity = np.sqrt((cx - pcx) ** 2 + (cy - pcy) ** 2)
                scores.append(float(np.clip(velocity / 40.0, 0, 1)) * 0.55)

                # 2. Sudden box size change (rushing toward camera)
                if parea > 0:
                    size_change = abs(area - parea) / parea
                    scores.append(float(np.clip(size_change * 4, 0, 1)) * 0.25)

                # 3. Aspect ratio shift (falling / crouching)
                scores.append(float(np.clip(abs(ratio - pratio) * 2, 0, 1)) * 0.20)

            self._prev_box[person_id] = (cx, cy, area, ratio)

        # ── Pose-based signals (when MediaPipe is available) ───────────────
        keypoints = self.pose_estimator.extract_keypoints(frame)
        if keypoints is not None:
            lwy = keypoints[15, 1]; rwy = keypoints[16, 1]
            lsy = keypoints[11, 1]; rsy = keypoints[12, 1]
            arm_raise = (float(lwy < lsy) + float(rwy < rsy)) / 2
            scores.append(arm_raise * 0.5)

            if person_id in self._prev_keypoints:
                delta = np.linalg.norm(
                    keypoints[:, :2] - self._prev_keypoints[person_id][:, :2]
                )
                scores.append(float(np.clip(delta * 4, 0, 1)))

            self._prev_keypoints[person_id] = keypoints

        return float(np.clip(sum(scores), 0.0, 1.0)) if scores else 0.3

    def close(self):
        self.pose_estimator.close()
