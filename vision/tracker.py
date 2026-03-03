from ultralytics import YOLO
import cv2
import os


class VideoTracker:
    """
    Tracks people across video frames using YOLOv8.
    Provides bounding box coordinates and stable track IDs
    that persist across frames.
    """

    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model = YOLO(model_path)

    def process_video(self, video_path: str, output_path: str, tension_scorer=None):
        """
        Run the full tracking pipeline on a video file.

        Args:
            video_path:     Path to input video.
            output_path:    Path where the annotated output video is saved.
            tension_scorer: Optional VisualTensionScorer instance.
                            If None, tension scores default to 0.3.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        out = None
        tension_scores: dict[int, float] = {}

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.track(frame, persist=True, verbose=False)
            annotated_frame = frame.copy()

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy()

                for i, box in enumerate(boxes):
                    person_id = int(ids[i])
                    x1, y1, x2, y2 = map(int, box)

                    if person_id not in tension_scores:
                        tension_scores[person_id] = 0.3

                    if tension_scorer is not None:
                        crop = frame[max(0, y1):y2, max(0, x1):x2]
                        if crop.size > 0:
                            tension_scores[person_id] = tension_scorer.score(
                                crop, person_id, box=(x1, y1, x2, y2)
                            )

                    tension = tension_scores[person_id]
                    color = (0, 255, 0) if tension < 0.6 else (0, 0, 255)

                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        annotated_frame,
                        f"ID {person_id} | Tension: {tension:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                    )
                    if tension >= 0.6:
                        cv2.putText(
                            annotated_frame, "ALERT",
                            (x1, y2 + 20),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2,
                        )

            if out is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(
                    output_path, fourcc, 20.0,
                    (annotated_frame.shape[1], annotated_frame.shape[0]),
                )

            out.write(annotated_frame)

        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        print(f"Annotated video saved to {output_path}")
