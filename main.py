"""
AuraSentinel MVP — Video Analysis Pipeline

Processes a video file through the full vision pipeline:
  1. Track people using YOLOv8
  2. Estimate body pose via MediaPipe Pose (if installed)
  3. Score visual tension per tracked individual
  4. Annotate and save the output video

For the federated learning simulation, run separately:
    python federated/simulate.py

For the staff alert dashboard, run:
    python dashboard/app.py
"""

from vision.tracker import VideoTracker
from vision.tension_scorer import VisualTensionScorer

VIDEO_INPUT = "data/shop_robbery.mp4"
VIDEO_OUTPUT = "output/annotated_video.mp4"
MODEL_PATH = "yolov8n.pt"


def main():
    print("AuraSentinel — Video Analysis Pipeline")
    print(f"Input  : {VIDEO_INPUT}")
    print(f"Output : {VIDEO_OUTPUT}\n")

    scorer = VisualTensionScorer()
    tracker = VideoTracker(model_path=MODEL_PATH)
    tracker.process_video(VIDEO_INPUT, VIDEO_OUTPUT, tension_scorer=scorer)
    scorer.close()


if __name__ == "__main__":
    main()
