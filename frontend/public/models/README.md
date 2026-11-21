# Local MediaPipe models

Place the compatible MediaPipe Tasks `.task` model files in this directory:

- `face_landmarker.task` — Face Landmarker model
- `pose_landmarker.task` — Pose Landmarker model
- `hand_landmarker.task` — Hand Landmarker model

The live page requests these exact paths from the frontend's `/models/` URL. If your downloaded pose model has a variant suffix (for example, `pose_landmarker_lite.task`), rename it to `pose_landmarker.task`. Keep the model file types compatible with the corresponding MediaPipe task.
