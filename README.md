# Corner-Based Face Aligner

A computer-vision project built around a YOLOv5-style face detector with four predicted corner landmarks. The detection script uses the landmark coordinates to locate a quadrilateral on the face and includes a perspective-transform helper for rectifying it.

> The repository name is `corener-based-aligner`; the project description uses the corrected spelling “corner-based.”

## Capabilities

- Face detection with bounding boxes, confidence scores, and four corner landmarks.
- PyTorch model loading from checkpoint weights.
- Non-maximum suppression and coordinate scaling for detections.
- Annotated image output written as `result.jpg`.
- Model export and evaluation utilities, including ONNX/TensorFlow Lite conversion helpers.

## Requirements

The code depends on Python, PyTorch, OpenCV, NumPy, and Pillow. Use versions compatible with this YOLOv5-derived codebase; dependency versions are not currently pinned in a root requirements file.

## Run detection

The default script expects a checkpoint at `runs/train/exp5/weights/last.pt` and an image at `data/images/test.jpg`. Supply your own image and trained checkpoint:

```bash
python detect_face.py --weights path/to/model.pt --image path/to/image.jpg
```

Optional arguments include `--img-size`, `--conf-thres`, and `--iou-thres`. The current entry point selects the CPU device in code. It saves the annotated result in the current working directory as `result.jpg`.

The repository does not guarantee that trained weights or sample images are included. Checkpoint architecture and class configuration must match the model being loaded.

## Perspective transform

The `four_point_transform` helper in `detect_face.py` maps the predicted four corners to a fixed 500 × 300 output canvas. It is currently not called by the default detection flow; integrate it into your application if you need rectified crops.

## Main files

- `detect_face.py`: image inference, drawing, and perspective-transform helper.
- `train.py`: training pipeline.
- `test.py`, `test_landmark.py`, `test_widerface.py`: evaluation scripts.
- `landmark_eval.py`: landmark evaluation utilities.
- `export.py`, `onnx2pb.py`, `pb2tflite.py`: model export/conversion helpers.
- `models/`, `utils/`: model definitions and supporting code.

## License

No license file is currently listed. Review upstream YOLOv5 licensing and contact the repository owner before reusing or redistributing this project.
