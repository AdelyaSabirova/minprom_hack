from pathlib import Path
import cv2
import torch
from ultralytics import YOLO
import boxmot as bx

def create_video_writer(video_cap, output_filename, demo=False):

    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if demo: frame_width *= 2
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))

    if not Path(output_filename).parent.exists():
        Path(output_filename).parent.mkdir(parents=True)

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # type: ignore
    writer = cv2.VideoWriter(output_filename, fourcc, fps,
                             (frame_width, frame_height))

    return writer


def get_device(device):
    if device == "cpu":
        device = torch.device(device)
    else:
        device = torch.device(f"cuda:{device}")
    return device


def get_yolo_model(yolo_cfg, device="cpu"):
    model_path = yolo_cfg.pop("model")
    model = YOLO(model_path).to(device)
    return model


def get_tracker(tracker_cfg, device="cpu", **kwargs):
    tracker_cfg = tracker_cfg.copy()

    tracker_mapping = {
        'strongsort': 'boxmot.trackers.strongsort.strong_sort.StrongSORT',
        'ocsort': 'boxmot.trackers.ocsort.ocsort.OCSort',
        'bytetrack': 'boxmot.trackers.bytetrack.byte_tracker.BYTETracker',
        'botsort': 'boxmot.trackers.botsort.botsort.BoTSORT',
        'deepocsort': 'boxmot.trackers.deepocsort.deep_ocsort.DeepOCSort',
        'hybridsort': 'boxmot.trackers.hybridsort.hybridsort.HybridSORT',
        'imprassoc': 'boxmot.trackers.imprassoc.impr_assoc_tracker.ImprAssocTrack'
    }

    tracker_name = tracker_cfg.pop("tracker")
    if "device" in tracker_cfg:
        tracker_cfg["device"] = device

    module_path, class_name = tracker_mapping[tracker_name].rsplit('.', 1)
    tracker_class = getattr(__import__(module_path, fromlist=[class_name]), class_name)

    tracker = tracker_class(**tracker_cfg)

    return tracker


def dashed_rectangle(image, start_point, end_point, color, thickness=1, dash_length=10):
    x1, y1 = start_point
    x2, y2 = end_point
    
    for x in range(x1, x2, dash_length * 2):
        cv2.line(image, (x, y1), (min(x + dash_length, x2), y1), color, thickness)
        cv2.line(image, (x, y2), (min(x + dash_length, x2), y2), color, thickness)
    
    for y in range(y1, y2, dash_length * 2):
        cv2.line(image, (x1, y), (x1, min(y + dash_length, y2)), color, thickness)
        cv2.line(image, (x2, y), (x2, min(y + dash_length, y2)), color, thickness)
