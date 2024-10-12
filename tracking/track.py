import cv2
import torch
import numpy as np
from tqdm import tqdm, trange

import yaml
from fire import Fire

from ultralytics import YOLO

import time

import boxmot as bx
from boxmot.trackers.basetracker import TrackState


from utils import (
    create_video_writer,
    get_device,
    get_yolo_model,
    get_tracker,
    dashed_rectangle
)


RED = (0, 0, 255)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)


def run(
    video_cap,
    writer,
    model,
    tracker,
    yolo_cfg,
    skip_ratio,
    skip_len,
    video_start_stop,
    show_detections,
    demo
):
    num_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))

    start_frame = 0
    if video_start_stop[0] is not None:
        start_frame = int(video_start_stop[0] * fps)
    end_frame = num_frames
    if video_start_stop[1] is not None:
        end_frame = min(num_frames, int(video_start_stop[1] * fps))

    active_hist = set()

    for i in range(start_frame):
        ret, frame = video_cap.read()
        if not ret:
            break

    det_times, track_times = [], []
    for i in trange(start_frame, end_frame):

        # Read
        ret, frame = video_cap.read()
        if not ret:
            break

        # Detect
        det_start = time.perf_counter()
        det = model.predict(
            frame,
            **yolo_cfg,
            verbose=False
        )[0].boxes.data.cpu().numpy()
        det_end = time.perf_counter()
        det_times.append(det_end - det_start)

        if len(det) == 0:
            det = np.empty((0, 6))

        if skip_ratio is not None or skip_len is not None: # artificially skip frames to visualize tracks
            assert skip_ratio is not None and skip_len is not None
            period = int(skip_len / skip_ratio)
            if period - skip_len <= i % period <= period:
                det = np.empty((0, 6))

        # Track
        track_start = time.perf_counter()
        tracks = tracker.update(det, frame) # [xtl, ytl, xbr, ybr, id, conf, class, index (from detections)]
        track_end = time.perf_counter()
        track_times.append(track_end - track_start)

        # Visualize
        img_h, img_w, _ = frame.shape

        caption_size = int(img_h * 0.03)
        caption_font_size = img_h * 0.0005
        caption_font_thickness = max(2, int(caption_font_size * 4))
        bbox_line_thickness = 2

        if demo:
            frame = np.concatenate([frame, frame], axis=1)
            bbox_line_thickness = 4

        if show_detections:
            for d in det.astype(int).copy():
                d[0:2] -= 2
                d[2:3] += 2
                d[[0,2]] = np.clip(d[[0,2]], 0, img_w)
                d[[1,3]] = np.clip(d[[1,3]], 0, img_h)
                xtl, ytl, xbr, ybr = d[:4]
                cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), RED, bbox_line_thickness)

        if tracks.shape[0] != 0:
            tracks[:, [0,2]] = np.clip(tracks[:, [0,2]], 0, img_w)
            tracks[:, [1,3]] = np.clip(tracks[:, [1,3]], 0, img_h)

        for track in tracks:
            xtl, ytl, xbr, ybr, track_id = track[:5].astype(int)
            cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), GREEN, bbox_line_thickness)

            if demo:
                cv2.rectangle(frame, (xtl + img_w, ytl), (xbr + img_w, ybr), GREEN, bbox_line_thickness)

            if not demo:
                cv2.rectangle(frame, (xtl, ytl - int(caption_size*0.7)), (xtl + caption_size, ytl),
                              GREEN, -1)
                cv2.putText(frame, str(track_id), (xtl + 5, ytl - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, caption_font_size, WHITE, caption_font_thickness)

        active_hist.update(track for track in tracker.active_tracks)
        for track in [track for track in active_hist if track.state == TrackState.Lost]:
            (xtl, ytl, xbr, ybr), track_id = track.xyxy.astype(int), track.id

            if not demo:
                cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), GRAY, bbox_line_thickness)
            else:
                cv2.rectangle(frame, (xtl + img_w, ytl), (xbr + img_w, ybr), GREEN, bbox_line_thickness)

            if not demo:
                cv2.rectangle(frame, (xtl, ytl - int(caption_size*0.7)), (xtl + caption_size, ytl),
                            GRAY, -1)
                cv2.putText(frame, str(track_id), (xtl + 5, ytl - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, caption_font_size, WHITE, caption_font_thickness)

        if not demo:
            cv2.putText(frame, repr({track.id: track.state for track in active_hist}), (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, caption_font_size*3, WHITE, caption_font_thickness*2)

        if demo:
            cv2.putText(frame, "Tracking OFF", (150, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, caption_font_size*5, RED, caption_font_thickness*4)
            cv2.putText(frame, "Tracking ON", (150 + img_w, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, caption_font_size*5, GREEN, caption_font_thickness*4)

        # Write
        writer.write(frame)

    det_times = np.array(det_times) * 1000
    track_times = np.array(track_times) * 1000

    print(f"Detection latency: {det_times.mean():.1f} +- {det_times.std():.1f} ms")
    print(f"Tracking latency: {track_times.mean():.1f} +- {track_times.std():.1f} ms")

    video_cap.release()
    writer.release()


def main(
    source,
    save_path,
    yolo_cfg,
    tracker_cfg,
    device="cpu",
    skip_ratio=None,
    skip_len=None,
    video_start_stop=(None, None),
    show_detections=False,
    demo=False
):
    
    device = get_device(device)
    
    with open(yolo_cfg, 'r') as f:
        yolo_cfg = yaml.load(f, Loader=yaml.SafeLoader)
        assert "device" not in yolo_cfg, \
            "Need to specify device once for the whole run by script CLI"
        yolo_cfg["device"] = device

    with open(tracker_cfg, 'r') as f:
        tracker_cfg = yaml.load(f, Loader=yaml.SafeLoader)
        assert "device" not in tracker_cfg, \
            "Need to specify device once for the whole run by script CLI"
    
    video_cap = cv2.VideoCapture(source)
    writer = create_video_writer(video_cap, save_path, demo=demo)
    
    model = get_yolo_model(yolo_cfg, device)  # type: ignore

    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    tracker = get_tracker(tracker_cfg, device, fps=fps)  # type: ignore

    assert not (show_detections and demo)

    run(
        video_cap=video_cap,
        writer=writer,
        model=model,
        tracker=tracker,
        yolo_cfg=yolo_cfg,
        skip_ratio=skip_ratio,
        skip_len=skip_len,
        video_start_stop=video_start_stop,
        show_detections=show_detections,
        demo=demo
    )


if __name__ == "__main__":
    Fire(main)
