import os
import torch
import numpy as np
import av
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
import json

def read_video_as_windows(video_path, fps=30, window_sec=2, stride_sec=1):
    """Slice a video into (T, H, W, C) sliding window chunks."""
    container = av.open(video_path)
    stream = container.streams.video[0]
    stream.thread_type = 'AUTO'

    window_size = fps * window_sec
    stride_size = fps * stride_sec

    frames = []
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format="rgb24"))

    frames = np.stack(frames)  # (num_frames, H, W, C)
    container.close()

    windows = []
    for start in range(0, len(frames) - window_size + 1, stride_size):
        clip = frames[start:start + window_size]
        windows.append(clip)

    return windows


def run_inference(model, processor, clip_np):
    """Run inference on a single sliding window (T, H, W, C)."""
    num_frames = model.config.num_frames
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Pad or trim clip to expected number of frames
    if clip_np.shape[0] > num_frames:
        clip_np = clip_np[:num_frames]
    elif clip_np.shape[0] < num_frames:
        last_frame = clip_np[-1]
        padding = np.repeat(last_frame[None], num_frames - clip_np.shape[0], axis=0)
        clip_np = np.concatenate([clip_np, padding], axis=0)

    # Convert to list of PIL-style frames
    frames = [clip_np[i] for i in range(clip_np.shape[0])]
    processed = processor(frames, return_tensors="pt")

    inputs = {k: v.to(device) for k, v in processed.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    return logits

def load_labels():
    with open("label_map.json", "r") as f:
        index_to_gloss = json.load(f)

    return index_to_gloss

def main():
    model_path = "./videomae-base-finetuned-ucf101-subset-run1/final_model"  # Or ./final_model
    video_path = "./yolo-wlasl-classify/val/116/4.mp4"

    model = VideoMAEForVideoClassification.from_pretrained(model_path)
    processor = VideoMAEImageProcessor.from_pretrained(model_path)

    windows = read_video_as_windows(video_path)
    print(f"Extracted {len(windows)} clips")

    label_map = load_labels()

    predictions = []
    for i, clip in enumerate(windows):
        logits = run_inference(model, processor, clip)
        pred = logits.argmax(-1).item()
        label = model.config.id2label[pred]
        predictions.append(label_map[label])
        print(f"[Window {i}] Predicted class: {label}")

    print("Final predictions across sliding windows:", predictions)


if __name__ == "__main__":
    main()
