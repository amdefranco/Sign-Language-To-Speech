import os
import shutil
import random
from datasets import load_dataset, Video
from collections import defaultdict
from tqdm import tqdm

# Load WLASL dataset and prevent auto-decoding
dataset = load_dataset("Voxel51/WLASL")
train_split = dataset["train"].cast_column("video", Video(decode=False))

# Output structure
output_root = "yolo-wlasl-classify"
train_dir = os.path.join(output_root, "train")
val_dir = os.path.join(output_root, "val")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Group samples by label
samples_by_label = defaultdict(list)
for i, sample in enumerate(train_split):
    label = sample["label"]
    path = sample["video"]["path"]
    samples_by_label[label].append(path)

# Split and copy
for label, video_paths in tqdm(samples_by_label.items(), desc="Copying samples"):
    random.shuffle(video_paths)
    split_idx = int(0.9 * len(video_paths))
    train_paths = video_paths[:split_idx]
    val_paths = video_paths[split_idx:]

    # Copy train videos
    train_label_dir = os.path.join(train_dir, str(label))
    os.makedirs(train_label_dir, exist_ok=True)
    for i, src in enumerate(train_paths):
        dst = os.path.join(train_label_dir, f"{i}.mp4")
        shutil.copy(src, dst)

    # Copy val videos
    val_label_dir = os.path.join(val_dir, str(label))
    os.makedirs(val_label_dir, exist_ok=True)
    for i, src in enumerate(val_paths):
        dst = os.path.join(val_label_dir, f"{i}.mp4")
        shutil.copy(src, dst)

print("✅ WLASL data split and copied to YOLOv8 classification format.")
