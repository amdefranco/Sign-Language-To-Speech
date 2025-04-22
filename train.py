import os
import torch
import numpy as np
import pathlib
import evaluate
from transformers import (
    VideoMAEImageProcessor, 
    VideoMAEForVideoClassification,
    TrainingArguments, 
    Trainer,
    pipeline
)

import pytorchvideo.data
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    UniformTemporalSubsample,
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)

# Set batch size
batch_size = 8
num_gpu = 1

def count_videos(dataset_root_path):
    """Count the number of videos in the dataset."""
    dataset_root_path = pathlib.Path(dataset_root_path)
    
    video_count_train = len(list(dataset_root_path.glob("train/*/*.mp4")))
    video_count_val = len(list(dataset_root_path.glob("val/*/*.mp4")))
    video_count_test = len(list(dataset_root_path.glob("test/*/*.mp4")))
    video_total = video_count_train + video_count_val + video_count_test
    
    print(f"Total videos: {video_total}")
    print(f"Train videos: {video_count_train}")
    print(f"Validation videos: {video_count_val}")
    print(f"Test videos: {video_count_test}")
    
    return video_count_train, video_count_val, video_count_test

def get_label_mappings():
    """Create label to id and id to label mappings."""
    class_labels = [str(i) for i in range(119)]
    label2id = {label: i for i, label in enumerate(class_labels)}
    id2label = {i: label for label, i in label2id.items()}
    
    print(f"Unique classes: {list(label2id.keys())}.")
    
    return label2id, id2label

def load_model(label2id, id2label, model_ckpt="MCG-NJU/videomae-base"):
    """Load the pre-trained model and image processor."""
    image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
    model = VideoMAEForVideoClassification.from_pretrained(
        model_ckpt,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
    )
    
    return model, image_processor

def prepare_datasets(dataset_root_path, image_processor, model):
    """Prepare the training, validation, and test datasets."""
    # Define constants
    mean = image_processor.image_mean
    std = image_processor.image_std
    if "shortest_edge" in image_processor.size:
        height = width = image_processor.size["shortest_edge"]
    else:
        height = image_processor.size["height"]
        width = image_processor.size["width"]
    resize_to = (height, width)
    
    num_frames_to_sample = model.config.num_frames
    sample_rate = 4
    fps = 30
    clip_duration = num_frames_to_sample * sample_rate / fps
    
    # Define training transformations
    train_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames_to_sample),
                        Lambda(lambda x: x / 255.0),
                        Normalize(mean, std),
                        RandomShortSideScale(min_size=256, max_size=320),
                        RandomCrop(resize_to),
                        RandomHorizontalFlip(p=0.5),
                    ]
                ),
            ),
        ]
    )
    
    # Define validation and test transformations
    val_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames_to_sample),
                        Lambda(lambda x: x / 255.0),
                        Normalize(mean, std),
                        Resize(resize_to),
                    ]
                ),
            ),
        ]
    )
    
    # Create datasets
    train_dataset = pytorchvideo.data.Ucf101(
        data_path=os.path.join(dataset_root_path, "train"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
        decode_audio=False,
        transform=train_transform,
    )
    
    val_dataset = pytorchvideo.data.Ucf101(
        data_path=os.path.join(dataset_root_path, "val"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
        decode_audio=False,
        transform=val_transform,
    )
    
  
    print(f"Train dataset videos: {train_dataset.num_videos}")
    print(f"Validation dataset videos: {val_dataset.num_videos}")
    
    return train_dataset, val_dataset, clip_duration


def collate_fn(examples):
    """Collate function for batching examples."""
    # permute to (num_frames, num_channels, height, width)
    pixel_values = torch.stack(
        [example["video"].permute(1, 0, 2, 3) for example in examples]
    )
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

def train_model(model, train_dataset, val_dataset, image_processor, model_ckpt, num_epochs=4):
    """Train the model using the Trainer class."""
    model_name = model_ckpt.split("/")[-1]
    new_model_name = f"{model_name}-continued-training"

    max_steps = (train_dataset.num_videos // (batch_size * num_gpu)) * (num_epochs*2)

    print("max steps", max_steps)
    args = TrainingArguments(
        new_model_name,
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,  # Set to True if you want to push to Hub
        max_steps=max_steps,
        log_level="debug",  # Global log level (Python's logging module)
        log_level_replica="debug",
        fp16=True,
        max_grad_norm=1.0,
        adam_epsilon=1e-6
    )
    
    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=image_processor,  # This is for saving the processor config
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )
    
    train_results = trainer.train()
    
    # Optionally push to hub
    # trainer.push_to_hub()
    
    return trainer, train_results

def run_inference(model, video):
    """Run inference on a video."""
    # (num_frames, num_channels, height, width)
    perumuted_sample_test_video = video.permute(1, 0, 2, 3)
    inputs = {
        "pixel_values": perumuted_sample_test_video.unsqueeze(0),
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model = model.to(device)
    
    # forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    return logits

def main():
    # Load metric
    global metric
    metric = evaluate.load("accuracy")
    
    dataset_root_path = "yolo-wlasl-classify"
    # Count videos
    video_count_train, video_count_val, video_count_test = count_videos(dataset_root_path)
    
    # Get label mappings
    label2id, id2label = get_label_mappings()
    
    # Load model
    model, image_processor = load_model(label2id, id2label, model_ckpt="./videomae-base-finetuned-ucf101-subset-run1/checkpoint-5372")
    
    # Prepare datasets
    train_dataset, val_dataset, clip_duration = prepare_datasets(
        dataset_root_path, image_processor, model
    )
    
    # Train model
    trainer, train_results = train_model(
        model, train_dataset, val_dataset, image_processor, "MCG-NJU/videomae-base"
    )
    
    # Save the model
    trainer.save_model("./continued_model")
    
    # Test on a sample video
    sample_test_video = next(iter(val_dataset))
    logits = run_inference(model, sample_test_video["video"])
    predicted_class_idx = logits.argmax(-1).item()
    print("Predicted class:", model.config.id2label[predicted_class_idx])
    
    # Create a pipeline for inference
    # print("Creating pipeline for inference...")
    # video_cls = pipeline(
    #     task="video-classification", 
    #     model="./final_model",
    #     device=0 if torch.cuda.is_available() else -1
    # )
    
    # You can use the pipeline like this:
    # result = video_cls("path/to/video.mp4")
    # print(result)

if __name__ == "__main__":
    main()