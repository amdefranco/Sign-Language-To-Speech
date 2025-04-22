import os
import torch
import numpy as np
import av
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
from transformers import pipeline
import json
from itertools import groupby
import torch.nn.functional as F
from espnet2.bin.tts_inference import Text2Speech
import soundfile as sf

import nltk
nltk.data.path.append("/home/rvmalhot/nltk_data")
nltk.download('averaged_perceptron_tagger_eng')

def read_video_as_windows(video_path, fps=30, window_sec=2, stride_sec=4):
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


def text_to_speech(tts, text, output_wav_path="output.wav"):
    # Run synthesis
    wav = tts(text)["wav"]

    # Save to WAV file
    sf.write(output_wav_path, wav.view(-1).cpu().numpy(), 22050)
    print(f"Saved synthesized audio to {output_wav_path}")
    return output_wav_path




def dedup_sentence(predictions):
    # Prompt the LLM to convert glosses into fluent English
    sentence = ' '.join(predictions)        
    deduped = " ".join([k for k,v in groupby(sentence.split())])
    print("deduped sentence:", deduped)
    return deduped

def inference_windows(windows, model, processor, label_map):
    predictions = []
    for i, clip in enumerate(windows):
        logits = run_inference(model, processor, clip)
        logits = torch.tensor(logits).squeeze(0)
        probs = F.softmax(torch.tensor(logits), dim=-1)  # convert logits to probabilities
        pred = probs.argmax(-1).item()
        confidence = probs[pred].item()  # this is the probability of the predicted label
        label = label_map[str(pred)]
        print(f"[Window {i}] Predicted class: {pred} {label} {confidence}")
        if confidence * 100 > 5:
            predictions.append(label)

    return predictions


def ensemble_llm(llm_pipe, sentence):
    prompt = f"Fix the grammer for the sentence, make minimal changes as possible\n\nSentence: {sentence}"
    result = llm_pipe(prompt)[0]['generated_text']
    print(f"Cleaned Sentence: {result}")
    return result


def main():
    model_path = "./videomae-base-finetuned-ucf101-subset-run1/final_model"  # Or ./final_model
    video_path = "./sign_example.mov"
    video_path = "./yolo-wlasl-classify/train/108/14.mp4"

    model = VideoMAEForVideoClassification.from_pretrained(model_path)
    processor = VideoMAEImageProcessor.from_pretrained(model_path)
    llm_model = "google/flan-t5-large"
    llm_pipe = pipeline("text2text-generation", model=llm_model)

    # Load pretrained TTS model (you can swap this out with any model in espnet_model_zoo)
    tts = Text2Speech.from_pretrained(
            model_tag="kan-bayashi/ljspeech_vits",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    
    windows = read_video_as_windows(video_path)
    print(f"Extracted {len(windows)} clips")
    label_map = load_labels()
    predictions = inference_windows(windows, model, processor, label_map)
    print("Final predictions across sliding windows:", predictions)
    sentence = dedup_sentence(predictions)
    result = ensemble_llm(llm_pipe, sentence)
    text_to_speech(tts, result)

if __name__ == "__main__":
    main()