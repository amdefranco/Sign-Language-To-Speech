# Sign-Language-To-Speech

### How to run

### Prepare data

```
python load_dataset.py
```

Re-run this if the job dies due to rate limit errors

```
python prepare_dataset.py
```

### Run train job

```
python train.py
```

or if using SLURM

```
sbatch train_job.sh
```

### Inference

```
python inference.py
```

### Run gradio demo

```
python demo.py
```

## Implementation details

We do an ensemble of three models

1. Video classification (MCG-NJU/videomae-base)
2. LLM (google/flan-t5-large)
3. TTS (kan-bayashi/ljspeech_vits)

### Video classification

1. Finetuned on WLASL (training details yet to come)
2. Performs word level classification
3. Do inference on video with 2 second sliding window and 1 second stride on window to classify words during the video
4. Choose words where the window classified passes a certain threshold

The idea of the sliding window is to capture indiviual words that occur in the duration the video

A transformer based model is used because sign language recognition requires identifying movement across multiple frames. We believe attention is a good method to recognize this.

### LLM

1. First we dedup words that occur consecutively in a sentence
2. We ask an LLM to fix grammer issues during classification using the prompt

```
f"Fix the grammer for the sentence, make minimal changes as possible\n\nSentence: {sentence}"
```

### TTS

The output from the LLM is passed to a TTS model to create an audio
