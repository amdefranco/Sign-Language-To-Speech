
import gradio as gr
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor, pipeline
from espnet2.bin.tts_inference import Text2Speech
import soundfile as sf
import json
import tempfile
import nltk
import torch
from inference import read_video_as_windows, load_labels, inference_windows, dedup_sentence, ensemble_llm, text_to_speech
nltk.data.path.append("/home/rvmalhot/nltk_data")
nltk.download('averaged_perceptron_tagger_eng')

# Load models outside the handler to avoid repeated initialization
model_path = "./videomae-base-finetuned-ucf101-subset-run1/final_model"
model = VideoMAEForVideoClassification.from_pretrained(model_path)
processor = VideoMAEImageProcessor.from_pretrained(model_path)
label_map = load_labels()
llm_pipe = pipeline("text2text-generation", model="google/flan-t5-large")
# llm_pipe = pipeline(task="text2text-generation", model="hassaanik/grammar-correction-model")



tts = Text2Speech.from_pretrained(model_tag="kan-bayashi/ljspeech_vits", device="cuda" if torch.cuda.is_available() else "cpu")


def process_video(video_file):
    try:
        windows = read_video_as_windows(video_file)
        print(f"Extracted {len(windows)} clips")
        label_map = load_labels()
        predictions = inference_windows(windows, model, processor, label_map)
        print("Final predictions across sliding windows:", predictions)
        
        sentence = dedup_sentence(predictions)
        result = ensemble_llm(llm_pipe, sentence)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            audio_path = text_to_speech(tts, result, tmp.name)

        return result, audio_path
    except Exception as e:
        print("exception", e)



iface = gr.Interface(
    fn=process_video,
    inputs=gr.Video(label="Upload Sign Language Video"),
    outputs=[
        gr.Text(label="Gloss Translation to English"),
        gr.Audio(label="Synthesized Audio")
    ],
    title="Sign Language to Speech",
    description="Upload a short sign language video to see its English gloss translated to natural language and spoken aloud."
)

if __name__ == "__main__":
    iface.launch(share=True, debug=True, show_error=True)
    # print(llm_pipe("Fix the grammer for the sentence, make minimal changes as possible\n\nSentence: woman"))
    