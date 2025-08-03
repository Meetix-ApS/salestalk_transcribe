from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq
import torch

def load_transcriber():
    model_id = "syvai/hviske-v3-conversation"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=dtype)
    processor = AutoProcessor.from_pretrained(model_id)

    return pipeline(
        "automatic-speech-recognition",
