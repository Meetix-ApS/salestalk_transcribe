from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import torch

# Brug GPU hvis tilgængelig
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# Model ID
model_id = "syvai/hviske-v3-conversation"

# Hent model og processor
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=dtype)
processor = AutoProcessor.from_pretrained(model_id)

# Lav pipeline
pipe = pipeline(
 "automatic-speech-recognition",
 model=model,
 tokenizer=processor.tokenizer,
 feature_extractor=processor.feature_extractor,
 chunk_length_s=30,
 batch_size=8,
 torch_dtype=dtype,
 device=0 if device == "cuda" else -1,
)

# Indsæt navnet på din lydfil her
AUDIO_PATH = "test_audio.wav"

# Kør transkription
result = pipe(AUDIO_PATH)
print("TRANSKRIPTION:\n", result["text"])
