from models.load_models import load_transcriber, load_diarizer
import torchaudio

def process_audio(path):
    transcriber = load_transcriber()
    diarizer = load_diarizer()
    waveform, sr = torchaudio.load(path)

    text = transcriber(waveform, sr)
    speakers = diarizer(waveform, sr)

    return {"transcript": text, "speakers": speakers}
