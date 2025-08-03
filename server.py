app = FastAPI()

@app.post("/transcribe")
async def transcribe(file: UploadFile):
    with open("audio/input.wav", "wb") as f:
        f.write(await file.read())
    result = process_audio("audio/input.wav")
    return result
