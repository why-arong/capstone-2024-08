import whisper


model = whisper.load_model("medium")
result = model.transcribe("audio.mp3")
print(result["text"])
