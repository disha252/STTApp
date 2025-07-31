from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import whisper
import yt_dlp
import os
import uuid
import tempfile

app = FastAPI()

# Load Whisper model once (base/small/medium/large)
model = whisper.load_model("base")

@app.post("/transcribe/audio")
async def transcribe_audio(file: UploadFile = File(...)):
    # Validate file type
    if not file.filename.endswith((".mp3", ".wav", ".m4a", ".webm")):
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    try:
        # Use a temp file for the uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            content = await file.read()
            temp_audio.write(content)
            temp_audio_path = temp_audio.name

        # Transcribe using Whisper
        result = model.transcribe(temp_audio_path)

        # Clean up
        os.remove(temp_audio_path)

        return {"text": result["text"]}
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/transcribe/url")
async def transcribe_from_url(video_url: str = Form(...)):
    # Generate unique filename for downloaded audio
    temp_id = uuid.uuid4().hex
    temp_file = f"{temp_id}.mp3"

    # yt-dlp options
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': temp_file,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True
    }

    try:
        # Download YouTube audio
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        # Check if file downloaded
        if not os.path.exists(temp_file):
            raise Exception("Download failed or file not created")

        # Transcribe using Whisper
        result = model.transcribe(temp_file)

        # Clean up
        os.remove(temp_file)

        return {"text": result["text"]}
    
    except Exception as e:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return JSONResponse(status_code=400, content={"error": str(e)})
