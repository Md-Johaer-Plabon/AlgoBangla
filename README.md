# 🎙 Running Whisper Locally: Step-by-Step Guide

## **Prerequisites**
Before running Whisper locally, ensure you have the following installed:
- Python (3.10.10 recommended, **Latest version of python has compatibility issue**)
- pip (Python package manager)
- FFmpeg (for audio processing)

### **1️⃣ Install Python and Dependencies**
Ensure you have Python installed. If not, download Python 3.10.10 from [Python's official site](https://www.python.org/downloads/release/python-31010/) and install it.

Check if Python is installed:
```sh
python --version
```

Upgrade pip:
```sh
pip install --upgrade pip
```

### **2️⃣ Install Whisper and Dependencies**
Install OpenAI Whisper along with dependencies (CPU only):
```sh
pip install openai-whisper torch torchvision torchaudio
```

If you're using a **GPU with CUDA**, install PyTorch with CUDA support:
```sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```


📌Ensure Chocolatey is installed:

With PowerShell, you must ensure ```Get-ExecutionPolicy``` is not Restricted. 
Run ```Get-ExecutionPolicy```. If it returns Restricted, then run ```Set-ExecutionPolicy AllSigned``` or ```Set-ExecutionPolicy Bypass -Scope Process```.


📌After installing choco, install FFmpeg:
```sh
choco install ffmpeg
```

📌Verify FFmpeg installation:
```sh
ffmpeg -version
```

### **3️⃣ Download OpenAI-Whisper**
```sh
pip install -U openai-whisper
```

📌For GPU acceleration:
```python
import whisper
model = whisper.load_model("medium").to("cuda")
```

### **4️⃣ Running Whisper Locally**
📌Run Whisper on an audio file: Whisper supports different models (`tiny`, `base`, `small`, `medium`, `large`). Download the desired model. 
Open the cmd in specific audio (audio.mp3 or .wav or .m4a) filepath
```sh
whisper audio.mp3 --model tiny
```
**Output: The model will bw downloaded and transcribe the audio.mp3 file.**

📌If you want to transcribe within a Python script:
```python
import whisper
model = whisper.load_model("tiny")
result = model.transcribe("audio.mp3")
print(result["text"])
```

### **5️⃣ Running Whisper as a FastAPI Server**
📌Install Dependencies:
```sh
pip install fastapi
pip install uvicorn
```

📌Create a file `app.py` with the following:
```python
from fastapi import FastAPI, UploadFile, File
import whisper

app = FastAPI()
model = whisper.load_model("tiny")  # Load model once for better performance

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    with open("temp_audio.mp3", "wb") as buffer:
        buffer.write(file.file.read())

    result = model.transcribe("temp_audio.mp3")
    return {"text": result["text"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
```

📌Run the server: Open cmd in the filepath of audio file.
```sh
python app.py
```

📌Browse the server:
```sh
http://127.0.0.1:8000/docs
```

### **6️⃣ Integrate in UWP app**
  ```git
  git pull
  ```
