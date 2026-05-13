"""
Jarvis Railway Server
- Receives audio from laptop mic (or ESP32 later)
- Whisper transcribes it
- Groq generates response
- Sends command/response back over WebSocket
"""

import os
import json
import tempfile
import logging
from typing import Set

import whisper
from groq import Groq
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

WS_MAX_SIZE = 10 * 1024 * 1024  # 10MB

# ── Load Whisper model once at startup ───────────────────────────────────────
logger.info("Loading Whisper model (v2)...")
whisper_model = whisper.load_model("tiny")  # tiny = fastest, good enough for commands
logger.info("Whisper ready.")

# ── Groq client ───────────────────────────────────────────────────────────────
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

# ── Connected clients ─────────────────────────────────────────────────────────
connected_clients: Set[WebSocket] = set()

# ── Known commands ────────────────────────────────────────────────────────────
# These get sent directly to the ESP32 / laptop as action commands
# Everything else gets answered by Groq as a question
COMMANDS = {
    "open":        "SERVO_OPEN",
    "open visor":  "SERVO_OPEN",
    "close":       "SERVO_CLOSE",
    "close visor": "SERVO_CLOSE",
    "shutdown":    "PC_SHUTDOWN",
    "sleep":       "PC_SLEEP",
    "lock":        "PC_LOCK",
    "volume up":   "PC_VOL_UP",
    "volume down": "PC_VOL_DOWN",
    "mute":        "PC_MUTE",
    "lights on":   "HOME_LIGHTS_ON",
    "lights off":  "HOME_LIGHTS_OFF",
}


def detect_command(transcript: str) -> str | None:
    """Check if transcript matches a known command."""
    t = transcript.lower().strip().rstrip(".,!?")
    for phrase, command in COMMANDS.items():
        if phrase in t:
            return command
    return None


def ask_groq(question: str) -> str:
    """Send a question to Groq and get a Jarvis-style response."""
    try:
        response = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are JARVIS, Tony Stark's AI assistant. "
                        "You are helpful, witty, and concise. "
                        "Keep responses under 3 sentences unless more detail is needed. "
                        "Never break character."
                    ),
                },
                {"role": "user", "content": question},
            ],
            max_tokens=200,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Groq error: {e}")
        return "I'm having trouble connecting to my knowledge base right now, sir."


def transcribe_audio(audio_bytes: bytes) -> str:
    """Write audio bytes to a temp file and transcribe with Whisper."""
    import numpy as np
    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        # Load WAV directly with soundfile (no ffmpeg needed)
        audio_data, sample_rate = sf.read(tmp_path, dtype="float32")
        # Whisper expects mono 16kHz
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        result = whisper_model.transcribe(audio_data, language="en", fp16=False)
        return result["text"].strip()
    except Exception as e:
        logger.error(f"Whisper error: {e}")
        return ""
    finally:
        os.unlink(tmp_path)


# ── WebSocket endpoint ────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    logger.info(f"Client connected. Total: {len(connected_clients)}")

    try:
        while True:
            # Receive a message — either audio bytes or a JSON control message
            data = await websocket.receive()

            if "bytes" in data:
                # Audio chunk received — transcribe and process
                audio_bytes = data["bytes"]
                logger.info(f"Received audio: {len(audio_bytes)} bytes")

                if len(audio_bytes) < 1000:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Audio too short, please try again."
                    }))
                    continue

                # Transcribe
                try:
                    transcript = transcribe_audio(audio_bytes)
                except Exception as e:
                    logger.error(f"Transcription crashed: {e}", exc_info=True)
                    await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
                    continue
                logger.info(f"Transcript: '{transcript}'")

                if not transcript:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Could not understand audio."
                    }))
                    continue

                # Send transcript back so UI can display it
                await websocket.send_text(json.dumps({
                    "type": "transcript",
                    "text": transcript,
                }))

                # Check if it's a command
                command = detect_command(transcript)
                if command:
                    logger.info(f"Command detected: {command}")
                    await websocket.send_text(json.dumps({
                        "type": "command",
                        "command": command,
                        "transcript": transcript,
                    }))
                else:
                    # It's a question — ask Groq
                    logger.info("Sending to Groq...")
                    answer = ask_groq(transcript)
                    logger.info(f"Groq response: {answer}")
                    await websocket.send_text(json.dumps({
                        "type": "response",
                        "transcript": transcript,
                        "answer": answer,
                    }))

            elif "text" in data:
                # Control message from client (e.g. ping)
                msg = json.loads(data["text"])
                if msg.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))

    except WebSocketDisconnect:
        connected_clients.discard(websocket)
        logger.info(f"Client disconnected. Total: {len(connected_clients)}")
    except Exception as e:
        connected_clients.discard(websocket)
        logger.error(f"WebSocket error: {e}")


@app.get("/health")
async def health():
    return {"status": "ok", "clients": len(connected_clients)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
