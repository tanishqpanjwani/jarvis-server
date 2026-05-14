"""
Jarvis Railway Server
- Receives audio from laptop mic (or ESP32 later)
- faster-whisper transcribes it (4x faster than whisper on CPU)
- Groq generates response
- Sends command/response back over WebSocket
"""

import os
import json
import tempfile
import logging
from typing import Set

from faster_whisper import WhisperModel
from groq import Groq
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

WS_MAX_SIZE = 10 * 1024 * 1024  # 10MB

# ── Load faster-whisper model once at startup ─────────────────────────────────
logger.info("Loading Whisper model...")
whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
logger.info("Whisper ready.")

# ── Groq client ───────────────────────────────────────────────────────────────
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

# ── Connected clients ─────────────────────────────────────────────────────────
connected_clients: Set[WebSocket] = set()

# ── Known commands ────────────────────────────────────────────────────────────
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
    t = transcript.lower().strip().rstrip(".,!?")
    for phrase, command in COMMANDS.items():
        if phrase in t:
            return command
    return None


def ask_groq(question: str) -> str:
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
    """Write audio bytes to temp WAV and transcribe with faster-whisper."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        segments, info = whisper_model.transcribe(
            tmp_path,
            language="en",
            beam_size=1,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        logger.info(f"Detected language: {info.language} ({info.language_probability:.2f})")
        transcript = " ".join(s.text for s in segments).strip()
        logger.info(f"Transcript: '{transcript}'")
        return transcript
    except Exception as e:
        logger.error(f"Whisper error: {e}", exc_info=True)
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
            data = await websocket.receive()

            if "bytes" in data:
                audio_bytes = data["bytes"]
                logger.info(f"Received audio: {len(audio_bytes)} bytes")

                if len(audio_bytes) < 1000:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Audio too short, please try again."
                    }))
                    continue

                transcript = transcribe_audio(audio_bytes)

                if not transcript:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Could not understand audio."
                    }))
                    continue

                await websocket.send_text(json.dumps({
                    "type": "transcript",
                    "text": transcript,
                }))

                command = detect_command(transcript)
                if command:
                    logger.info(f"Command detected: {command}")
                    await websocket.send_text(json.dumps({
                        "type": "command",
                        "command": command,
                        "transcript": transcript,
                    }))
                else:
                    logger.info("Sending to Groq...")
                    answer = ask_groq(transcript)
                    logger.info(f"Groq response: {answer}")
                    await websocket.send_text(json.dumps({
                        "type": "response",
                        "transcript": transcript,
                        "answer": answer,
                    }))

            elif "text" in data:
                msg = json.loads(data["text"])
                if msg.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))

    except WebSocketDisconnect:
        connected_clients.discard(websocket)
        logger.info(f"Client disconnected. Total: {len(connected_clients)}")
    except Exception as e:
        connected_clients.discard(websocket)
        logger.error(f"WebSocket error: {e}", exc_info=True)


@app.get("/health")
async def health():
    return {"status": "ok", "clients": len(connected_clients)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
