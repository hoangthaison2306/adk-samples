"""Voice front-end for travel_concierge.

Bridges the bidi-demo audio/WebSocket layer to travel_concierge:

    browser mic --(PCM)--> Live session (STT only) --transcript-->
        travel_concierge (text) --response--> browser

Why STT-only instead of attaching root_agent to the live session:
travel_concierge drives its sub-agents through AgentTool, and
AgentTool.run_async() uses the NON-live runner path, which requires a model
with `generateContent`. Native-audio Live models expose only
`bidiGenerateContent`, so every AgentTool call would fail. Transcribing first
and then invoking travel_concierge over text keeps routing and sub-agents
untouched, exactly as designed.

Env vars:
    GOOGLE_GENAI_MODEL   text model for travel_concierge (e.g. gemini-3.6-flash)
    VOICE_STT_MODEL      Live model used for transcription
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from google.adk.agents import Agent, LiveRequestQueue
from google.adk.agents.run_config import RunConfig
from google.adk.runners import InMemoryRunner
from google.genai import types

from travel_concierge.agent import root_agent

logger = logging.getLogger(__name__)

APP_NAME = "travel_concierge_voice"
STT_MODEL = os.getenv("VOICE_STT_MODEL", "gemini-2.5-flash-native-audio-preview-12-2025")

# Thin agent whose only job is to hold the Live session open so we can read
# input transcriptions off the event stream.
stt_agent = Agent(
    model=STT_MODEL,
    name="stt_agent",
    description="Transcribes the traveler's speech.",
    instruction="Only acknowledge briefly. Never answer questions.",
)

app = FastAPI(title="travel-concierge voice bridge")

_text_runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)
_stt_runner = InMemoryRunner(agent=stt_agent, app_name=APP_NAME + "_stt")


async def ask_travel_concierge(
    user_id: str, session_id: str, text: str
) -> dict[str, object]:
    """Run one text turn through travel_concierge.

    Returns the reply plus the routing trace, so the test harness can assert
    that a spoken question reached the right sub-agent rather than merely
    getting *some* answer back.
    """
    session = await _text_runner.session_service.get_session(
        app_name=APP_NAME, user_id=user_id, session_id=session_id
    )
    if session is None:
        session = await _text_runner.session_service.create_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id
        )

    message = types.Content(role="user", parts=[types.Part(text=text)])
    reply: list[str] = []
    agents: list[str] = []
    transfers: list[str] = []
    tools: list[str] = []

    started = time.monotonic()
    async for event in _text_runner.run_async(
        user_id=user_id, session_id=session.id, new_message=message
    ):
        if event.author and event.author not in agents:
            agents.append(event.author)
        if event.actions and event.actions.transfer_to_agent:
            transfers.append(event.actions.transfer_to_agent)
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    reply.append(part.text)
                if part.function_call and part.function_call.name:
                    tools.append(part.function_call.name)

    return {
        "text": "".join(reply).strip(),
        "agents": agents,
        "transfers": transfers,
        "tools": tools,
        "latency_ms": round((time.monotonic() - started) * 1000),
    }


@app.websocket("/ws/{session_id}")
async def voice_ws(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    user_id = f"voice-{session_id}"

    stt_session = await _stt_runner.session_service.create_session(
        app_name=APP_NAME + "_stt", user_id=user_id, session_id=session_id
    )
    live_queue = LiveRequestQueue()
    run_config = RunConfig(
        response_modalities=["AUDIO"],
        input_audio_transcription=types.AudioTranscriptionConfig(),
    )

    transcript_parts: list[str] = []

    async def downstream() -> None:
        """Live events -> transcript -> travel_concierge -> client.

        A failure here (bad Live model, quota, no key) must not take down the
        text path: the harness still needs to drive travel_concierge without a
        mic.
        """
        try:
            await _stream_transcripts()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("live/STT stream failed")
            await websocket.send_text(
                json.dumps(
                    {"type": "stt_unavailable", "text": f"{type(exc).__name__}: {exc}"}
                )
            )

    async def _stream_transcripts() -> None:
        async for event in _stt_runner.run_live(
            session=stt_session,
            live_request_queue=live_queue,
            run_config=run_config,
        ):
            if event.input_transcription and event.input_transcription.text:
                transcript_parts.append(event.input_transcription.text)
                await websocket.send_text(
                    json.dumps(
                        {"type": "partial_transcript", "text": event.input_transcription.text}
                    )
                )

            if event.turn_complete and transcript_parts:
                question = "".join(transcript_parts).strip()
                transcript_parts.clear()
                if not question:
                    continue
                await websocket.send_text(
                    json.dumps({"type": "transcript", "text": question})
                )
                try:
                    answer = await ask_travel_concierge(user_id, session_id, question)
                except Exception as exc:  # surface to the client, keep socket open
                    logger.exception("travel_concierge failed")
                    await websocket.send_text(
                        json.dumps({"type": "error", "text": f"{type(exc).__name__}: {exc}"})
                    )
                    continue
                await websocket.send_text(json.dumps({"type": "response", **answer}))

    async def upstream() -> None:
        """Client audio -> Live session."""
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            mime_type = msg.get("mime_type", "")
            data = msg.get("data", "")
            if mime_type.startswith("audio/pcm"):
                live_queue.send_realtime(
                    types.Blob(mime_type=mime_type, data=base64.b64decode(data))
                )
            elif mime_type == "text/plain":
                # lets the harness bypass the mic and send a typed question
                try:
                    answer = await ask_travel_concierge(user_id, session_id, data)
                except Exception as exc:  # keep the socket alive for the next question
                    logger.exception("travel_concierge failed")
                    await websocket.send_text(
                        json.dumps({"type": "error", "text": f"{type(exc).__name__}: {exc}"})
                    )
                    continue
                await websocket.send_text(json.dumps({"type": "response", **answer}))

    tasks = [asyncio.create_task(downstream()), asyncio.create_task(upstream())]
    try:
        done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for task in done:
            task.result()
    except WebSocketDisconnect:
        logger.info("client disconnected")
    except Exception:
        logger.exception("voice session failed")
    finally:
        for task in tasks:
            task.cancel()
        live_queue.close()


# Mounted last on purpose: a StaticFiles mount at "/" matches every path, so
# registering it before /ws/{session_id} would shadow the WebSocket route.
_STATIC = Path(__file__).parent / "static"
if _STATIC.is_dir():
    app.mount("/", StaticFiles(directory=_STATIC, html=True), name="static")
