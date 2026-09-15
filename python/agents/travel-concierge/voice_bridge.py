"""Voice front-end for travel_concierge, built on the bidi-demo streaming layer.

This is a port of `python/agents/bidi-demo/app/main.py` with the attached agent
swapped from `google_search_agent` to travel_concierge's `root_agent`. The
WebSocket protocol is unchanged, so bidi-demo's own frontend (`app.js`,
`audio-recorder.js`, the PCM worklets) drives it as-is:

    client -> server   binary frames = 16kHz mono 16-bit PCM
                       text frames   = {"type": "text", "text": ...}
    server -> client   raw ADK Event JSON (exclude_none, by_alias)

Why root_agent is not simply handed to run_live():
travel_concierge reaches its sub-agents through AgentTool, and
AgentTool.run_async() takes the NON-live runner path, which requires a model
exposing `generateContent`. Native-audio Live models expose only
`bidiGenerateContent`, so every sub-agent call inside a live session fails.

So the live session holds a transcription-only gateway agent, and each finished
input transcription is passed to travel_concierge through its ordinary free-text
entry point. Routing, sub-agents and tools are untouched. travel_concierge's
events are forwarded verbatim, in the same shape the live events have, so the
existing frontend renders them without a single change.

Env vars:
    GOOGLE_GENAI_MODEL   text model for travel_concierge (e.g. gemini-3.6-flash)
    VOICE_STT_MODEL      Live model used for transcription
    BIDI_STATIC_DIR      override the frontend directory
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import warnings
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from google.adk.agents import Agent
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

load_dotenv(Path(__file__).parent / ".env")

# Imported after load_dotenv: travel_concierge resolves its backend at import.
from travel_concierge.agent import root_agent  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
# Under `uvicorn voice_bridge:app` the server configures logging before this
# module is imported, so basicConfig above is a no-op and the root logger stays
# at WARNING -- which would hide every line below. Setting the level here works
# either way: propagated records are filtered by the originating logger, not by
# the root logger's level.
logger.setLevel(logging.INFO)
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

APP_NAME = "travel-concierge-voice"
STT_MODEL = os.getenv(
    "VOICE_STT_MODEL", "gemini-2.5-flash-native-audio-preview-12-2025"
)

# Reuse bidi-demo's frontend rather than shipping a second copy of it.
_DEFAULT_STATIC = (
    Path(__file__).resolve().parents[1] / "bidi-demo" / "app" / "static"
)
STATIC_DIR = Path(os.getenv("BIDI_STATIC_DIR", _DEFAULT_STATIC))

# ========================================
# Phase 1: Application Initialization (once at startup)
# ========================================

app = FastAPI(title="travel-concierge voice bridge")

_INDEX = STATIC_DIR / "index.html"
# Report a missing frontend at startup rather than as a 500 on the first page
# load: the usual cause is an interrupted checkout, which is worth naming.
if _INDEX.is_file():
    logger.info("serving frontend from %s", STATIC_DIR)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
else:
    logger.error(
        "frontend not found: %s does not exist. The WebSocket still works, so "
        "harness.py can run, but the browser UI cannot be served. Expected "
        "bidi-demo's frontend alongside this package; set BIDI_STATIC_DIR to "
        "point elsewhere.",
        _INDEX,
    )

session_service = InMemorySessionService()

# Holds the live session open so input transcriptions arrive. It must never
# answer anything: travel_concierge owns the conversation.
gateway_agent = Agent(
    name="voice_gateway",
    model=STT_MODEL,
    description="Transcribes the traveler's speech.",
    instruction="Stay silent. Never speak, answer, or acknowledge. Say nothing.",
)

live_runner = Runner(
    app_name=APP_NAME + "-stt", agent=gateway_agent, session_service=session_service
)
text_runner = Runner(
    app_name=APP_NAME, agent=root_agent, session_service=session_service
)


# ========================================
# HTTP Endpoints
# ========================================


@app.get("/")
async def root():
    """Serve bidi-demo's index.html."""
    if not _INDEX.is_file():
        return PlainTextResponse(
            f"Frontend not found at {_INDEX}\n\n"
            "bidi-demo's frontend is missing from this checkout. Restore it with:\n"
            "    git checkout HEAD -- python/agents/bidi-demo/app/static\n\n"
            "Or point BIDI_STATIC_DIR at a directory containing index.html.\n"
            "The WebSocket endpoint works regardless, so harness.py can still run.",
            status_code=503,
        )
    return FileResponse(_INDEX)


# ========================================
# WebSocket Endpoint
# ========================================


@app.websocket("/ws/{user_id}/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    user_id: str,
    session_id: str,
    proactivity: bool = False,
    affective_dialog: bool = False,
) -> None:
    """Bidirectional streaming endpoint, protocol-identical to bidi-demo."""
    logger.info(
        "WebSocket request: user_id=%s session_id=%s proactivity=%s "
        "affective_dialog=%s",
        user_id, session_id, proactivity, affective_dialog,
    )
    await websocket.accept()

    # ========================================
    # Phase 2: Session Initialization (once per streaming session)
    # ========================================

    # Native audio models only support the AUDIO response modality; half-cascade
    # models also do TEXT, which is faster. Either way we need input
    # transcription, since the transcript *is* the input to travel_concierge.
    is_native_audio = "native-audio" in STT_MODEL.lower()
    if is_native_audio:
        run_config = RunConfig(
            streaming_mode=StreamingMode.BIDI,
            response_modalities=["AUDIO"],
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            session_resumption=types.SessionResumptionConfig(),
            proactivity=(
                types.ProactivityConfig(proactive_audio=True) if proactivity else None
            ),
            enable_affective_dialog=affective_dialog if affective_dialog else None,
        )
    else:
        run_config = RunConfig(
            streaming_mode=StreamingMode.BIDI,
            response_modalities=["TEXT"],
            input_audio_transcription=types.AudioTranscriptionConfig(),
            session_resumption=types.SessionResumptionConfig(),
        )
        if proactivity or affective_dialog:
            logger.warning(
                "proactivity/affective dialog need a native-audio model; "
                "%s is half-cascade, ignoring them", STT_MODEL
            )

    for app_name in (APP_NAME, APP_NAME + "-stt"):
        if not await session_service.get_session(
            app_name=app_name, user_id=user_id, session_id=session_id
        ):
            await session_service.create_session(
                app_name=app_name, user_id=user_id, session_id=session_id
            )

    live_request_queue = LiveRequestQueue()
    heard: list[str] = []

    async def send_event(event: Event) -> None:
        await websocket.send_text(
            event.model_dump_json(exclude_none=True, by_alias=True)
        )

    async def ask_travel_concierge(question: str) -> None:
        """Feed one utterance into travel_concierge's free-text entry point.

        Its events go to the client in the same shape the live events have, so
        the frontend renders the reply and the harness can read `author` and
        `actions.transferToAgent` to see which sub-agent handled the question.
        """
        logger.info("-> travel_concierge: %s", question)
        message = types.Content(role="user", parts=[types.Part(text=question)])
        invocation_id = ""
        try:
            async for event in text_runner.run_async(
                user_id=user_id, session_id=session_id, new_message=message
            ):
                invocation_id = event.invocation_id or invocation_id
                await send_event(event)
        except Exception as exc:
            # Report and keep the socket open; one bad turn must not end the call.
            logger.exception("travel_concierge failed")
            await websocket.send_text(
                json.dumps({"error": f"{type(exc).__name__}: {exc}"})
            )
        # run_async never sets turn_complete; the frontend needs it to close the
        # bubble and re-enable input.
        await send_event(
            Event(
                author=root_agent.name,
                invocation_id=invocation_id or "voice",
                turn_complete=True,
            )
        )

    # ========================================
    # Phase 3: Active Session (concurrent bidirectional communication)
    # ========================================

    async def upstream_task() -> None:
        """WebSocket -> LiveRequestQueue (audio) or travel_concierge (text)."""
        while True:
            message = await websocket.receive()

            if "bytes" in message:
                live_request_queue.send_realtime(
                    types.Blob(mime_type="audio/pcm;rate=16000", data=message["bytes"])
                )

            elif "text" in message:
                json_message = json.loads(message["text"])
                kind = json_message.get("type")

                # Typed text skips the live session entirely: it is already the
                # transcript, and this is the path the harness drives.
                if kind == "text":
                    await ask_travel_concierge(json_message["text"])

                elif kind == "image":
                    live_request_queue.send_realtime(
                        types.Blob(
                            mime_type=json_message.get("mimeType", "image/jpeg"),
                            data=base64.b64decode(json_message["data"]),
                        )
                    )

    async def downstream_task() -> None:
        """Live events -> transcript -> travel_concierge.

        Only transcription and control events are forwarded. The gateway agent's
        own content is dropped: it exists to transcribe, and letting its audio
        through would talk over travel_concierge.
        """
        async for event in live_runner.run_live(
            user_id=user_id,
            session_id=session_id,
            live_request_queue=live_request_queue,
            run_config=run_config,
        ):
            transcription = event.input_transcription
            if transcription and transcription.text:
                heard.append(transcription.text)
                await send_event(event)
            elif event.interrupted:
                await send_event(event)

            # `finished` is the reliable marker; turn_complete is the fallback
            # for models that never set it.
            complete = (transcription and transcription.finished) or event.turn_complete
            if complete and heard:
                question = "".join(heard).strip()
                heard.clear()
                if question:
                    await ask_travel_concierge(question)

    async def guarded_downstream() -> None:
        """Keep an STT failure from taking the text path down with it.

        A missing key, a quota error or a model without bidiGenerateContent all
        surface here, and the typed-text path still has to work afterwards.
        """
        try:
            await downstream_task()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("live/STT stream failed")
            await websocket.send_text(
                json.dumps({"sttUnavailable": f"{type(exc).__name__}: {exc}"})
            )

    upstream = asyncio.create_task(upstream_task())
    downstream = asyncio.create_task(guarded_downstream())
    try:
        # Not gather(): upstream must survive a dead STT stream.
        await upstream
    except WebSocketDisconnect:
        logger.info("client disconnected")
    except Exception:
        logger.exception("voice session failed")
    finally:
        # ========================================
        # Phase 4: Session Termination
        # ========================================
        downstream.cancel()
        live_request_queue.close()
