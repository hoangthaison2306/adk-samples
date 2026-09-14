"""Week 2 acceptance harness for the travel_concierge voice bridge.

Drives N sample questions end-to-end through the bridge's WebSocket and checks
two things per question:

  1. ROUTING  - did it reach the expected sub-agent?
  2. LATENCY  - how long did the round trip take?

Two input modes:

  text   (default)  send the question as text, exercising
                    transcript -> travel_concierge -> response
  audio             send a WAV file as PCM, exercising the full
                    mic -> STT -> transcript -> travel_concierge path

Usage:
    uv run python harness.py                       # text mode, built-in questions
    uv run python harness.py --audio clips/        # audio mode, one .wav per question
    uv run python harness.py --url ws://host:8000  # non-default bridge

The bridge must already be running:
    uv run uvicorn voice_bridge:app --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import statistics
import sys
import time
import wave
from pathlib import Path

import websockets

# (question, sub-agent we expect to end up handling it)
# root_agent answering directly is acceptable for greetings/chitchat.
QUESTIONS: list[tuple[str, str | None]] = [
    ("Need some destination ideas for the Americas", "inspiration_agent"),
    ("What are some fun things to do in Seattle?", "inspiration_agent"),
    ("I want to plan a trip to Seattle in June", "planning_agent"),
    ("Find me flights from San Diego to Seattle", "planning_agent"),
    ("I'd like a window seat please", "planning_agent"),
    ("Show me hotels near downtown Seattle", "planning_agent"),
    ("I'm ready to pay for the booking", "booking_agent"),
    ("Do I need a visa for this trip?", "pre_trip_agent"),
    ("What should I pack?", "pre_trip_agent"),
    ("How do I get from my hotel to the Space Needle?", "in_trip_agent"),
]

CHUNK_MS = 200
SAMPLE_RATE = 16000


def _read_wav_as_pcm(path: Path) -> bytes:
    """Return 16kHz mono 16-bit PCM bytes, or raise with a clear message."""
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ValueError(f"{path.name}: need mono 16-bit WAV")
        if wav.getframerate() != SAMPLE_RATE:
            raise ValueError(
                f"{path.name}: need {SAMPLE_RATE}Hz, got {wav.getframerate()}"
            )
        return wav.readframes(wav.getnframes())


async def _collect_response(ws, timeout: float) -> dict | None:
    """Read until a response/error arrives, returning the parsed message."""
    deadline = time.monotonic() + timeout
    transcript = None
    while time.monotonic() < deadline:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=deadline - time.monotonic())
        except TimeoutError:
            return None
        msg = json.loads(raw)
        kind = msg.get("type")
        if kind == "transcript":
            transcript = msg.get("text")
        elif kind == "stt_unavailable":
            print(f"    ! STT unavailable: {msg.get('text', '')[:120]}")
        elif kind in ("response", "error"):
            if transcript:
                msg["heard"] = transcript
            return msg
    return None


async def run_question(
    url: str, idx: int, question: str, expected: str | None,
    audio: Path | None, timeout: float,
) -> dict:
    """Run one question through a fresh session; return a result row."""
    result: dict = {"question": question, "expected": expected}
    started = time.monotonic()
    try:
        async with websockets.connect(f"{url}/ws/harness-{idx}") as ws:
            if audio is not None:
                pcm = _read_wav_as_pcm(audio)
                step = int(SAMPLE_RATE * 2 * CHUNK_MS / 1000)
                for pos in range(0, len(pcm), step):
                    await ws.send(json.dumps({
                        "mime_type": f"audio/pcm;rate={SAMPLE_RATE}",
                        "data": base64.b64encode(pcm[pos:pos + step]).decode(),
                    }))
                    await asyncio.sleep(CHUNK_MS / 1000)
            else:
                await ws.send(json.dumps({"mime_type": "text/plain", "data": question}))

            msg = await _collect_response(ws, timeout)
    except Exception as exc:
        result.update(status="ERROR", detail=f"{type(exc).__name__}: {exc}")
        result["latency_ms"] = round((time.monotonic() - started) * 1000)
        return result

    result["latency_ms"] = round((time.monotonic() - started) * 1000)

    if msg is None:
        result.update(status="TIMEOUT", detail=f"no response within {timeout}s")
        return result
    if msg.get("type") == "error":
        result.update(status="ERROR", detail=msg.get("text", "")[:200])
        return result

    agents = msg.get("agents", [])
    transfers = msg.get("transfers", [])
    result.update(
        reply=msg.get("text", ""),
        heard=msg.get("heard"),
        agents=agents,
        transfers=transfers,
        tools=msg.get("tools", []),
        agent_latency_ms=msg.get("latency_ms"),
    )

    reached = set(agents) | set(transfers)
    if expected is None:
        result["status"] = "PASS"
    elif expected in reached:
        result["status"] = "PASS"
    elif not result["reply"]:
        result.update(status="FAIL", detail="empty reply")
    else:
        result.update(
            status="ROUTED-ELSEWHERE",
            detail=f"expected {expected}, got {sorted(reached) or 'none'}",
        )
    return result


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8000")
    parser.add_argument(
        "--audio",
        type=Path,
        help="directory of 16kHz mono WAVs named 01.wav, 02.wav ... one per question",
    )
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--limit", type=int, help="only run the first N questions")
    args = parser.parse_args()

    questions = QUESTIONS[: args.limit] if args.limit else QUESTIONS
    mode = "audio" if args.audio else "text"

    print(f"\ntravel_concierge voice harness - {mode} mode, {len(questions)} questions")
    print(f"bridge: {args.url}\n")

    rows = []
    for idx, (question, expected) in enumerate(questions, start=1):
        clip = None
        if args.audio:
            clip = args.audio / f"{idx:02d}.wav"
            if not clip.exists():
                print(f"[{idx:2d}] SKIP - missing {clip}")
                rows.append({
                    "question": question, "expected": expected,
                    "status": "SKIP", "detail": f"missing {clip.name}",
                    "latency_ms": 0,
                })
                continue

        print(f"[{idx:2d}] {question}")
        row = await run_question(args.url, idx, question, expected, clip, args.timeout)
        rows.append(row)

        mark = {"PASS": "OK", "ROUTED-ELSEWHERE": "~~", "SKIP": "--"}.get(
            row["status"], "XX"
        )
        print(f"     {mark} {row['status']}  {row['latency_ms']}ms")
        if row.get("heard"):
            print(f"     heard : {row['heard']}")
        if row.get("agents"):
            print(f"     agents: {' -> '.join(row['agents'])}")
        if row.get("tools"):
            print(f"     tools : {', '.join(row['tools'])}")
        if row.get("reply"):
            print(f"     reply : {row['reply'][:140]}")
        if row.get("detail"):
            print(f"     note  : {row['detail']}")
        print()

    print("=" * 68)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print("  " + "   ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    timings = [r["latency_ms"] for r in rows if r["status"] in ("PASS", "ROUTED-ELSEWHERE")]
    if timings:
        timings.sort()
        p95 = timings[min(len(timings) - 1, int(len(timings) * 0.95))]
        print(
            f"  latency: median {statistics.median(timings):.0f}ms  "
            f"p95 {p95}ms  max {max(timings)}ms"
        )
    print("=" * 68 + "\n")

    Path("harness_results.json").write_text(json.dumps(rows, indent=2))
    print("wrote harness_results.json\n")

    passed = counts.get("PASS", 0)
    print(f"ACCEPTANCE: {passed}/{len(rows)} questions routed as expected")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
