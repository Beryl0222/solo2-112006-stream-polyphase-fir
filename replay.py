"""把交接采集回放进流式变采样引擎, 逐步打印裁决与结果(JSONL)。

用法::

    python -m replay samples/capture.json
"""
from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Tuple

from read_capture import read_capture
from resampler import GAP, Handoff


def _jsonable(obj: Any) -> Any:
    if obj is GAP:
        return None  # 空档在 JSON 中记为 null, 状态字段标明 "gap"
    if isinstance(obj, Fraction):
        return f"{obj.numerator}/{obj.denominator}"
    if isinstance(obj, dict):
        return {key: _jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(value) for value in obj]
    return obj


def build_engine(capture: Dict[str, Any]) -> Handoff:
    """按采集中的滤波器材质与保留窗口建引擎, 轨名取事件中的首次出现顺序。"""
    tracks: List[str] = []
    for event in capture["events"]:
        if event[0] in ("clock", "chunk", "gap", "safe") and event[1] not in tracks:
            tracks.append(event[1])
    return Handoff(
        tracks=tracks,
        offsets=capture["offsets"],
        taps=capture["taps"],
        retention=capture["retention"],
    )


def replay(capture: Dict[str, Any]) -> Tuple[Handoff, List[Dict[str, Any]]]:
    """按接收顺序回放事件, 返回引擎与每步的记录。"""
    engine = build_engine(capture)
    transcript: List[Dict[str, Any]] = []
    for event in capture["events"]:
        kind = event[0]
        if kind == "clock":
            _, track, revision, start, end, num, den, stride = event
            outcome = engine.clock(track, revision, start, end, num, den, stride)
            transcript.append({
                "event": event,
                "accepted": outcome.accepted,
                "code": outcome.code,
                "detail": outcome.detail,
            })
        elif kind == "chunk":
            outcome = engine.chunk(event[1], event[2], event[3])
            transcript.append({
                "event": event,
                "accepted": outcome.accepted,
                "code": outcome.code,
                "detail": outcome.detail,
            })
        elif kind == "gap":
            outcome = engine.gap(event[1], event[2], event[3])
            transcript.append({
                "event": event,
                "accepted": outcome.accepted,
                "code": outcome.code,
                "detail": outcome.detail,
            })
        elif kind == "safe":
            outcome = engine.safe(event[1], event[2])
            transcript.append({
                "event": event,
                "accepted": outcome.accepted,
                "code": outcome.code,
                "detail": outcome.detail,
            })
        elif kind == "preview":
            transcript.append({"event": event, "result": engine.preview(event[1], event[2])})
        elif kind == "publish":
            transcript.append({"event": event, "result": engine.publish()})
        else:
            transcript.append({"event": event, "error": "未知事件"})
    return engine, transcript


def main() -> None:
    parser = argparse.ArgumentParser(description="回放交接采集并打印每步裁决与结果")
    parser.add_argument("capture", type=Path)
    args = parser.parse_args()
    capture = read_capture(args.capture)
    engine, transcript = replay(capture)
    for entry in transcript:
        print(json.dumps(_jsonable(entry), ensure_ascii=False))
    summary = {
        "delivered": engine.delivered,
        "archive": engine.published(0, engine.delivered),
        "blockers": engine.blockers(),
    }
    print(json.dumps(_jsonable({"summary": summary}), ensure_ascii=False))


if __name__ == "__main__":
    main()
