"""Step 4a - transcribe the movie audio with faster-whisper."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import PATHS, WHISPER


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str
    words: list[Word]


def transcribe(
    video_path: Path,
    *,
    model_size: str = WHISPER.model_size,
    language: str | None = WHISPER.language,
    device: str = WHISPER.device,
    compute_type: str = WHISPER.compute_type,
    cache_path: Path | None = None,
) -> list[Segment]:
    """Run faster-whisper and return word-level segments.

    Caches the JSON next to the video by default so subsequent rank/render
    invocations don't re-transcribe.
    """
    PATHS.ensure()
    cache = cache_path or (PATHS.transcripts / f"{video_path.stem}.json")
    if cache.exists():
        return _load_cache(cache)

    # Imported lazily because faster-whisper pulls in heavy native deps.
    from faster_whisper import WhisperModel

    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        download_root=str(PATHS.models),
    )
    segments_iter, _info = model.transcribe(
        str(video_path),
        language=language,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )

    segments: list[Segment] = []
    for seg in segments_iter:
        words: list[Word] = []
        for w in seg.words or []:
            text = (w.word or "").strip()
            if not text:
                continue
            words.append(Word(start=float(w.start), end=float(w.end), text=text))
        segments.append(
            Segment(
                start=float(seg.start),
                end=float(seg.end),
                text=(seg.text or "").strip(),
                words=words,
            )
        )

    _save_cache(cache, segments)
    return segments


def _save_cache(cache: Path, segments: list[Segment]) -> None:
    cache.parent.mkdir(parents=True, exist_ok=True)
    serialised = [
        {
            "start": s.start,
            "end": s.end,
            "text": s.text,
            "words": [asdict(w) for w in s.words],
        }
        for s in segments
    ]
    cache.write_text(json.dumps(serialised, ensure_ascii=False))


def _load_cache(cache: Path) -> list[Segment]:
    raw = json.loads(cache.read_text())
    out: list[Segment] = []
    for s in raw:
        words = [Word(**w) for w in s.get("words", [])]
        out.append(Segment(start=s["start"], end=s["end"], text=s["text"], words=words))
    return out


def words_in_window(segments: list[Segment], start: float, end: float) -> list[Word]:
    out: list[Word] = []
    for seg in segments:
        if seg.end < start or seg.start > end:
            continue
        for w in seg.words:
            if w.end < start or w.start > end:
                continue
            out.append(w)
    return out
