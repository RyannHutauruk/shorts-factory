# shorts-factory

A pipeline that turns **public-domain feature films** (sourced from the Internet
Archive) into vertical short-form clips ready for YouTube Shorts / TikTok /
Reels.

## What it does

1. **Discover** — searches `archive.org` for public-domain feature films matching
   a query, ranked by community ratings and download count.
2. **Download** — pulls the best-quality MP4 with `yt-dlp`.
3. **Detect scenes** — splits the film into shots with `PySceneDetect`
   (`ContentDetector`).
4. **Transcribe + rank** — runs `faster-whisper` to get word-level transcripts
   and ranks scenes by a heuristic score (dialogue density, audio energy,
   duration sweet-spot).
5. **Reframe + caption** — re-encodes the top scenes to 1080×1920 9:16
   (smart crop with motion-weighted center-of-mass), burns in word-level
   captions, and writes them to `out/`.

> ⚠️ This project only works with content you have the right to reuse.
> The defaults target the Internet Archive's public-domain feature-film
> collection. Do not point it at copyrighted material.

## Quickstart

```bash
# system deps
sudo apt-get install -y ffmpeg

# python deps
uv sync                           # or: pip install -e ".[dev]"

# end-to-end run
uv run shorts-factory run \
    --query "night of the living dead" \
    --max-clips 5 \
    --clip-seconds 40
```

Outputs land in `out/<movie-id>/short_<NN>.mp4`.

## Sub-commands

```bash
shorts-factory search   --query "..." --limit 10
shorts-factory download --identifier night_of_the_living_dead
shorts-factory scenes   --video work/<file>.mp4
shorts-factory rank     --video work/<file>.mp4 --top 5
shorts-factory render   --video work/<file>.mp4 --start 1234.5 --duration 40
shorts-factory run      --query "..."          # full pipeline
```

## Layout

```
src/shorts_factory/
    cli.py          # typer entrypoint, sub-commands
    config.py       # paths + tunables
    discovery.py    # step 1 - archive.org search
    download.py     # step 2 - yt-dlp wrapper
    scenes.py       # step 3 - PySceneDetect
    transcribe.py   # step 4a - faster-whisper
    rank.py         # step 4b - heuristic scoring
    render.py       # step 5 - 9:16 reframe + captions (ffmpeg)
    pipeline.py     # orchestrator that wires 1->5 together
    util.py
```

## Roadmap

- LLM-based highlight ranker (plug into `rank.py`)
- Face-tracking smart-crop (currently motion-COM)
- Auto-upload to YouTube / TikTok / IG (intentionally not in v1)
- LangChain-style metadata generator (titles, descriptions, tags)
