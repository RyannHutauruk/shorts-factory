# shorts-factory

A pipeline that turns content **you have the right to reuse** — public-domain
feature films from the Internet Archive, or Creative-Commons-licensed YouTube
videos — into vertical short-form clips ready for YouTube Shorts / TikTok /
Reels.

## What it does

1. **Discover** — searches `archive.org` for public-domain feature films, **or**
   the YouTube Data API for videos uploaded under a Creative Commons license
   (`videoLicense=creativeCommon`), filtered by duration / view count.
2. **Download** — pulls the best-quality MP4 (direct HTTP for archive.org;
   `yt-dlp` for YouTube). The YouTube path re-verifies `status.license ==
   "creativeCommon"` before downloading and refuses anything else.
3. **Detect scenes** — splits the film into shots with `PySceneDetect`
   (`ContentDetector`), cached to disk.
4. **Transcribe + rank** — runs `faster-whisper` to get word-level transcripts
   and ranks scenes by a heuristic score (dialogue density, audio energy,
   duration sweet-spot, motion).
5. **Reframe + caption** — re-encodes the top scenes to 1080×1920 9:16
   (blurred-bars + centered foreground), burns in word-level
   captions, writes them to `out/`, and emits an `ATTRIBUTION.txt` next to
   each clip when the source requires credit (CC-BY).

> ⚠️ This project will refuse to download YouTube videos that aren't licensed
> Creative Commons, and the archive.org defaults target public-domain
> collections only. Do not point it at copyrighted material — even with
> mirroring / pitch-shift / subtitle overlays, YouTube's Content ID system
> will still match copyrighted audio and you will lose the channel.

## Two pipelines in one repo

| Pipeline | Source | When to use |
| --- | --- | --- |
| **archive.org / movies** | Public-domain feature films | Auto-clipped 9:16 shorts from CC/PD movies |
| **YouTube CC-BY** | YouTube Data API + yt-dlp | Same as above but using modern CC-BY YouTube content |
| **Niche (`niche`)** | **Topic → Gemini script → Piper TTS → Wikimedia B-roll → ffmpeg** | **Faceless-niche explainer shorts (true crime / disasters / history). Most monetizable; doesn't reuse other people's footage.** |

## Faceless-niche pipeline (recommended)

Generates a 35-45 second narrated explainer short from a single topic. No video source needed — the pipeline writes a script with Gemini, narrates it with Piper TTS, fetches CC-licensed images from Wikimedia Commons, and composes them into a 1080×1920 short with burned-in captions and a clickable title/description.

```bash
export GEMINI_API_KEY="..."   # free key: https://aistudio.google.com/apikey

# one-shot
shorts-factory niche --topic "Hindenburg disaster" --niche true_crime

# batch (one short per topic, with anti-repetition across the run)
shorts-factory niche-batch --topic "Tenerife airport disaster" \
                           --topic "Edmund Fitzgerald" \
                           --topic "Halifax explosion"

# from a topics file (one per line)
shorts-factory niche-batch --topics-file topics.txt --niche true_crime

# topic discovery (Gemini suggests N fresh ideas per niche)
shorts-factory topics --niche history --count 30 --out topics.txt

# fallback: hand-written script, no LLM
shorts-factory niche --script-file my_script.json
```

### Niches

`--niche` accepts: `true_crime`, `history`, `science`, `mysteries`,
`weird_facts`, `biographies`, `tech_history`, `space`. Each has its own
prompt template tuned for tone + factual accuracy; all generate the same
hook / 3-5 beats / payoff structure (90-130 words, 35-50 seconds).

## Local scheduler + YouTube uploader

For a hands-off "post 2 shorts/day" channel, run the scheduler **on your
own machine** (NOT this project's dev VM — uploads from datacenter IPs
get rate-limited and can flag your channel). The scheduler reads a TOML
config and fires off `generate -> upload` jobs at the configured slots.

### One-time setup

1. **Get a Gemini key** at <https://aistudio.google.com/apikey>. Free
   tier (1000 req/day on `gemini-2.5-flash-lite`) is enough for ~6
   shorts/day.
2. **Create a YouTube OAuth client** (uploads need OAuth2; an API key
   alone cannot upload):
   - Go to <https://console.cloud.google.com/apis/credentials>
   - **Create credentials -> OAuth client ID -> Desktop app**
   - Click **Download JSON** and save it locally to
     `~/.config/shorts-factory/client_secret.json`. **Do NOT paste it
     into chat or commit it.**
   - On the OAuth consent screen tab, add your YouTube Google account
     as a test user and ensure the `https://www.googleapis.com/auth/youtube.upload`
     scope is enabled.
3. **Initialise the schedule config**:
   ```bash
   shorts-factory schedule init                       # writes ~/.config/shorts-factory/schedule.toml
   $EDITOR ~/.config/shorts-factory/schedule.toml     # set slots, niche, timezone
   ```
4. **First-run consent** (opens a browser tab, caches refresh token to
   `~/.config/shorts-factory/youtube_token.json`):
   ```bash
   # generate one short and upload it as PRIVATE so you can review before going live
   shorts-factory schedule tick
   ```

### Schedule config (`~/.config/shorts-factory/schedule.toml`)

```toml
slots = ["09:30", "20:00"]   # 2 uploads/day, in your local timezone
timezone = "America/Los_Angeles"
queue_path = "~/shorts-factory/queue.txt"
out_root = "~/shorts-factory/out"
history_path = "~/.config/shorts-factory/history.json"
niche = "history"            # or any other niche name above
privacy_status = "public"    # public | unlisted | private
upload = true                # set false to generate only
auto_refill_topics = true    # auto-call Gemini topic discovery when queue empties
refill_count = 30
```

### Run the scheduler

```bash
# foreground (Ctrl-C to stop)
shorts-factory schedule run

# or as a systemd --user service
shorts-factory schedule systemd > ~/.config/systemd/user/shorts-factory.service
systemctl --user daemon-reload && systemctl --user enable --now shorts-factory
```

### One-off upload (no scheduler)

```bash
shorts-factory youtube-upload \
  --video out/niche_hindenburg-disaster/hindenburg-disaster.mp4 \
  --metadata out/niche_hindenburg-disaster/METADATA.txt \
  --privacy unlisted
```

Every upload sets `containsSyntheticMedia=true` (YouTube's mandatory AI
disclosure flag). Title is hard-capped at 100 chars, description at
5000, tags at 30. Quota: each upload costs ~1,600 of YouTube's free
10,000/day, so the **API hard-caps you at ~6 uploads/day** — which
matches the algorithmic sweet spot anyway (1-3/day, never more than
~6).

Each run produces:

```
out/niche_<slug>/
├── <slug>.mp4         # 1080×1920 H.264 short
├── METADATA.txt       # title, description, hashtags, image credits
├── _broll/            # downloaded source images
├── _narration/        # per-sentence WAVs + concatenated narration.wav
└── _assemble/         # captions.ass + intermediate visual.mp4
```

Voice model (one-time, ~120 MB):

```bash
mkdir -p work/voices
curl -sSL -o work/voices/en_US-ryan-high.onnx \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/high/en_US-ryan-high.onnx"
curl -sSL -o work/voices/en_US-ryan-high.onnx.json \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/high/en_US-ryan-high.onnx.json"
```

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

### archive.org sources (public domain)

```bash
shorts-factory search   --query "..." --limit 10
shorts-factory download --identifier night_of_the_living_dead
shorts-factory scenes   --video work/<file>.mp4
shorts-factory rank     --video work/<file>.mp4 --top 5
shorts-factory render   --video work/<file>.mp4 --start 1234.5 --duration 40
shorts-factory run      --query "..."          # full archive.org pipeline
```

### YouTube CC-BY sources (modern, color, modern resolution)

Requires a free YouTube Data API v3 key in `YOUTUBE_API_KEY`. Get one at
<https://console.cloud.google.com/apis/credentials> (enable
"YouTube Data API v3" first).

```bash
export YOUTUBE_API_KEY="AIza..."

# discover CC-BY videos
shorts-factory youtube-search   --query "tears of steel"  --min-duration 300
# fetch one video by ID (refuses non-CC videos)
shorts-factory youtube-download --id bjYbA1bWjeE
# end-to-end pipeline
shorts-factory youtube-run      --query "tears of steel" --max-clips 5
```

The YouTube path always emits `out/yt_<id>/ATTRIBUTION.txt` with the
required CC-BY credit line — paste it into the short's description before
publishing.

## Layout

```
src/shorts_factory/
    cli.py          # typer entrypoint, sub-commands
    config.py       # paths + tunables
    discovery.py    # step 1 - archive.org search
    download.py     # step 2 - archive.org metadata-API + best-mp4 picker
    youtube.py      # step 1+2 - YouTube Data API CC-BY search + yt-dlp download
    scenes.py       # step 3 - PySceneDetect
    transcribe.py   # step 4a - faster-whisper
    rank.py         # step 4b - heuristic scoring
    render.py       # step 5 - 9:16 reframe + captions (ffmpeg)
    pipeline.py     # orchestrator (archive.org source)
    yt_pipeline.py  # orchestrator (YouTube CC source)
    util.py
```

## Roadmap

- LLM-based highlight ranker (plug into `rank.py`)
- Face-tracking smart-crop (currently blurred-bars)
- Auto-upload to YouTube / TikTok / IG (intentionally not in v1)
- LangChain-style metadata generator (titles, descriptions, tags)
