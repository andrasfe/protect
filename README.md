# protect

Black out sensitive personal information — names, ID numbers, dates and the
like — that appears on a screen or other flat surface in a video, even when the
camera is handheld and the surface moves around in the frame.

`protect` tracks the surface in every frame, blacks out the regions you mark,
removes audio and container metadata, and then checks its own output. When it
cannot locate the surface with confidence it **fails closed**: it hides more, up
to the whole frame, never less.

Everything runs locally. No frames, text or metadata leave your machine.

## How it works

1. **Reference.** You pick one sharp frame of the video and mark two kinds of
   rectangles on it:
   * `track` regions — static, textured parts of the surface (labels, logos,
     borders) that look the same throughout the video;
   * `redact` regions — what must be hidden.
2. **Tracking.** For every frame, SIFT features from the `track` regions are
   matched and a homography (the perspective mapping between the reference and
   the frame) is fitted with RANSAC. It is then refined to sub-pixel accuracy
   with ECC image alignment around the `redact` regions; the alignment
   correlation becomes the frame's confidence score. Frames are independent, so
   tracking runs on all CPU cores.
3. **Planning.** Each frame gets one status:

   | status     | meaning                                                         | output                              |
   |------------|-----------------------------------------------------------------|-------------------------------------|
   | `ok`       | tracked with confidence                                         | regions blacked out                 |
   | `filled`   | not trusted alone, but within `max_gap` frames of tracked ones  | larger regions covering the possible positions |
   | `kept`     | untracked, inside a `keep` range you reviewed                   | left as is                          |
   | `lost`     | untracked and not reviewed                                      | **whole frame black**               |
   | `blackout` | inside a `blackout` range                                       | whole frame black                   |

   Tracked frames that jump away from their neighbours are distrusted. Frames
   where the regions are confidently outside the picture count as tracked.
4. **Rendering.** Regions are filled with solid black (not blurred, since blurred
   text can sometimes be recovered). The video is re-encoded from decoded frames
   only. Audio is dropped unless you opt in, and titles, dates, locations and
   chapters are not copied.
5. **Verification.**
   * Every frame: the planned regions must be black in the output, and planned
     black frames must be black.
   * Sampled frames: OCR (tesseract) reads the area around the regions in the
     original and in the output and looks for words you list. Hits in the
     original show how well OCR can read this footage; any hit in the output fails
     the check.
   * Contact sheets of the output around the regions, for you to look at.

## Install

Requirements: Python 3.10+, [ffmpeg](https://ffmpeg.org/) on `PATH`, and
optionally [tesseract](https://github.com/tesseract-ocr/tesseract) for the OCR
check.

```bash
git clone <this repository> && cd protect
python3 -m venv .venv && . .venv/bin/activate
pip install -e .            # add [dev] for the test suite
```

## Usage

Keep job files, reference frames, word lists and outputs **outside** this
repository (for example next to the videos). They are derived from the footage.

```bash
# 1. create a job from a sharp frame that shows everything you need to hide
protect init /videos/clip.mp4 --at 12.5          # writes clip.job.json + clip.job.ref.png

# 2. mark regions (opens a window; or edit the JSON by hand)
protect select /videos/clip.job.json track
protect select /videos/clip.job.json redact
protect preview /videos/clip.job.json            # check the boxes on the reference

# 3. locate the surface in every frame and see what will happen
protect track /videos/clip.job.json
protect plan /videos/clip.job.json --sheets      # contact sheets of untracked frames

# 4. optionally list reviewed ranges under "keep", then render and verify
protect render /videos/clip.job.json
protect verify /videos/clip.job.json --terms /videos/clip.terms.txt
```

`protect run JOB --terms FILE` does track (if needed), render and verify in one
go. `verify` exits with status 2 if any check fails.

The terms file is plain text, one word per line, `#` for comments. Use the
words that must not be readable (for example the name and ID number shown on the
screen). Reports only refer to them as `term1`, `term2`, …

### Untracked frames and `keep` ranges

Frames where the surface cannot be found are blacked out entirely. That's the
right default but it also blacks out unrelated footage, for example when the
camera turns away from the screen. Run `protect plan JOB --sheets`, look at the
contact sheets of untracked frames, and add a time range to `keep` only where
nothing sensitive is visible. Leave transitions (where the surface is still
partly in view) out of the range.

## Job file

```json
{
  "input": "clip.mp4",
  "output": "clip.redacted.mp4",
  "reference": "clip.job.ref.png",
  "track":  [[100, 40, 900, 90], [700, 600, 120, 80]],
  "redact": [{"label": "line", "box": [180, 60, 300, 40]}],
  "padding": 2,
  "search": null,
  "refine": null,
  "keep": [[12.0, 30.5]],
  "blackout": [],
  "audio": false,
  "options": {}
}
```

* Boxes are `[x, y, width, height]` in reference-image pixels; paths are
  relative to the job file. Redact boxes may extend past the reference image.
* `padding` grows every redact box (reference pixels).
* `search` limits tracking to part of the frame, for example to skip black
  bars. Pixels outside it are never treated as picture content.
* `refine` (optional `[x, y, w, h]`) is the area used for sub-pixel alignment
  and the confidence score. By default it is the redact regions plus a margin.
  Set it when that margin would catch things that aren't flat on the surface,
  such as a raised frame or bezel, or content that changes during the video.
* `keep` / `blackout` are `[start, end]` in seconds.
* `audio: true` copies the original audio track.
* `options` overrides tuning defaults (see `src/protect/job.py`): matching
  thresholds, `min_score`, `max_gap`, padding of bridged frames, encoder `crf`
  and `preset`, and so on.

Changing the video, reference image, regions or tracking options invalidates
stored tracking data. `render` refuses to use stale data, and `run` re-tracks
automatically.

## Limitations

* The surface must be flat (a screen, a sign, a sheet of paper) and the hidden
  regions must stay at the same place on it. Content that scrolls or moves
  across the surface is not followed.
* The reference frame needs enough static texture in the `track` regions. A
  thin band of text works. A uniform panel doesn't.
* OCR verification is a backstop, not proof. On small or blurry text it may not
  be able to read the original either, and the report says so. Look at the
  contact sheets.
* `keep` ranges are human decisions. Review them carefully.

## Development

```bash
pip install -e '.[dev]'
pytest
```

The tests build synthetic footage (a textured panel moving under known
homographies), so no real video is needed. The end-to-end tests need ffmpeg.

## License

MIT. See [LICENSE](LICENSE).
