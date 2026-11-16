"""Build the visuals used in the README. Run from the repo root after the final run of all clips:

    python3 backend/make_readme_assets.py

Writes docs/gifs/<clip>.gif, docs/images/light_timeline.png, speed.png, pipeline.png and violations_gallery.jpg.
"""
import csv
import re
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402
from PIL import Image  # noqa: E402

from config import CLIPS  # noqa: E402
from light import LightReader  # noqa: E402

GIF_DIR, IMG_DIR = Path("docs/gifs"), Path("docs/images")
GIF_FPS = 10                  # frames per second in the GIFs
GIF_MAX_MB = 2.0              # shrink a GIF until it is below this size
HERO = "clip1"                # the clip shown large at the top of the README
STATE_COLORS = {"red": "#d62728", "yellow": "#f2c40f", "green": "#2ca02c", "unknown": "#bdbdbd"}


def clip_names():
    names = [c for c in CLIPS if Path(f"data/clips/{c}.mp4").exists()]
    return sorted(names, key=lambda c: int(re.sub(r"\D", "", c) or 0))


def read_events(clip):
    for folder in ("results", "outputs"):
        p = Path(folder) / f"{clip}_events.csv"
        if p.exists():
            with open(p) as f:
                return [{"time_s": float(r["time_s"]), "event": r["event"], "track_id": r["track_id"]}
                        for r in csv.DictReader(f)]
    return []


# --- 1. GIFs of the annotated videos ------------------------------------------------------------
def make_gif(clip, events):
    src = Path("outputs") / f"{clip}_violations.mp4"
    if not src.exists():
        print(f"  {clip}: no {src}, skipped (run detect_violations.py with the video on first)")
        return
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    flagged = [e for e in events if e["event"] != "LEGAL"]
    if flagged:                                   # a few seconds around the first violation
        t0 = flagged[0]["time_s"]
        start, end = max(t0 - 3.0, 0.0), t0 + 2.5
    else:                                         # no violation: show the start of the clip
        start, end = 0.0, 6.0
    step = max(int(round(fps / GIF_FPS)), 1)
    first, last = int(start * fps), min(int(end * fps), n)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    raw = []
    for i in range(first, last):
        ok, frame = cap.read()
        if not ok:
            break
        if (i - first) % step == 0:
            raw.append(frame)
    cap.release()
    if not raw:
        print(f"  {clip}: no frames in the chosen window, skipped")
        return
    dst = GIF_DIR / f"{clip}.gif"
    for width in ((720, 640, 560, 480) if clip == HERO else (480, 400, 320, 260)):
        small = [cv2.cvtColor(cv2.resize(f, (width, int(f.shape[0] * width / f.shape[1])), interpolation=cv2.INTER_AREA),
                              cv2.COLOR_BGR2RGB) for f in raw]
        pil = [Image.fromarray(s).convert("P", palette=Image.Palette.ADAPTIVE, colors=128, dither=Image.Dither.NONE)
               for s in small]
        pil[0].save(dst, save_all=True, append_images=pil[1:], duration=int(1000 / (fps / step)),
                    loop=0, optimize=True, disposal=2)
        if dst.stat().st_size <= GIF_MAX_MB * 1e6:
            break
    print(f"  {clip}: {dst} ({dst.stat().st_size / 1e6:.1f} MB, {len(raw)} frames, {width}px wide)")


# --- 2. traffic-light timeline with the crossings ----------------------------------------------------
def light_states(clip):
    cap = cv2.VideoCapture(f"data/clips/{clip}.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reader = LightReader(CLIPS[clip]["light"], w, h)
    states, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        states.append(reader.update(frame, i / fps)[0])
        i += 1
    cap.release()
    return states, fps


def plot_light_timeline(names):
    fig, ax = plt.subplots(figsize=(11, 0.6 * len(names) + 2.2))
    markers = {"VIOLATION": ("X", "black", "white", "violation"), "LATE_ENTRY": ("D", "#ff9800", "black", "late entry"),
               "NEEDS_REVIEW": ("s", "#9c27b0", "white", "needs review"), "LEGAL": ("o", "white", "black", "legal crossing")}
    for row, clip in enumerate(names):
        states, fps = light_states(clip)
        i = 0
        while i < len(states):                       # draw each run of the same colour as one bar
            j = i
            while j < len(states) and states[j] == states[i]:
                j += 1
            ax.barh(row, (j - i) / fps, left=i / fps, height=0.62, color=STATE_COLORS.get(states[i], "#bdbdbd"), linewidth=0)
            i = j
        for e in read_events(clip):
            m, face, edge, _ = markers.get(e["event"], markers["NEEDS_REVIEW"])
            ax.plot(e["time_s"], row, marker=m, markersize=10, markerfacecolor=face, markeredgecolor=edge,
                    markeredgewidth=1.4, linestyle="none", zorder=3)
        print(f"  {clip}: light timeline done ({len(states)} frames)")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("time in the clip (s)")
    ax.set_title("Traffic-light colour read by the detector, and each car crossing the stop line", fontsize=12)
    handles = [Patch(color=STATE_COLORS[k], label=f"light {k}") for k in ("red", "yellow", "green", "unknown")]
    handles += [Line2D([], [], marker=m, markersize=9, markerfacecolor=f, markeredgecolor=ed, linestyle="none", label=lab)
                for m, f, ed, lab in markers.values()]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4, frameon=False, fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "light_timeline.png", dpi=160)
    plt.close(fig)


# --- 3. speed -----------------------------------------------------------------------------------------
def parse_speeds(path):
    speeds, clip = {}, None
    p = Path(path)
    if not p.exists():
        return speeds
    for line in p.read_text().splitlines():
        m = re.match(r"==\s*(clip\d+)", line)
        if m:
            clip = m.group(1)
            continue
        m = re.search(r"Processing speed:\s*([\d.]+) FPS", line)
        if m and clip:
            speeds[clip] = float(m.group(1))
    return speeds


def plot_speed(names):
    series = [("1280 px, video written in the main loop", "outputs/run_all_engine2.log", "#c9d6e8"),
              ("1280 px, video written in a background thread", "outputs/run_all_threaded.log", "#7fa3d1"),
              ("960 px (final)", "results/run_all_final.log", "#1f5fa8")]
    series = [(lab, parse_speeds(p), c) for lab, p, c in series]
    series = [s for s in series if s[1]]
    if not series:
        print("  no speed logs found, speed plot skipped")
        return
    fig, ax = plt.subplots(figsize=(11, 4.6))
    w = 0.8 / len(series)
    x = np.arange(len(names))
    for k, (lab, sp, col) in enumerate(series):
        vals = [sp.get(c, np.nan) for c in names]
        bars = ax.bar(x + (k - (len(series) - 1) / 2) * w, vals, w, label=lab, color=col)
        if k == len(series) - 1:
            for b, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(b.get_x() + b.get_width() / 2, v + 0.2, f"{v:.1f}", ha="center", fontsize=8)
    ax.axhline(15, color="#d62728", linestyle="--", linewidth=1.2)
    ax.text(len(names) - 0.5, 15.2, "15 FPS target", color="#d62728", ha="right", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("frames per second")
    ax.set_ylim(0, 17)
    ax.set_title("Processing speed on the Jetson Orin Nano (1080p clips, annotated video written)", fontsize=12)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=9, ncol=3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "speed.png", dpi=160)
    plt.close(fig)


# --- 4. pipeline diagram ------------------------------------------------------------------------------
def draw_pipeline():
    fig, ax = plt.subplots(figsize=(15, 4.4))
    ax.set_xlim(0, 15.0)
    ax.set_ylim(0, 4.4)
    ax.axis("off")
    bw, bh = 2.05, 1.15
    col = [0.2 + i * 2.5 for i in range(6)]
    top, mid, bot = 3.0, 2.2, 1.0
    blue, amber, green, gray = "#dbe8f7", "#fdeec2", "#d6efd8", "#e8e8e8"
    boxes = {
        "frame": (col[0], mid, "Video frame\n(1080p)", gray),
        "yolo": (col[1], top, "YOLOv8n (TensorRT FP16)\ndetects cars, buses,\ntrucks, motorcycles", blue),
        "track": (col[2], top, "ByteTrack\none ID per vehicle", blue),
        "cross": (col[3], top, "Stop-line check\nrear of the car crosses\nthe line, moving ahead", blue),
        "lread": (col[1], bot, "Light reader\nlamp blobs in HSV,\n5-frame vote", amber),
        "lstate": (col[2], bot, "Light state\nred / yellow / green\n+ seconds since red", amber),
        "verdict": (col[4], mid, "Verdict\nLEGAL, LATE_ENTRY,\nVIOLATION,\nNEEDS_REVIEW", green),
        "out": (col[5], mid, "Annotated video\nevents CSV\nsnapshots, dashboard", gray),
    }
    for x, y, text, color in boxes.values():
        ax.add_patch(FancyBboxPatch((x, y - bh / 2), bw, bh, boxstyle="round,pad=0.03,rounding_size=0.12",
                                    facecolor=color, edgecolor="#555555", linewidth=1.1))
        ax.text(x + bw / 2, y, text, ha="center", va="center", fontsize=8.6)

    def arrow(a, b, side_a="r", side_b="l"):
        xa, ya, _, _ = boxes[a]
        xb, yb, _, _ = boxes[b]
        pa = (xa + bw if side_a == "r" else xa, ya)
        pb = (xb if side_b == "l" else xb + bw, yb)
        ax.annotate("", xy=pb, xytext=pa, arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.4, shrinkA=2, shrinkB=2))

    for a, b in [("frame", "yolo"), ("frame", "lread"), ("yolo", "track"), ("track", "cross"), ("lread", "lstate"),
                 ("cross", "verdict"), ("lstate", "verdict"), ("verdict", "out")]:
        arrow(a, b)
    ax.text(col[1] + bw / 2, top + bh / 2 + 0.2, "detect + track", ha="center", fontsize=9, color="#2a5a99")
    ax.text(col[1] + bw / 2, bot - bh / 2 - 0.2, "read the light", ha="center", fontsize=9, color="#9a6a00")
    fig.savefig(IMG_DIR / "pipeline.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# --- 5. gallery of the violation snapshots --------------------------------------------------------------
def make_gallery():
    files = sorted(Path("results/snapshots").glob("clip*.jpg"),
                   key=lambda p: int(re.match(r"clip(\d+)", p.name).group(1)))
    if not files:
        print("  no snapshots in results/snapshots, gallery skipped")
        return
    cols = 3
    rows = (len(files) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 2.9 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, f in zip(axes, files):
        m = re.match(r"(clip\d+)_id(\d+)_(\w+)\.jpg", f.name)
        ax.imshow(Image.open(f))
        ax.set_title(f"{m.group(1)}: car #{m.group(2)} {m.group(3).replace('_', ' ').lower()}" if m else f.stem, fontsize=11)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "violations_gallery.jpg", dpi=110, pil_kwargs={"quality": 85})
    plt.close(fig)


if __name__ == "__main__":
    GIF_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    names = clip_names()
    print("GIFs")
    for c in names:
        make_gif(c, read_events(c))
    print("Light timeline")
    plot_light_timeline(names)
    print("Speed plot")
    plot_speed(names)
    print("Pipeline diagram")
    draw_pipeline()
    print("Violation gallery")
    make_gallery()
    print("Done: docs/gifs and docs/images")