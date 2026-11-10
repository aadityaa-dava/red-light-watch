"""RedLightWatch dashboard (step 1): run the detector on a clip and review the results."""
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent       # repo root (detect_violations.py expects to run here)
sys.path.insert(0, str(ROOT / "backend"))
from config import CLIPS  # noqa: E402

OUT = ROOT / "outputs"
ENGINE = ROOT / "yolov8n_960.engine"
HIGH_RISK_THRESHOLD = 3

st.set_page_config(page_title="RedLightWatch", page_icon="🚦", layout="wide")
st.title("🚦 RedLightWatch")
st.caption("Edge red-light violation detector: YOLOv8n + ByteTrack, stop-line crossing and traffic-light colour.")


def run_detector(clip, use_engine, write_video):
    env = dict(os.environ)
    if use_engine:
        env["RLW_MODEL"], env["RLW_IMGSZ"] = str(ENGINE), "960"
    if not write_video:
        env["RLW_NOVIDEO"] = "1"
    proc = subprocess.run([sys.executable, "backend/detect_violations.py", clip],
                          cwd=ROOT, env=env, capture_output=True, text=True)
    m = re.search(r"Processing speed: ([\d.]+) FPS", proc.stdout)
    return proc.returncode, (float(m.group(1)) if m else None), proc.stderr[-1500:]


def web_video(clip):
    """Browsers cannot play OpenCV's mp4v video, so convert a copy to H.264 once."""
    src, dst = OUT / f"{clip}_violations.mp4", OUT / f"{clip}_web.mp4"
    if not src.exists():
        return None
    if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-preset", "veryfast", str(dst)])
        if r.returncode != 0:
            return None
    return dst


# --- sidebar -----------------------------------------------------------------
with st.sidebar:
    st.header("Run")
    clip = st.selectbox("Clip", list(CLIPS.keys()))
    use_engine = st.checkbox("Use TensorRT engine (faster)", value=ENGINE.exists(), disabled=not ENGINE.exists())
    write_video = st.checkbox("Write annotated video", value=True)
    go = st.button("Run detection", type="primary", width="stretch")

if go:
    with st.spinner(f"Processing {clip} ..."):
        code, fps, err = run_detector(clip, use_engine, write_video)
    if code != 0:
        st.error("The detector failed.")
        st.code(err)
    else:
        st.session_state["speed"] = {clip: fps}
        st.session_state["last_clip"] = clip

csv_path = OUT / f"{clip}_events.csv"
if not csv_path.exists():
    st.info("No results for this clip yet. Press **Run detection** in the sidebar.")
    st.stop()

events = pd.read_csv(csv_path)
n = events["event"].value_counts() if len(events) else pd.Series(dtype=int)
count = lambda k: int(n.get(k, 0))  # noqa: E731

if count("VIOLATION") >= HIGH_RISK_THRESHOLD:
    st.error("INTERSECTION FLAGGED: HIGH-RISK")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Violations", count("VIOLATION"))
c2.metric("Late entries", count("LATE_ENTRY"))
c3.metric("Needs review", count("NEEDS_REVIEW"))
c4.metric("Legal crossings", count("LEGAL"))
speed = st.session_state.get("speed", {}).get(clip)
c5.metric("Speed", f"{speed:.1f} FPS" if speed else "n/a")

left, right = st.columns([3, 2])
with left:
    st.subheader("Annotated video")
    video = web_video(clip)
    if video:
        st.video(str(video))
    else:
        st.caption("No annotated video for this run (video writing was off).")
with right:
    st.subheader("Events")
    if len(events):
        st.dataframe(events.drop(columns=["snapshot"]), hide_index=True, width="stretch")
    else:
        st.caption("No vehicle crossed the stop line.")
    st.download_button("Download events CSV", csv_path.read_bytes(), file_name=csv_path.name,
                       mime="text/csv", width="stretch")

flagged = events[events["event"] != "LEGAL"] if len(events) else events
if len(flagged):
    st.subheader("Snapshots")
    cols = st.columns(min(len(flagged), 3))
    for col, (_, ev) in zip(cols * ((len(flagged) // 3) + 1), flagged.iterrows()):
        if isinstance(ev["snapshot"], str) and (ROOT / ev["snapshot"]).exists():
            col.image(str(ROOT / ev["snapshot"]), caption=f"{ev['event']} · car #{ev['track_id']} · {ev['time_s']}s · light {ev['light']}")