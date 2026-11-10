"""RedLightWatch dashboard: run the detector on a sample clip or an uploaded video and review the results."""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except ImportError:           # without the package the user types the coordinates instead of clicking
    streamlit_image_coordinates = None

ROOT = Path(__file__).resolve().parent.parent       # repo root (detect_violations.py expects to run here)
sys.path.insert(0, str(ROOT / "backend"))
from config import CLIPS  # noqa: E402

OUT = ROOT / "outputs"
CLIPS_DIR = ROOT / "data" / "clips"
CUSTOM = "custom"          # name used for an uploaded video
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


def first_frame(path, t):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(int(t * fps), max(n - 1, 0)))
    ok, frame = cap.read()
    cap.release()
    return (frame if ok else None), fps, n


def draw_setup(frame, line, light, points=()):
    img = frame.copy()
    th = max(2, frame.shape[1] // 400)
    fs = max(0.7, frame.shape[1] / 1600)
    cv2.line(img, line[0], line[1], (255, 255, 0), th)
    cv2.polylines(img, [np.array(light, np.int32)], True, (0, 215, 255), th)
    cv2.putText(img, "stop line", (line[0][0], max(line[0][1] - 12, 20)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 0), th)
    cv2.putText(img, "traffic light", (light[0][0], max(light[0][1] - 12, 20)), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 215, 255), th)
    for p in points:                                   # the points just clicked
        cv2.circle(img, p, max(6, frame.shape[1] // 160), (0, 0, 255), -1)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# --- sidebar -----------------------------------------------------------------
with st.sidebar:
    st.header("Run")
    source = st.radio("Source", ["Sample clips", "Upload video"], horizontal=True)
    if source == "Sample clips":
        clip = st.selectbox("Clip", list(CLIPS.keys()))
        upload = None
    else:
        clip = CUSTOM
        upload = st.file_uploader("Video file", type=["mp4", "mov", "avi", "mkv"])
    use_engine = st.checkbox("Use TensorRT engine (faster)", value=ENGINE.exists(), disabled=not ENGINE.exists())
    write_video = st.checkbox("Write annotated video", value=True)
    go = st.button("Run detection", type="primary", width="stretch")

setup_ok = True
if source == "Upload video":
    if upload is None:
        st.info("Upload a video in the sidebar, then mark the stop line and the traffic light.")
        st.stop()
    vid_id = f"{upload.name}_{upload.size}"
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    if st.session_state.get("vid_id") != vid_id:           # a new upload: save it, drop old results
        (CLIPS_DIR / f"{CUSTOM}.mp4").write_bytes(upload.getbuffer())
        for f in OUT.glob(f"{CUSTOM}_*"):
            f.unlink()
        st.session_state["vid_id"] = vid_id
    video_path = CLIPS_DIR / f"{CUSTOM}.mp4"

    st.subheader("1. Mark the stop line and the traffic light")
    probe, vfps, nframes = first_frame(video_path, 0)
    if probe is None:
        st.error("Could not read that video.")
        st.stop()
    H, W = probe.shape[:2]
    duration = max(nframes / vfps - 0.1, 0.1)
    t_show = st.slider("Preview time (s): pick a moment when the light is visible", 0.0, float(duration), 0.0, 0.1)
    frame, _, _ = first_frame(video_path, t_show)

    k = vid_id
    names = ("lx1", "ly1", "lx2", "ly2", "tx1", "ty1", "tx2", "ty2")
    defaults = (int(W * 0.2), int(H * 0.7), int(W * 0.8), int(H * 0.7),
                int(W * 0.45), int(H * 0.05), int(W * 0.55), int(H * 0.15))
    key = {n: f"{k}_{n}" for n in names}
    for n, d in zip(names, defaults):
        st.session_state.setdefault(key[n], d)
    val = {n: int(st.session_state[key[n]]) for n in names}

    target = st.radio("Click on the picture to mark:", ["Stop line", "Traffic light"], horizontal=True, key=f"{k}_target")
    pts = st.session_state.setdefault(f"{k}_pts", {"Stop line": [], "Traffic light": []})
    done = len(pts[target]) % 2 == 0 and len(pts[target]) > 0
    if target == "Stop line":
        hint = "Marked. Click again to redo it." if done else ("Now click the OTHER END of the stop line." if pts[target] else "Click one END of the stop line (where cars must stop).")
    else:
        hint = "Marked. Click again to redo it." if done else ("Now click the OPPOSITE corner of the light." if pts[target] else "Click one CORNER of a box around the traffic light lamps.")
    st.info(hint)

    line = [(val["lx1"], val["ly1"]), (val["lx2"], val["ly2"])]
    light_poly = [(val["tx1"], val["ty1"]), (val["tx2"], val["ty1"]), (val["tx2"], val["ty2"]), (val["tx1"], val["ty2"])]
    marks = [tuple(p) for lst in pts.values() for p in lst]
    preview = draw_setup(frame, line, light_poly, marks)

    if streamlit_image_coordinates is not None:
        click = streamlit_image_coordinates(preview, width=min(W, 1100), key=f"{k}_img")
        if click and click.get("unix_time") != st.session_state.get(f"{k}_last"):
            st.session_state[f"{k}_last"] = click["unix_time"]
            scale = W / click["width"]                       # the picture is shown smaller than the video
            px = min(max(int(round(click["x"] * scale)), 0), W)
            py = min(max(int(round(click["y"] * scale)), 0), H)
            lst = pts[target]
            if len(lst) >= 2:
                lst.clear()
            lst.append((px, py))
            if target == "Stop line":
                st.session_state[key["lx1" if len(lst) == 1 else "lx2"]] = px
                st.session_state[key["ly1" if len(lst) == 1 else "ly2"]] = py
            elif len(lst) == 2:
                st.session_state[key["tx1"]], st.session_state[key["tx2"]] = sorted((lst[0][0], lst[1][0]))
                st.session_state[key["ty1"]], st.session_state[key["ty2"]] = sorted((lst[0][1], lst[1][1]))
            st.rerun()
    else:
        st.image(preview, width="stretch")
        st.warning("Clicking needs the package streamlit-image-coordinates (pip install streamlit-image-coordinates). Type the numbers below instead.")
    st.caption(f"Preview at {t_show:.1f}s ({W}x{H}). Use the slider above to find a frame where the light is visible. "
               "Cars should drive from the bottom of the picture toward the top, across the line.")

    with st.expander("Coordinates (fine-tune by typing)"):
        a, b = st.columns(2)
        with a:
            st.markdown("**Stop line**")
            l1, l2, l3, l4 = st.columns(4)
            lx1 = l1.number_input("x1", min_value=0, max_value=W, key=key["lx1"])
            ly1 = l2.number_input("y1", min_value=0, max_value=H, key=key["ly1"])
            lx2 = l3.number_input("x2", min_value=0, max_value=W, key=key["lx2"])
            ly2 = l4.number_input("y2", min_value=0, max_value=H, key=key["ly2"])
        with b:
            st.markdown("**Traffic light**")
            t1, t2, t3, t4 = st.columns(4)
            tx1 = t1.number_input("left", min_value=0, max_value=W, key=key["tx1"])
            ty1 = t2.number_input("top", min_value=0, max_value=H, key=key["ty1"])
            tx2 = t3.number_input("right", min_value=0, max_value=W, key=key["tx2"])
            ty2 = t4.number_input("bottom", min_value=0, max_value=H, key=key["ty2"])
    line = [(int(lx1), int(ly1)), (int(lx2), int(ly2))]
    light_poly = [(int(tx1), int(ty1)), (int(tx2), int(ty1)), (int(tx2), int(ty2)), (int(tx1), int(ty2))]
    setup_ok = (lx1, ly1) != (lx2, ly2) and tx1 < tx2 and ty1 < ty2
    if not setup_ok:
        st.warning("The stop line needs two different points, and the light box needs right > left and bottom > top.")
    st.divider()

if go and not setup_ok:
    st.stop()
if go:
    if source == "Upload video":
        (CLIPS_DIR / f"{CUSTOM}.json").write_text(json.dumps({"line": line, "light": light_poly}))
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