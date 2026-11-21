import io
import os
import threading
import time
from datetime import datetime

import av
import cv2
import pandas as pd
import plotly.express as px
import streamlit as st

from database import create_table
from detector import CONFIDENCE_THRESHOLD, detect_frame, get_model_status
from logger import log_detections, log_violation

try:
    from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, WebRtcMode, webrtc_streamer
except ImportError:
    RTCConfiguration = None
    VideoProcessorBase = object
    WebRtcMode = None
    webrtc_streamer = None


DB_PATH = "data/violations.db"
SNAPSHOT_DIR = "snapshots"
_stats_lock = threading.Lock()
_shared_stats = {
    "connected": False,
    "frames": 0,
    "total_detections": 0,
    "safe_count": 0,
    "violation_count": 0,
    "latest_safe": 0,
    "latest_violation": 0,
    "average_confidence": 0.0,
    "threshold": float(CONFIDENCE_THRESHOLD),
    "last_update": None,
}


class HelmetVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_log = 0.0
        self.last_snapshot = 0.0

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        with _stats_lock:
            threshold = float(_shared_stats["threshold"])
        annotated, safe, violations, detections = detect_frame(
            image,
            confidence_threshold=threshold,
        )
        confidences = [item["confidence"] for item in detections]
        average_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        now = time.time()
        with _stats_lock:
            _shared_stats["connected"] = True
            _shared_stats["frames"] += 1
            _shared_stats["latest_safe"] = safe
            _shared_stats["latest_violation"] = violations
            _shared_stats["safe_count"] += safe
            _shared_stats["violation_count"] += violations
            _shared_stats["total_detections"] += len(detections)
            _shared_stats["average_confidence"] = average_confidence
            _shared_stats["last_update"] = datetime.now().strftime("%H:%M:%S")

        if detections and now - self.last_log >= 2.0:
            log_detections(detections, source="webcam")
            self.last_log = now

        if violations and now - self.last_snapshot >= 5.0:
            os.makedirs(SNAPSHOT_DIR, exist_ok=True)
            image_path = os.path.join(
                SNAPSHOT_DIR,
                f"violation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
            )
            cv2.imwrite(image_path, annotated)
            log_violation(
                _shared_stats["total_detections"],
                _shared_stats["safe_count"],
                _shared_stats["violation_count"],
                image_path,
                average_confidence,
            )
            self.last_snapshot = now

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


def get_stats():
    with _stats_lock:
        return dict(_shared_stats)


def load_summary():
    create_table()
    violations = pd.read_sql_query(
        "SELECT * FROM violations ORDER BY id DESC",
        sqlite_connection(),
    )
    detections = pd.read_sql_query(
        "SELECT * FROM detections ORDER BY id DESC",
        sqlite_connection(),
    )
    return violations, detections


def sqlite_connection():
    import sqlite3
    os.makedirs("data", exist_ok=True)
    return sqlite3.connect(DB_PATH)


st.set_page_config(
    page_title="Helmet Intelligence",
    page_icon="",
    layout="wide",
)

st.markdown(
    """
    <style>
    html, body, [class*="css"] { font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    .stApp { background: #f4f7fb; }
    .block-container { max-width: 1500px; padding-top: 1.6rem; }
    [data-testid="stSidebar"] { background: #111827; }
    [data-testid="stSidebar"] * { color: #dbe3ef !important; }
    .hero {
        background: linear-gradient(120deg, #0f172a 0%, #172554 58%, #1e3a8a 100%);
        border-radius: 18px;
        padding: 24px 28px;
        color: white;
        margin-bottom: 18px;
    }
    .hero h1 { margin: 0; font-size: 32px; letter-spacing: -0.8px; }
    .hero p { margin: 7px 0 0; color: #b9c7df; }
    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #e6ebf2;
        border-radius: 14px;
        padding: 16px;
        box-shadow: 0 5px 18px rgba(15,23,42,.05);
    }
    [data-testid="stMetricLabel"] { color: #64748b; }
    [data-testid="stMetricValue"] { color: #0f172a; }
    .status-pill {
        display: inline-block;
        border-radius: 999px;
        padding: 7px 12px;
        font-size: 13px;
        font-weight: 700;
        background: #dcfce7;
        color: #166534;
    }
    .status-danger {
        background: #fee2e2;
        color: #991b1b;
    }
    .panel {
        background: white;
        border: 1px solid #e6ebf2;
        border-radius: 14px;
        padding: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("## Helmet Intelligence")
st.sidebar.caption("YOLO helmet monitoring")
threshold = st.sidebar.slider(
    "Detection confidence",
    min_value=0.25,
    max_value=0.95,
    value=float(CONFIDENCE_THRESHOLD),
    step=0.05,
)
with _stats_lock:
    _shared_stats["threshold"] = float(threshold)
st.sidebar.markdown("---")
model_ready, model_detail = get_model_status()
if model_ready:
    st.sidebar.success(f"Model ready: {model_detail}")
else:
    st.sidebar.error(model_detail)

st.markdown(
    """
    <div class="hero">
      <h1>Helmet Detection Command Center</h1>
      <p>Real-time rider safety monitoring with confidence-based YOLO detection</p>
    </div>
    """,
    unsafe_allow_html=True,
)

stats = get_stats()
summary_status = "LIVE" if stats["connected"] else "WAITING"
status_class = "" if stats["connected"] else "status-danger"
st.markdown(
    f'<span class="status-pill {status_class}">Camera {summary_status}</span>',
    unsafe_allow_html=True,
)

with st.container():
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Helmet detections", stats["safe_count"])
    col2.metric("No-helmet violations", stats["violation_count"])
    col3.metric("Total detections", stats["total_detections"])
    col4.metric(
        "Average confidence",
        f"{stats['average_confidence']:.0%}",
    )

st.markdown("<br>", unsafe_allow_html=True)
left, right = st.columns([2.15, 1], gap="large")

with left:
    st.subheader("Live camera")
    if webrtc_streamer is None:
        st.error(
            "streamlit-webrtc is not installed. Install requirements and restart Streamlit."
        )
    elif not model_ready:
        st.error(
            f"Add a trained YOLO model at models/best.pt or set HELMET_MODEL_PATH. "
            f"{model_detail}"
        )
    else:
        rtc_configuration = None
        if RTCConfiguration is not None:
            rtc_configuration = RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            )
        webrtc_streamer(
            key="helmet-live-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_configuration,
            video_processor_factory=HelmetVideoProcessor,
            media_stream_constraints={
                "video": {"width": {"ideal": 1280}, "height": {"ideal": 720}},
                "audio": False,
            },
            async_processing=True,
        )

with right:
    st.subheader("Detection status")
    live_safe = stats["latest_safe"]
    live_violation = stats["latest_violation"]
    if live_violation:
        st.error(f"{live_violation} no-helmet detection(s)")
    elif live_safe:
        st.success(f"{live_safe} helmet detection(s)")
    else:
        st.info("Waiting for detections")

    st.caption(
        f"Last frame: {stats['last_update'] or 'No camera data'}  |  "
        f"Confidence threshold: {threshold:.0%}"
    )
    st.progress(min(stats["average_confidence"], 1.0))

violations_df, detections_df = load_summary()

st.markdown("---")
chart_col, log_col = st.columns([1.35, 1], gap="large")

with chart_col:
    st.subheader("Recent activity")
    if violations_df.empty:
        st.info("Violation history will appear here after the camera detects an unsafe rider.")
    else:
        chart_df = violations_df.head(30).copy()
        chart_df["time"] = chart_df["time"].fillna("")
        fig = px.bar(
            chart_df,
            x="time",
            y=["safe_count", "violation_count"],
            barmode="group",
            labels={"value": "Count", "time": "Time"},
            color_discrete_map={
                "safe_count": "#2ac782",
                "violation_count": "#2c60ff",
            },
        )
        fig.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend_title_text="",
        )
        st.plotly_chart(fig, use_container_width=True)

with log_col:
    st.subheader("Confidence distribution")
    if detections_df.empty:
        st.info("Detection confidence scores will be listed here.")
    else:
        fig = px.histogram(
            detections_df.head(300),
            x="confidence",
            nbins=20,
            color="status",
            color_discrete_map={
                "safe": "#2ac782",
                "violation": "#2c60ff",
            },
        )
        fig.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_tickformat=".0%",
        )
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
table_col, export_col = st.columns([3, 1])
with table_col:
    st.subheader("Detection log")
    if detections_df.empty:
        st.info("No recorded detections.")
    else:
        display_df = detections_df.head(100).copy()
        display_df["confidence"] = display_df["confidence"].map(
            lambda value: f"{float(value):.1%}"
        )
        st.dataframe(display_df, use_container_width=True, hide_index=True)

with export_col:
    st.subheader("Report")
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        violations_df.to_excel(writer, index=False, sheet_name="Violations")
        detections_df.to_excel(writer, index=False, sheet_name="Detections")
    st.download_button(
        "Download Excel report",
        data=buffer.getvalue(),
        file_name=f"helmet_report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
        mime="application/vnd.ms-excel",
        use_container_width=True,
    )

time.sleep(2)
st.rerun()
