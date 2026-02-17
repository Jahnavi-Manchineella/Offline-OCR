# streamlit_app.py

import tempfile
import time

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import easyocr

# ─────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────
TROCR_MODEL_PATH  = "trocr_model"
EASYOCR_LANGS     = ['en']
CONF_THRESHOLD    = 0.35
IOU_THRESHOLD     = 0.40
MULTI_SCALES      = [0.75, 1.0, 1.5, 2.0]

# ─────────────────────────────────────────────
#  Page Setup
# ─────────────────────────────────────────────
st.set_page_config(
    page_title  = "Advanced OCR System",
    page_icon   = "🔬",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

# ─────────────────────────────────────────────
#  Styling
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp { background: #0a0a0f; color: #c9c9d4; }

section[data-testid="stSidebar"] {
    background: #0e0e18;
    border-right: 1px solid #1e1e32;
}

/* ── Header ── */
.ocr-header {
    background: linear-gradient(120deg, #0e0e18 0%, #111128 100%);
    border: 1px solid #2a2a50;
    border-top: 3px solid #6c63ff;
    border-radius: 0 0 14px 14px;
    padding: 1.8rem 2.2rem 1.6rem;
    margin-bottom: 1.8rem;
    display: flex;
    align-items: center;
    gap: 1.2rem;
}
.ocr-header-icon { font-size: 2.4rem; line-height: 1; }
.ocr-header h1 {
    font-family: 'Space Mono', monospace;
    font-size: 1.55rem;
    color: #e8e8ff;
    margin: 0;
    letter-spacing: -0.5px;
}
.ocr-header p { color: #5a5a8a; font-size: 0.82rem; margin: 0.3rem 0 0; }

/* ── Stat pills ── */
.pill-row { display: flex; gap: 0.6rem; flex-wrap: wrap; margin-bottom: 1.4rem; }
.pill {
    padding: 4px 12px;
    border-radius: 20px;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.3px;
}
.pill-gpu  { background:#12122a; color:#6c63ff; border:1px solid #6c63ff44; }
.pill-easy { background:#0e2218; color:#39d98a; border:1px solid #39d98a44; }
.pill-trocr{ background:#1a1210; color:#ff8c5a; border:1px solid #ff8c5a44; }
.pill-offline { background:#1a1a1a; color:#888; border:1px solid #333; }

/* ── Panels ── */
.panel {
    background: #0e0e18;
    border: 1px solid #1e1e32;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
.panel-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #5a5a8a;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 0.8rem;
}

/* ── Result JSON box ── */
.json-outer {
    background: #070710;
    border: 1px solid #1e1e32;
    border-left: 3px solid #6c63ff;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #9898c8;
    max-height: 420px;
    overflow-y: auto;
    white-space: pre-wrap;
    line-height: 1.75;
}

/* ── Metric cards ── */
.metrics-row { display: flex; gap: 0.8rem; margin-bottom: 1rem; }
.mc {
    flex: 1;
    background: #0e0e18;
    border: 1px solid #1e1e32;
    border-radius: 8px;
    padding: 0.9rem;
    text-align: center;
}
.mc-val {
    font-family: 'Space Mono', monospace;
    font-size: 1.45rem;
    color: #6c63ff;
    font-weight: 700;
}
.mc-lbl { font-size: 0.7rem; color: #44446a; margin-top: 3px; text-transform: uppercase; letter-spacing: 0.8px; }

/* ── Video label ── */
.live-badge {
    display: inline-block;
    background: #2a0000;
    color: #ff4444;
    border: 1px solid #ff444444;
    border-radius: 4px;
    padding: 2px 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 1px;
    margin-bottom: 0.6rem;
}

/* ── Buttons ── */
.stButton > button {
    background: #6c63ff18;
    color: #6c63ff;
    border: 1px solid #6c63ff55;
    border-radius: 7px;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: #6c63ff28;
    border-color: #6c63ff;
    color: #a09bff;
}

hr { border-color: #1e1e32 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────
st.markdown("""
<div class="ocr-header">
    <div class="ocr-header-icon">🔬</div>
    <div>
        <h1>Advanced OCR System</h1>
        <p>Multi-scale detection · EasyOCR + TrOCR consensus · Image &amp; Video · Fully offline</p>
    </div>
</div>
<div class="pill-row">
    <span class="pill pill-easy">EasyOCR</span>
    <span class="pill pill-trocr">TrOCR · trocr-base-printed</span>
    <span class="pill pill-offline">⛔ No Internet Required</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Device
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with st.sidebar:
    st.markdown("### ⚙️ System")
    device_label = "🟢 CUDA GPU" if torch.cuda.is_available() else "🔵 CPU"
    st.markdown(f'<span class="pill pill-gpu">{device_label}</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🎚 Detection Settings")
    conf_thresh = st.slider("Min Confidence", 0.10, 0.90, CONF_THRESHOLD, 0.05)
    iou_thresh  = st.slider("IoU Fusion Threshold", 0.10, 0.90, IOU_THRESHOLD, 0.05)

    st.markdown("---")
    st.markdown("### 📐 Scales Used")
    st.caption(f"`{MULTI_SCALES}`")
    st.caption("Multi-scale detection helps catch small and large text simultaneously.")

    st.markdown("---")
    st.markdown("### ℹ️ Pipeline")
    st.caption("1. Multi-enhance (CLAHE + sharpen)\n2. Multi-scale EasyOCR\n3. IoU box fusion\n4. TrOCR per crop\n5. Consensus merge\n6. Line reconstruction")

# ─────────────────────────────────────────────
#  Model Loading
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    reader = easyocr.Reader(EASYOCR_LANGS, gpu=torch.cuda.is_available())

    processor = TrOCRProcessor.from_pretrained(
        TROCR_MODEL_PATH,
        local_files_only=True
    )
    model = VisionEncoderDecoderModel.from_pretrained(
        TROCR_MODEL_PATH,
        local_files_only=True
    )
    model.to(device)
    model.eval()

    return reader, processor, model


with st.spinner("Loading OCR models from local storage..."):
    reader, processor, model = load_models()

# ─────────────────────────────────────────────
#  Image Enhancement
# ─────────────────────────────────────────────
def multi_enhance(image: np.ndarray) -> list[np.ndarray]:
    versions = [image]

    # CLAHE on luminance
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(3.0, (8, 8))
    cl    = clahe.apply(gray)
    versions.append(cv2.cvtColor(cl, cv2.COLOR_GRAY2BGR))

    # Unsharp-mask sharpening
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    versions.append(cv2.filter2D(image, -1, kernel))

    return versions

# ─────────────────────────────────────────────
#  Multi-Scale Detection
# ─────────────────────────────────────────────
def multi_scale_detect(image: np.ndarray, conf_threshold: float) -> list[dict]:
    detections = []

    for scale in MULTI_SCALES:
        resized = cv2.resize(image, None, fx=scale, fy=scale)
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        results = reader.readtext(rgb)

        for (bbox, text, conf) in results:
            if conf < conf_threshold:
                continue

            bbox = (np.array(bbox) / scale).astype(int)
            x1 = int(np.min(bbox[:, 0]))
            y1 = int(np.min(bbox[:, 1]))
            x2 = int(np.max(bbox[:, 0]))
            y2 = int(np.max(bbox[:, 1]))

            detections.append({
                "box"      : [x1, y1, x2, y2],
                "text_easy": text.strip(),
                "conf_easy": float(conf)
            })

    return detections

# ─────────────────────────────────────────────
#  IoU + Box Fusion
# ─────────────────────────────────────────────
def iou(boxA: list, boxB: list) -> float:
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])

    if xB <= xA or yB <= yA:
        return 0.0

    inter     = (xB - xA) * (yB - yA)
    area_a    = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    area_b    = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(area_a + area_b - inter)


def fuse_boxes(detections: list[dict], iou_threshold: float) -> list[dict]:
    detections = sorted(detections, key=lambda x: -x["conf_easy"])
    final = []

    for d in detections:
        if all(iou(d["box"], f["box"]) <= iou_threshold for f in final):
            final.append(d)

    return final

# ─────────────────────────────────────────────
#  TrOCR Inference
# ─────────────────────────────────────────────
def recognize_trocr(crop: np.ndarray) -> str:
    if crop.size == 0:
        return ""

    rgb          = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pixel_values = processor(images=rgb, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        generated_ids = model.generate(pixel_values)

    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

# ─────────────────────────────────────────────
#  Consensus
# ─────────────────────────────────────────────
def consensus(easy_text: str, trocr_text: str) -> str:
    easy_text  = easy_text.upper().strip()
    trocr_text = trocr_text.upper().strip()

    if not trocr_text: return easy_text
    if not easy_text:  return trocr_text
    if easy_text in trocr_text:  return trocr_text
    if trocr_text in easy_text:  return easy_text

    return trocr_text if len(trocr_text) > len(easy_text) else easy_text

# ─────────────────────────────────────────────
#  Line Reconstruction
# ─────────────────────────────────────────────
def reconstruct_lines(results: list[dict]) -> list[dict]:
    results = sorted(results, key=lambda r: (r["box"][1], r["box"][0]))
    used    = [False] * len(results)
    lines   = []

    for i, r1 in enumerate(results):
        if used[i]:
            continue

        x1, y1, x2, y2 = r1["box"]
        h1  = y2 - y1
        cy1 = (y1 + y2) / 2
        current_line = [r1]
        used[i] = True

        for j, r2 in enumerate(results):
            if used[j]:
                continue
            x3, y3, x4, y4 = r2["box"]
            h2  = y4 - y3
            cy2 = (y3 + y4) / 2

            if abs(cy1 - cy2) < max(h1, h2) * 0.6:
                current_line.append(r2)
                used[j] = True

        current_line = sorted(current_line, key=lambda r: r["box"][0])

        seen, words = set(), []
        for r in current_line:
            if r["text"] not in seen:
                words.append(r["text"])
                seen.add(r["text"])

        lines.append({
            "text"      : " ".join(words),
            "confidence": float(np.mean([r["confidence"] for r in current_line]))
        })

    return lines

# ─────────────────────────────────────────────
#  Full OCR Pipeline
# ─────────────────────────────────────────────
def extract_text_from_image(
    image_np       : np.ndarray,
    conf_threshold : float = CONF_THRESHOLD,
    iou_threshold  : float = IOU_THRESHOLD
) -> tuple[np.ndarray, list[dict], float]:

    t0 = time.perf_counter()

    # 1 ── Enhancement
    enhanced_versions = multi_enhance(image_np)

    # 2 ── Multi-scale detection across all enhanced versions
    detections = []
    for v in enhanced_versions:
        detections.extend(multi_scale_detect(v, conf_threshold))

    # 3 ── Fuse overlapping boxes
    detections = fuse_boxes(detections, iou_threshold)

    # 4 ── TrOCR refinement + consensus
    final_results = []
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        crop       = image_np[y1:y2, x1:x2]
        trocr_text = recognize_trocr(crop)
        final_text = consensus(d["text_easy"], trocr_text)

        final_results.append({
            "text"      : final_text,
            "confidence": float(d["conf_easy"]),
            "box"       : [x1, y1, x2, y2]
        })

    # 5 ── Line reconstruction
    lines = reconstruct_lines(final_results)

    # 6 ── Draw bounding boxes
    output = image_np.copy()
    for r in final_results:
        x1, y1, x2, y2 = r["box"]
        cv2.rectangle(output, (x1, y1), (x2, y2), (108, 99, 255), 2)
        cv2.putText(output, r["text"], (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (108, 99, 255), 2)

    elapsed = time.perf_counter() - t0
    return output, lines, elapsed

# ─────────────────────────────────────────────
#  File Upload
# ─────────────────────────────────────────────
st.markdown('<div class="panel-title">📂 Upload</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drop an image or video file",
    type=["jpg", "jpeg", "png", "mp4"],
    label_visibility="collapsed"
)

if not uploaded_file:
    st.markdown("""
    <div style="text-align:center; padding:3rem 0; color:#2a2a4a;
                font-family:'Space Mono',monospace; font-size:0.82rem;">
        ↑ Upload an image (JPG / PNG) or video (MP4) to start
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
#  Image Mode
# ─────────────────────────────────────────────
if "image" in uploaded_file.type:

    image    = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    bgr_np   = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    col_input, col_output = st.columns(2, gap="large")

    with col_input:
        st.markdown('<div class="panel-title">🖼 Input Image</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.caption(f"`{uploaded_file.name}` · {image.width}×{image.height}px")

    with col_output:
        st.markdown('<div class="panel-title">📄 OCR Output</div>', unsafe_allow_html=True)
        run = st.button("▶  Run OCR", use_container_width=True)

        if run:
            with st.spinner("Running multi-scale OCR pipeline..."):
                output_img, lines, elapsed = extract_text_from_image(
                    bgr_np, conf_thresh, iou_thresh
                )
                output_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)

            # Metrics
            st.markdown(f"""
            <div class="metrics-row">
                <div class="mc">
                    <div class="mc-val">{elapsed:.2f}s</div>
                    <div class="mc-lbl">Total Time</div>
                </div>
                <div class="mc">
                    <div class="mc-val">{len(lines)}</div>
                    <div class="mc-lbl">Lines Found</div>
                </div>
                <div class="mc">
                    <div class="mc-val">{
                        f"{np.mean([l['confidence'] for l in lines])*100:.0f}%" if lines else "—"
                    }</div>
                    <div class="mc-lbl">Avg Confidence</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.image(output_rgb, caption="Annotated Result", use_container_width=True)

            # Full text view
            full_text = "\n".join([l["text"] for l in lines if l["text"].strip()])
            if full_text:
                st.markdown("**Extracted Text**")
                st.markdown(f'<div class="json-outer">{full_text}</div>', unsafe_allow_html=True)
                st.download_button(
                    "⬇ Download extracted text",
                    full_text,
                    file_name="ocr_output.txt",
                    mime="text/plain",
                    use_container_width=True
                )

            # Raw JSON
            with st.expander("📋 Raw JSON (line data)"):
                st.json(lines)

        else:
            st.markdown("""
            <div style="color:#2a2a4a; font-family:'Space Mono',monospace;
                        font-size:0.78rem; padding:1.2rem 0;">
                Press <strong style="color:#44447a;">▶ Run OCR</strong> to begin extraction.
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Video Mode
# ─────────────────────────────────────────────
elif "video" in uploaded_file.type:

    st.markdown('<div class="live-badge">● LIVE PROCESSING</div>', unsafe_allow_html=True)
    st.info("Each frame is processed through the full OCR pipeline in real-time.")

    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())

    cap     = cv2.VideoCapture(tfile.name)
    stframe = st.empty()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress     = st.progress(0, text="Processing video...")
    frame_idx    = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        output_frame, _, _ = extract_text_from_image(frame, conf_thresh, iou_thresh)
        rgb_frame          = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)

        stframe.image(rgb_frame, channels="RGB", use_container_width=True)

        frame_idx += 1
        if total_frames > 0:
            progress.progress(
                min(frame_idx / total_frames, 1.0),
                text=f"Frame {frame_idx} / {total_frames}"
            )

    cap.release()
    progress.empty()
    st.success(f"✔ Video processing complete — {frame_idx} frames processed.")

# ─────────────────────────────────────────────
#  Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; font-family:'Space Mono',monospace;
            font-size:0.7rem; color:#22223a; padding-bottom:0.5rem;">
    Advanced OCR System · EasyOCR + TrOCR · Fully Offline · No data transmitted
</div>
""", unsafe_allow_html=True)