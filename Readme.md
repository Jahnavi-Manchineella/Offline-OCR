🔬 Advanced OCR System
EasyOCR + TrOCR (Fully Offline · Multi-Scale · Streamlit UI)

An advanced offline Optical Character Recognition (OCR) system that combines EasyOCR (detection) and Microsoft TrOCR (recognition refinement) using a multi-scale + consensus-based pipeline.

This system supports:

✅ Image OCR (JPG / PNG)

✅ Video OCR (MP4 frame-by-frame)

✅ Multi-scale text detection

✅ IoU box fusion

✅ EasyOCR + TrOCR consensus

✅ Line reconstruction

✅ Fully offline execution

✅ Streamlit UI

📦 1️⃣ Submission Files (Mandatory)

Your submission includes:

✅ Complete project ZIP folder

✅ Detailed documentation (PDF/DOC)

✅ Sample outputs inside /outputs/

✅ README.md (this file)

✅ requirements.txt

✅ Working Streamlit app (app.py)

⭐ GitHub repository (recommended)

🗂 2️⃣ Required Project Structure
project/
├── datasets/
│   └── sample_images/
├── models/
│   └── trocr_model/
├── test_videos/
├── outputs/
├── main.py
├── app.py
├── requirements.txt
└── README.md


Note: In your implementation, trocr_model/ contains locally saved HuggingFace model weights to ensure offline execution.

⚙️ Installation Instructions
1️⃣ Create Virtual Environment (Recommended)
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Download Models (First Time Only)

Run:

python main.py


This will:

Initialize EasyOCR

Download microsoft/trocr-base-printed

Save model locally inside /trocr_model/

Prepare system for offline inference

After this step → Internet is NOT required.

🚀 Running the Application
streamlit run app.py


Then open browser at:

http://localhost:8501


Upload:

JPG / PNG image

MP4 video

Click Run OCR

🧠 Model Selection Justification
Why EasyOCR?

Strong detection capability

Works offline

Lightweight compared to heavy detection models

Supports multi-scale inference

No cloud API required

Why TrOCR?

Transformer-based OCR model

State-of-the-art recognition accuracy

Robust to noisy / distorted text

Works fully offline after download

Improves recognition quality over EasyOCR raw output

Why Not YOLO?

Assignment restriction:

❌ Do NOT use YOLO

Additionally:

YOLO requires object detection training

Not necessary for text-only detection

Adds training complexity

📊 Dataset Justification

This system is inference-based and does not use COCO or ImageNet.

Why?

❌ COCO/ImageNet are general object detection datasets

❌ Not optimized for text recognition

❌ Assignment restriction

Instead:

System uses pre-trained OCR models

Tested on:

Custom sample images

Real-world text samples

Printed documents

Industrial stencil samples

🔄 Inference Pipeline (Step-by-Step)

The complete OCR pipeline:

1️⃣ Image Preprocessing

Convert to grayscale

Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)

Apply sharpening filter

Create multiple enhanced versions

2️⃣ Multi-Scale Detection

Scales used:

[0.75, 1.0, 1.5, 2.0]


For each scale:

Resize image

Run EasyOCR

Rescale bounding boxes back

Filter by confidence threshold

3️⃣ IoU Box Fusion

Remove overlapping bounding boxes

Keep highest confidence

Reduce duplicate detections

4️⃣ TrOCR Recognition

For each cropped text region:

Convert to RGB

Pass through TrOCR processor

Generate text using transformer decoder

Decode tokens to final text

5️⃣ Consensus Merge

Final text = best match between:

EasyOCR output

TrOCR output

Rules:

If one contains the other → keep longer

Else → keep higher quality prediction

6️⃣ Line Reconstruction

Group boxes by vertical alignment

Sort left-to-right

Remove duplicates

Build clean readable lines

📄 Inference Script (Standalone CLI Version)

You can also run OCR without Streamlit:

# inference.py

import cv2
from main import extract_text_from_image
import numpy as np
from PIL import Image

image_path = "datasets/sample_images/Picture3.jpg"

image = Image.open(image_path).convert("RGB")
image_np = np.array(image)
bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

output_img, lines, elapsed = extract_text_from_image(bgr)

print("Time:", elapsed)
print("\nExtracted Text:\n")

for line in lines:
    print(line["text"], "| Confidence:", line["confidence"])

cv2.imwrite("outputs/result.jpg", output_img)


Run:

python inference.py

📈 Metrics Logged

For each run:

Total inference time

Number of lines detected

Average confidence score

Bounding box visualization

Structured JSON output

Example JSON:

[
  {
    "text": "ADVANCED OCR SYSTEM",
    "confidence": 0.91
  },
  {
    "text": "FULLY OFFLINE",
    "confidence": 0.88
  }
]

🔒 Offline Constraint (Strictly Followed)

❌ No Google Vision

❌ No AWS Textract

❌ No Cloud API

❌ No Internet during inference

✅ All models stored locally

✅ local_files_only=True enforced

System works completely offline after first model download.

⚠️ Challenges Faced
1️⃣ Duplicate Detections

Multi-scale detection caused repeated boxes.

✔ Solution:
Implemented IoU-based fusion.

2️⃣ Noisy / Low Contrast Text

Low-quality images reduced accuracy.

✔ Solution:

CLAHE contrast enhancement

Sharpening

Multi-enhancement pipeline

3️⃣ Recognition Errors

EasyOCR sometimes produced short/incorrect words.

✔ Solution:
Consensus merge with TrOCR refinement.

4️⃣ Video Processing Performance

Frame-by-frame OCR is computationally heavy.

✔ Solution:

Efficient caching

Model loaded once

GPU support enabled

🔧 Possible Improvements

Add language auto-detection

Add beam search tuning in TrOCR

Add batch frame processing

Add PDF batch processing

Add layout-aware text grouping

Implement confidence-weighted consensus

📊 Performance Summary
Component	Description
Detection	EasyOCR Multi-Scale
Recognition	TrOCR Transformer
Fusion	IoU-based
Runtime	~1–3 sec per image (CPU)
GPU Support	Yes
Offline	Fully
🎯 AI Technical Assignment Compliance
Requirement	Status
No YOLO	✅
No COCO/ImageNet	✅
No Cloud APIs	✅
Offline execution	✅
Proper folder structure	✅
Streamlit runs without error	✅
Structured output	✅
Documentation included	✅
🧪 Sample Outputs

Stored inside:

/outputs/


Includes:

Annotated images

Extracted text files

Screenshots

👩‍💻 Author

Advanced OCR System
AI Technical Assignment Submission

✅ Final Verification Checklist

Before submission:

 ZIP folder created

 outputs/ contains sample results

 requirements.txt updated

 README.md included

 Streamlit app runs

 Models downloaded locally

 Documentation PDF added