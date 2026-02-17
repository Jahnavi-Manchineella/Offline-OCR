# main.py
import os
import sys
import time
import torch
import easyocr
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# ─────────────────────────────────────────────
#  Configuration  (must match streamlit_app.py)
# ─────────────────────────────────────────────
TROCR_MODEL_ID   = "microsoft/trocr-base-printed"
TROCR_SAVE_PATH  = "trocr_model"          # ← loaded with local_files_only=True in app
EASYOCR_LANGS    = ['en']

# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
def banner(text: str) -> None:
    width = 52
    print("\n" + "═" * width)
    print(f"  {text}")
    print("═" * width)

def step(idx: int, total: int, label: str) -> None:
    print(f"\n  [{idx}/{total}]  {label}")
    print("  " + "─" * 44)

def ok(msg: str) -> None:
    print(f"  ✔  {msg}")

def info(msg: str) -> None:
    print(f"  ℹ  {msg}")

def elapsed(t0: float) -> str:
    return f"{time.perf_counter() - t0:.1f}s"

# ─────────────────────────────────────────────
#  Step 1 — EasyOCR
# ─────────────────────────────────────────────
def download_easyocr() -> easyocr.Reader:
    step(1, 3, "Initialising EasyOCR")
    info(f"Languages : {EASYOCR_LANGS}")
    info(f"GPU       : {torch.cuda.is_available()}")

    t0     = time.perf_counter()
    reader = easyocr.Reader(EASYOCR_LANGS, gpu=torch.cuda.is_available())

    ok(f"EasyOCR ready  ({elapsed(t0)})")
    return reader

# ─────────────────────────────────────────────
#  Step 2 — TrOCR download
# ─────────────────────────────────────────────
def download_trocr() -> tuple[TrOCRProcessor, VisionEncoderDecoderModel]:
    step(2, 3, "Downloading TrOCR from HuggingFace")
    info(f"Model ID  : {TROCR_MODEL_ID}")

    t0        = time.perf_counter()
    processor = TrOCRProcessor.from_pretrained(TROCR_MODEL_ID)
    model     = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_ID)
    model.eval()

    ok(f"TrOCR downloaded  ({elapsed(t0)})")
    return processor, model

# ─────────────────────────────────────────────
#  Step 3 — Save locally (local_files_only=True)
# ─────────────────────────────────────────────
def save_trocr(
    processor : TrOCRProcessor,
    model     : VisionEncoderDecoderModel,
    save_path : str
) -> None:

    step(3, 3, f"Saving TrOCR → ./{save_path}/")

    # Warn if folder already exists (will overwrite)
    if os.path.exists(save_path):
        info(f"Folder '{save_path}' already exists — overwriting.")

    os.makedirs(save_path, exist_ok=True)

    t0 = time.perf_counter()
    processor.save_pretrained(save_path)
    model.save_pretrained(save_path)

    # Verify essential files are present
    required = ["config.json", "pytorch_model.bin"]
    missing  = [f for f in required if not os.path.exists(os.path.join(save_path, f))]

    if missing:
        print(f"\n  ⚠  Warning: expected files not found: {missing}")
        print("     The app may fail to load with local_files_only=True")
    else:
        ok(f"All files saved successfully  ({elapsed(t0)})")
        ok(f"Path: {os.path.abspath(save_path)}")

# ─────────────────────────────────────────────
#  Summary
# ─────────────────────────────────────────────
def print_summary(save_path: str) -> None:
    print("\n" + "═" * 52)
    print("  ✅  All models downloaded successfully!")
    print("═" * 52)
    print()
    print("  Your streamlit_app.py will load models from:")
    print(f"    └── ./{save_path}/   (local_files_only=True)")
    print()
    print("  To launch the OCR app, run:")
    print("    streamlit run streamlit_app.py")
    print()

# ─────────────────────────────────────────────
#  Guard — skip re-download if already saved
# ─────────────────────────────────────────────
def already_saved(save_path: str) -> bool:
    required = ["config.json", "pytorch_model.bin"]
    return all(os.path.exists(os.path.join(save_path, f)) for f in required)

# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
def main() -> None:
    banner("OCR Model Download  ·  Advanced OCR System")

    # ── EasyOCR ──────────────────────────────
    reader = download_easyocr()

    # ── TrOCR ────────────────────────────────
    if already_saved(TROCR_SAVE_PATH):
        print(f"\n  ⏩  TrOCR already saved at ./{TROCR_SAVE_PATH} — skipping download.")
        print("     Delete the folder to force a fresh download.")
    else:
        processor, model = download_trocr()
        save_trocr(processor, model, TROCR_SAVE_PATH)

    print_summary(TROCR_SAVE_PATH)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  ✖  Download cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n  ✖  Download failed: {e}")
        sys.exit(1)