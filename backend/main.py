import base64
import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from seal_comparator import SealComparator
from seal_detector import SealDetector

app = FastAPI(title="인감 도장 대조 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 서버 시작 시 모델을 한 번만 로드
detector   = SealDetector()
comparator = SealComparator()

ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png"}


def _img_to_b64(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode()


def _check_ext(filename: str) -> None:
    import os
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"지원하지 않는 파일 형식입니다: {ext}")


@app.post("/api/compare")
async def compare_seals(
    certificate: UploadFile = File(..., description="인감 증명서 (PDF/JPG/PNG)"),
    contract:    UploadFile = File(..., description="계약서 (PDF/JPG/PNG)"),
):
    _check_ext(certificate.filename)
    _check_ext(contract.filename)

    cert_bytes = await certificate.read()
    cont_bytes = await contract.read()

    cert_seal, cert_preview = detector.extract_seal(cert_bytes, certificate.filename)
    cont_seal, cont_preview = detector.extract_seal(cont_bytes, contract.filename)

    if cert_seal is None:
        raise HTTPException(422, "인감 증명서에서 도장을 찾을 수 없습니다.")
    if cont_seal is None:
        raise HTTPException(422, "계약서에서 도장을 찾을 수 없습니다.")

    score, details = comparator.compare(cert_seal, cont_seal)

    # 점수 구간별 판정 (0~100 기준)
    if score >= 95:
        verdict, verdict_en, verdict_color = "패스",   "pass",    "#22c55e"
    elif score >= 89:
        verdict, verdict_en, verdict_color = "양호",   "good",    "#84cc16"
    elif score >= 75:
        verdict, verdict_en, verdict_color = "주의",   "caution", "#f59e0b"
    else:
        verdict, verdict_en, verdict_color = "불일치", "fail",    "#ef4444"

    return {
        "score":               score,
        "verdict":             verdict,
        "verdict_en":          verdict_en,
        "verdict_color":       verdict_color,
        "details":             details,
        "certificate_seal":    _img_to_b64(cert_seal),
        "contract_seal":       _img_to_b64(cont_seal),
        "certificate_preview": _img_to_b64(cert_preview),
        "contract_preview":    _img_to_b64(cont_preview),
    }


@app.post("/api/debug/yolo")
async def debug_yolo(file: UploadFile = File(...)):
    """YOLO 검출 결과를 그대로 반환 — 디버그용."""
    _check_ext(file.filename)
    file_bytes = await file.read()
    img = detector._load_image(file_bytes, file.filename)
    if img is None:
        raise HTTPException(400, "이미지를 읽을 수 없습니다.")
    h, w = img.shape[:2]
    detections = detector.yolo_detections(img)
    return {"image_size": {"w": w, "h": h}, "detections": detections}


@app.get("/api/health")
def health():
    return {"status": "ok"}