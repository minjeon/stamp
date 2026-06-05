import os
import sys
import io
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

try:
    import fitz
    USE_FITZ = True
except ImportError:
    from pdf2image import convert_from_path
    USE_FITZ = False


# ── 파일 로드 ─────────────────────────────────────────────────────────────────

def load_images_from_file(file_path: str) -> list[np.ndarray]:
    """PDF / JPG / PNG → BGR numpy 배열 리스트."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        if USE_FITZ:
            doc = fitz.open(file_path)
            images = []
            for page in doc:
                pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
                pil_img = Image.open(io.BytesIO(pix.tobytes("ppm")))
                images.append(cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR))
            return images
        else:
            return [cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR)
                    for p in convert_from_path(file_path, dpi=300)]
    elif ext in (".jpg", ".jpeg", ".png"):
        img = cv2.imread(file_path)
        if img is None:
            raise FileNotFoundError(f"이미지를 읽을 수 없음: {file_path}")
        return [img]
    else:
        raise ValueError(f"지원하지 않는 형식: {ext}")


def load_images_from_bytes(data: bytes, filename: str) -> list[np.ndarray]:
    """업로드된 파일 바이트 → BGR numpy 배열 리스트."""
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        if USE_FITZ:
            doc = fitz.open(stream=data, filetype="pdf")
            images = []
            for page in doc:
                pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
                pil_img = Image.open(io.BytesIO(pix.tobytes("ppm")))
                images.append(cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR))
            return images
        else:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                f.write(data)
                tmp_path = f.name
            imgs = [cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR)
                    for p in convert_from_path(tmp_path, dpi=300)]
            os.unlink(tmp_path)
            return imgs
    elif ext in (".jpg", ".jpeg", ".png"):
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"이미지 디코딩 실패: {filename}")
        return [img]
    else:
        raise ValueError(f"지원하지 않는 형식: {ext}")


# ── CLAHE 전처리 ──────────────────────────────────────────────────────────────

def apply_clahe(bgr_img: np.ndarray,
                clip_limit: float = 2.0,
                tile_size: tuple = (8, 8)) -> np.ndarray:
    """LAB 밝기(L) 채널에만 CLAHE 적용 — 잉크 색 왜곡 없이 대비 향상."""
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    lab_enhanced = cv2.merge([clahe.apply(l_ch), a_ch, b_ch])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


# ── YOLOv10 멀티스케일 추론 + NMS ─────────────────────────────────────────────

def detect_with_yolo(bgr_img: np.ndarray,
                     model: YOLO,
                     conf_threshold: float = 0.5,
                     scale_factor: float = 1.5,
                     nms_iou_threshold: float = 0.5) -> list[list[int]]:
    """원본 + 1.5× 확대 두 번 추론 후 NMS로 중복 박스 제거."""
    h_orig, w_orig = bgr_img.shape[:2]
    all_boxes, all_scores = [], []

    def _infer(img: np.ndarray, is_scaled: bool = False):
        for result in model(img, verbose=False):
            if result.boxes is None:
                continue
            for box, score in zip(result.boxes.xyxy, result.boxes.conf):
                score_val = float(score)
                if score_val < conf_threshold:
                    continue
                x1, y1, x2, y2 = map(int, box.tolist())
                if is_scaled:
                    x1, y1 = int(x1 / scale_factor), int(y1 / scale_factor)
                    x2, y2 = int(x2 / scale_factor), int(y2 / scale_factor)
                x1 = max(0, min(x1, w_orig - 1))
                y1 = max(0, min(y1, h_orig - 1))
                x2 = max(0, min(x2, w_orig))
                y2 = max(0, min(y2, h_orig))
                all_boxes.append([x1, y1, x2, y2])
                all_scores.append(score_val)

    _infer(bgr_img, is_scaled=False)
    img_scaled = cv2.resize(bgr_img,
                            (int(w_orig * scale_factor), int(h_orig * scale_factor)),
                            interpolation=cv2.INTER_LINEAR)
    _infer(img_scaled, is_scaled=True)

    if not all_boxes:
        return []

    boxes_xywh = [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in all_boxes]
    keep = cv2.dnn.NMSBoxes(boxes_xywh, all_scores, conf_threshold, nms_iou_threshold)
    if len(keep) == 0:
        return []

    final = [all_boxes[i] for i in keep.flatten()]
    print(f"  [YOLO] {len(final)}개 검출 (conf≥{conf_threshold})")
    return final


# ── 5단계 폴백 검출 ───────────────────────────────────────────────────────────

def _get_red_mask(bgr_img: np.ndarray) -> np.ndarray:
    """HSV에서 빨간 픽셀 이진 마스크 추출."""
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0,   80, 50]), np.array([10,  255, 255])),
        cv2.inRange(hsv, np.array([160, 80, 50]), np.array([180, 255, 255])),
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _mask_to_boxes(mask: np.ndarray, min_area: int = 500) -> list[list[int]]:
    """이진 마스크 컨투어 → [x1,y1,x2,y2] 박스 리스트."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append([x, y, x + w, y + h])
    return boxes


def _hough_boxes(mask: np.ndarray, img_h: int, img_w: int,
                 param1: int, param2: int) -> list[list[int]]:
    """HoughCircles 결과를 박스 리스트로 변환."""
    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT,
                                dp=1.2, minDist=50,
                                param1=param1, param2=param2,
                                minRadius=20, maxRadius=300)
    if circles is None:
        return []
    boxes = []
    for cx, cy, r in np.round(circles[0]).astype(int):
        pad = int(r * 0.1)
        boxes.append([max(0, cx-r-pad), max(0, cy-r-pad),
                      min(img_w, cx+r+pad), min(img_h, cy+r+pad)])
    return boxes


def fallback_detect(bgr_img: np.ndarray) -> list[list[int]]:
    """
    YOLO 실패 시 5단계 rule-based 검출.
    1. ColorHough     : HSV 빨간 마스크 + HoughCircles
    2. ColorContour   : HSV 빨간 마스크 + 컨투어
    3. GrayscaleHough : 그레이 + HoughCircles
    4. AdaptiveThresh : 적응형 이진화 + 컨투어
    5. Contour        : Otsu 이진화 컨투어 (최후 수단)
    """
    h, w = bgr_img.shape[:2]
    red_mask = _get_red_mask(bgr_img)

    boxes = _hough_boxes(red_mask, h, w, param1=50, param2=30)
    if boxes:
        print(f"  [폴백] 1단계 성공 (ColorHough): {len(boxes)}개")
        return boxes

    boxes = _mask_to_boxes(red_mask, min_area=800)
    if boxes:
        print(f"  [폴백] 2단계 성공 (ColorContour): {len(boxes)}개")
        return boxes

    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), sigmaX=2)
    boxes = _hough_boxes(blurred, h, w, param1=80, param2=40)
    if boxes:
        print(f"  [폴백] 3단계 성공 (GrayscaleHough): {len(boxes)}개")
        return boxes

    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, blockSize=15, C=4)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    boxes = _mask_to_boxes(cv2.dilate(thresh, kernel, iterations=2), min_area=1000)
    if boxes:
        print(f"  [폴백] 4단계 성공 (AdaptiveThresh): {len(boxes)}개")
        return boxes

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    boxes = _mask_to_boxes(otsu, min_area=1500)
    if boxes:
        boxes.sort(key=lambda b: (b[2]-b[0]) * (b[3]-b[1]), reverse=True)
        print(f"  [폴백] 5단계 성공 (Contour): {min(3, len(boxes))}개")
        return boxes[:3]

    print("  [폴백] 전 단계 실패")
    return []


# ── 형태 기반 필터 ────────────────────────────────────────────────────────────

def filter_by_shape(bgr_img: np.ndarray,
                    boxes: list[list[int]],
                    min_aspect: float = 0.5,
                    max_aspect: float = 2.0,
                    min_ink_density: float = 0.03,
                    min_circularity: float = 0.1) -> list[list[int]]:
    """
    가로세로 비율 · 잉크 밀도 · 원형도 세 조건으로 텍스트·노이즈 박스 제거.
    색깔 무관하게 동작 (빨강/파랑/흑백 도장 모두 대응).
    """
    filtered = []
    for box in boxes:
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        if bw == 0 or bh == 0:
            continue

        aspect = bw / bh
        if not (min_aspect <= aspect <= max_aspect):
            print(f"  [필터] 제거 — 비율 {aspect:.2f}  box={box}")
            continue

        gray_roi = cv2.cvtColor(bgr_img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        if np.sum(gray_roi < 180) / (bw * bh) < min_ink_density:
            print(f"  [필터] 제거 — 잉크 밀도 부족  box={box}")
            continue

        _, binary = cv2.threshold(gray_roi, 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            perim = cv2.arcLength(largest, closed=True)
            circ = (4 * np.pi * area / perim ** 2) if perim > 0 else 0
            if circ < min_circularity:
                print(f"  [필터] 제거 — 원형도 {circ:.2f}  box={box}")
                continue

        print(f"  [필터] 통과  box={box}")
        filtered.append(box)
    return filtered


# ── 박스 시각화 ───────────────────────────────────────────────────────────────

def draw_seal_boxes(img: np.ndarray, boxes: list[list[int]]) -> np.ndarray:
    """
    원본 이미지에 검출된 도장 박스를 빨간 사각형으로 그려 반환한다.
    원본 이미지를 수정하지 않도록 복사본에 그림.
    """
    preview = img.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        # 빨간 사각형 (BGR: 0, 0, 255)
        # thickness=3 : 선 두께. 이미지 크기에 따라 조정.
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 0, 255), 3)

        # 박스 번호 라벨 (배경 채워서 가독성 확보)
        label = f"seal {i+1}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(preview,
                      (x1, y1 - lh - 8), (x1 + lw + 6, y1),
                      (0, 0, 255), -1)
        cv2.putText(preview, label,
                    (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return preview


# ── SealDetector 클래스 ───────────────────────────────────────────────────────

class SealDetector:
    """
    YOLO + 5단계 폴백으로 도장을 검출하고 박스 크롭을 반환한다.
    """

    def __init__(self,
                 yolo_model_path: str = r"C:\min\stamp3\weights\best.pt",
                 sam_checkpoint: str = "sam_vit_b_01ec64.pth",
                 sam_model_type: str = "vit_b",
                 device: str = "cpu",
                 yolo_conf: float = 0.5,
                 yolo_conf_retry: float = 0.3):

        self.yolo_conf       = yolo_conf
        self.yolo_conf_retry = yolo_conf_retry

        print(f"[초기화] YOLO 로드: {yolo_model_path}")
        self.yolo = YOLO(yolo_model_path)

        print(f"[초기화] SAM 로드: {sam_checkpoint}")
        sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.sam_predictor = SamPredictor(sam)
        print("[초기화] 완료")

    def _process_page(self, img: np.ndarray) -> tuple[list[np.ndarray], list[list[int]]]:
        """
        단일 이미지에서 도장 크롭 리스트와 박스 좌표 리스트를 함께 반환한다.

        Returns
        -------
        crops : 각 도장의 크롭 이미지 리스트
        boxes : 각 도장의 [x1,y1,x2,y2] 좌표 리스트 (crops와 인덱스 대응)
        """
        img_clahe = apply_clahe(img)

        # YOLO 결과는 필터 없이 바로 사용
        boxes = detect_with_yolo(img_clahe, self.yolo,
                                  conf_threshold=self.yolo_conf)
        if not boxes:
            print(f"  [검출] YOLO 1차 실패 → conf {self.yolo_conf_retry}로 재시도")
            boxes = detect_with_yolo(img_clahe, self.yolo,
                                      conf_threshold=self.yolo_conf_retry)

        # 폴백 결과만 형태 필터 적용
        if not boxes:
            print("  [검출] YOLO 완전 실패 → 폴백 시작")
            fallback_boxes = fallback_detect(img_clahe)
            boxes = filter_by_shape(img_clahe, fallback_boxes)

        if not boxes:
            print("  [검출] 도장 검출 실패")
            return [], []

        h_img, w_img = img_clahe.shape[:2]
        crops, valid_boxes = [], []
        for box in boxes:
            x1, y1, x2, y2 = box
            pad = 10
            crop = img_clahe[max(0, y1-pad):min(h_img, y2+pad),
                             max(0, x1-pad):min(w_img, x2+pad)]
            if crop.size > 0:
                crops.append(crop)
                valid_boxes.append(box)
                print(f"  [크롭] box={box} → {crop.shape}")

        return crops, valid_boxes

    def extract_seal(self, data: bytes, filename: str):
        """
        FastAPI에서 호출하는 메서드.

        Returns
        -------
        seal_crop : 가장 큰 도장 크롭 이미지 (비교용). 실패 시 None.
        preview   : 원본 이미지에 검출 박스를 그린 프리뷰 이미지. 실패 시 None.
        """
        images = load_images_from_bytes(data, filename)
        if not images:
            return None, None

        # 첫 페이지만 처리 (인감증명서·계약서는 단일 페이지)
        img = images[0]
        crops, boxes = self._process_page(img)

        if not crops:
            return None, None

        # 가장 큰 도장 선택
        best_idx = max(range(len(crops)),
                       key=lambda i: crops[i].shape[0] * crops[i].shape[1])
        best_crop = crops[best_idx]

        # 원본 이미지에 모든 검출 박스를 그려서 프리뷰 생성
        preview = draw_seal_boxes(img, [boxes[best_idx]])

        return best_crop, preview

    def detect(self, file_path: str) -> list[list[np.ndarray]]:
        """파일 경로로 검출. 페이지별 크롭 리스트의 리스트를 반환."""
        images = load_images_from_file(file_path)
        results = []
        for i, img in enumerate(images):
            print(f"\n[페이지 {i+1}]")
            crops, _ = self._process_page(img)
            results.append(crops)
            print(f"  → {len(crops)}개 도장")
        return results

    def detect_from_bytes(self, data: bytes, filename: str) -> list[list[np.ndarray]]:
        """바이트 데이터로 검출. 페이지별 크롭 리스트의 리스트를 반환."""
        images = load_images_from_bytes(data, filename)
        results = []
        for i, img in enumerate(images):
            print(f"\n[페이지 {i+1}]")
            crops, _ = self._process_page(img)
            results.append(crops)
            print(f"  → {len(crops)}개 도장")
        return results

    def yolo_detections(self, img: np.ndarray) -> list[dict]:
        """디버그용 — YOLO 검출 결과를 dict 리스트로 반환."""
        boxes = detect_with_yolo(img, self.yolo,
                                  conf_threshold=self.yolo_conf)
        return [{"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]} for b in boxes]

    def _load_image(self, data: bytes, filename: str) -> np.ndarray | None:
        """디버그용 — 바이트를 단일 BGR 이미지로 변환."""
        try:
            imgs = load_images_from_bytes(data, filename)
            return imgs[0] if imgs else None
        except Exception:
            return None


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("사용법: python seal_detector.py <입력파일> <yolo.pt> <sam.pth>")
        sys.exit(0)

    detector = SealDetector(
        yolo_model_path=sys.argv[2],
        sam_checkpoint=sys.argv[3],
        device="cpu",
    )
    results = detector.detect(sys.argv[1])

    os.makedirs("output_crops", exist_ok=True)
    total = 0
    for pi, crops in enumerate(results):
        for ci, crop in enumerate(crops):
            path = f"output_crops/page{pi+1}_seal{ci+1}.png"
            cv2.imwrite(path, crop)
            print(f"저장: {path}")
            total += 1
    print(f"\n총 {total}개 저장 완료")