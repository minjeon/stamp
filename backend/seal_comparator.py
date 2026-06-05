import os
import sys
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.morphology import skeletonize
import torch
from transformers import AutoModel, AutoProcessor
from seal_detector import SealDetector


# ── 정렬 ─────────────────────────────────────────────────────────────────────

def _to_gray(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _rotate(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderValue=(255, 255, 255))


# def align_images(ref: np.ndarray, cmp: np.ndarray,
#                  processor=None, model=None, device: str = "cpu") -> np.ndarray:
#     """
#     DINOv2 코사인 유사도 기반 회전 정렬.
#     SSIM은 원형 도장에서 각도 구분이 어려워 DINOv2로 교체.
#     Coarse(30° 간격, 0~360°) → Fine(5° 간격, ±15°)
#     """
#     h, w = ref.shape[:2]
#     cmp = cv2.resize(cmp, (w, h))

#     def _score(img_a, img_b):
#         if processor is not None and model is not None:
#             return score_dinov2(img_a, img_b, processor, model, device)
#         # DINOv2 없으면 SSIM fallback
#         ga = _to_gray(img_a)
#         gb = _to_gray(img_b)
#         if ga.shape != gb.shape:
#             gb = cv2.resize(gb, (ga.shape[1], ga.shape[0]))
#         s, _ = ssim(ga, gb, full=True)
#         return s

#     # Coarse : 0°~330°, 30° 간격
#     best_angle, best_score = 0.0, -1.0
#     for angle in range(0, 360, 15):
#         s = _score(ref, _rotate(cmp, float(angle)))
#         if s > best_score:
#             best_score, best_angle = s, float(angle)

#     # Fine : best_angle ±15°, 5° 간격
#     for angle in np.arange(best_angle - 15, best_angle + 16, 2):
#         s = _score(ref, _rotate(cmp, angle))
#         if s > best_score:
#             best_score, best_angle = s, angle

#     print(f"  [정렬] 최적 회전각: {best_angle:.1f}°  (점수={best_score:.3f})")
#     aligned = _rotate(cmp, best_angle)

#     # PhaseCorrelation 평행이동 보정
#     (dx, dy), _ = cv2.phaseCorrelate(
#         np.float32(_to_gray(ref)),
#         np.float32(_to_gray(aligned))
#     )
#     aligned = cv2.warpAffine(
#         aligned,
#         np.float32([[1, 0, -dx], [0, 1, -dy]]),
#         (w, h),
#         borderValue=(255, 255, 255)
#     )
#     print(f"  [정렬] 평행이동 보정: dx={dx:.1f}px, dy={dy:.1f}px")
#     return aligned

def align_images_sift(ref: np.ndarray, cmp: np.ndarray) -> np.ndarray:
    """
    SIFT 특징점 매칭 기반 어파인 정렬.
    DINOv2 회전 정렬보다 기하학적으로 정확.
    매칭 실패 시 원본 cmp 반환 (fallback).
    """
    h, w = ref.shape[:2]
    cmp_r = cv2.resize(cmp, (w, h))

    sift = cv2.SIFT_create(nfeatures=500)
    ga = _to_gray(ref)
    gb = _to_gray(cmp_r)

    kp_a, des_a = sift.detectAndCompute(ga, None)
    kp_b, des_b = sift.detectAndCompute(gb, None)

    if des_a is None or des_b is None or len(kp_a) < 4 or len(kp_b) < 4:
        print("  [정렬] SIFT 특징점 부족 → 정렬 스킵")
        return cmp_r

    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw = bf.knnMatch(des_a, des_b, k=2)
    good = [m for m, n in raw if len((m, n)) == 2 and m.distance < 0.75 * n.distance]

    if len(good) < 4:
        print(f"  [정렬] 매칭 {len(good)}개 부족 → 정렬 스킵")
        return cmp_r

    pts_a = np.float32([kp_a[m.queryIdx].pt for m in good])
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in good])

    # estimateAffinePartial2D: 회전+스케일+평행이동만 허용 (왜곡 방지)
    M, inliers = cv2.estimateAffinePartial2D(pts_b, pts_a,
                                              method=cv2.RANSAC,
                                              ransacReprojThreshold=3.0)
    if M is None:
        print("  [정렬] 어파인 추정 실패 → 정렬 스킵")
        return cmp_r

    n_inliers = int(inliers.sum()) if inliers is not None else 0
    print(f"  [정렬] SIFT 매칭 {len(good)}개, 인라이어 {n_inliers}개")

    aligned = cv2.warpAffine(cmp_r, M, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderValue=(255, 255, 255))
    return aligned

# ── 유사도 지표 ──────────────────────────────────────────────────────────────

def _to_gray_224(img: np.ndarray) -> np.ndarray:
    return cv2.resize(_to_gray(img), (224, 224), interpolation=cv2.INTER_AREA)


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return b > 0


def score_ink_ssim(ga: np.ndarray, gb: np.ndarray) -> float:
    """잉크 픽셀 bounding box 안에서만 SSIM."""
    mask = _ink_mask(ga) | _ink_mask(gb)
    if mask.sum() < 100:
        return 0.0
    rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    ca, cb = ga[r0:r1+1, c0:c1+1], gb[r0:r1+1, c0:c1+1]
    if ca.shape[0] < 7 or ca.shape[1] < 7:
        pa, pb = ga[mask], gb[mask]
        return float(np.corrcoef(pa, pb)[0, 1])
    s, _ = ssim(ca, cb, full=True)
    return max(0.0, float(s))


def score_skeleton_ssim(ga: np.ndarray, gb: np.ndarray) -> float:
    """세선화(1픽셀 중심선) 후 SSIM — 잉크 굵기 변화 무시."""
    sa = skeletonize(_ink_mask(ga)).astype(np.uint8) * 255
    sb = skeletonize(_ink_mask(gb)).astype(np.uint8) * 255
    if sa.sum() < 255 or sb.sum() < 255:
        return 0.0
    s, _ = ssim(sa, sb, full=True)
    return max(0.0, float(s))


def score_grid_ssim(ga: np.ndarray, gb: np.ndarray, grid: int = 4) -> float:
    """grid×grid 구역별 SSIM 평균 — 부분 위변조 감지."""
    h, w = ga.shape
    th, tw = h // grid, w // grid
    scores = []
    for r in range(grid):
        for c in range(grid):
            ta = ga[r*th:(r+1)*th, c*tw:(c+1)*tw]
            tb = gb[r*th:(r+1)*th, c*tw:(c+1)*tw]
            if ta.shape[0] < 7 or ta.shape[1] < 7:
                continue
            s, _ = ssim(ta, tb, full=True)
            scores.append(max(0.0, float(s)))
    return float(np.mean(scores)) if scores else 0.0


def score_dinov2(img_a: np.ndarray, img_b: np.ndarray,
                 processor, model, device: str) -> float:
    """DINOv2 CLS 토큰 코사인 유사도 — 해상도·굵기 변화에 강건."""
    def _feat(img):
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inp = processor(images=rgb, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inp)
        return out.last_hidden_state[:, 0, :].squeeze()

    fa, fb = _feat(img_a), _feat(img_b)
    return float(torch.nn.functional.cosine_similarity(
        fa.unsqueeze(0), fb.unsqueeze(0)
    ).item())

def score_sift(img_a: np.ndarray, img_b: np.ndarray,
               ratio_thresh: float = 0.75) -> float:
    """
    SIFT 특징점 매칭 비율 점수 (0~1).
    
    - Lowe's ratio test로 불량 매칭 제거
    - 매칭 수 / min(kp_a, kp_b) 로 정규화
    - 도장처럼 세밀한 획/구조 차이를 구분하는 데 강점
    """
    sift = cv2.SIFT_create(nfeatures=500)

    # 잉크 마스크: 배경(흰색) 영역은 특징점 탐지에서 제외
    def _ink_mask_u8(gray: np.ndarray) -> np.ndarray:
        _, m = cv2.threshold(gray, 0, 255,
                             cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # SIFT mask는 uint8, 255=탐지허용
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        return cv2.dilate(m, kernel, iterations=2)

    ga = _to_gray(img_a)
    gb = _to_gray(img_b)

    if ga.shape != gb.shape:
        gb = cv2.resize(gb, (ga.shape[1], ga.shape[0]))

    mask_a = _ink_mask_u8(ga)
    mask_b = _ink_mask_u8(gb)

    kp_a, des_a = sift.detectAndCompute(ga, mask_a)
    kp_b, des_b = sift.detectAndCompute(gb, mask_b)

    # 특징점이 너무 적으면 판단 불가
    if des_a is None or des_b is None:
        return 0.0
    if len(kp_a) < 8 or len(kp_b) < 8:
        return 0.0

    # BFMatcher + Lowe's ratio test
    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = bf.knnMatch(des_a, des_b, k=2)

    good = []
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio_thresh * n.distance:
                good.append(m)

    # 정규화: 더 적은 쪽 특징점 수 기준
    denom = min(len(kp_a), len(kp_b))
    ratio = len(good) / denom if denom > 0 else 0.0

    print(f"  [SIFT] kp_a={len(kp_a)}, kp_b={len(kp_b)}, "
          f"good={len(good)}, ratio={ratio:.3f}")

    # 0.3 이상이면 사실상 동일 도장 수준 → sigmoid로 0~1 스케일
    # ratio가 0.05~0.40 범위에서 주로 분포하므로 선형 클리핑으로 충분
    return float(np.clip(ratio / 0.35, 0.0, 1.0))


# ── SealComparator 클래스 ────────────────────────────────────────────────────

class SealComparator:
    """
    두 도장 이미지(numpy 배열)를 받아 유사도 점수를 반환한다.
    FastAPI main.py에서 detector.extract_seal()로 얻은 크롭을 직접 넘긴다.
    """

    WEIGHTS = {
        "dinov2"       : 0.40,
        "sift"         : 0.60,
        "skeleton_ssim": 0.00,
        "grid_ssim"    : 0.00,
        "ink_ssim"     : 0.00,
    }

    def __init__(self,
                 dinov2_model_name: str = "facebook/dinov2-base",
                 device: str = "cpu",
                 threshold: float = 0.70):
        self.device    = device
        self.threshold = threshold

        print(f"[초기화] DINOv2 로드 중: {dinov2_model_name}")
        self.processor  = AutoProcessor.from_pretrained(dinov2_model_name)
        self.dino_model = AutoModel.from_pretrained(dinov2_model_name).to(device)
        self.dino_model.eval()
        print("[초기화] 완료")

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        """
        잉크 픽셀의 최소 외접원으로 도장 영역만 추출.
        HoughCircles와 달리 타원·찌그러진 도장도 정확하게 감쌈.
        """
        h, w = crop.shape[:2]
        pad = 5

        # 빨간 잉크 픽셀 마스크 추출
        b, g, r_ch = cv2.split(crop)
        red_mask = (
            (r_ch > 80) &
            (r_ch.astype(int) - g.astype(int) > 20) &
            (r_ch.astype(int) - b.astype(int) > 20)
        ).astype(np.uint8) * 255

        # 보라색 마스크 추가 (R↑ B↑ G↓)
        purple_mask = (
            (r_ch.astype(int) > 80) &
            (b.astype(int) > 80) &
            (r_ch.astype(int) - g.astype(int) > 20) &
            (b.astype(int) - g.astype(int) > 20)
        ).astype(np.uint8) * 255

        # 모폴로지로 마스크 구멍 메우기 (번진 잉크 연결)
        color_mask = cv2.bitwise_or(red_mask, purple_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)

        ys, xs = np.where(color_mask > 0)

        if len(ys) > 50:
            # 빨간 잉크 픽셀 전체를 감싸는 최소 외접원
            points = np.column_stack([xs, ys]).astype(np.float32)
            (cx, cy), r = cv2.minEnclosingCircle(points)
            cx, cy, r = int(cx), int(cy), int(r)
        else:
            # 빨간 잉크 부족 → 그레이스케일 잉크 픽셀로 폴백
            print("  [전처리] 빨간 잉크 부족 → 그레이스케일 폴백")
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255,
                                       cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            ys, xs = np.where(binary > 0)
            if len(ys) > 50:
                points = np.column_stack([xs, ys]).astype(np.float32)
                (cx, cy), r = cv2.minEnclosingCircle(points)
                cx, cy, r = int(cx), int(cy), int(r)
            else:
                # 잉크 자체가 없으면 그냥 전체 크롭 사용
                cx, cy, r = w // 2, h // 2, min(h, w) // 2

        # 원형 마스크 적용 — 원 바깥 흰색으로
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), r, 255, -1)
        white_bg = np.full_like(crop, 255)
        result = np.where(mask[:, :, None] > 0, crop, white_bg)

        # 원 bounding box로 크롭
        result = result[max(0, cy-r-pad):min(h, cy+r+pad),
                        max(0, cx-r-pad):min(w, cx+r+pad)]

        # 정사각형 패딩 후 224×224
        rh, rw = result.shape[:2]
        side = max(rh, rw)
        square = np.full((side, side, 3), 255, dtype=np.uint8)
        square[(side-rh)//2:(side-rh)//2+rh,
               (side-rw)//2:(side-rw)//2+rw] = result

        return cv2.resize(square, (224, 224), interpolation=cv2.INTER_AREA)

    def compare(self, seal_ref: np.ndarray, seal_cmp: np.ndarray) -> tuple[float, dict]:
        """
        두 도장 크롭 이미지를 비교해 (점수 0~100, 상세 dict)를 반환한다.

        Parameters
        ----------
        seal_ref : 기준 도장 (인감증명서에서 검출된 크롭)
        seal_cmp : 비교 도장 (계약서에서 검출된 크롭)

        Returns
        -------
        score   : 0~100 점수 (main.py 판정 기준에 맞춤)
        details : 각 지표별 점수 dict
        """
        ref = self._preprocess(seal_ref)
        cmp = self._preprocess(seal_cmp)

        # ── 여기 추가 (확인 후 지워도 됨) ──────────────
        import os
        os.makedirs("output_debug", exist_ok=True)
        cv2.imwrite("output_debug/ref_preprocessed.png", ref)
        cv2.imwrite("output_debug/cmp_preprocessed.png", cmp)
        print("  [디버그] 전처리 이미지 저장: output_debug/")
        # ───────────────────────────────────────────────

        # cmp_aligned = align_images(ref, cmp,
        #                     processor=self.processor,
        #                     model=self.dino_model,
        #                     device=self.device)

        cmp_aligned = align_images_sift(ref, cmp)

        cv2.imwrite("output_debug/cmp_aligned.png", cmp_aligned)

        gray_ref = _to_gray_224(ref)
        gray_cmp = _to_gray_224(cmp_aligned)

        s_dino     = score_dinov2(ref, cmp_aligned,
                                  self.processor, self.dino_model, self.device)
        s_sift     = score_sift(ref, cmp_aligned)
        s_skeleton = score_skeleton_ssim(gray_ref, gray_cmp)
        s_grid     = score_grid_ssim(gray_ref, gray_cmp)
        s_ink      = score_ink_ssim(gray_ref, gray_cmp)

        raw = (
            self.WEIGHTS["dinov2"]        * s_dino     +
            self.WEIGHTS["sift"]          * s_sift     +
            self.WEIGHTS["skeleton_ssim"] * s_skeleton +
            self.WEIGHTS["grid_ssim"]     * s_grid     +
            self.WEIGHTS["ink_ssim"]      * s_ink
        )
        score = round(raw * 100, 2)  # 0~1 → 0~100

        details = {
            "dinov2"       : round(s_dino * 100,     2),
            "sift"         : round(s_sift * 100,      2),
            "skeleton_ssim": round(s_skeleton * 100, 2),
            "grid_ssim"    : round(s_grid * 100,     2),
            "ink_ssim"     : round(s_ink * 100,      2),
        }

        print(f"\n  DINOv2        : {details['dinov2']:.1f}")
        print(f"  SIFT          : {details['sift']:.1f}")  
        print(f"  Skeleton SSIM : {details['skeleton_ssim']:.1f}")
        print(f"  Grid SSIM     : {details['grid_ssim']:.1f}")
        print(f"  잉크 SSIM     : {details['ink_ssim']:.1f}")
        print(f"  최종 점수     : {score:.1f} / 100")

        return score, details


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("사용법: python seal_comparator.py <인감증명서> <계약서> <yolo.pt> <sam.pth>")
        sys.exit(0)

    detector = SealDetector(
        yolo_model_path=sys.argv[3],
        sam_checkpoint=sys.argv[4],
        device="cpu",
    )
    comparator = SealComparator(device="cpu")

    with open(sys.argv[1], "rb") as f:
        cert_data = f.read()
    with open(sys.argv[2], "rb") as f:
        cont_data = f.read()

    cert_seal, cert_preview = detector.extract_seal(cert_data, sys.argv[1])
    cont_seal, cont_preview = detector.extract_seal(cont_data, sys.argv[2])

    if cert_seal is None or cont_seal is None:
        print("도장 검출 실패")
        sys.exit(1)

    score, details = comparator.compare(cert_seal, cont_seal)
    print(f"\n판정: {'✓ 동일 도장' if score >= comparator.threshold * 100 else '✗ 다른 도장'}")

    os.makedirs("output_compare", exist_ok=True)
    cv2.imwrite("output_compare/ref_seal.png", cert_seal)
    cv2.imwrite("output_compare/cmp_seal.png", cont_seal)
    print("크롭 이미지 저장 완료: output_compare/")