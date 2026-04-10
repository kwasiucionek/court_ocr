"""Preprocessing zdjęć dokumentów sądowych.

Obsługuje: korekcję perspektywy, deskew, denoising, binaryzację,
poprawę kontrastu — kluczowe dla zdjęć z telefonu.
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from dataclasses import dataclass

from config import PREPROCESS_TARGET_DPI, PREPROCESS_MAX_DIMENSION


@dataclass
class PreprocessResult:
    """Wynik preprocessingu."""
    image: np.ndarray
    original_size: tuple[int, int]
    processed_size: tuple[int, int]
    corrections_applied: list[str]


def preprocess_image(
    image: np.ndarray | Image.Image | str | Path,
    auto_deskew: bool = True,
    auto_denoise: bool = True,
    auto_contrast: bool = True,
    auto_perspective: bool = True,
) -> PreprocessResult:
    """Główna funkcja preprocessingu — przyjmuje obraz w dowolnym formacie."""

    img = _load_image(image)
    original_size = (img.shape[1], img.shape[0])
    corrections = []

    # 1. Resize jeśli za duży
    img = _resize_if_needed(img)

    # 2. Korekcja perspektywy (dla zdjęć z telefonu)
    if auto_perspective:
        img_corrected, was_corrected = _correct_perspective(img)
        if was_corrected:
            img = img_corrected
            corrections.append("perspective_correction")

    # 3. Deskew (korekcja nachylenia)
    if auto_deskew:
        img_deskewed, angle = _deskew(img)
        if abs(angle) > 0.3:
            img = img_deskewed
            corrections.append(f"deskew_{angle:.1f}deg")

    # 4. Denoising
    if auto_denoise:
        img = _denoise(img)
        corrections.append("denoise")

    # 5. Poprawa kontrastu (CLAHE)
    if auto_contrast:
        img = _enhance_contrast(img)
        corrections.append("contrast_enhancement")

    processed_size = (img.shape[1], img.shape[0])

    return PreprocessResult(
        image=img,
        original_size=original_size,
        processed_size=processed_size,
        corrections_applied=corrections,
    )


def _load_image(image) -> np.ndarray:
    """Wczytaj obraz z różnych źródeł."""
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            raise ValueError(f"Nie można wczytać obrazu: {image}")
        return img
    elif isinstance(image, Image.Image):
        img_array = np.array(image)
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
            return cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        return img_array
    elif isinstance(image, np.ndarray):
        return image.copy()
    else:
        raise TypeError(f"Nieobsługiwany typ obrazu: {type(image)}")


def _resize_if_needed(img: np.ndarray) -> np.ndarray:
    """Zmniejsz obraz jeśli przekracza maksymalny wymiar."""
    h, w = img.shape[:2]
    max_dim = max(h, w)
    if max_dim > PREPROCESS_MAX_DIMENSION:
        scale = PREPROCESS_MAX_DIMENSION / max_dim
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    return img


def _correct_perspective(img: np.ndarray) -> tuple[np.ndarray, bool]:
    """Korekcja perspektywy — wykrywa kontury dokumentu na zdjęciu."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edged = cv2.dilate(edged, kernel, iterations=2)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return img, False

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours[:5]:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:
            area = cv2.contourArea(approx)
            img_area = img.shape[0] * img.shape[1]

            if area < img_area * 0.2:
                continue

            pts = approx.reshape(4, 2).astype(np.float32)
            rect = _order_points(pts)
            warped = _four_point_transform(img, rect)
            return warped, True

    return img, False


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Uporządkuj 4 punkty: TL, TR, BR, BL."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def _four_point_transform(img: np.ndarray, rect: np.ndarray) -> np.ndarray:
    """Transformacja perspektywy na podstawie 4 punktów."""
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (max_width, max_height))


def _deskew(img: np.ndarray) -> tuple[np.ndarray, float]:
    """Korekcja nachylenia tekstu.

    Używa dwóch metod — najpierw Hough Lines (odporniejszy na zabrudzenia
    z kserokopii), fallback na minAreaRect.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    angle = _deskew_angle_via_hough(gray)
    if angle is None:
        angle = _deskew_angle_via_minarearect(gray)
    if angle is None or abs(angle) < 0.3:
        return img, 0.0
    if abs(angle) > 20:
        return img, 0.0

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, angle


def _deskew_angle_via_hough(gray: np.ndarray) -> float | None:
    """Szacowanie kąta przez wykrycie linii poziomych (Hough).

    Odporniejszy na szum i zabrudzenia niż minAreaRect, bo operuje
    na długich liniach tekstu a nie na wszystkich ciemnych pikselach.
    """
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=gray.shape[1] // 4,
        maxLineGap=20,
    )
    if lines is None:
        return None

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) <= 15:
            angles.append(angle)

    if not angles:
        return None

    median_angle = float(np.median(angles))
    filtered = [a for a in angles if abs(a - median_angle) < 3]
    return -float(np.mean(filtered)) if filtered else None


def _deskew_angle_via_minarearect(gray: np.ndarray) -> float | None:
    """Szacowanie kąta przez minAreaRect na tekstowych pikselach (fallback)."""
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 100:
        return None
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    return angle


def _denoise(img: np.ndarray) -> np.ndarray:
    """Usuwanie szumu — fastNlMeans dla zdjęć z telefonu i kopii ksero."""
    if len(img.shape) == 3:
        return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    return cv2.fastNlMeansDenoising(img, None, 10, 7, 21)


def _enhance_contrast(img: np.ndarray) -> np.ndarray:
    """Poprawa kontrastu przez CLAHE."""
    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        merged = cv2.merge([l_channel, a, b])
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img)
