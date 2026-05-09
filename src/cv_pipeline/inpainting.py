"""
Inpainting Module - Remove text from presentation slides using OpenCV.
Used as pre-processing before HTML generation to ensure
clean background (no duplicated text).

Approach:
  1. Create binary mask from text bounding boxes
  2. Dilate mask to fully cover text area
  3. Use cv2.INPAINT_TELEA to fill masked area with surrounding content
"""

import cv2
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def create_text_mask(
    image: np.ndarray,
    bboxes: list[tuple[int, int, int, int]],
    dilate_px: int = 4,
) -> np.ndarray:
    """
    Create binary mask (white = text area, black = background).

    Args:
        image   : Original image (H x W x C)
        bboxes  : List of (x_min, y_min, x_max, y_max)
        dilate_px: Padding around bbox (pixels)

    Returns:
        mask: uint8 array (H x W), value 0 or 255
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for (x1, y1, x2, y2) in bboxes:
        # Clamp to image boundaries
        x1 = max(0, x1 - dilate_px)
        y1 = max(0, y1 - dilate_px)
        x2 = min(w, x2 + dilate_px)
        y2 = min(h, y2 + dilate_px)
        mask[y1:y2, x1:x2] = 255

    # Slight dilation with kernel for smooth edges
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def inpaint_image(
    image: np.ndarray,
    mask: np.ndarray,
    radius: int = 5,
) -> np.ndarray:
    """
    Remove text from image using TELEA inpainting algorithm.

    Args:
        image  : Original BGR image
        mask   : Binary mask (255 = area to be removed)
        radius : Neighborhood radius for inpainting

    Returns:
        clean_image: BGR image with text removed
    """
    return cv2.inpaint(image, mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)


def remove_text_from_image(
    image_path: str,
    bboxes: list[tuple[int, int, int, int]],
    output_path: str | None = None,
    dilate_px: int = 4,
) -> tuple[np.ndarray, str]:
    """
    Complete pipeline: load image -> create mask -> inpaint -> save.

    Args:
        image_path  : Path to input image
        bboxes      : List of (x_min, y_min, x_max, y_max) from OCR
        output_path : Path to save result (auto-generate if None)
        dilate_px   : bbox padding before masking

    Returns:
        (clean_image, output_path): numpy array + saved file path
    """
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not bboxes:
        logger.warning("No bbox found - image returned without changes")
        clean = image.copy()
    else:
        mask = create_text_mask(image, bboxes, dilate_px=dilate_px)
        clean = inpaint_image(image, mask)
        logger.info(f"Inpainting finished: {len(bboxes)} areas removed")

    # Determine output path
    if output_path is None:
        stem = image_path.stem
        output_path = str(image_path.parent / f"{stem}_clean{image_path.suffix}")

    cv2.imwrite(output_path, clean)
    logger.info(f"Clean background saved: {output_path}")

    return clean, output_path


def easyocr_to_bboxes(ocr_results: list) -> list[tuple[int, int, int, int]]:
    """
    Convert EasyOCR results to (x_min, y_min, x_max, y_max) format.

    EasyOCR output format:
        [([[x1,y1], [x2,y1], [x2,y2], [x1,y2]], text, confidence), ...]
    """
    bboxes = []
    for (points, text, confidence) in ocr_results:
        text_lower = text.lower().strip()
        # Heuristic: exclude logo from inpainting so it remains on the background
        if text_lower in ["pis", "ofpis", "qfpis", "qfois"] or ("pis" in text_lower and len(text_lower) <= 7):
            logger.info(f"Skipping logo text from inpainting: '{text}'")
            continue

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        bboxes.append((int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))))
    return bboxes
