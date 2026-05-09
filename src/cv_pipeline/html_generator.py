"""
HTML Generator - Layout-Aware Text Extraction
Converts OCR results (text + bounding box + style) into an HTML file
that renders text as an overlay on top of the original image background.

Features:
  - Text is positioned with CSS absolute positioning
  - Text color is extracted from the original image (dominant pixel sampling)
  - Font size is estimated from bounding box height
  - contenteditable="true" for bonus text editing feature
  - Background uses clean image (post-inpainting) if available
"""

import cv2
import numpy as np
from pathlib import Path
import logging
from jinja2 import Template

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# Style Extraction
# ─────────────────────────────────────────────────────────

def extract_text_color(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    n_samples: int = 5,
) -> str:
    """
    Extract dominant text color from bounding box area.

    Strategy:
      - Crop bbox area from image
      - Convert to grayscale
      - Sample dark pixels (text) -> find original color in BGR
      - Return color in #RRGGBB format

    Args:
        image  : Original BGR image (BEFORE inpainting)
        bbox   : (x_min, y_min, x_max, y_max)
        n_samples: Number of pixels to sample

    Returns:
        color_hex: String "#RRGGBB"
    """
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]

    # Clamp to image boundaries
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return "#000000"

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return "#000000"

    # Convert to grayscale to detect dark pixels (text)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Find darkest pixels (text candidates)
    flat_gray = gray.flatten()
    sorted_indices = np.argsort(flat_gray)  # ascending: darkest first

    # Take median of N darkest pixels
    sample_indices = sorted_indices[:max(1, len(sorted_indices) // 10)]
    sample_indices = sample_indices[:n_samples]

    crop_flat = crop.reshape(-1, 3)
    sampled_colors = crop_flat[sample_indices]  # BGR
    median_color = np.median(sampled_colors, axis=0).astype(int)

    b, g, r = int(median_color[0]), int(median_color[1]), int(median_color[2])

    # If color is too bright -> use black (for dark slide background becoming white)
    # Heuristic: if background area is dark, text is likely bright
    bg_brightness = np.mean(gray)
    if bg_brightness < 128:
        # Dark background -> text likely bright
        light_samples = sorted_indices[-n_samples:]
        light_colors = crop_flat[light_samples]
        median_light = np.median(light_colors, axis=0).astype(int)
        b, g, r = int(median_light[0]), int(median_light[1]), int(median_light[2])

    return f"#{r:02X}{g:02X}{b:02X}"


def estimate_font_size(bbox: tuple[int, int, int, int], scale: float = 0.55) -> int:
    """
    Estimate font size from bounding box height.
    Font size ~ 55% of bbox height (EasyOCR boxes have padding)

    Args:
        bbox   : (x_min, y_min, x_max, y_max) in pixels
        scale  : Conversion factor height -> font-size

    Returns:
        font_size_px: Integer font size in px
    """
    x1, y1, x2, y2 = bbox
    height = max(1, y2 - y1)
    return max(8, int(height * scale))


# ─────────────────────────────────────────────────────────
# HTML Template
# ─────────────────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="id">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{{ title }} - Layout Aware OCR</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }

    body {
      background: #f0f2f5;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
      font-family: 'Inter', sans-serif;
      overflow: hidden;
    }

    h1 {
      position: absolute;
      top: 15px;
      left: 20px;
      color: #333;
      font-size: 16px;
      font-weight: 700;
      z-index: 100;
    }

    .slide-container {
      position: relative;
      /* Dimensions will be set by JS */
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .slide-wrapper {
      position: absolute;
      top: 0; left: 0;
      width: {{ width }}px;
      height: {{ height }}px;
      background: white;
      box-shadow: 0 10px 40px rgba(0,0,0,0.15);
      border-radius: 4px;
      transform-origin: top left;
      /* Scale is handled purely by JavaScript for reliability */
    }

    .background-layer {
      position: absolute;
      top: 0; left: 0;
      width: 100%;
      height: 100%;
      object-fit: fill;
      z-index: 0;
    }

    .text-overlay {
      position: absolute;
      top: 0; left: 0;
      width: 100%;
      height: 100%;
      z-index: 1;
      pointer-events: none;
    }

    .text-block {
      position: absolute;
      cursor: text;
      pointer-events: all;
      outline: none;
      border: 2px dashed transparent;
      border-radius: 4px;
      padding: 0px 4px;
      line-height: 1.1;
      white-space: nowrap;
      overflow: visible;
      display: flex;
      align-items: center;
      transition: border-color 0.15s, background 0.15s;
    }

    .text-block:hover {
      border-color: rgba(59, 130, 246, 0.5);
      background: rgba(59, 130, 246, 0.05);
    }

    .text-block:focus, .text-block.active {
      border-color: #3b82f6;
      background: rgba(255, 255, 255, 0.9);
      box-shadow: 0 4px 12px rgba(0,0,0,0.1);
      z-index: 999;
    }

    /* Floating Toolbar UI */
    .toolbar {
      position: fixed;
      display: flex;
      align-items: center;
      gap: 5px;
      background: white;
      padding: 6px 12px;
      border-radius: 8px;
      box-shadow: 0 4px 20px rgba(0,0,0,0.15);
      border: 1px solid #e5e7eb;
      z-index: 1000;
      opacity: 0;
      pointer-events: none;
      transform: translateY(10px);
      transition: all 0.2s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }

    .toolbar.visible {
      opacity: 1;
      pointer-events: all;
      transform: translateY(0);
    }

    .toolbar select {
      padding: 6px 8px;
      border: 1px solid #e5e7eb;
      border-radius: 4px;
      font-family: 'Inter', sans-serif;
      font-size: 14px;
      outline: none;
      cursor: pointer;
    }

    .toolbar .divider {
      width: 1px;
      height: 24px;
      background: #e5e7eb;
      margin: 0 4px;
    }

    .toolbar button {
      background: transparent;
      border: none;
      border-radius: 4px;
      width: 32px;
      height: 32px;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      font-size: 16px;
      color: #374151;
      transition: background 0.1s;
    }

    .toolbar button:hover {
      background: #f3f4f6;
    }
    
    .toolbar button.active {
      background: #e0e7ff;
      color: #4f46e5;
    }

    .toolbar input[type="text"] {
      width: 40px;
      text-align: center;
      border: 1px solid #e5e7eb;
      border-radius: 4px;
      padding: 6px 0;
      font-size: 14px;
    }

  </style>
</head>
<body>
  <h1>{{ title }}</h1>

  <!-- Floating Editor Toolbar -->
  <div id="editor-toolbar" class="toolbar">
    <select id="fontFamily">
      <option value="'Inter', sans-serif">Inter</option>
      <option value="Arial, sans-serif">Arial</option>
      <option value="'Times New Roman', serif">Times New Roman</option>
    </select>
    <div class="divider"></div>
    <button id="btnMinus" title="Decrease Font Size">-</button>
    <input type="text" id="fontSizeVal" value="20" readonly>
    <button id="btnPlus" title="Increase Font Size">+</button>
    <div class="divider"></div>
    <button id="btnBold" title="Bold"><b>B</b></button>
    <button id="btnItalic" title="Italic"><i>I</i></button>
    <button id="btnUnderline" title="Underline"><u>U</u></button>
  </div>

  <div class="slide-container" id="slideContainer">
    <div class="slide-wrapper" id="slideWrapper">
      <!-- Background Layer: clean image (post-inpainting) -->
      <img class="background-layer"
           src="{{ background_src }}"
           alt="Slide Background" />

      <!-- Text Overlay Layer -->
      <div class="text-overlay">
        {% for block in text_blocks %}
        <div
          class="text-block"
          contenteditable="true"
          spellcheck="false"
          data-page="{{ block.page }}"
          data-confidence="{{ block.confidence }}"
          style="
            left: {{ block.left }}px;
            top: {{ block.top }}px;
            width: {{ block.width }}px;
            height: {{ block.height }}px;
            font-size: {{ block.font_size }}px;
            color: {{ block.color }};
            font-family: 'Inter', sans-serif;
            font-weight: {{ block.font_weight }};
          "
        >{{ block.text }}</div>
        {% endfor %}
      </div>
    </div>
  </div>

  <script>
    // 1. PERFECT SCALING SCRIPT
    const originalWidth = {{ width }};
    const originalHeight = {{ height }};
    const slideWrapper = document.getElementById('slideWrapper');
    const slideContainer = document.getElementById('slideContainer');
    let currentScale = 1;

    function resizeSlide() {
      // Scale to fit 90% of window width and 85% of window height
      const scaleX = (window.innerWidth * 0.90) / originalWidth;
      const scaleY = (window.innerHeight * 0.85) / originalHeight;
      currentScale = Math.min(scaleX, scaleY);

      slideWrapper.style.transform = `scale(${currentScale})`;
      slideContainer.style.width = `${originalWidth * currentScale}px`;
      slideContainer.style.height = `${originalHeight * currentScale}px`;
    }

    window.addEventListener('resize', resizeSlide);
    resizeSlide(); // Call immediately

    // 2. FLOATING TOOLBAR LOGIC
    const toolbar = document.getElementById('editor-toolbar');
    let activeBlock = null;

    // Formatting Buttons
    const btnBold = document.getElementById('btnBold');
    const btnItalic = document.getElementById('btnItalic');
    const btnUnderline = document.getElementById('btnUnderline');
    const btnMinus = document.getElementById('btnMinus');
    const btnPlus = document.getElementById('btnPlus');
    const fontSizeVal = document.getElementById('fontSizeVal');
    const fontFamily = document.getElementById('fontFamily');

    document.querySelectorAll('.text-block').forEach(block => {
      block.addEventListener('focus', (e) => {
        // Remove active class from previous
        if(activeBlock) activeBlock.classList.remove('active');
        activeBlock = block;
        activeBlock.classList.add('active');
        
        // Update toolbar state to match clicked block
        const style = window.getComputedStyle(block);
        
        // Parse font size and divide by currentScale to get actual raw pixels!
        const displayedFontSize = parseFloat(style.fontSize);
        const rawFontSize = Math.round(displayedFontSize / currentScale);
        fontSizeVal.value = rawFontSize;
        
        btnBold.classList.toggle('active', style.fontWeight === '700' || style.fontWeight === 'bold');
        btnItalic.classList.toggle('active', style.fontStyle === 'italic');
        btnUnderline.classList.toggle('active', style.textDecorationLine === 'underline');

        // Position toolbar slightly above the block
        const rect = block.getBoundingClientRect();
        toolbar.style.left = `${Math.max(10, rect.left)}px`;
        toolbar.style.top = `${Math.max(10, rect.top - 55)}px`;
        toolbar.classList.add('visible');
      });
      
      // Auto-grow box width if user types long text
      block.addEventListener('input', (e) => {
        block.style.width = 'auto'; // allow expansion
      });
    });

    // Hide toolbar when clicking outside
    document.addEventListener('mousedown', (e) => {
      if (!toolbar.contains(e.target) && !e.target.classList.contains('text-block')) {
        toolbar.classList.remove('visible');
        if(activeBlock) {
          activeBlock.classList.remove('active');
          activeBlock = null;
        }
      }
    });

    // Toolbar Actions
    btnBold.addEventListener('click', () => {
      if(!activeBlock) return;
      const isBold = activeBlock.style.fontWeight === 'bold';
      activeBlock.style.fontWeight = isBold ? 'normal' : 'bold';
      btnBold.classList.toggle('active', !isBold);
    });

    btnItalic.addEventListener('click', () => {
      if(!activeBlock) return;
      const isItalic = activeBlock.style.fontStyle === 'italic';
      activeBlock.style.fontStyle = isItalic ? 'normal' : 'italic';
      btnItalic.classList.toggle('active', !isItalic);
    });

    btnUnderline.addEventListener('click', () => {
      if(!activeBlock) return;
      const isUnder = activeBlock.style.textDecoration === 'underline';
      activeBlock.style.textDecoration = isUnder ? 'none' : 'underline';
      btnUnderline.classList.toggle('active', !isUnder);
    });

    function changeFontSize(delta) {
      if(!activeBlock) return;
      let currentRaw = parseInt(fontSizeVal.value);
      currentRaw += delta;
      if(currentRaw < 8) currentRaw = 8;
      fontSizeVal.value = currentRaw;
      // We set the raw font size on the block, the CSS transform handles the rest
      activeBlock.style.fontSize = `${currentRaw}px`;
    }

    btnMinus.addEventListener('click', () => changeFontSize(-2));
    btnPlus.addEventListener('click', () => changeFontSize(2));

    fontFamily.addEventListener('change', (e) => {
      if(!activeBlock) return;
      activeBlock.style.fontFamily = e.target.value;
    });

  </script>
</body>
</html>
"""


# ─────────────────────────────────────────────────────────
# Main Generator
# ─────────────────────────────────────────────────────────

def generate_html(
    original_image_path: str,
    ocr_results: list,
    output_path: str,
    clean_background_path: str | None = None,
    min_confidence: float = 0.3,
) -> str:
    """
    Generate HTML file with text overlay on top of background image.

    Args:
        original_image_path   : Original image path (for style extraction)
        ocr_results           : EasyOCR output [(points, text, confidence), ...]
        output_path           : Output HTML file path
        clean_background_path : Post-inpainting image path (if any)
        min_confidence        : Filter OCR with confidence below threshold

    Returns:
        output_path: Generated HTML file path
    """
    original_path = Path(original_image_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load original image for style extraction
    original_image = cv2.imread(str(original_path))
    if original_image is None:
        raise FileNotFoundError(f"Image not found: {original_path}")

    h, w = original_image.shape[:2]

    # Use clean background if available, otherwise use original
    bg_path = Path(clean_background_path) if clean_background_path else original_path
    # Create relative path from HTML output to background image
    try:
        bg_rel = bg_path.relative_to(output_path.parent)
        background_src = str(bg_rel).replace("\\", "/")
    except ValueError:
        background_src = str(bg_path).replace("\\", "/")

    # Build text blocks dari OCR results
    text_blocks = []
    for item in ocr_results:
        points, text, confidence = item

        if confidence < min_confidence:
            continue
        if not text.strip():
            continue

        text_lower = text.lower().strip()
        # Heuristic: exclude logo from HTML overlay so it remains on the background
        if text_lower in ["pis", "ofpis", "qfpis", "qfois"] or ("pis" in text_lower and len(text_lower) <= 7):
            logger.info(f"Skipping logo text from HTML: '{text}'")
            continue

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x1, y1 = int(min(xs)), int(min(ys))
        x2, y2 = int(max(xs)), int(max(ys))

        bbox = (x1, y1, x2, y2)
        color = extract_text_color(original_image, bbox)
        font_size = estimate_font_size(bbox)

        # Font weight heuristic: large text -> bold
        font_weight = "bold" if font_size >= 20 else "normal"

        text_blocks.append({
            "text": text,
            "left": x1,
            "top": y1,
            "width": max(10, x2 - x1),
            "height": max(10, y2 - y1),
            "font_size": font_size,
            "color": color,
            "font_weight": font_weight,
            "confidence": round(confidence, 2),
            "page": original_path.stem,
        })

    logger.info(f"Generating HTML with {len(text_blocks)} text blocks")

    # Render template
    template = Template(HTML_TEMPLATE)
    html_content = template.render(
        title=original_path.stem,
        width=w,
        height=h,
        background_src=background_src,
        text_blocks=text_blocks,
    )

    output_path.write_text(html_content, encoding="utf-8")
    logger.info(f"HTML saved: {output_path}")

    return str(output_path)
