import os
import cv2
import numpy as np
import re
from ultralytics import YOLO
from utils import preprocess_for_ocr, encode_image_base64, get_text_region_candidates

MODEL_PATH = "../runs/detect/train3/weights/best.pt"
TEXT_MODEL_PATH = "text_region_model.pt"

model = None
text_model = None

try:
    # Load model once entirely globally (persists through multiple identical requests)
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Warning: Could not load YOLO model from {MODEL_PATH}. Error: {e}")

try:
    if os.path.exists(TEXT_MODEL_PATH):
        text_model = YOLO(TEXT_MODEL_PATH)
except Exception as e:
    pass

try:
    import easyocr
    reader = easyocr.Reader(['en'], gpu=False)
except ImportError:
    reader = None
    print("EasyOCR not installed. Skipping fallback.")

try:
    from paddleocr import PaddleOCR
    try:
        paddle_reader = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    except (TypeError, ValueError):
        paddle_reader = PaddleOCR(use_angle_cls=True, lang='en')
except ImportError:
    paddle_reader = None
    print("PaddleOCR not installed. Skipping fallback.")

def normalize_ocr_text(text):
    text = re.sub(r'[^0-9a-zA-Z\-\/\#\,\s]', '', text)
    text = text.upper().strip()
    # Strip common label prefixes that might leak into the serial value
    for label in ["SERIAL NO", "SERIAL NUMBER", "SERIALNO", "SR NO", "SRNO", "SRL", "SER NO", "S/N", "S N"]:
        text = re.sub(r'(?i)^' + re.escape(label) + r'[\s\:\-]*', '', text)
    return text.strip()

def normalize_fallback_text(text):
    """Aggressive normalization for fallback OCR (PaddleOCR/EasyOCR) outputs."""
    text = re.sub(r'\s+', '', text)  # Remove ALL spaces (fallback OCR adds junk spaces)
    text = re.sub(r'[^0-9a-zA-Z\-\/\#\,]', '', text)
    return text.upper().strip()

def collapse_consecutive_duplicates(text):
    if not text:
        return text
    result = [text[0]]
    for ch in text[1:]:
        if ch != result[-1] or ch == ' ':
            result.append(ch)
    return ''.join(result)

CONFUSION_MAP = {
    'S': '5', '5': 'S',
    'B': '8', '8': 'B',
    'O': '0', '0': 'O',
    'I': '1', '1': 'I',
}

def apply_confusion_correction(text):
    if not text:
        return text
    clean = text.replace(' ', '')
    digit_count = sum(1 for c in clean if c.isdigit())
    is_mostly_numeric = digit_count > len(clean) * 0.6
    corrected = list(text)
    for i, ch in enumerate(corrected):
        if ch in CONFUSION_MAP:
            replacement = CONFUSION_MAP[ch]
            if is_mostly_numeric and ch.isalpha():
                corrected[i] = replacement
            elif not is_mostly_numeric and ch.isdigit():
                corrected[i] = replacement
    candidate = ''.join(corrected)
    orig_unique = len(set(clean)) / max(len(clean), 1)
    new_clean = candidate.replace(' ', '')
    new_unique = len(set(new_clean)) / max(len(new_clean), 1)
    if new_unique >= orig_unique:
        return candidate
    return text

def validate_output(text):
    clean = text.replace(' ', '')
    if len(clean) < 4 or len(clean) > 20:
        return False
    if len(clean) > 0 and len(set(clean)) / float(len(clean)) < 0.4:
        return False
    if len(clean) > 3 and len(set(clean)) == 1:
        return False
    return True

def generate_image_variants(img):
    variants = []
    variants.append({"image": img, "label": "original"})
    variants.append({"image": cv2.rotate(img, cv2.ROTATE_180), "label": "rotated_180"})
    enhanced = cv2.convertScaleAbs(img, alpha=1.3, beta=10)
    variants.append({"image": enhanced, "label": "contrast_enhanced"})
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    variants.append({"image": blurred, "label": "light_blur"})
    return variants

def extract_string_from_boxes(boxes_data):
    if not boxes_data:
        return "", []
    
    # Compute y_center and box height for every detection
    for bd in boxes_data:
        bd["y_center"] = (bd["box"][1] + bd["box"][3]) / 2
        bd["box_h"] = bd["box"][3] - bd["box"][1]
    
    avg_box_h = sum(bd["box_h"] for bd in boxes_data) / len(boxes_data)
    threshold = max(0.5 * avg_box_h, 8)  # Dynamic threshold, minimum 8px
    
    # Single-line safeguard: if vertical spread is tiny, treat as one line
    y_centers = [bd["y_center"] for bd in boxes_data]
    y_spread = max(y_centers) - min(y_centers)
    
    if len(boxes_data) < 20 and y_spread < threshold * 2:
        # All boxes belong to a single line
        boxes_data.sort(key=lambda x: x["box"][0])
        final_string = "".join([b["char"] for b in boxes_data])
        print(f"[Line Grouping] Single-line detected | {len(boxes_data)} chars | y-spread: {y_spread:.1f}px | threshold: {threshold:.1f}px")
        return final_string.strip(), [boxes_data]
    
    # Multi-line clustering: sort by y_center, group by mean-y proximity
    boxes_data.sort(key=lambda x: x["y_center"])
    lines = [[boxes_data[0]]]
    
    for bd in boxes_data[1:]:
        # Compare against the MEAN y_center of the current line
        line_mean_y = sum(b["y_center"] for b in lines[-1]) / len(lines[-1])
        if abs(bd["y_center"] - line_mean_y) < threshold:
            lines[-1].append(bd)
        else:
            lines.append([bd])
    
    # Fallback override: if too many lines detected, collapse into single line
    if len(lines) > 3:
        print(f"[Line Grouping] WARNING: {len(lines)} lines detected — collapsing to single line")
        all_boxes = [b for line in lines for b in line]
        all_boxes.sort(key=lambda x: x["box"][0])
        final_string = "".join([b["char"] for b in all_boxes])
        return final_string.strip(), [all_boxes]
    
    # Sort lines top→bottom by average y, then chars left→right within each line
    lines.sort(key=lambda line: sum(b["y_center"] for b in line) / len(line))
    for line in lines:
        line.sort(key=lambda x: x["box"][0])
    
    # Build final string
    line_strings = []
    for line in lines:
        line_strings.append("".join([b["char"] for b in line]))
    final_string = " ".join(line_strings)
    
    # Debug logging
    print(f"[Line Grouping] {len(lines)} line(s) | threshold: {threshold:.1f}px | y-spread: {y_spread:.1f}px")
    for i, line in enumerate(lines):
        y_vals = [b["y_center"] for b in line]
        print(f"  Line {i+1}: {len(line)} chars | y-range: [{min(y_vals):.1f}, {max(y_vals):.1f}]")
    
    return final_string.strip(), lines

def compute_final_score(text, conf):
    if not text: return 0
    clean_text = text.replace(" ", "")
    if not clean_text: return 0
    
    length = len(clean_text)
    unique_chars = len(set(clean_text))
    diversity = unique_chars / float(length)
    
    # Penalty accumulator
    penalty = 0.0
    
    # Length validation (expected serial: 6-15 chars)
    if length < 6 or length > 15:
        penalty += 2.0
    valid_length_score = 1.0 if 6 <= length <= 15 else 0.2
    
    # Repeated character penalty
    from collections import Counter
    char_counts = Counter(clean_text)
    max_char_count = max(char_counts.values())
    repeat_ratio = max_char_count / float(length)
    if repeat_ratio > 0.4:
        penalty += 2.0
    
    # Low diversity penalty
    if diversity < 0.4:
        penalty += 1.5
    
    # Excessive whitespace penalty
    if text.count(" ") > 3:
        penalty += 1.0
    
    # Pattern sanity: must have at least 1 digit AND 1 letter
    has_digit = any(c.isdigit() for c in clean_text)
    has_letter = any(c.isalpha() for c in clean_text)
    if not (has_digit and has_letter):
        penalty += 2.0
    
    score = (conf * 0.5) + (valid_length_score * 0.2) + (diversity * 0.2) - (penalty * 0.5)
    return score

def compute_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    xA = max(x1_min, x2_min)
    yA = max(y1_min, y2_min)
    xB = min(x1_max, x2_max)
    yB = min(y1_max, y2_max)

    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (x1_max - x1_min) * (y1_max - y1_min)
    box2Area = (x2_max - x2_min) * (y2_max - y2_min)
    unionArea = box1Area + box2Area - interArea
    
    if unionArea == 0:
        return 0
    return interArea / unionArea

SERIAL_KEYWORDS = ["serial", "serial no", "s/n", "sr no", "sr.no", "srl", "ser no", "ser.no"]

def detect_serial_region(img_bgr):
    """
    Use PaddleOCR or EasyOCR to scan for 'Serial No' / 'S/N' keyword.
    Returns a cropped image of the serial number value region, or None.
    """
    h, w = img_bgr.shape[:2]
    detected_texts = []
    
    # Try PaddleOCR first (more reliable on structured plates)
    if paddle_reader is not None:
        try:
            try:
                p_res = paddle_reader.ocr(img_bgr, cls=True)
            except TypeError:
                p_res = paddle_reader.ocr(img_bgr)
            if p_res and p_res[0]:
                for line in p_res[0]:
                    try:
                        bbox = line[0]  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                        text_info = line[1]
                        if isinstance(text_info, (list, tuple)):
                            text = str(text_info[0])
                        elif isinstance(text_info, str):
                            text = text_info
                        else:
                            text = str(text_info)
                        # Convert polygon to bounding rect
                        xs = [p[0] for p in bbox]
                        ys = [p[1] for p in bbox]
                        detected_texts.append({
                            "text": text,
                            "box": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
                        })
                    except (IndexError, TypeError, ValueError):
                        continue
        except Exception as e:
            print(f"PaddleOCR pre-scan failed: {e}")
    
    # Fallback to EasyOCR
    if not detected_texts and reader is not None:
        try:
            e_res = reader.readtext(img_bgr)
            for (bbox, text, conf) in e_res:
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
                detected_texts.append({
                    "text": text,
                    "box": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
                })
        except Exception as e:
            print(f"EasyOCR pre-scan failed: {e}")
    
    if not detected_texts:
        return None
    
    # Search for serial keyword
    keyword_box = None
    for dt in detected_texts:
        text_lower = dt["text"].lower().strip()
        for kw in SERIAL_KEYWORDS:
            if kw in text_lower:
                keyword_box = dt
                print(f"[Serial ROI] Found keyword '{dt['text']}' at {dt['box']}")
                break
        if keyword_box:
            break
    
    if keyword_box:
        kx1, ky1, kx2, ky2 = keyword_box["box"]
        kw_width = kx2 - kx1
        kw_height = ky2 - ky1
        
        # Expand region: keep left edge, extend right by 3x keyword width,
        # and add vertical padding to capture the value field
        pad_y = 20
        roi_x1 = max(0, kx1)
        roi_y1 = max(0, ky1 - pad_y)
        roi_x2 = min(w, kx2 + kw_width * 3)
        roi_y2 = min(h, ky2 + pad_y)
        
        # Ensure minimum viable crop
        if (roi_x2 - roi_x1) > 30 and (roi_y2 - roi_y1) > 15:
            crop = img_bgr[roi_y1:roi_y2, roi_x1:roi_x2]
            print(f"[Serial ROI] Cropped region: ({roi_x1},{roi_y1}) -> ({roi_x2},{roi_y2})")
            return crop
    
    # Fallback: return central horizontal band (middle 40% height)
    band_y1 = int(h * 0.3)
    band_y2 = int(h * 0.7)
    print(f"[Serial ROI] No keyword found. Using central band [{band_y1}:{band_y2}]")
    return img_bgr[band_y1:band_y2, :]

def run_inference(img_bgr):
    if not model:
        return {"error": "Model not loaded"}
        
    def score(result):
        boxes = result[0].boxes
        if boxes is None or len(boxes) == 0:
            return 0
        return float(boxes.conf.mean()) * len(boxes)

    def is_cropped_image(img, threshold=0.06):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = cv2.countNonZero(edges)
        total_pixels = img.shape[0] * img.shape[1]
        ratio = edge_pixels / float(total_pixels)
        print(f"Edge Density Ratio: {ratio:.4f}")
        return ratio > threshold
        
    candidates = []
    
    # Step 0: Try keyword-based serial number ROI detection first
    serial_crop = detect_serial_region(img_bgr)
    if serial_crop is not None:
        candidates.append({
            "image": serial_crop,
            "offset": (0, 0),
            "source": "serial_keyword_roi"
        })

    if is_cropped_image(img_bgr):
        print("Detected as CROPPED text image. Bypassing regional scanning...")
        candidates.append({
            "image": img_bgr,
            "offset": (0, 0),
            "source": "original_cropped_bypass"
        })
    else:
        print("Detected as FULL-OBJECT image. Running regional scanning...")
        # 1. Heuristic region candidates (Canny edge)
        candidates = get_text_region_candidates(img_bgr, top_k=3)
        
        # 2. Optional Model-based text region detection
        if text_model is not None:
            t_results = text_model(img_bgr, verbose=False)
            t_boxes = []
            for r in t_results:
                for b in r.boxes:
                    t_boxes.append({
                        "box": b.xyxy[0].tolist(),
                        "conf": float(b.conf[0])
                    })
            t_boxes.sort(key=lambda x: x["conf"], reverse=True)
            for tb in t_boxes[:2]: # top 1-2 regions by conf
                x1, y1, x2, y2 = map(int, tb["box"])
                pad = 15
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(img_bgr.shape[1], x2 + pad)
                y2 = min(img_bgr.shape[0], y2 + pad)
                candidates.append({
                    "image": img_bgr[y1:y2, x1:x2],
                    "offset": (x1, y1),
                    "source": "model_detector"
                })
                
        # 3. Always append original image as fallback safe baseline
        candidates.append({
            "image": img_bgr,
            "offset": (0, 0),
            "source": "original_fallback"
        })
    
    best_overall_score = -1
    best_overall_result = None
    best_overall_preprocessed = None
    best_overall_variant_label = None
    best_source = None
    
    for cand in candidates:
        crop_img = cand["image"]
        variants = generate_image_variants(crop_img)
        
        for var in variants:
            var_img = preprocess_for_ocr(var["image"])
            res = model(var_img, device="mps", verbose=False)
            s = score(res)
            
            if s > best_overall_score:
                best_overall_score = s
                best_overall_result = res
                best_overall_preprocessed = var_img
                best_overall_variant_label = var["label"]
                best_source = cand["source"]
            
    print(f"Winning Crop: {best_source} | Variant: {best_overall_variant_label} | Score: {best_overall_score:.3f}")
    
    best_results = best_overall_result
    preprocessed = best_overall_preprocessed
        
    # Initial detection extraction
    boxes_data = []
    for r in best_results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            char = model.names[cls_id]
            boxes_data.append({
                "box": [x1, y1, x2, y2],
                "conf": conf,
                "char": char
            })

    # Duplication filtering / Non-Max Suppression
    filtered_boxes = []
    boxes_data.sort(key=lambda x: x["conf"], reverse=True)
    
    for bd in boxes_data:
        overlap = False
        for fbd in filtered_boxes:
            if compute_iou(bd["box"], fbd["box"]) > 0.4:
                overlap = True
                break
        if not overlap:
            filtered_boxes.append(bd)
            
    # Final String construction (Multi-line heuristic)
    yolo_text, structured_lines = extract_string_from_boxes(filtered_boxes)
    yolo_text = normalize_ocr_text(yolo_text)
    yolo_text = collapse_consecutive_duplicates(yolo_text)
    yolo_text = apply_confusion_correction(yolo_text)
    
    confs = [b["conf"] for b in filtered_boxes]
    ret_boxes = [b["box"] for b in filtered_boxes]
    
    # -------------------------------------------------------------
    # RELIABILITY CHECK & ENSEMBLE FALLBACK
    # -------------------------------------------------------------
    yolo_conf = sum(confs) / len(confs) if confs else 0.0
    yolo_len = len(yolo_text.replace(" ", ""))
    yolo_unique = len(set(yolo_text.replace(" ", "")))
    
    yolo_unreliable = False
    if yolo_conf < 0.6: yolo_unreliable = True
    elif yolo_len < 4: yolo_unreliable = True
    elif yolo_len > 0 and yolo_unique == 1: yolo_unreliable = True
    elif yolo_len > 0 and (yolo_unique / float(yolo_len)) < 0.4: yolo_unreliable = True
    
    candidates_eval = []
    candidates_eval.append({
        "engine": "YOLO (Primary)",
        "text": yolo_text,
        "score": compute_final_score(yolo_text, yolo_conf),
        "valid": validate_output(yolo_text)
    })
    
    if yolo_unreliable:
        print("⚠️  YOLO prediction Unreliable. Firing Fallback Ensembles...")
        
        # EasyOCR Sweep
        if reader is not None:
            e_res = reader.readtext(preprocessed)
            e_text = " ".join([r[1] for r in e_res])
            e_text = normalize_fallback_text(e_text)
            e_text = collapse_consecutive_duplicates(e_text)
            e_text = apply_confusion_correction(e_text)
            e_conf = sum([r[2] for r in e_res]) / len(e_res) if e_res else 0.5
            e_score = compute_final_score(e_text, e_conf)
            candidates_eval.append({
                "engine": "EasyOCR",
                "text": e_text,
                "score": e_score,
                "valid": validate_output(e_text)
            })
            
        # PaddleOCR Sweep
        if paddle_reader is not None:
            try:
                p_res = paddle_reader.ocr(preprocessed, cls=True)
            except TypeError:
                p_res = paddle_reader.ocr(preprocessed)
            p_text, p_confs = "", []
            if p_res and p_res[0]:
                for line in p_res[0]:
                    try:
                        text_info = line[1]
                        if isinstance(text_info, (list, tuple)):
                            p_text += str(text_info[0]) + " "
                            p_confs.append(float(text_info[1]))
                        elif isinstance(text_info, str):
                            p_text += text_info + " "
                            p_confs.append(0.5)
                        else:
                            p_text += str(text_info) + " "
                            p_confs.append(0.5)
                    except (IndexError, TypeError, ValueError):
                        continue
            p_text = normalize_fallback_text(p_text)
            p_text = collapse_consecutive_duplicates(p_text)
            p_text = apply_confusion_correction(p_text)
            p_conf = sum(p_confs) / len(p_confs) if p_confs else 0.5
            p_score = compute_final_score(p_text, p_conf)
            candidates_eval.append({
                "engine": "PaddleOCR",
                "text": p_text,
                "score": p_score,
                "valid": validate_output(p_text)
            })
    
    # Hard rejection: remove candidates with penalty > 3.0 (score will be very negative)
    # Prefer valid candidates first, then highest score
    viable_candidates = [c for c in candidates_eval if c["score"] > -0.5]
    if not viable_candidates:
        viable_candidates = candidates_eval  # fallback: keep all if everything is bad
    
    valid_candidates = [c for c in viable_candidates if c["valid"]]
    if valid_candidates:
        best_candidate = max(valid_candidates, key=lambda x: x["score"])
    else:
        best_candidate = max(candidates_eval, key=lambda x: x["score"])
        
    print(f"\n--- MULTI-CANDIDATE RESULTS ---")
    for c in candidates_eval:
        status = "✓" if c["valid"] else "✗"
        clean = c['text'].replace(' ', '')
        div = len(set(clean)) / max(len(clean), 1)
        print(f"  [{status}] [{c['engine']}] Score: {c['score']:.2f} | Len: {len(clean)} | Div: {div:.2f} | '{c['text']}'")
    print(f"  ✅ WINNER: {best_candidate['engine']} -> {best_candidate['text']}\n")
    
    detected_string = best_candidate['text']
    
    # Graphic Generation Display Component with Line-Aware Colors
    annotated = preprocessed.copy()
    
    # Palette definition for lines: Green, Blue, Cyan, Yellow, Purple
    colors = [(0, 255, 0), (255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    
    # If YOLO string generation failed parsing structure safely, fallback flat rendering:
    if not structured_lines:
        structured_lines = [filtered_boxes]
        
    for idx, line in enumerate(structured_lines):
        line_color = colors[idx % len(colors)]
        for bd in line:
            x1, y1, x2, y2 = map(int, bd["box"])
            char = bd["char"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), line_color, 2)
            cv2.putText(annotated, char, (x1, max(y1-10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, line_color, 2)
        
    base64_out = encode_image_base64(annotated)
    preprocessed_base64 = encode_image_base64(preprocessed)
    
    return {
        "text": detected_string,
        "boxes": ret_boxes,
        "confidences": confs,
        "annotated_image": f"data:image/png;base64,{base64_out}",
        "preprocessed_image": f"data:image/png;base64,{preprocessed_base64}"
    }
