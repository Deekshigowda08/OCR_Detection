import os
import random
import string
import cv2
import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont

# ---------------- CONFIG ----------------
TOTAL_IMAGES = 10000
BLUR_COUNT = random.randint(200, 500)
NOISE_COUNT = random.randint(600, 800)
ROTATION_COUNT = random.randint(200, 500)

IMG_DIR = "dataset/images"
LBL_DIR = "dataset/labels"
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LBL_DIR, exist_ok=True)

FONT_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"  # Mac path
IMG_SIZE = (800, 200)

CLASSES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CLASS_TO_IDX = {ch: i for i, ch in enumerate(CLASSES)}

# ----------------------------------------

def generate_text():
    length = random.randint(10, 15)
    mode = random.choice(["num", "alnum"])

    if mode == "num":
        chars = string.digits
    else:
        chars = string.ascii_uppercase + string.digits

    return ''.join(random.choices(chars, k=length))


def render_text(text):
    # Industrial style dark background
    bg_color = (random.randint(20, 50), random.randint(20, 50), random.randint(20, 50))
    img = Image.new("RGB", IMG_SIZE, bg_color)
    draw = ImageDraw.Draw(img)

    font_scale = random.randint(60, 90)
    try:
        font = ImageFont.truetype(FONT_PATH, font_scale)
    except IOError:
        font = ImageFont.load_default() # fail-safe

    boxes = []
    
    # Starting coordinates
    x = random.randint(20, 100)
    y = random.randint(IMG_SIZE[1]//2 - font_scale//2 - 20, IMG_SIZE[1]//2 - font_scale//2 + 20)
    
    # Text color (light coloring for industrial style)
    text_color = (random.randint(180, 255), random.randint(180, 255), random.randint(180, 255))

    for ch in text:
        # Get bounding box for character
        bbox = draw.textbbox((x, y), ch, font=font)
        left, top, right, bottom = bbox
        
        # draw text
        draw.text((x, y), ch, fill=text_color, font=font)
        
        # Store box: char, x_min, y_min, x_max, y_max
        boxes.append((ch, left, top, right, bottom))
        
        # increment x (adding some random spacing)
        x_advance = right - left
        x += x_advance + random.randint(2, 8)
        
        if x > IMG_SIZE[0] - 50: # boundary check
            break

    # Convert to OpenCV format BGR
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img_bgr, boxes


def transform_boxes(boxes, M, width, height):
    new_boxes = []
    for ch, x_min, y_min, x_max, y_max in boxes:
        # Corner points: 3D homogeneous coordinates
        pts = np.array([
            [x_min, y_min, 1],
            [x_max, y_min, 1],
            [x_max, y_max, 1],
            [x_min, y_max, 1]
        ], dtype=np.float32)
        
        # apply transformation
        new_pts = M.dot(pts.T).T
        
        # Get new min/max
        nx_min = np.min(new_pts[:, 0])
        nx_max = np.max(new_pts[:, 0])
        ny_min = np.min(new_pts[:, 1])
        ny_max = np.max(new_pts[:, 1])
        
        # clip to boundary
        nx_min = max(0, min(nx_min, width - 1))
        nx_max = max(0, min(nx_max, width - 1))
        ny_min = max(0, min(ny_min, height - 1))
        ny_max = max(0, min(ny_max, height - 1))
        
        if nx_max > nx_min and ny_max > ny_min:
            new_boxes.append((ch, nx_min, ny_min, nx_max, ny_max))
            
    return new_boxes


# ----------- AUGMENTATIONS ---------------

def add_blur(img):
    k = random.choice([5, 7, 9])
    return cv2.GaussianBlur(img, (k, k), 0)

def motion_blur(img):
    size = random.choice([9, 11, 15, 19])
    kernel = np.zeros((size, size))
    kernel[int((size-1)/2), :] = np.ones(size)
    kernel = kernel / size
    return cv2.filter2D(img, -1, kernel)

def add_noise(img):
    noise = np.random.normal(0, random.randint(30, 70), img.shape).astype(np.float32)
    img_float = img.astype(np.float32) + noise
    return np.clip(img_float, 0, 255).astype(np.uint8)

def low_contrast(img):
    alpha = random.uniform(0.5, 0.8)
    beta = random.randint(-40, -10)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def add_occlusion(img):
    h, w = img.shape[:2]
    # small rectangles that don't cover whole letters completely ideally
    for _ in range(random.randint(1,4)):
        x1 = random.randint(0, w-20)
        y1 = random.randint(0, h-10)
        x2 = x1 + random.randint(5, 20)
        y2 = y1 + random.randint(5, 20)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,0), -1)
    return img

def rotate(img, boxes):
    angle = random.uniform(-30, 30)
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    
    # warpAffine expects white border or constant border for background
    bg_val = int(np.mean(img)) 
    rotated_img = cv2.warpAffine(img, M, (w, h), borderValue=(bg_val, bg_val, bg_val))
    
    new_boxes = transform_boxes(boxes, M, w, h)
    return rotated_img, new_boxes

def apply_augmentations(img, boxes):
    if random.random() < 0.5:
        img = add_blur(img)

    if random.random() < 0.4:
        img = motion_blur(img)

    if random.random() < 0.6:
        img = add_noise(img)

    if random.random() < 0.5:
        img = low_contrast(img)

    if random.random() < 0.4:
        img = add_occlusion(img)

    if random.random() < 0.8: # high chance to rotate since it's common
        img, boxes = rotate(img, boxes)

    return img, boxes


# ----------- LABEL WRITING ---------------

def write_yolo_labels(boxes, label_path, width, height):
    with open(label_path, "w") as f:
        for ch, x_min, y_min, x_max, y_max in boxes:
            if ch not in CLASS_TO_IDX:
                continue
                
            class_id = CLASS_TO_IDX[ch]
            
            # center coordinates and normalized
            x_center = (x_min + x_max) / 2.0 / width
            y_center = (y_min + y_max) / 2.0 / height
            w = (x_max - x_min) / width
            h = (y_max - y_min) / height
            
            # avoid out of bounds mathematically
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))
            
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")


# ----------- MAIN GENERATION -------------

def main():
    blur_indices = set(random.sample(range(TOTAL_IMAGES), BLUR_COUNT))
    noise_indices = set(random.sample(range(TOTAL_IMAGES), NOISE_COUNT))
    rot_indices = set(random.sample(range(TOTAL_IMAGES), ROTATION_COUNT))

    for i in range(TOTAL_IMAGES):
        text = generate_text()
        img, boxes = render_text(text)

        if i in blur_indices:
            if random.random() < 0.5:
                img = add_blur(img)
            else:
                img = motion_blur(img)
        
        if i in noise_indices:
            img = add_noise(img)
            
        if i in rot_indices:
            img, boxes = rotate(img, boxes)

        filename = f"image_{i+1:04d}"
        img_path = os.path.join(IMG_DIR, f"{filename}.png")
        lbl_path = os.path.join(LBL_DIR, f"{filename}.txt")
        
        cv2.imwrite(img_path, img)
        write_yolo_labels(boxes, lbl_path, IMG_SIZE[0], IMG_SIZE[1])

        if (i+1) % 500 == 0:
            print(f"Generated {i+1} images...")

    print(f"\nDone: {TOTAL_IMAGES} images generated")
    print(f"Blur: {BLUR_COUNT}, Noise: {NOISE_COUNT}, Rotation: {ROTATION_COUNT}")


if __name__ == "__main__":
    main()