#!/usr/bin/env python3
"""Bengali ALPR - Automatic License Plate Recognition"""

from pathlib import Path
import cv2
import json
from ultralytics.models import YOLO
from extract_license_text import extract_license_text

# Load YOLO model
print("Loading model...")
model = YOLO('models/yolo.pt')

# Get all images from images folder
images = sorted(Path('images').glob('*.jpg')) + \
         sorted(Path('images').glob('*.jpeg')) + \
         sorted(Path('images').glob('*.png'))

print(f"Found {len(images)} images\n")

# Store results
results = []

# Process each image
for img_path in images:
    print(f"Processing: {img_path.name}")
    
    # Load and detect
    img = cv2.imread(str(img_path))
    detection = model.predict(img, conf=0.5, verbose=False)
    
    if len(detection[0].boxes) == 0:
        print("  ✗ No plate detected\n")
        continue
    
    # Crop license plate
    bbox = detection[0].boxes.data.cpu().numpy()[0]
    x1, y1, x2, y2 = bbox[:4].astype(int)
    confidence = float(bbox[4])
    cropped = img[y1:y2, x1:x2]
    
    # Extract text
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    area, number = extract_license_text(gray)
    
    # Store result
    results.append({
        "image": img_path.name,
        "output": f"{area} {number}",
        "confidence": f"{confidence * 100:.2f}%"
    })
    
    print(f"  ✓ {area} {number}")
    print()

# Save results to JSON
with open('results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"✓ Results saved to results.json")
