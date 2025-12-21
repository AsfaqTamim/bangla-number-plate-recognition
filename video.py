#!/usr/bin/env python3
"""Bengali ALPR - Video Processing"""

import cv2
import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from ultralytics.models import YOLO
from extract_license_text import extract_license_text

# Configuration
VIDEO_PATH = "video/input.mp4"
FRAME_SKIP = 5  # Process every 5th frame for smoother detection
OUTPUT_JSON = "video_results.json"
OUTPUT_VIDEO = "video/output.mp4"
DETECTION_PERSIST = 90  # Keep detection visible for 90 frames (~3 seconds at 30fps)

# COCO class IDs for vehicles
VEHICLE_CLASSES = {
    2: 'Car',
    3: 'Motorcycle',
    5: 'Bus',
    7: 'Truck'
}

print("Loading models...")
vehicle_model = YOLO('yolov8n.pt')  # For vehicle detection
plate_model = YOLO('models/yolo.pt')  # For license plate detection
print("Models loaded!")

def draw_bengali_text(img, text, position, font_size=30, color=(0, 255, 0)):
    """Draw Bengali text on image using PIL"""
    # Create a copy to avoid modifying original
    img_copy = img.copy()
    img_pil = Image.fromarray(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # Try different font paths for cross-platform support
    font = None
    font_paths = [
        # macOS
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        # Windows
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/Arial.ttf",
        # Linux
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    
    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, font_size)
            break
        except:
            continue
    
    if font is None:
        font = ImageFont.load_default()
    
    draw.text(position, text, font=font, fill=color)
    # Convert back with higher quality
    result = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return result

# Open video
video_path = Path(VIDEO_PATH)
if not video_path.exists():
    print(f"Error: Video file not found at {VIDEO_PATH}")
    print("Please place your video in the 'video' folder or update VIDEO_PATH")
    exit(1)

cap = cv2.VideoCapture(str(video_path))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video: {video_path.name}")
print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
print(f"Processing every {FRAME_SKIP} frames...\n")

# Setup video writer with high quality settings
fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec for better quality
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

# Check if video writer opened successfully
if not out.isOpened():
    print("Warning: H.264 codec failed, trying mp4v...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

results = []
frame_count = 0
processed = 0
current_detection = None
detection_frames_left = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Process detection every FRAME_SKIP frames
    if frame_count % FRAME_SKIP == 0:
        processed += 1
        timestamp = frame_count / fps
        
        print(f"Frame {frame_count} ({timestamp:.2f}s)", end=" ")
        
        # First detect vehicle type
        vehicle_detection = vehicle_model.predict(frame, conf=0.5, verbose=False, classes=list(VEHICLE_CLASSES.keys()))
        vehicle_type = 'Unknown'
        
        if len(vehicle_detection[0].boxes) > 0:
            # Get the most confident vehicle detection
            vehicle_box = vehicle_detection[0].boxes.data.cpu().numpy()[0]
            vehicle_class_id = int(vehicle_box[5])
            vehicle_type = VEHICLE_CLASSES.get(vehicle_class_id, 'Unknown')
        
        # Then detect license plate
        detection = plate_model.predict(frame, conf=0.5, verbose=False)
        
        if len(detection[0].boxes) > 0:
            bbox = detection[0].boxes.data.cpu().numpy()[0]
            x1, y1, x2, y2 = bbox[:4].astype(int)
            confidence = float(bbox[4])
            cropped = frame[y1:y2, x1:x2]
            
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            area, number = extract_license_text(gray)
            
            # Update current detection
            current_detection = {
                'bbox': (x1, y1, x2, y2),
                'plate': f"{area} {number}",
                'vehicle_type': vehicle_type,
                'confidence': confidence
            }
            detection_frames_left = DETECTION_PERSIST
            
            results.append({
                "frame": frame_count,
                "timestamp": f"{timestamp:.2f}s",
                "plate": f"{area} {number}",
                "vehicle_type": vehicle_type,
                "confidence": f"{confidence * 100:.2f}%"
            })
            
            print(f"✓ {area} {number} - {vehicle_type}")
        else:
            print("✗ No plate")
    
    # Draw annotations if we have an active detection
    if current_detection and detection_frames_left > 0:
        x1, y1, x2, y2 = current_detection['bbox']
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Create overlay for text background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (550, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw all text at once to minimize PIL conversions
        text_lines = [
            (current_detection['plate'], (20, 20), 32, (0, 255, 0)),
            (f"Type: {current_detection['vehicle_type']}", (20, 60), 24, (255, 255, 0)),
            (f"Conf: {current_detection['confidence']*100:.1f}%", (20, 90), 20, (255, 255, 255))
        ]
        
        # Single PIL conversion for all text
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)
        
        # Try different font paths for cross-platform support
        font_large = font_medium = font_small = None
        font_paths = [
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/Library/Fonts/Arial Unicode.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
        
        for font_path in font_paths:
            try:
                font_large = ImageFont.truetype(font_path, 32)
                font_medium = ImageFont.truetype(font_path, 24)
                font_small = ImageFont.truetype(font_path, 20)
                break
            except:
                continue
        
        if font_large is None:
            font_large = font_medium = font_small = ImageFont.load_default()
        
        fonts = [font_large, font_medium, font_small]
        
        for idx, (text, pos, size, color) in enumerate(text_lines):
            draw.text(pos, text, font=fonts[idx], fill=color)
        
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        
        detection_frames_left -= 1
    
    out.write(frame)

cap.release()
out.release()

# Save results
with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n{'='*50}")
print(f"Processing complete!")
print(f"Total frames: {total_frames}")
print(f"Processed: {processed}")
print(f"Detected: {len(results)}")
print(f"Output video: {OUTPUT_VIDEO}")
print(f"Results saved to: {OUTPUT_JSON}")
print(f"{'='*50}")
