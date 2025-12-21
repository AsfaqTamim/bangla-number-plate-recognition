# Bangla Number Plate Recognition with Vehicle Types

**Bangla number plate recognition with vehicle types. You can use both video or image. If your image quality is good then it will give you 100% accurate numbers.**

A powerful Python-based system for detecting and recognizing Bangladeshi license plates with automatic vehicle type classification using YOLOv8 and EasyOCR.

## Features

- 🚗 **License Plate Detection**: Uses fine-tuned YOLOv8 model
- � **Vehicle Detection**: YOLOv8n for Car, Bus, Truck, Motorcycle detection
- 🔤 **Bengali Text Recognition**: Custom EasyOCR model for Bengali characters
- 🎯 **Smart Text Extraction**: Auto-corrects area names using difflib
- ⚡ **Simple & Fast**: Just run one command to process all images
- 🎥 **Video Processing**: Annotated output with real-time vehicle & plate detection
- 💻 **Cross-Platform**: Works on Windows, macOS, and Linux

## Installation

### 1. Clone or navigate to the project

```bash
cd bengali-alpr
```

### 2. Create a virtual environment

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the models

**macOS/Linux:**
```bash
gdown 1ujSYC3tEC3VNoxqUIWO2PG7Lg5HAdYqn
unzip models.zip
rm models.zip
```

**Windows:**
```bash
gdown 1ujSYC3tEC3VNoxqUIWO2PG7Lg5HAdYqn
tar -xf models.zip
del models.zip
```

## Usage

### Process Images

Simply run the script to process all images in the `images/` folder:

**macOS/Linux:**
```bash
python3 main.py
```

**Windows:**
```bash
python main.py
```

### Process Video

1. Place your video file in the `video/` folder (e.g., `video/input.mp4`)
2. Update the `VIDEO_PATH` in `video.py` if needed
3. Run the video processing script:

**macOS/Linux:**
```bash
python3 video.py
```

**Windows:**
```bash
python video.py
```

The script will:
- Process every 5th frame by default (configurable via `FRAME_SKIP`)
- Detect vehicles using YOLOv8n (Car, Bus, Truck, Motorcycle)
- Detect license plates in each processed frame
- Save results to `video_results.json` with frame number and timestamp
- Create annotated video at `video/output.mp4`

That's it! The script will automatically:
- Load the YOLO model
- Find all images in the `images/` folder (for main.py)
- Process video frames (for video.py)
- Detect and extract license plate text
- Display and save the results

## Example Output

```
Loading model...
Found 6 images

Processing: dataset-cover.jpg
  ✓ ঢাকা মেট্রো-ল ২১১৫৮১৭

Processing: test_image.jpg
  ✓ ঢাকা মেট্রো-গ ১৯-০৪৭৬

Processing: banglanp.jpeg
  ✓ ঢাকা মেট্রো-গ ৩১-৯৯৫৭

Processing: images (1).jpeg
  ✓ ঢাকা মেট্রো-গ ২০-০১১

Processing: images (2).jpeg
  ✓ গাজীপুর-জ -৪০২৬

Processing: images.jpeg
  ✓ ঢাকা মেট্রো-গ ১১-৭৫৩
```

## Project Structure

```
bengali-alpr/
├── main.py                  # Main script for images
├── video.py                 # Video processing script
├── extract_license_text.py  # Text extraction module
├── areas.txt                # Valid area names for correction
├── requirements.txt         # Project dependencies
├── README.md                # Documentation
├── LICENSE                  # License file
├── models/                  # Pre-trained models
│   ├── yolo.pt             # YOLOv8 license plate detector
│   └── EasyOCR/            # Custom Bengali OCR model
├── images/                  # Test images
└── video/                   # Video input folder
```

## How It Works

1. **Detection**: YOLOv8 detects the license plate region in the image
2. **Cropping**: Extracts the bounding box coordinates
3. **OCR**: EasyOCR recognizes Bengali characters from the cropped plate
4. **Extraction**: Parses area name and number using pattern matching
5. **Correction**: Uses difflib to match closest valid area name

## Technical Details

- **Detection Model**: YOLOv8 fine-tuned on Bangladeshi license plates
- **Vehicle Detection**: YOLOv8n (COCO dataset - Car, Bus, Truck, Motorcycle)
- **OCR Model**: Custom EasyOCR model (`bn_license_tps.pth`)
- **Device**: CPU mode (MPS/GPU has compatibility issues on some systems)
- **Python Version**: Tested with Python 3.11+
- **Platforms**: Windows, macOS, Linux

## License Plate Format

Bangladeshi license plates have two lines:
- **Line 1**: Area name + Vehicle class (e.g., "ঢাকা মেট্রো-গ")
- **Line 2**: 6-digit number in format `dd-dddd` (e.g., "১৯-০৪৭৬")

## Troubleshooting

### Model not found error
Download the models using the command in step 4 of installation.

### MPS/GPU errors on macOS
The project is configured to use CPU mode by default to avoid Apple Silicon GPU compatibility issues.

### Import errors
Make sure all dependencies are installed: `pip install -r requirements.txt`

### Font rendering issues on Windows
The project will automatically fallback to default fonts if system fonts are not found. For best Bengali text rendering, ensure Arial Unicode MS or similar Unicode fonts are installed.

### Video codec issues
If video output fails, the script will automatically try alternative codecs (avc1 → mp4v).

## Credits

- Detection: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- OCR: [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- Training framework: [CLOVA AI Deep Text Recognition](https://github.com/clovaai/deep-text-recognition-benchmark)

## Python Version

Tested with Python 3.11.5 and 3.14.0
