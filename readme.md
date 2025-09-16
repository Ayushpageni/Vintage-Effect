# 🎬 Vintage Video Effects App

**Author:** Ayush Pageni
**Language:** Python
**GUI Library:** Tkinter
**Dependencies:** OpenCV, PIL (Pillow), NumPy

---

## Overview

The **Vintage Video Effects App** allows users to upload a video and apply retro-style effects such as:

- Sepia tones
- Black & White
- Film grain, scratches, and vignette overlays
- Optional audio merging from the original video

Users can preview the processed video and save it in MP4 format. The app also allows controlling the FPS to achieve a retro 15 FPS look.

---

## Features

1. **Video Upload:** Select video files (`.mp4`, `.avi`, `.mov`, `.mkv`, `.wmv`) to process.
2. **Preview:** View a real-time preview of the applied vintage effect.
3. **Effect Options:**

   - Sepia
   - Black & White

4. **Vintage Effects:**

   - Film grain and noise
   - Dust and scratch overlays
   - Vignette for old camera look
   - Slight brightness and contrast adjustment

5. **Save Processed Video:**

   - Save as `.mp4`
   - Automatically merge audio from the original video if `ffmpeg` is installed

6. **Progress Tracking:** Progress bar and status messages during processing and saving.

---

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd vintage-video-app
```

2. Install dependencies:

```bash
pip install opencv-python pillow numpy
```

3. **Optional:** Install `ffmpeg` for audio merging:

- **Windows:** [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
- **macOS:** `brew install ffmpeg`
- **Linux:** `sudo apt install ffmpeg`

---

## Usage

1. Run the app:

```bash
python vintage_video_app.py
```

2. Click **📁 Upload Video** to select a video file.
3. Choose an effect (Sepia or Black & White).
4. Click **✨ Apply Vintage Effect** to start processing.
5. Preview the video in the canvas section.
6. Click **💾 Save Video** to export the processed video (MP4 format).

---

## File Structure

```
vintage-video-app/
├── vintage_video_app.py   # Main Python script
├── README.md              # Project documentation
└── requirements.txt       # Optional: Python dependencies
```

---

## Notes

- The preview is scaled to 640x360 for performance.
- Processing may take time depending on video length and resolution.
- Audio merging requires `ffmpeg` installed on your system.

---

## License

This project is created and maintained by **Ayush Pageni**. You can use it for personal and academic purposes.

---

## Contact

For questions or suggestions, contact:
**Ayush Pageni**
