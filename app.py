import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
import os
import threading
from PIL import Image, ImageTk
import tempfile
import random
import subprocess

class VintageVideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vintage Video Effects")
        self.root.geometry("900x700")
        self.root.configure(bg="#2c3e50")

        self.video_path = None
        self.processed_frames = []  # list of BGR uint8 frames
        self.preview_frames = []    # smaller RGB frames for preview
        self.preview_index = 0
        self.preview_running = False
        self.target_fps = 15  # retro fps
        self.temp_dir = tempfile.mkdtemp()

        self.setup_ui()

    def setup_ui(self):
        # Title
        title_label = tk.Label(
            self.root,
            text="🎬 Vintage Video Effects",
            font=("Arial", 24, "bold"),
            bg="#2c3e50",
            fg="#ecf0f1"
        )
        title_label.pack(pady=12)

        # Upload section
        upload_frame = tk.Frame(self.root, bg="#2c3e50")
        upload_frame.pack(pady=6)

        self.upload_btn = tk.Button(
            upload_frame,
            text="📁 Upload Video",
            font=("Arial", 14, "bold"),
            bg="#3498db",
            fg="white",
            padx=24,
            pady=8,
            command=self.upload_video,
            cursor="hand2"
        )
        self.upload_btn.pack(side="left")

        self.file_label = tk.Label(
            upload_frame,
            text="No video selected",
            font=("Arial", 10),
            bg="#2c3e50",
            fg="#95a5a6",
            padx=12
        )
        self.file_label.pack(side="left")

        # Preview section
        preview_frame = tk.Frame(self.root, bg="#34495e", relief="sunken", bd=2)
        preview_frame.pack(pady=12, padx=20, fill="both", expand=True)

        preview_title = tk.Label(
            preview_frame,
            text="Preview",
            font=("Arial", 16, "bold"),
            bg="#34495e",
            fg="#ecf0f1"
        )
        preview_title.pack(pady=8)

        # Canvas size chosen to nicely fit videos (will scale)
        self.preview_canvas = tk.Canvas(
            preview_frame,
            bg="#2c3e50",
            width=640,
            height=360,
            highlightthickness=0
        )
        self.preview_canvas.pack(pady=8)

        # Effects options
        options_frame = tk.Frame(self.root, bg="#2c3e50")
        options_frame.pack(pady=6)

        tk.Label(
            options_frame,
            text="Effect Style:",
            font=("Arial", 12),
            bg="#2c3e50",
            fg="#ecf0f1"
        ).pack(side="left", padx=(0, 10))

        self.effect_var = tk.StringVar(value="sepia")
        effect_options = [("Sepia", "sepia"), ("Black & White", "bw")]
        for text, value in effect_options:
            tk.Radiobutton(
                options_frame,
                text=text,
                variable=self.effect_var,
                value=value,
                bg="#2c3e50",
                fg="#ecf0f1",
                selectcolor="#34495e",
                font=("Arial", 10)
            ).pack(side="left", padx=8)

        # Control buttons
        control_frame = tk.Frame(self.root, bg="#2c3e50")
        control_frame.pack(pady=12)

        self.process_btn = tk.Button(
            control_frame,
            text="✨ Apply Vintage Effect",
            font=("Arial", 14, "bold"),
            bg="#e74c3c",
            fg="white",
            padx=24,
            pady=8,
            command=self.process_video,
            cursor="hand2",
            state="disabled"
        )
        self.process_btn.pack(side="left", padx=8)

        self.save_btn = tk.Button(
            control_frame,
            text="💾 Save Video",
            font=("Arial", 14, "bold"),
            bg="#27ae60",
            fg="white",
            padx=24,
            pady=8,
            command=self.save_video,
            cursor="hand2",
            state="disabled"
        )
        self.save_btn.pack(side="left", padx=8)

        # Progress bar & status
        self.progress = ttk.Progressbar(
            self.root,
            length=560,
            mode='determinate'
        )
        self.progress.pack(pady=10)

        self.status_label = tk.Label(
            self.root,
            text="Ready",
            font=("Arial", 10),
            bg="#2c3e50",
            fg="#95a5a6"
        )
        self.status_label.pack(pady=(0, 12))

    def upload_video(self):
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.video_path = file_path
            filename = os.path.basename(file_path)
            self.file_label.config(text=f"Selected: {filename}")
            self.process_btn.config(state="normal")
            # show first frame preview
            self.load_preview()

    def load_preview(self):
        if not self.video_path:
            return
        try:
            cap = cv2.VideoCapture(self.video_path)
            ret, frame = cap.read()
            if ret:
                # Resize to canvas size while preserving aspect
                canvas_w = int(self.preview_canvas['width'])
                canvas_h = int(self.preview_canvas['height'])
                frame = self._fit_frame_to_canvas(frame, canvas_w, canvas_h)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(pil_image)
                self.preview_canvas.delete("all")
                self.preview_canvas.create_image(canvas_w//2, canvas_h//2, image=photo)
                self.preview_canvas.image = photo
            cap.release()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load preview: {str(e)}")

    def _fit_frame_to_canvas(self, frame, canvas_w, canvas_h):
        h, w = frame.shape[:2]
        scale = min(canvas_w / w, canvas_h / h)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        # create black background and center
        bg = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        x = (canvas_w - new_w)//2
        y = (canvas_h - new_h)//2
        bg[y:y+new_h, x:x+new_w] = resized
        return bg

    # --- Effects helpers (grain, dust, vignette) ---
    def create_noise_texture(self, width, height, intensity=0.12):
        noise = np.random.randn(height, width).astype(np.float32) * 255 * intensity
        noise = noise.astype(np.int16)  # allow negative before adding
        return noise

    def create_scratch_texture(self, width, height):
        texture = np.zeros((height, width), dtype=np.uint8)
        for _ in range(random.randint(5, 15)):
            cx = random.randint(0, width-1)
            cy = random.randint(0, height-1)
            r = random.randint(1, 3)
            intensity = random.randint(60, 150)
            cv2.circle(texture, (cx, cy), r, intensity, -1)
        return texture

    def create_vignette(self, width, height):
        center_x, center_y = width // 2, height // 2
        max_distance = np.sqrt(center_x**2 + center_y**2)
        y, x = np.ogrid[:height, :width]
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        normalized_distance = distance / max_distance
        vignette = 1 - (normalized_distance * 0.7)
        vignette = np.clip(vignette, 0.25, 1)
        return vignette

    def apply_vintage_effect(self, frame, effect_type="sepia"):
        height, width = frame.shape[:2]
        # Convert to grayscale base
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if effect_type == "sepia":
            # Convert original frame to sepia using kernel
            sepia_kernel = np.array([[0.272, 0.534, 0.131],
                                     [0.349, 0.686, 0.168],
                                     [0.393, 0.769, 0.189]])
            sepia = cv2.transform(frame, sepia_kernel)
            sepia = np.clip(sepia, 0, 255).astype(np.uint8)
            base = sepia
        else:
            base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Add film grain (random noise)
        noise = self.create_noise_texture(width, height, intensity=0.12)
        # Add noise per channel
        noisy = base.astype(np.int16)
        for c in range(3):
            noisy[:, :, c] = np.clip(noisy[:, :, c].astype(np.int16) + noise, 0, 255)
        noisy = noisy.astype(np.uint8)

        # Occasionally overlay dust spots (reduced frequency)
        if random.random() < 0.35:
            dust = self.create_scratch_texture(width, height)
            for c in range(3):
                noisy[:, :, c] = cv2.addWeighted(noisy[:, :, c], 0.92, dust, 0.08, 0)

        # Vignette
        vign = self.create_vignette(width, height)
        for c in range(3):
            noisy[:, :, c] = (noisy[:, :, c].astype(np.float32) * vign).astype(np.uint8)

        # Slight contrast/brightness tweak
        final = cv2.convertScaleAbs(noisy, alpha=0.95, beta=8)

        return final

    # --- Processing pipeline ---
    def process_video(self):
        if not self.video_path:
            messagebox.showwarning("Warning", "Please select a video first!")
            return

        self.process_btn.config(state="disabled")
        self.save_btn.config(state="disabled")
        self.status_label.config(text="Processing video...")
        self.progress.config(value=0, maximum=100)
        thread = threading.Thread(target=self._process_video_thread, daemon=True)
        thread.start()

    def _process_video_thread(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise RuntimeError("Could not open video file.")

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0 or np.isnan(fps):
                fps = 30.0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Force retro fps (15)
            target_fps = int(self.target_fps)
            frame_skip = max(1, int(round(fps / target_fps)))

            processed_frames = []
            preview_frames = []
            frame_idx = 0
            processed_count = 0

            # We'll iterate through frames and skip as needed
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_skip == 0:
                    # Resize to 75% to speed up processing (like your original)
                    h, w = frame.shape[:2]
                    new_w = max(1, int(w * 0.75))
                    new_h = max(1, int(h * 0.75))
                    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                    vintage = self.apply_vintage_effect(resized, self.effect_var.get())
                    processed_frames.append(vintage)
                    # prepare a smaller preview copy (RGB)
                    preview_rgb = cv2.cvtColor(cv2.resize(vintage, (640, 360), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                    preview_frames.append(preview_rgb)
                    processed_count += 1

                frame_idx += 1

                # update progress every few frames
                if frame_idx % 10 == 0 or frame_idx == frame_count:
                    progress_pct = min(100, int((frame_idx / max(1, frame_count)) * 100))
                    self.root.after(0, lambda p=progress_pct: self.progress.config(value=p))
                    self.root.after(0, lambda s=f"Processing frame {frame_idx}/{frame_count}": self.status_label.config(text=s))

            cap.release()

            self.processed_frames = processed_frames
            self.preview_frames = preview_frames
            self.preview_index = 0

            if self.preview_frames:
                # start preview loop
                self.preview_running = True
                self._start_preview_loop()
            else:
                self.root.after(0, lambda: self.preview_canvas.delete("all"))

            self.root.after(0, lambda: self.process_btn.config(state="normal"))
            self.root.after(0, lambda: self.save_btn.config(state="normal"))
            self.root.after(0, lambda: self.status_label.config(text="Processing complete"))
            self.root.after(0, lambda: self.progress.config(value=100))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Processing failed: {e}"))
            self.root.after(0, lambda: self.process_btn.config(state="normal"))
            self.root.after(0, lambda: self.status_label.config(text="Processing failed"))

    # --- Preview playback loop ---
    def _start_preview_loop(self):
        if not self.preview_frames:
            return
        # show current preview frame
        frame_rgb = self.preview_frames[self.preview_index]
        pil = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(pil)
        self.preview_canvas.delete("all")
        cw = int(self.preview_canvas['width'])
        ch = int(self.preview_canvas['height'])
        # center it
        self.preview_canvas.create_image(cw//2, ch//2, image=photo)
        self.preview_canvas.image = photo
        # advance
        self.preview_index = (self.preview_index + 1) % len(self.preview_frames)
        # schedule next frame based on target fps
        delay_ms = int(1000 / max(1, self.target_fps))
        if self.preview_running:
            self.root.after(delay_ms, self._start_preview_loop)

    # --- Saving ---
    def save_video(self):
        if not self.processed_frames:
            messagebox.showwarning("Warning", "No processed video to save!")
            return

        save_path = filedialog.asksaveasfilename(
            title="Save Vintage Video",
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
        )
        if not save_path:
            return

        # Disable save button while writing
        self.save_btn.config(state="disabled")
        self.status_label.config(text="Saving video...")
        thread = threading.Thread(target=self._save_video_thread, args=(save_path,), daemon=True)
        thread.start()

    def _save_video_thread(self, save_path):
        try:
            # Determine frame size from processed frames
            h, w = self.processed_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # widely supported
            out = cv2.VideoWriter(save_path, fourcc, float(self.target_fps), (w, h))

            total = len(self.processed_frames)
            for i, frame in enumerate(self.processed_frames):
                out.write(frame)
                if i % 5 == 0 or i == total-1:
                    prog = int((i / max(1, total)) * 100)
                    self.root.after(0, lambda p=prog: self.progress.config(value=p))
            out.release()

            # Try to merge audio from original using ffmpeg if available
            merged = self._try_add_audio(save_path)
            final_path = merged if merged else save_path

            self.root.after(0, lambda: messagebox.showinfo("Success", f"Video saved to:\n{final_path}"))
            self.root.after(0, lambda: self.status_label.config(text="Video saved successfully!"))
            self.root.after(0, lambda: self.save_btn.config(state="normal"))
            self.root.after(0, lambda: self.progress.config(value=0))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to save video: {e}"))
            self.root.after(0, lambda: self.save_btn.config(state="normal"))
            self.root.after(0, lambda: self.status_label.config(text="Save failed"))

    def _try_add_audio(self, video_path):
        """
        If ffmpeg exists, extract audio from original and mux into the newly saved video.
        Returns path to merged file on success, else None.
        """
        try:
            # Build temp path
            base, ext = os.path.splitext(video_path)
            merged_path = base + "_with_audio.mp4"

            # ffmpeg command: take video from new file and audio from original
            cmd = [
                "ffmpeg",
                "-y",
                "-i", video_path,
                "-i", self.video_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-shortest",
                merged_path
            ]
            # run
            completed = subprocess.run(cmd, capture_output=True, text=True)
            if completed.returncode == 0 and os.path.exists(merged_path):
                # replace video_path with merged_path
                try:
                    os.replace(merged_path, video_path)
                except Exception:
                    pass
                return video_path
            else:
                # ffmpeg failed
                return None
        except FileNotFoundError:
            # ffmpeg not installed
            return None
        except Exception:
            return None

def main():
    root = tk.Tk()
    app = VintageVideoApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
