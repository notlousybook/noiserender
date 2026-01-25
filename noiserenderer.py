import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import multiprocessing
import subprocess
import os
import time
import math
import queue
import glob

# --- HELPER: FIND FFMPEG ---
def find_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return "ffmpeg"
    except:
        if os.path.exists("ffmpeg.exe"): return os.path.abspath("ffmpeg.exe")
    return None

# --- WORKER PROCESS ---
def render_chunk(chunk_id, start_frame, end_frame, input_path, 
                 internal_w, internal_h, inverted, fps, progress_queue):
    try:
        # Stagger start
        time.sleep(chunk_id * 0.05) 

        chunk_filename = f"temp_chunk_{chunk_id:03d}.avi"
        preview_filename = f"temp_preview_{chunk_id}.jpg"
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened(): return (chunk_id, False, "Open Fail")
        
        # SEEKING SAFETY:
        # Sometimes seeking to exact frame fails on compressed MP4s.
        # We try to seek.
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # We render at INTERNAL (Low) Resolution for speed
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(chunk_filename, fourcc, fps, (internal_w, internal_h))

        # Reusable buffers (Low RAM usage)
        static_gray = np.random.randint(0, 256, (internal_h, internal_w), dtype=np.uint8)
        static_noise = cv2.merge([static_gray, static_gray, static_gray])
        active_gray = np.zeros((internal_h, internal_w), dtype=np.uint8)

        current_frame = start_frame
        
        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret: 
                # If we run out of frames early, stop writing to prevent corruption
                break

            # Resize to INTERNAL size (FAST)
            # Use INTER_NEAREST for that crispy retro noise look (and speed)
            frame_resized = cv2.resize(frame, (internal_w, internal_h), interpolation=cv2.INTER_NEAREST)
            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
            mask_bool = (mask > 0)[:, :, np.newaxis]

            # Generate Noise (Fastest Method)
            cv2.randu(active_gray, 0, 256)
            active_noise = cv2.merge([active_gray, active_gray, active_gray])

            # Composite
            if not inverted:
                final = np.where(mask_bool, active_noise, static_noise)
            else:
                final = np.where(mask_bool, static_noise, active_noise)

            out.write(final)
            
            # Preview (Resize to tiny for UI)
            if current_frame % 45 == 0:
                try:
                    thumb = cv2.resize(final, (320, 240), interpolation=cv2.INTER_NEAREST) 
                    cv2.imwrite(preview_filename, thumb)
                except: pass

            current_frame += 1
            progress_queue.put(1)

        cap.release()
        out.release()
        
        if os.path.exists(preview_filename):
            try: os.remove(preview_filename)
            except: pass
        
        return (chunk_id, True, "Success")
    except Exception as e:
        return (chunk_id, False, str(e))

# --- MAIN APP ---
class NoiseRenderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bad Apple: SYNC FIXED")
        self.root.geometry("450x680") 
        self.root.configure(bg="#121212")
        self.root.resizable(False, False)

        # Variables
        self.file_path = tk.StringVar()
        
        # DEFAULTS UPDATED PER REQUEST
        self.target_width = tk.IntVar(value=1920)  # 1080p Default
        self.internal_scale = tk.IntVar(value=4)   # 4x Faster Default
        self.fps_display = tk.StringVar(value="Auto")     
        self.chunk_count = tk.IntVar(value=30) 
        self.is_inverted = tk.BooleanVar(value=False)
        
        # State
        self.status = tk.StringVar(value="Idle")
        self.eta = tk.StringVar(value="--:--")
        self.is_running = False
        self.total = 0 
        self.processed = 0
        self.src_fps_val = 30.0 # Default fallback
        
        self.setup_ui()

    def setup_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TLabel", background="#121212", foreground="#eee")
        style.configure("TButton", background="#333", foreground="#fff", borderwidth=1)
        style.configure("Horizontal.TProgressbar", troughcolor="#333", background="#00dd00")

        tk.Label(self.root, text="SYNC FIX RENDERER", font=("Impact", 22), bg="#121212", fg="#ffcc00").pack(pady=10)

        # Source
        f_frame = tk.Frame(self.root, bg="#121212")
        f_frame.pack(fill="x", padx=10)
        tk.Entry(f_frame, textvariable=self.file_path, bg="#222", fg="#fff").pack(side="left", fill="x", expand=True)
        ttk.Button(f_frame, text="Browse", command=self.browse).pack(side="right", padx=5)

        # Config
        c_frame = tk.LabelFrame(self.root, text="Settings", bg="#121212", fg="#888")
        c_frame.pack(fill="x", padx=10, pady=10)

        # Output Size
        tk.Label(c_frame, text="Output Width:", bg="#121212", fg="#eee").grid(row=0, column=0, sticky="e", padx=5)
        tk.Entry(c_frame, textvariable=self.target_width, bg="#222", fg="#fff", width=8).grid(row=0, column=1, sticky="w")

        # Optimization Level (Internal Scale)
        tk.Label(c_frame, text="Speed Factor:", bg="#121212", fg="#00ffff").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        
        opt_frame = tk.Frame(c_frame, bg="#121212")
        opt_frame.grid(row=1, column=1, columnspan=3, sticky="w")
        
        modes = [("1x", 1), ("2x", 2), ("4x (Retro)", 4)]
        for txt, val in modes:
            tk.Radiobutton(opt_frame, text=txt, variable=self.internal_scale, value=val, 
                           bg="#121212", fg="#eee", selectcolor="#444", activebackground="#121212", activeforeground="#fff").pack(side="left")

        # FPS & Chunks
        tk.Label(c_frame, text="FPS:", bg="#121212", fg="#eee").grid(row=2, column=0, sticky="e")
        # Read-only FPS entry because we force auto-detect
        tk.Entry(c_frame, textvariable=self.fps_display, bg="#333", fg="#aaa", width=8, state="readonly").grid(row=2, column=1, sticky="w")
        
        tk.Label(c_frame, text="Chunks:", bg="#121212", fg="#eee").grid(row=2, column=2, sticky="e")
        tk.Entry(c_frame, textvariable=self.chunk_count, bg="#222", fg="#fff", width=5).grid(row=2, column=3, sticky="w")

        ttk.Checkbutton(c_frame, text="Invert Logic", variable=self.is_inverted).grid(row=3, column=0, columnspan=2)

        # Preview
        self.lbl_preview = tk.Label(self.root, bg="#000", text="Preview", fg="#444", height=10)
        self.lbl_preview.pack(fill="both", padx=10, pady=5)

        # Progress
        tk.Label(self.root, textvariable=self.status, bg="#121212", fg="#0f0").pack()
        self.p_bar = ttk.Progressbar(self.root, length=100, mode='determinate')
        self.p_bar.pack(fill="x", padx=10)
        tk.Label(self.root, textvariable=self.eta, bg="#121212", fg="#0ff").pack()

        # Run
        self.btn_run = tk.Button(self.root, text="LAUNCH RENDER", command=self.start, 
                                 bg="#b00", fg="#fff", font=("Arial", 12, "bold"))
        self.btn_run.pack(fill="x", padx=40, pady=15)
        
        tk.Label(self.root, text="made by lousybook01 cuz yes", bg="#121212", fg="#444", font=("Arial", 8)).pack(side="bottom")

    def browse(self):
        f = filedialog.askopenfilename()
        if f: 
            self.file_path.set(f)
            # Auto-detect FPS immediately upon browse
            self.detect_fps(f)

    def detect_fps(self, path):
        try:
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                self.src_fps_val = fps
                self.fps_display.set(f"{fps:.2f} (Auto)")
            cap.release()
        except:
            self.fps_display.set("Unknown")

    def start(self):
        if not self.file_path.get(): return
        
        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            messagebox.showerror("Error", "FFmpeg not found!")
            return

        self.btn_run.config(state="disabled")
        self.manager = multiprocessing.Manager()
        self.queue = self.manager.Queue()
        
        # Init counters
        self.processed = 0
        self.total = 0 
        self.start_time = time.time()
        
        import threading
        threading.Thread(target=self.run_logic, args=(ffmpeg,)).start()
        
        self.is_running = True
        self.update_ui()

    def update_ui(self):
        if not self.is_running: return
        try:
            while True:
                _ = self.queue.get_nowait()
                self.processed += 1
        except queue.Empty: pass

        if self.total > 0:
            pct = (self.processed / self.total) * 100
            self.p_bar['value'] = pct
            
            elapsed = time.time() - self.start_time
            if elapsed > 1 and self.processed > 0:
                fps = self.processed / elapsed
                rem = self.total - self.processed
                eta = rem / fps if fps > 0 else 0
                m, s = divmod(int(eta), 60)
                self.eta.set(f"ETA: {m:02d}:{s:02d} ({fps:.1f} fps)")
                self.status.set(f"{int(pct)}% Complete")
        else:
            self.status.set("Initializing...")

        # Preview
        previews = glob.glob("temp_preview_*.jpg")
        if previews:
            try:
                latest = max(previews, key=os.path.getmtime)
                img = Image.open(latest)
                w_box = self.root.winfo_width() - 20
                h_box = 150
                img.thumbnail((w_box, h_box))
                tk_img = ImageTk.PhotoImage(img)
                self.lbl_preview.config(image=tk_img, text="", height=0)
                self.lbl_preview.image = tk_img
            except: pass

        self.root.after(100, self.update_ui)

    def run_logic(self, ffmpeg_exe):
        # 1. Setup
        src = self.file_path.get()
        final_w = self.target_width.get()
        scale = self.internal_scale.get()
        
        # Internal Resolution
        calc_w = int(final_w // scale)
        calc_h = int(calc_w * (3/4))
        
        # Final Resolution
        final_h = int(final_w * (3/4))

        chunks = self.chunk_count.get()
        inv = self.is_inverted.get()

        cap = cv2.VideoCapture(src)
        self.total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # CRITICAL: USE EXACT SOURCE FPS
        real_fps = cap.get(cv2.CAP_PROP_FPS)
        if real_fps <= 0 or math.isnan(real_fps): real_fps = 30.0 # Fallback
        
        cap.release()
        
        self.status.set(f"Processing @ {real_fps:.2f} FPS (Synced)...")

        # 2. Multiprocessing
        per_chunk = math.ceil(self.total / chunks)
        tasks = []
        for i in range(chunks):
            s = i * per_chunk
            e = min(s + per_chunk, self.total)
            if s >= self.total: break
            # Pass real_fps to worker
            tasks.append((i, s, e, src, calc_w, calc_h, inv, real_fps, self.queue))

        active = min(chunks, os.cpu_count())
        
        with multiprocessing.Pool(active) as pool:
            pool.starmap(render_chunk, tasks)

        # 3. Stitching
        self.is_running = False
        self.status.set("Stitching & Upscaling...")
        
        with open("list.txt", "w") as f:
            for i in range(len(tasks)):
                f.write(f"file 'temp_chunk_{i:03d}.avi'\n")
        
        out_name = f"BadApple_SyncFixed_{final_w}p.mp4"

        # FFmpeg Command
        cmd = [
            ffmpeg_exe, '-y',
            '-f', 'concat', '-safe', '0', '-i', 'list.txt',
            '-i', src,
            '-map', '0:v', '-map', '1:a',
            
            # Use detected FPS
            '-r', str(real_fps),
            
            # Upscale Filter
            '-vf', f'scale={final_w}:{final_h}:flags=neighbor',
            
            '-c:v', 'libx264',
            '-crf', '28', 
            '-preset', 'ultrafast',
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac', '-b:a', '128k',
            '-shortest',
            out_name
        ]

        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        try:
            subprocess.run(cmd, check=True, startupinfo=si)
            
            # Cleanup
            os.remove("list.txt")
            for f in glob.glob("temp_chunk_*.avi"): 
                try: os.remove(f)
                except: pass
            for f in glob.glob("temp_preview_*.jpg"): 
                try: os.remove(f)
                except: pass
            
            messagebox.showinfo("Success", f"Done!\n{out_name}")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
        
        self.btn_run.config(state="normal")
        self.lbl_preview.config(image='')

if __name__ == "__main__":
    multiprocessing.freeze_support()
    root = tk.Tk()
    app = NoiseRenderApp(root)
    root.mainloop()
