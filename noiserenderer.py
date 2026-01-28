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
import io

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
                 internal_w, internal_h, inverted, fps, progress_queue,
                 noise_type, scroll_direction, scroll_speed):
    try:
        time.sleep(chunk_id * 0.05)

        chunk_filename = f"temp_chunk_{chunk_id:03d}.avi"
        preview_filename = f"temp_preview_{chunk_id}.jpg"

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened(): return (chunk_id, False, "Open Fail")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(chunk_filename, fourcc, fps, (internal_w, internal_h))

        # --- STATIC NOISE SETUP ---
        static_gray = np.random.randint(0, 256, (internal_h, internal_w), dtype=np.uint8)
        static_noise = cv2.merge([static_gray, static_gray, static_gray]).astype(np.float32)
        
        active_gray_buffer = np.zeros((internal_h, internal_w), dtype=np.uint8)

        if noise_type == "Scrolling":
            scrolling_gray = np.random.randint(0, 256, (internal_h, internal_w), dtype=np.uint8)

        current_frame = start_frame

        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret: break

            frame_resized = cv2.resize(frame, (internal_w, internal_h), interpolation=cv2.INTER_NEAREST)
            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            gray_normalized = (gray.astype(np.float32) / 255.0)[:, :, np.newaxis]

            if noise_type == "Random":
                cv2.randu(active_gray_buffer, 0, 256)
                active_noise = cv2.merge([active_gray_buffer, active_gray_buffer, active_gray_buffer]).astype(np.float32)
            else: # Scrolling Noise
                if scroll_direction == 'Down':
                    scrolling_gray = np.roll(scrolling_gray, shift=scroll_speed, axis=0)
                elif scroll_direction == 'Up':
                    scrolling_gray = np.roll(scrolling_gray, shift=-scroll_speed, axis=0)
                elif scroll_direction == 'Right':
                    scrolling_gray = np.roll(scrolling_gray, shift=scroll_speed, axis=1)
                elif scroll_direction == 'Left':
                    scrolling_gray = np.roll(scrolling_gray, shift=-scroll_speed, axis=1)
                
                active_noise = cv2.merge([scrolling_gray, scrolling_gray, scrolling_gray]).astype(np.float32)

            if not inverted:
                final_float = (active_noise * gray_normalized) + (static_noise * (1.0 - gray_normalized))
            else:
                final_float = (static_noise * gray_normalized) + (active_noise * (1.0 - gray_normalized))

            final = cv2.convertScaleAbs(final_float)
            out.write(final)

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

# --- THUMBNAIL GENERATOR WINDOW ---
class ThumbnailGeneratorWindow(tk.Toplevel):
    def __init__(self, app):
        super().__init__(app.root)
        self.app = app 
        
        self.title("Thumbnail Generator")
        self.configure(bg="#1e1e1e")

        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        win_w = int(screen_w * 0.8)
        win_h = int(screen_h * 0.8)
        x_pos = (screen_w - win_w) // 2
        y_pos = (screen_h - win_h) // 2
        self.geometry(f"{win_w}x{win_h}+{x_pos}+{y_pos}")
        self.minsize(600, 500)

        self.video_path = self.app.file_path.get()
        self.cap = cv2.VideoCapture(self.video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.internal_w = int(self.app.target_width.get() // self.app.internal_scale.get())
        self.internal_h = int(self.internal_w * (3/4))

        self.current_frame_num = tk.IntVar(value=0)
        self.improve_visibility = tk.BooleanVar(value=False)
        self.visibility_amount = tk.DoubleVar(value=0.3)

        _static_gray = np.random.randint(0, 256, (self.internal_h, self.internal_w), dtype=np.uint8)
        self.static_noise = cv2.merge([_static_gray, _static_gray, _static_gray])
        
        self.processed_frame = None

        self.setup_ui()
        self.after(100, self.update_preview)

    def setup_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)

        self.preview_frame = tk.Frame(self, bg="#000")
        self.preview_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        self.lbl_preview = tk.Label(self.preview_frame, bg="#000", text="Loading...", fg="#555")
        self.lbl_preview.pack(expand=True, fill="both")

        controls_frame = tk.LabelFrame(self, text="Controls", bg="#1e1e1e", fg="#eee")
        controls_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=10)

        tk.Label(controls_frame, text="Timeline Position:", bg="#1e1e1e", fg="#ccc").pack(anchor="w", padx=10)
        self.slider = ttk.Scale(controls_frame, from_=0, to=self.total_frames - 1, variable=self.current_frame_num, orient="horizontal", command=lambda s: self.update_preview())
        self.slider.pack(fill="x", padx=10, pady=(0, 15))

        vis_frame = tk.Frame(controls_frame, bg="#1e1e1e")
        vis_frame.pack(fill="x", padx=10, pady=5)
        ttk.Checkbutton(vis_frame, text="Blend Original Video", variable=self.improve_visibility, command=self.update_preview).pack(side="left")
        tk.Label(vis_frame, text="Blend Amount:", bg="#1e1e1e", fg="#ccc").pack(side="left", padx=(20, 5))
        self.vis_slider = ttk.Scale(vis_frame, from_=0, to=1.0, variable=self.visibility_amount, orient="horizontal", command=lambda s: self.update_preview(), length=200)
        self.vis_slider.pack(side="left")

        btn_frame = tk.Frame(self, bg="#1e1e1e")
        btn_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
        ttk.Button(btn_frame, text="Randomize Noise Seed", command=self.reroll_noise).pack(side="left")
        ttk.Button(btn_frame, text="SAVE THUMBNAIL (<2MB)", command=self.save_thumbnail).pack(side="right")

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.preview_frame.bind("<Configure>", self.on_resize)
        self.last_resize_time = 0

    def on_resize(self, event):
        current_time = time.time()
        if current_time - self.last_resize_time > 0.1:
            self.last_resize_time = current_time
            self.update_preview()

    def reroll_noise(self):
        _static_gray = np.random.randint(0, 256, (self.internal_h, self.internal_w), dtype=np.uint8)
        self.static_noise = cv2.merge([_static_gray, _static_gray, _static_gray])
        self.update_preview()

    def generate_noise_frame(self, original_frame):
        frame_resized = cv2.resize(original_frame, (self.internal_w, self.internal_h), interpolation=cv2.INTER_NEAREST)
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        gray_normalized = (gray.astype(np.float32) / 255.0)[:, :, np.newaxis]

        _active_gray = np.random.randint(0, 256, (self.internal_h, self.internal_w), dtype=np.uint8)
        active_noise = cv2.merge([_active_gray, _active_gray, _active_gray])

        final_float = (active_noise.astype(np.float32) * gray_normalized) + (self.static_noise.astype(np.float32) * (1.0 - gray_normalized))
        noise_frame = cv2.convertScaleAbs(final_float)

        if self.improve_visibility.get():
            blend_factor = self.visibility_amount.get()
            noise_frame = cv2.addWeighted(noise_frame, 1 - blend_factor, frame_resized, blend_factor, 0)

        return noise_frame

    def update_preview(self, event=None):
        frame_pos = self.current_frame_num.get()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = self.cap.read()
        if ret:
            self.processed_frame = self.generate_noise_frame(frame)
            img_rgb = cv2.cvtColor(self.processed_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img_rgb)
            
            max_w = self.preview_frame.winfo_width() - 4
            max_h = self.preview_frame.winfo_height() - 4
            
            if max_w < 10 or max_h < 10: return

            img_w, img_h = img.size
            ratio = min(max_w / img_w, max_h / img_h)
            new_w = int(img_w * ratio)
            new_h = int(img_h * ratio)
            
            img = img.resize((new_w, new_h), Image.Resampling.NEAREST)

            self.tk_img = ImageTk.PhotoImage(image=img)
            self.lbl_preview.config(image=self.tk_img, text="")
            self.lbl_preview.image = self.tk_img

    def save_thumbnail(self):
        if self.processed_frame is None: return

        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")], title="Save Thumbnail", parent=self)
        if not save_path: return

        target_w, target_h = 1920, 1080
        scaled_h = target_h
        scaled_w = int(target_h * (self.internal_w / self.internal_h))
        upscaled_image = cv2.resize(self.processed_frame, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)

        _bg_gray = np.random.randint(0, 256, (target_h, target_w), dtype=np.uint8)
        canvas = cv2.merge([_bg_gray, _bg_gray, _bg_gray])
        
        x_offset = (target_w - scaled_w) // 2
        canvas[:, x_offset:x_offset + scaled_w] = upscaled_image

        canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(canvas_rgb)

        try:
            colors = 256 
            buffer = io.BytesIO()
            pil_img.save(buffer, format="PNG", optimize=True)
            size_mb = buffer.tell() / (1024 * 1024)
            final_img = pil_img
            
            while size_mb > 2.0 and colors >= 8:
                buffer.seek(0)
                buffer.truncate(0)
                final_img = pil_img.quantize(colors=colors, method=Image.Quantize.MAXCOVERAGE)
                final_img.save(buffer, format="PNG", optimize=True)
                size_mb = buffer.tell() / (1024 * 1024)
                print(f"Trying {colors} colors... Size: {size_mb:.2f} MB")
                colors //= 2

            with open(save_path, "wb") as f:
                f.write(buffer.getbuffer())

            msg = f"Saved: {save_path}\nFinal Size: {size_mb:.2f} MB"
            if colors < 256:
                msg += f"\n(Colors reduced to {colors * 2} to fit limit)"
            
            messagebox.showinfo("Success", msg, parent=self)

        except Exception as e:
            messagebox.showerror("Error", str(e), parent=self)
            
    def on_close(self):
        self.cap.release()
        self.destroy()

# --- MAIN APP (SCROLLABLE FIX) ---
class NoiseRenderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bad Apple: SYNC FIXED")
        
        # 1. ADAPTIVE HEIGHT CALCULATION
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        
        # Target height is 850, but clamp it to screen height - 80px (for taskbar)
        target_h = 850
        max_allowed_h = sh - 80
        final_h = min(target_h, max_allowed_h)
        
        w = 480 # Slightly wider for scrollbar
        x = (sw - w) // 2
        y = (sh - final_h) // 2
        
        self.root.geometry(f"{w}x{final_h}+{x}+{y}")
        self.root.configure(bg="#121212")
        # 2. MAKE RESIZABLE
        self.root.resizable(True, True)

        self.file_path = tk.StringVar()
        self.target_width = tk.IntVar(value=1920)
        self.internal_scale = tk.IntVar(value=4)
        self.fps_display = tk.StringVar(value="Auto")
        self.chunk_count = tk.IntVar(value=30)
        self.is_inverted = tk.BooleanVar(value=False)
        self.noise_type = tk.StringVar(value="Random")
        self.scroll_direction = tk.StringVar(value="Down")
        self.scroll_speed = tk.IntVar(value=4)
        self.status = tk.StringVar(value="Idle")
        self.eta = tk.StringVar(value="--:--")
        self.is_running = False
        self.total = 0
        self.processed = 0
        self.src_fps_val = 30.0

        self.setup_scrollable_ui()

    def setup_scrollable_ui(self):
        # --- SCROLLBAR INFRASTRUCTURE ---
        # Main container
        self.main_container = tk.Frame(self.root, bg="#121212")
        self.main_container.pack(fill="both", expand=True)

        # Canvas
        self.canvas = tk.Canvas(self.main_container, bg="#121212", highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)

        # Scrollbar
        self.scrollbar = ttk.Scrollbar(self.main_container, orient="vertical", command=self.canvas.yview)
        self.scrollbar.pack(side="right", fill="y")

        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Scrollable Frame (Holds actual content)
        self.scrollable_frame = tk.Frame(self.canvas, bg="#121212")
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        # Bindings for resizing and scrolling
        self.scrollable_frame.bind("<Configure>", self.on_frame_configure)
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        self.root.bind_all("<MouseWheel>", self._on_mousewheel)

        # --- BUILD UI INSIDE SCROLLABLE FRAME ---
        self.populate_ui(self.scrollable_frame)

    def on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_canvas_configure(self, event):
        # Ensure width matches canvas
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def populate_ui(self, parent):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TLabel", background="#121212", foreground="#eee")
        style.configure("TButton", background="#333", foreground="#fff", borderwidth=1)
        style.configure("Horizontal.TProgressbar", troughcolor="#333", background="#00dd00")
        style.configure("TRadiobutton", background="#121212", foreground="#eee")
        style.configure("TCheckbutton", background="#121212", foreground="#eee")
        style.configure("TScale", background="#121212")

        tk.Label(parent, text="SYNC FIX RENDERER", font=("Impact", 22), bg="#121212", fg="#ffcc00").pack(pady=10)

        f_frame = tk.Frame(parent, bg="#121212")
        f_frame.pack(fill="x", padx=10)
        tk.Entry(f_frame, textvariable=self.file_path, bg="#222", fg="#fff").pack(side="left", fill="x", expand=True)
        ttk.Button(f_frame, text="Browse", command=self.browse).pack(side="right", padx=5)

        c_frame = tk.LabelFrame(parent, text="Settings", bg="#121212", fg="#888")
        c_frame.pack(fill="x", padx=10, pady=10)
        tk.Label(c_frame, text="Output Width:", bg="#121212", fg="#eee").grid(row=0, column=0, sticky="e", padx=5)
        tk.Entry(c_frame, textvariable=self.target_width, bg="#222", fg="#fff", width=8).grid(row=0, column=1, sticky="w")
        tk.Label(c_frame, text="Speed Factor:", bg="#121212", fg="#00ffff").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        opt_frame = tk.Frame(c_frame, bg="#121212")
        opt_frame.grid(row=1, column=1, columnspan=3, sticky="w")
        for txt, val in [("1x", 1), ("2x", 2), ("4x (Retro)", 4)]:
            tk.Radiobutton(opt_frame, text=txt, variable=self.internal_scale, value=val, bg="#121212", fg="#eee", selectcolor="#444").pack(side="left")
        tk.Label(c_frame, text="FPS:", bg="#121212", fg="#eee").grid(row=2, column=0, sticky="e")
        tk.Entry(c_frame, textvariable=self.fps_display, bg="#333", fg="#aaa", width=8, state="readonly").grid(row=2, column=1, sticky="w")
        tk.Label(c_frame, text="Chunks:", bg="#121212", fg="#eee").grid(row=2, column=2, sticky="e")
        tk.Entry(c_frame, textvariable=self.chunk_count, bg="#222", fg="#fff", width=5).grid(row=2, column=3, sticky="w")
        ttk.Checkbutton(c_frame, text="Invert Logic", variable=self.is_inverted).grid(row=3, column=0, columnspan=2, sticky="w")

        n_frame = tk.LabelFrame(parent, text="Noise Options", bg="#121212", fg="#888")
        n_frame.pack(fill="x", padx=10, pady=5)
        noise_type_frame = tk.Frame(n_frame, bg="#121212")
        noise_type_frame.pack(fill="x")
        tk.Label(noise_type_frame, text="Noise Type:", bg="#121212", fg="#eee").pack(side="left", padx=5)
        ttk.Radiobutton(noise_type_frame, text="Random", variable=self.noise_type, value="Random", command=self.toggle_scroll_options).pack(side="left")
        ttk.Radiobutton(noise_type_frame, text="Scrolling", variable=self.noise_type, value="Scrolling", command=self.toggle_scroll_options).pack(side="left")
        self.scroll_frame = tk.LabelFrame(n_frame, text="Scrolling Noise Settings", bg="#121212", fg="#888")
        dir_frame = tk.Frame(self.scroll_frame, bg="#121212")
        dir_frame.pack()
        tk.Label(dir_frame, text="Direction:", bg="#121212", fg="#eee").pack(side="left", padx=5)
        for d in ["Down", "Up", "Left", "Right"]:
            ttk.Radiobutton(dir_frame, text=d, variable=self.scroll_direction, value=d).pack(side="left")
        speed_frame = tk.Frame(self.scroll_frame, bg="#121212")
        speed_frame.pack(pady=5)
        tk.Label(speed_frame, text="Speed:", bg="#121212", fg="#eee").pack(side="left", padx=5)
        ttk.Scale(speed_frame, from_=1, to=20, orient="horizontal", variable=self.scroll_speed, length=150).pack(side="left")
        ttk.Label(speed_frame, textvariable=self.scroll_speed, width=3).pack(side="left", padx=5)

        self.lbl_preview = tk.Label(parent, bg="#000", text="Preview", fg="#444", height=10)
        self.lbl_preview.pack(fill="both", padx=10, pady=5)

        tk.Label(parent, textvariable=self.status, bg="#121212", fg="#0f0").pack()
        self.p_bar = ttk.Progressbar(parent, length=100, mode='determinate')
        self.p_bar.pack(fill="x", padx=10)
        tk.Label(parent, textvariable=self.eta, bg="#121212", fg="#0ff").pack()
        
        buttons_frame = tk.Frame(parent, bg="#121212")
        buttons_frame.pack(fill="x", padx=40, pady=10)
        self.btn_run = tk.Button(buttons_frame, text="LAUNCH RENDER", command=self.start, bg="#b00", fg="#fff", font=("Arial", 12, "bold"))
        self.btn_run.pack(side="left", fill="x", expand=True, ipady=5)
        self.btn_thumb = ttk.Button(buttons_frame, text="Generate Thumbnail", command=self.open_thumbnail_generator)
        self.btn_thumb.pack(side="right", padx=10)

        tk.Label(parent, text="made by lousybook01 cuz yes", bg="#121212", fg="#444", font=("Arial", 8)).pack(side="bottom", pady=10)

        # Initial scroll option check
        self.toggle_scroll_options()

    def toggle_scroll_options(self):
        if self.noise_type.get() == "Scrolling": self.scroll_frame.pack(fill="x", padx=5, pady=5)
        else: self.scroll_frame.pack_forget()

    def open_thumbnail_generator(self):
        if not self.file_path.get() or not os.path.exists(self.file_path.get()):
            messagebox.showerror("Error", "Please select a valid video file first.")
            return
        ThumbnailGeneratorWindow(self)

    def browse(self):
        f = filedialog.askopenfilename()
        if f: self.file_path.set(f); self.detect_fps(f)

    def detect_fps(self, path):
        try:
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                self.src_fps_val = cap.get(cv2.CAP_PROP_FPS)
                self.fps_display.set(f"{self.src_fps_val:.2f} (Auto)")
            cap.release()
        except: self.fps_display.set("Unknown")

    def start(self):
        if not self.file_path.get(): return
        ffmpeg = find_ffmpeg()
        if not ffmpeg: messagebox.showerror("Error", "FFmpeg not found!"); return

        self.btn_run.config(state="disabled")
        self.manager = multiprocessing.Manager()
        self.queue = self.manager.Queue()
        self.processed, self.total = 0, 0
        self.start_time = time.time()
        
        import threading
        threading.Thread(target=self.run_logic, args=(ffmpeg,)).start()
        
        self.is_running = True
        self.update_ui()

    def update_ui(self):
        if not self.is_running: return
        try:
            while True: self.processed += 1; self.queue.get_nowait()
        except queue.Empty: pass

        if self.total > 0:
            pct = (self.processed / self.total) * 100
            self.p_bar['value'] = pct
            elapsed = time.time() - self.start_time
            if elapsed > 1 and self.processed > 0:
                fps = self.processed / elapsed
                eta = (self.total - self.processed) / fps if fps > 0 else 0
                m, s = divmod(int(eta), 60)
                self.eta.set(f"ETA: {m:02d}:{s:02d} ({fps:.1f} fps)")
                self.status.set(f"{int(pct)}% Complete")
        else: self.status.set("Initializing...")

        previews = glob.glob("temp_preview_*.jpg")
        if previews:
            try:
                latest = max(previews, key=os.path.getmtime)
                img = Image.open(latest)
                # Resize for main window preview
                w_box = self.lbl_preview.winfo_width() - 10
                if w_box > 10:
                    img.thumbnail((w_box, 150))
                tk_img = ImageTk.PhotoImage(img)
                self.lbl_preview.config(image=tk_img, text="")
                self.lbl_preview.image = tk_img
            except: pass

        self.root.after(100, self.update_ui)

    def run_logic(self, ffmpeg_exe):
        src = self.file_path.get(); final_w = self.target_width.get(); scale = self.internal_scale.get()
        calc_w = int(final_w // scale); calc_h = int(calc_w * (3/4)); final_h = int(final_w * (3/4))
        chunks = self.chunk_count.get(); inv = self.is_inverted.get()
        nt, sd, ss = self.noise_type.get(), self.scroll_direction.get(), self.scroll_speed.get()

        cap = cv2.VideoCapture(src)
        self.total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        real_fps = cap.get(cv2.CAP_PROP_FPS); cap.release()
        if real_fps <= 0 or math.isnan(real_fps): real_fps = 30.0
        self.status.set(f"Processing @ {real_fps:.2f} FPS (Synced)...")

        per_chunk = math.ceil(self.total / chunks)
        tasks = [(i, i * per_chunk, min((i + 1) * per_chunk, self.total), src, calc_w, calc_h, inv, real_fps, self.queue, nt, sd, ss) for i in range(chunks) if i * per_chunk < self.total]

        with multiprocessing.Pool(min(chunks, os.cpu_count())) as pool:
            pool.starmap(render_chunk, tasks)

        self.is_running = False
        self.status.set("Stitching & Upscaling...")
        with open("list.txt", "w") as f:
            for i in range(len(tasks)): f.write(f"file 'temp_chunk_{i:03d}.avi'\n")
        
        out_name = f"BadApple_SyncFixed_{final_w}p.mp4"
        cmd = [ffmpeg_exe, '-y', '-f', 'concat', '-safe', '0', '-i', 'list.txt', '-i', src, '-map', '0:v', '-map', '1:a', '-r', str(real_fps), '-vf', f'scale={final_w}:{final_h}:flags=neighbor', '-c:v', 'libx264', '-crf', '28', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '128k', '-shortest', out_name]
        
        si = subprocess.STARTUPINFO(); si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        
        try:
            print("\n--- Starting FFmpeg Stitching Process ---")
            print(f"Command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, startupinfo=si)
            messagebox.showinfo("Success", f"Done!\n{out_name}")
            
        except subprocess.CalledProcessError:
            messagebox.showerror("Error", "FFmpeg failed during the stitching process. Please check the console window for specific error messages.")
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            print("--- FFmpeg Process Finished ---\n")
            for f in glob.glob("temp_chunk_*.avi") + glob.glob("temp_preview_*.jpg") + ["list.txt"]:
                try: os.remove(f)
                except: pass
            self.btn_run.config(state="normal")
            self.lbl_preview.config(image='')

if __name__ == "__main__":
    multiprocessing.freeze_support()
    root = tk.Tk()
    app = NoiseRenderApp(root)
    root.mainloop()
