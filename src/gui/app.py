import customtkinter as ctk
from tkinter import filedialog, messagebox
import threading
import os
import sys
import time
import numpy as np
import tarfile
import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.encoder import ToneEncoder
from core.decoder import ToneDecoder
from crypto.engine import CryptoEngine

class ToneCryptApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- TOKYO NIGHT DESIGN SYSTEM ---
        self.bg_color = "#1a1b26"
        self.bg_dark = "#16161e"
        self.card_bg = "#1f2335"
        self.accent_blue = "#7aa2f7"
        self.accent_cyan = "#7dcfff"
        self.accent_green = "#9ece6a"
        self.accent_red = "#f7768e"
        self.accent_purple = "#bb9af7"
        self.text_color = "#c0caf5"
        self.text_dim = "#565f89"

        self.title("TONECRYPT ELITE | v3.0 Sonic Vault")
        self.geometry("1350x900")
        self.minsize(1100, 800)
        self.configure(fg_color=self.bg_color)

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.queue = []
        self.carrier_path = None
        self.processing = False
        self.recording = False
        self.recorded_data = []
        
        self.setup_sidebar()
        self.setup_main_view()
        self.setup_visualizer()
        
        self.show_page("encrypt")
        self.log("System v3.0 ELITE Online. Physical Interface Ready.")
        self.start_heartbeat()

    def setup_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0, fg_color=self.bg_dark, border_width=1, border_color="#24283b")
        self.sidebar.grid(row=0, column=0, sticky="nsew", rowspan=2)
        self.sidebar.grid_propagate(False)
        
        # Header
        self.title_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.title_frame.pack(pady=40, padx=20, fill="x")
        
        ctk.CTkLabel(self.title_frame, text="TONECRYPT", 
                     font=ctk.CTkFont(family="Courier", size=28, weight="bold"),
                     text_color=self.accent_blue).pack()
        ctk.CTkLabel(self.title_frame, text="SONIC MODEM INTERFACE", 
                     font=ctk.CTkFont(family="Courier", size=10, weight="bold"),
                     text_color=self.text_dim).pack()

        # Navigation
        self.nav_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.nav_frame.pack(fill="x", padx=20)

        self.enc_btn = self.create_nav_button("🔒 VAULT DATA", "encrypt", self.accent_blue)
        self.dec_btn = self.create_nav_button("🔓 RESTORE VAULT", "decrypt", self.accent_cyan)
        self.phys_btn = self.create_nav_button("📼 PHYSICAL MEDIA", "physical", self.accent_purple)

        ctk.CTkLabel(self.sidebar, text="").pack(expand=True)

        # Dashboard Stats
        self.dash_frame = ctk.CTkFrame(self.sidebar, fg_color=self.card_bg, corner_radius=10, border_width=1, border_color="#414868")
        self.dash_frame.pack(pady=30, padx=20, fill="x")
        
        self.create_stat_line(self.dash_frame, "DSP STATUS", "OPTIMAL", self.accent_green)
        self.create_stat_line(self.dash_frame, "ENCRYPTION", "AES-256", self.accent_purple)
        self.create_stat_line(self.dash_frame, "ECC LAYER", "RS-32", self.accent_cyan)
        self.create_stat_line(self.dash_frame, "INPUT", "LINE-IN", self.accent_blue)

    def create_nav_button(self, text, page, color):
        btn = ctk.CTkButton(self.nav_frame, text=text, 
                            command=lambda: self.show_page(page),
                            anchor="w",
                            height=50,
                            font=ctk.CTkFont(size=14, weight="bold"),
                            fg_color="transparent",
                            text_color=self.text_color,
                            hover_color="#24283b",
                            border_spacing=10)
        btn.pack(fill="x", pady=5)
        return btn

    def create_stat_line(self, parent, label, val, color):
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.pack(fill="x", padx=15, pady=5)
        ctk.CTkLabel(f, text=label, font=("Courier", 10), text_color=self.text_dim).pack(side="left")
        ctk.CTkLabel(f, text=val, font=("Courier", 10, "bold"), text_color=color).pack(side="right")

    def setup_main_view(self):
        self.main_container = ctk.CTkFrame(self, fg_color="transparent")
        self.main_container.grid(row=0, column=1, padx=30, pady=30, sticky="nsew")
        self.main_container.grid_columnconfigure(0, weight=1)
        self.main_container.grid_rowconfigure(0, weight=1)

        self.pages = {}
        
        # --- ENCRYPT PAGE ---
        enc_page = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.pages["encrypt"] = enc_page
        enc_page.grid_columnconfigure(0, weight=1)
        enc_page.grid_rowconfigure(3, weight=1)

        pass_card = ctk.CTkFrame(enc_page, fg_color=self.card_bg, corner_radius=15, border_width=1, border_color="#414868")
        pass_card.grid(row=0, column=0, pady=(0, 20), sticky="ew")
        ctk.CTkLabel(pass_card, text="SECURE CHANNEL INITIALIZATION", font=ctk.CTkFont(size=14, weight="bold"), text_color=self.accent_blue).pack(pady=(15, 5))
        self.enc_pass = ctk.CTkEntry(pass_card, placeholder_text="ENTER MASTER ACCESS KEY", show="*", width=600, height=45, fg_color=self.bg_dark, border_color="#414868")
        self.enc_pass.pack(pady=10, padx=30)
        self.enc_pass.bind("<KeyRelease>", self.update_pass_strength)
        self.strength_bar = ctk.CTkProgressBar(pass_card, width=500, height=6, fg_color=self.bg_dark)
        self.strength_bar.set(0); self.strength_bar.pack(pady=(0, 15))

        opt_card = ctk.CTkFrame(enc_page, fg_color=self.card_bg, corner_radius=15, border_width=1, border_color="#414868")
        opt_card.grid(row=1, column=0, pady=(0, 20), sticky="ew")
        self.stealth_toggle = ctk.CTkCheckBox(opt_card, text="STEALTH ENCODING", text_color=self.accent_cyan, command=self.toggle_stealth, font=ctk.CTkFont(size=12, weight="bold"))
        self.stealth_toggle.pack(side="left", padx=30, pady=20)
        self.music_btn = ctk.CTkButton(opt_card, text="SELECT CARRIER MUSIC", command=self.select_music, state="disabled", fg_color="#24283b", height=35)
        self.music_btn.pack(side="right", padx=30, pady=20)

        queue_card = ctk.CTkFrame(enc_page, fg_color=self.card_bg, corner_radius=15, border_width=1, border_color="#414868")
        queue_card.grid(row=3, column=0, sticky="nsew")
        queue_card.grid_columnconfigure(0, weight=1); queue_card.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(queue_card, text="DATA STAGING PIPELINE", font=ctk.CTkFont(size=14, weight="bold"), text_color=self.text_color).grid(row=0, column=0, pady=10)
        self.queue_box = ctk.CTkTextbox(queue_card, fg_color=self.bg_dark, text_color=self.text_color, font=("Courier", 13), border_width=0)
        self.queue_box.grid(row=1, column=0, padx=15, pady=5, sticky="nsew")
        q_ctrl = ctk.CTkFrame(queue_card, fg_color="transparent")
        q_ctrl.grid(row=2, column=0, pady=15, padx=15, sticky="ew"); q_ctrl.grid_columnconfigure((0,1,2), weight=1)
        ctk.CTkButton(q_ctrl, text="+ FILES", command=self.add_to_queue_files, fg_color="#24283b").grid(row=0, column=0, padx=5, sticky="ew")
        ctk.CTkButton(q_ctrl, text="+ FOLDERS", command=self.add_to_queue_folder, fg_color="#24283b").grid(row=0, column=1, padx=5, sticky="ew")
        ctk.CTkButton(q_ctrl, text="CLEAR", command=self.clear_queue, fg_color="transparent", border_width=1, border_color=self.accent_red, text_color=self.accent_red).grid(row=0, column=2, padx=5, sticky="ew")

        self.total_size_label = ctk.CTkLabel(enc_page, text="PIPELINE LOAD: 0.00 MB", font=("Courier", 14, "bold"), text_color=self.accent_purple)
        self.total_size_label.grid(row=4, column=0, pady=10)
        self.start_enc_btn = ctk.CTkButton(enc_page, text="🚀 EXECUTE DATA SONIFICATION", command=self.start_encryption, height=65, font=ctk.CTkFont(size=20, weight="bold"), fg_color="#414868", state="disabled", corner_radius=10)
        self.start_enc_btn.grid(row=5, column=0, pady=(0, 10), padx=100, sticky="ew")

        # --- DECRYPT PAGE ---
        dec_page = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.pages["decrypt"] = dec_page
        dec_page.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(dec_page, text="SIGNAL ANALYSIS & RESTORATION", font=ctk.CTkFont(size=24, weight="bold"), text_color=self.accent_cyan).grid(row=0, column=0, pady=30)
        dec_card = ctk.CTkFrame(dec_page, fg_color=self.card_bg, corner_radius=20, border_width=1, border_color="#414868")
        dec_card.grid(row=1, column=0, padx=100, sticky="ew")
        self.dec_pass = ctk.CTkEntry(dec_card, placeholder_text="MASTER DECRYPTION KEY", show="*", width=500, height=50, fg_color=self.bg_dark, border_color=self.accent_cyan)
        self.dec_pass.pack(pady=30, padx=50)
        ctk.CTkButton(dec_card, text="📁 LOAD .WAV DATA VAULT", command=self.select_vault, fg_color=self.accent_cyan, text_color="#16161e", height=50, font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(0, 10), padx=50, fill="x")
        self.vault_label = ctk.CTkLabel(dec_card, text="STATUS: WAITING FOR INPUT", text_color=self.text_dim, font=("Courier", 12))
        self.vault_label.pack(pady=20)
        self.start_dec_btn = ctk.CTkButton(dec_page, text="🚀 BEGIN SIGNAL RECOVERY", command=self.start_decryption, height=70, font=ctk.CTkFont(size=22, weight="bold"), fg_color="#414868", state="disabled", corner_radius=15)
        self.start_dec_btn.grid(row=2, column=0, pady=50, padx=150, sticky="ew")

        # --- PHYSICAL MEDIA PAGE ---
        phys_page = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.pages["physical"] = phys_page
        phys_page.grid_columnconfigure((0,1), weight=1)
        phys_page.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(phys_page, text="PHYSICAL MEDIA STATION", font=ctk.CTkFont(size=24, weight="bold"), text_color=self.accent_purple).grid(row=0, column=0, columnspan=2, pady=30)

        # 1. Capture Card (Recording)
        cap_card = ctk.CTkFrame(phys_page, fg_color=self.card_bg, corner_radius=20, border_width=1, border_color="#414868")
        cap_card.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        ctk.CTkLabel(cap_card, text="LIVE SIGNAL CAPTURE", font=ctk.CTkFont(size=16, weight="bold"), text_color=self.accent_red).pack(pady=20)
        
        self.record_btn = ctk.CTkButton(cap_card, text="🔴 START RECORDING", command=self.toggle_recording, height=60, fg_color="#414868", font=ctk.CTkFont(weight="bold"))
        self.record_btn.pack(pady=20, padx=40, fill="x")
        
        self.rec_status = ctk.CTkLabel(cap_card, text="READY: Select source in System Settings", text_color=self.text_dim, font=("Courier", 10))
        self.rec_status.pack(pady=10)

        self.save_rec_btn = ctk.CTkButton(cap_card, text="💾 SAVE RECORDED VAULT", command=self.save_recording, state="disabled", fg_color="#24283b")
        self.save_rec_btn.pack(pady=20, padx=40, fill="x")

        # 2. Broadcast Card (Playback)
        broad_card = ctk.CTkFrame(phys_page, fg_color=self.card_bg, corner_radius=20, border_width=1, border_color="#414868")
        broad_card.grid(row=1, column=1, padx=20, pady=10, sticky="nsew")
        ctk.CTkLabel(broad_card, text="SIGNAL BROADCAST", font=ctk.CTkFont(size=16, weight="bold"), text_color=self.accent_blue).pack(pady=20)
        
        ctk.CTkButton(broad_card, text="📂 SELECT VAULT TO PLAY", command=self.select_broadcast_vault, fg_color="#24283b").pack(pady=10, padx=40, fill="x")
        self.broad_label = ctk.CTkLabel(broad_card, text="NO VAULT LOADED", text_color=self.text_dim, font=("Courier", 10))
        self.broad_label.pack(pady=5)

        self.play_btn = ctk.CTkButton(broad_card, text="▶️ BROADCAST TO TAPE/CD", command=self.play_vault, height=60, fg_color="#414868", state="disabled", font=ctk.CTkFont(weight="bold"))
        self.play_btn.pack(pady=20, padx=40, fill="x")

        # 3. Calibration Card
        cal_card = ctk.CTkFrame(phys_page, fg_color=self.bg_dark, corner_radius=15)
        cal_card.grid(row=2, column=0, columnspan=2, pady=30, padx=20, sticky="ew")
        ctk.CTkButton(cal_card, text="🔊 SEND CALIBRATION TONE", command=self.send_calibration, width=250, fg_color="transparent", border_width=1).pack(side="left", padx=20, pady=15)
        ctk.CTkLabel(cal_card, text="Use calibration tone to set perfect volume levels on your physical recorder.", text_color=self.text_dim, font=("Courier", 10)).pack(side="left", padx=10)

    def setup_visualizer(self):
        self.viz_container = ctk.CTkFrame(self, height=220, fg_color=self.bg_dark, border_width=1, border_color="#24283b")
        self.viz_container.grid(row=1, column=1, padx=30, pady=(0, 30), sticky="nsew")
        self.viz_container.grid_columnconfigure(0, weight=1); self.viz_container.grid_columnconfigure(1, weight=1)
        self.console_text = ctk.CTkTextbox(self.viz_container, font=("Courier", 12), text_color=self.accent_green, fg_color="black", border_width=0)
        self.console_text.grid(row=0, column=0, padx=15, pady=15, sticky="nsew")
        self.fig = Figure(figsize=(5, 3), dpi=80, facecolor="black")
        self.ax = self.fig.add_subplot(111); self.ax.set_facecolor("black")
        self.ax.get_xaxis().set_visible(False); self.ax.get_yaxis().set_visible(False)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.viz_container)
        self.canvas.get_tk_widget().grid(row=0, column=1, padx=15, pady=15, sticky="nsew")
        self.spec_data = np.zeros((64, 128))
        self.im = self.ax.imshow(self.spec_data, aspect='auto', origin='lower', cmap='viridis', vmin=0, vmax=10)
        self.fig.tight_layout(pad=0)

    def update_spectrogram(self, activity=1.0, data=None):
        if data is not None:
            # Show actual waveform intensity if data is provided
            intensity = np.abs(np.fft.rfft(data[:1024]))[:64]
            new_col = intensity.reshape(-1, 1) * 2
        else:
            new_col = np.random.rand(64, 1) * (activity * 10)
        
        self.spec_data = np.append(self.spec_data[:, 1:], new_col, axis=1)
        self.im.set_array(self.spec_data)
        self.canvas.draw()

    def start_heartbeat(self):
        if not self.processing and not self.recording: self.update_spectrogram(activity=0.15)
        self.after(120, self.start_heartbeat)

    def log(self, msg):
        timestamp = time.strftime("%H:%M:%S")
        self.console_text.insert("end", f">>> [{timestamp}] {msg}\n"); self.console_text.see("end")

    def show_page(self, name):
        for p in self.pages.values(): p.grid_forget()
        self.pages[name].grid(row=0, column=0, sticky="nsew")
        self.enc_btn.configure(fg_color=self.accent_blue if name == "encrypt" else "transparent", text_color="#16161e" if name == "encrypt" else self.text_color)
        self.dec_btn.configure(fg_color=self.accent_cyan if name == "decrypt" else "transparent", text_color="#16161e" if name == "decrypt" else self.text_color)
        self.phys_btn.configure(fg_color=self.accent_purple if name == "physical" else "transparent", text_color="#16161e" if name == "physical" else self.text_color)

    # --- PHYSICAL MEDIA LOGIC ---
    def toggle_recording(self):
        if not self.recording:
            self.recording = True
            self.recorded_data = []
            self.record_btn.configure(text="⏹️ STOP CAPTURE", fg_color=self.accent_red)
            self.log("REC: Initializing line-in capture stream...")
            threading.Thread(target=self.run_record_thread).start()
        else:
            self.recording = False
            self.record_btn.configure(text="🔴 START RECORDING", fg_color="#414868")
            self.save_rec_btn.configure(state="normal", fg_color=self.accent_green, text_color="#16161e")
            self.log(f"REC: Capture complete. Buffer: {len(self.recorded_data)/44100:.1f}s")

    def run_record_thread(self):
        def callback(indata, frames, time, status):
            if self.recording:
                self.recorded_data.extend(indata[:, 0])
                self.after(0, lambda: self.update_spectrogram(data=indata[:, 0]))
        
        with sd.InputStream(samplerate=44100, channels=1, callback=callback):
            while self.recording: sd.sleep(100)

    def save_recording(self):
        path = filedialog.asksaveasfilename(defaultextension=".wav", title="Save Captured Vault")
        if path:
            sf.write(path, np.array(self.recorded_data), 44100)
            self.log(f"REC: Vault saved to disk: {os.path.basename(path)}")
            messagebox.showinfo("Success", "Captured Signal Saved!")

    def select_broadcast_vault(self):
        path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if path:
            self.broad_path = path
            self.broad_label.configure(text=f"LOADED: {os.path.basename(path)}", text_color=self.accent_blue)
            self.play_btn.configure(state="normal", fg_color=self.accent_blue, text_color="#16161e")

    def play_vault(self):
        self.log(f"PLAY: Broadcasting {os.path.basename(self.broad_path)} to output...")
        data, fs = sf.read(self.broad_path)
        sd.play(data, fs)
        self.processing = True
        threading.Thread(target=self.monitor_playback, args=(data,)).start()

    def monitor_playback(self, data):
        # Update viz while playing
        chunk_size = 1024
        for i in range(0, len(data), chunk_size):
            if not self.processing: break
            self.after(0, lambda d=data[i:i+chunk_size]: self.update_spectrogram(data=d))
            time.sleep(chunk_size/44100)
        self.processing = False
        self.after(0, self.reset_ui)

    def send_calibration(self):
        self.log("CAL: Sending 1000Hz reference tone...")
        t = np.linspace(0, 5, 44100 * 5)
        tone = 0.5 * np.sin(2 * np.pi * 1000 * t)
        sd.play(tone, 44100)

    # --- STANDARD LOGIC ---
    def toggle_stealth(self):
        if self.stealth_toggle.get(): self.music_btn.configure(state="normal", fg_color=self.accent_cyan, text_color="#16161e")
        else: self.music_btn.configure(state="disabled", fg_color="#24283b", text_color=self.text_color); self.carrier_path = None

    def select_music(self):
        path = filedialog.askopenfilename(filetypes=[("Audio files", "*.wav *.mp3")])
        if path: self.carrier_path = path; self.music_btn.configure(text=f"🎵 {os.path.basename(path)[:15]}...")

    def update_pass_strength(self, event=None):
        password = self.enc_pass.get()
        if not password: self.strength_bar.set(0); return
        score = sum([len(password) >= 10, any(c.isdigit() for c in password), any(c.isupper() for c in password), any(c in "!@#$%^&*()" for c in password)])
        colors = [self.accent_red, "#e67e22", "#f1c40f", self.accent_green, self.accent_cyan]
        self.strength_bar.set((score + 1) / 5); self.strength_bar.configure(progress_color=colors[min(score, 4)])

    def add_to_queue_files(self):
        paths = filedialog.askopenfilenames()
        if paths:
            for p in paths:
                if p not in self.queue: self.queue.append(p)
            self.refresh_queue_view()

    def add_to_queue_folder(self):
        path = filedialog.askdirectory()
        if path:
            if path not in self.queue: self.queue.append(path)
            self.refresh_queue_view()

    def clear_queue(self): self.queue = []; self.refresh_queue_view()

    def refresh_queue_view(self):
        self.queue_box.configure(state="normal"); self.queue_box.delete("0.0", "end")
        total_size = 0
        if not self.queue: self.queue_box.insert("0.0", "--- PIPELINE STANDBY ---"); self.start_enc_btn.configure(state="disabled", fg_color="#414868")
        else:
            for p in self.queue:
                size = self.get_size(p); total_size += size
                self.queue_box.insert("end", f" [LOADED] {os.path.basename(p)} ({size/1024/1024:.2f} MB)\n")
            self.start_enc_btn.configure(state="normal", fg_color=self.accent_blue, text_color="#16161e")
        self.total_size_label.configure(text=f"PIPELINE LOAD: {total_size/1024/1024:.2f} MB")
        self.queue_box.configure(state="disabled")

    def get_size(self, start_path):
        total_size = 0
        if os.path.isfile(start_path): return os.path.getsize(start_path)
        for dirpath, dirnames, filenames in os.walk(start_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp): total_size += os.path.getsize(fp)
        return total_size

    def start_encryption(self):
        password = self.enc_pass.get()
        if not password: return
        out_p = filedialog.asksaveasfilename(defaultextension=".wav", title="Save Sonic Vault")
        if not out_p: return
        self.processing = True; self.start_enc_btn.configure(state="disabled", text="⚡ MODULATING SIGNAL...")
        threading.Thread(target=self.run_enc_thread, args=(password, out_p)).start()

    def run_enc_thread(self, password, out_p):
        temp_tar = out_p + ".staging.tar.gz"
        try:
            self.log("INIT: Compressing binary payload...")
            with tarfile.open(temp_tar, "w:gz") as tar:
                for item in self.queue: tar.add(item, arcname=os.path.basename(item))
            crypto = CryptoEngine(password)
            with open(temp_tar, 'rb') as f: data = f.read()
            enc_data = crypto.encrypt_data(data)
            temp_enc = out_p + ".payload"
            with open(temp_enc, 'wb') as f: f.write(enc_data)
            self.log("DSP: Handshaking with v3.0 Modem...")
            encoder = ToneEncoder()
            def progress(p):
                self.after(0, lambda: self.update_spectrogram(activity=1.0))
                if int(p*100) % 20 == 0: self.log(f"STREAM: {int(p*100)}% Synchronized")
            encoder.encode_file(temp_enc, out_p, carrier_music_path=self.carrier_path, progress_callback=progress)
            os.remove(temp_tar); os.remove(temp_enc)
            self.after(0, lambda: messagebox.showinfo("Success", "VAULT SEALED SUCCESSFULLY")); self.after(0, self.clear_queue)
        except Exception as e: self.log(f"CRITICAL: {str(e)}")
        finally: self.processing = False; self.after(0, self.reset_ui)

    def select_vault(self):
        path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if path:
            self.decrypt_path = path
            self.vault_label.configure(text=f"VAULT DETECTED: {os.path.basename(path)}", text_color=self.accent_green)
            self.start_dec_btn.configure(state="normal", fg_color=self.accent_cyan, text_color="#16161e")

    def start_decryption(self):
        password = self.dec_pass.get()
        if not password: return
        out_dir = filedialog.askdirectory(title="Select Restoration Site")
        if not out_dir: return
        self.processing = True; self.start_dec_btn.configure(state="disabled", text="⚡ CAPTURING SIGNAL...")
        threading.Thread(target=self.run_dec_thread, args=(password, out_dir)).start()

    def run_dec_thread(self, password, out_dir):
        temp_dec = os.path.join(out_dir, "restoration.tmp")
        try:
            self.log("SYNC: Searching for Barker Preamble...")
            decoder = ToneDecoder()
            for _ in range(10): self.after(0, lambda: self.update_spectrogram(activity=1.0)); time.sleep(0.1)
            decoder.decode_wav(self.decrypt_path, temp_dec)
            crypto = CryptoEngine(password)
            with open(temp_dec, 'rb') as f: enc_data = f.read()
            clean_tar_data = crypto.decrypt_data(enc_data)
            temp_tar = temp_dec + ".tar.gz"
            with open(temp_tar, 'wb') as f: f.write(clean_tar_data)
            with tarfile.open(temp_tar, "r:gz") as tar: tar.extractall(path=out_dir)
            os.remove(temp_dec); os.remove(temp_tar)
            self.log(f"✅ SUCCESS: Restoration complete at {out_dir}")
            self.after(0, lambda: messagebox.showinfo("Success", "DATA RESTORED PERFECTLY"))
        except Exception as e: self.log(f"FAIL: {str(e)}")
        finally: self.processing = False; self.after(0, self.reset_ui)

    def reset_ui(self):
        self.start_enc_btn.configure(state="normal" if self.queue else "disabled", text="🚀 EXECUTE DATA SONIFICATION")
        self.start_dec_btn.configure(state="normal", text="🚀 BEGIN SIGNAL RECOVERY")
        self.play_btn.configure(text="▶️ BROADCAST TO TAPE/CD")

if __name__ == "__main__":
    app = ToneCryptApp()
    app.mainloop()
