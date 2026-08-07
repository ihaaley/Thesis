# =============================================================================
#  HINWEIS ZU DIESER DATEI  (auf Deutsch)
# =============================================================================
#  DIESE DATEI IST VOLLSTÄNDIG AUSKOMMENTIERT UND WIRD NICHT AUSGEFÜHRT.
#  Jede Zeile beginnt mit '#', Python überspringt sie also komplett.
#
#  WAS ES IST:
#  Ein früheres, eigenständiges Experiment aus derselben Lernphase — eine
#  Gesichtserkennungs-Oberfläche mit Tkinter:
#     • YOLOv8n-face findet Gesichter im Videobild
#     • DeepFace (Modell "SFace") berechnet daraus einen Merkmalsvektor
#     • per Kosinus-Ähnlichkeit wird gegen hinterlegte Referenzbilder verglichen
#     • bei einer Übereinstimmung erscheint ein Warnhinweis
#
#  WAS ES MIT DER THESIS ZU TUN HAT:
#  Inhaltlich nichts — hier geht es um Gesichter, nicht um Longboard-Tricks.
#  Methodisch war es die Vorübung: dieselben Bausteine (YOLO, OpenCV-Videoschleife,
#  Analyse in einem Hintergrund-Thread, grafische Oberfläche) tauchen in
#  main.py und viewer.py in ausgereifter Form wieder auf. Der Umstieg von
#  Tkinter auf PyQt5 in viewer.py ist eine direkte Folge dieser Erfahrung.
#
#  FÜR DIE ABGABE:
#  Die Datei kann bedenkenlos gelöscht werden — kein anderes Skript importiert
#  sie. Behalten lohnt sich nur, wenn die Entwicklungsschritte dokumentiert
#  werden sollen.
# =============================================================================

# """
# Face Detection GUI — YOLOv8n-face + Tkinter
# ============================================
# Layout:
#   Left  — live video feed with bounding boxes
#   Right — scrollable panel of every detected face crop

# Run:
#     python test.py --source data/video.mp4
#     python test.py --source 0              # webcam
#     python test.py --source data/video.mp4 --conf 0.5 --model-size s --save
# """

# import argparse
# import os
# import threading
# import time
# import tkinter as tk
# from tkinter import ttk, filedialog, messagebox

# import cv2
# import numpy as np
# import torch
# from PIL import Image, ImageTk
# from ultralytics import YOLO

# # ── Constants ──────────────────────────────────────────────────────────────────
# BOX_COLOR = (0, 220, 255)   # unknown face — cyan
# ALERT_COLOR = (0, 0, 255)     # known face   — red
# THUMB_SIZE = (120, 120)
# MAX_THUMBS = 200
# THUMBS_PER_ROW = 2
# RECOG_THRESH = 0.40            # cosine similarity threshold — SFace: 1 - 0.593 ≈ 0.40
# # run recognition every N frames (speed vs latency)
# RECOG_EVERY = 5


# # ── Helpers ───────────────────────────────────────────────────────────────────
# def cv2_to_tk(img_bgr, max_w, max_h):
#     """Resize a BGR frame to fit (max_w × max_h) and convert to ImageTk."""
#     h, w = img_bgr.shape[:2]
#     scale = min(max_w / w, max_h / h, 1.0)
#     new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
#     img_rgb = cv2.cvtColor(cv2.resize(
#         img_bgr, (new_w, new_h)), cv2.COLOR_BGR2RGB)
#     return ImageTk.PhotoImage(Image.fromarray(img_rgb))


# def crop_face(frame, box, pad=10):
#     """Return a padded face crop from the frame."""
#     h, w = frame.shape[:2]
#     x1, y1, x2, y2 = map(int, box.xyxy[0])
#     x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
#     x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
#     return frame[y1:y2, x1:x2]


# # ── GUI Application ────────────────────────────────────────────────────────────
# class FaceDetectorApp(tk.Tk):
#     def __init__(self, source, conf, model_size, save, output):
#         super().__init__()
#         self.title("Face Detection — YOLOv8")
#         self.geometry("1280x720")
#         self.resizable(True, True)
#         self.configure(bg="#1e1e1e")

#         # If no source given, ask the user at startup
#         if source is None:
#             source = self._ask_source()
#             if source is None:
#                 self.destroy()
#                 return

#         self.file_source = source    # original file/CLI source
#         self.source = source
#         self.conf = conf
#         self.save = save
#         self.output = output
#         self.running = True
#         self.paused = False
#         self.face_count = 0
#         self.fps_val = 0.0
#         self._last_frame = None
#         self._using_camera = False
#         self._switch_requested = False
#         self._next_source = None

#         # ── Face recognition state ────────────────────────────────────────────
#         self.ref_embeddings = {}    # {name: embedding_vector}
#         self._frame_idx = 0
#         self._alert_until = 0       # time.time() until which alert banner shows

#         # ── Device ────────────────────────────────────────────────────────────
#         self.device = ("mps" if torch.backends.mps.is_available() else
#                        "cuda" if torch.cuda.is_available() else "cpu")

#         # ── Model ─────────────────────────────────────────────────────────────
#         model_name = f"yolov8{model_size}-face.pt"
#         print(f"[INFO] Loading {model_name}  |  device: {self.device}")
#         self.model = YOLO(model_name)

#         # ── Layout ────────────────────────────────────────────────────────────
#         self._build_ui()

#         # ── Start capture thread ──────────────────────────────────────────────
#         self.cap = None
#         self.writer = None
#         # Defer thread start so the window is fully drawn first
#         self.after(100, self._start_thread)

#         self.protocol("WM_DELETE_WINDOW", self._on_close)

#     def _start_thread(self):
#         self.thread = threading.Thread(target=self._capture_loop, daemon=True)
#         self.thread.start()

#     def _ask_source(self):
#         """Show a simple dialog to pick a file or use the camera."""
#         dialog = tk.Toplevel(self)
#         dialog.title("Select Source")
#         dialog.configure(bg="#2d2d2d")
#         dialog.resizable(False, False)
#         dialog.grab_set()   # modal

#         chosen = [None]

#         tk.Label(dialog, text="Choose a video source",
#                  fg="white", bg="#2d2d2d",
#                  font=("Helvetica", 14, "bold")).pack(pady=(20, 4), padx=30)
#         tk.Label(dialog, text="Open a local video file or use your system camera.",
#                  fg="#aaa", bg="#2d2d2d",
#                  font=("Helvetica", 10)).pack(pady=(0, 20), padx=30)

#         def pick_file():
#             path = filedialog.askopenfilename(
#                 parent=dialog,
#                 title="Select video file",
#                 filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),
#                            ("All files", "*.*")])
#             if path:
#                 chosen[0] = path
#                 dialog.destroy()

#         def use_camera():
#             chosen[0] = "0"
#             dialog.destroy()

#         btn_style = dict(relief=tk.FLAT, font=(
#             "Helvetica", 12), padx=16, pady=8)
#         tk.Button(dialog, text="📂  Open Video File", bg="#1a4a6b", fg="white",
#                   command=pick_file, **btn_style).pack(pady=6, ipadx=10)
#         tk.Button(dialog, text="📷  Use System Camera", bg="#1a6b3c", fg="white",
#                   command=use_camera, **btn_style).pack(pady=6, ipadx=10)
#         tk.Button(dialog, text="Cancel", bg="#555", fg="white",
#                   command=dialog.destroy, **btn_style).pack(pady=(6, 20))

#         self.wait_window(dialog)
#         return chosen[0]

#     # ── UI Construction ───────────────────────────────────────────────────────
#     def _build_ui(self):
#         # ── Top bar ───────────────────────────────────────────────────────────
#         bar = tk.Frame(self, bg="#2d2d2d", pady=6)
#         bar.pack(fill=tk.X)

#         self.lbl_fps = tk.Label(bar, text="FPS: --", fg="#00ff88", bg="#2d2d2d",
#                                 font=("Helvetica", 12, "bold"))
#         self.lbl_fps.pack(side=tk.LEFT, padx=12)

#         self.lbl_count = tk.Label(bar, text="Faces: 0", fg="#00cfff", bg="#2d2d2d",
#                                   font=("Helvetica", 12, "bold"))
#         self.lbl_count.pack(side=tk.LEFT, padx=12)

#         self.lbl_total = tk.Label(bar, text="Crops saved: 0", fg="#ffcc00",
#                                   bg="#2d2d2d", font=("Helvetica", 12))
#         self.lbl_total.pack(side=tk.LEFT, padx=12)

#         tk.Button(bar, text="⏸  Pause / Resume", command=self._toggle_pause,
#                   bg="#444", fg="white", relief=tk.FLAT,
#                   font=("Helvetica", 11), padx=10).pack(side=tk.RIGHT, padx=12)

#         tk.Button(bar, text="🗑  Clear Crops", command=self._clear_crops,
#                   bg="#444", fg="white", relief=tk.FLAT,
#                   font=("Helvetica", 11), padx=10).pack(side=tk.RIGHT, padx=4)

#         self.btn_cam = tk.Button(bar, text="📷  Switch to Camera",
#                                  command=self._toggle_camera,
#                                  bg="#1a6b3c", fg="white", relief=tk.FLAT,
#                                  font=("Helvetica", 11), padx=10)
#         self.btn_cam.pack(side=tk.RIGHT, padx=4)

#         tk.Button(bar, text="📂  Open File", command=self._open_file,
#                   bg="#1a4a6b", fg="white", relief=tk.FLAT,
#                   font=("Helvetica", 11), padx=10).pack(side=tk.RIGHT, padx=4)

#         tk.Button(bar, text="🧑  Load Reference Faces", command=self._load_references,
#                   bg="#6b3a1a", fg="white", relief=tk.FLAT,
#                   font=("Helvetica", 11), padx=10).pack(side=tk.RIGHT, padx=4)

#         self.lbl_refs = tk.Label(bar, text="Refs: 0", fg="#ff9944",
#                                  bg="#2d2d2d", font=("Helvetica", 11))
#         self.lbl_refs.pack(side=tk.RIGHT, padx=6)

#         self.lbl_source = tk.Label(bar, text="▶ FILE", fg="#aaaaaa",
#                                    bg="#2d2d2d", font=("Helvetica", 10))
#         self.lbl_source.pack(side=tk.RIGHT, padx=8)

#         # ── Alert banner (hidden initially) ───────────────────────────────────
#         self.alert_bar = tk.Frame(self, bg="#cc0000", pady=4)
#         # not packed yet — shown on match
#         self.alert_lbl = tk.Label(self.alert_bar, text="", fg="white",
#                                   bg="#cc0000", font=("Helvetica", 13, "bold"))
#         self.alert_lbl.pack()

#         # ── Main area ─────────────────────────────────────────────────────────
#         main = tk.Frame(self, bg="#1e1e1e")
#         main.pack(fill=tk.BOTH, expand=True)

#         # ── Left — video canvas ───────────────────────────────────────────────
#         left = tk.Frame(main, bg="#1e1e1e")
#         left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)

#         tk.Label(left, text="LIVE VIDEO", fg="#888", bg="#1e1e1e",
#                  font=("Helvetica", 9, "bold")).pack(anchor=tk.W)

#         self.canvas = tk.Canvas(left, bg="#000000", highlightthickness=1,
#                                 highlightbackground="#444")
#         self.canvas.pack(fill=tk.BOTH, expand=True)

#         # ── Right — detected faces panel ──────────────────────────────────────
#         right = tk.Frame(main, bg="#1e1e1e", width=290)
#         right.pack(side=tk.RIGHT, fill=tk.Y, padx=6, pady=6)
#         right.pack_propagate(False)

#         tk.Label(right, text="DETECTED FACES", fg="#888", bg="#1e1e1e",
#                  font=("Helvetica", 9, "bold")).pack(anchor=tk.W)

#         # Scrollable canvas inside right panel
#         container = tk.Frame(right, bg="#141414")
#         container.pack(fill=tk.BOTH, expand=True)

#         self.faces_canvas = tk.Canvas(container, bg="#141414",
#                                       highlightthickness=1,
#                                       highlightbackground="#444")
#         scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL,
#                                   command=self.faces_canvas.yview)
#         self.faces_canvas.configure(yscrollcommand=scrollbar.set)
#         scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
#         self.faces_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

#         # Inner frame holds the thumbnail grid
#         self.faces_frame = tk.Frame(self.faces_canvas, bg="#141414")
#         self._fw_id = self.faces_canvas.create_window(
#             (0, 0), window=self.faces_frame, anchor=tk.NW)
#         self.faces_frame.bind("<Configure>",
#                               lambda e: self.faces_canvas.configure(
#                                   scrollregion=self.faces_canvas.bbox("all")))

#         # Mouse-wheel scroll (macOS sends <MouseWheel>, Linux sends Button-4/5)
#         self.faces_canvas.bind("<MouseWheel>",
#                                lambda e: self.faces_canvas.yview_scroll(-1 * (e.delta // 120), "units"))

#         self._thumb_col = 0
#         self._thumb_row = 0

#         # ── Reference faces list (below detected faces) ───────────────────────
#         tk.Label(right, text="REFERENCE FACES", fg="#888", bg="#1e1e1e",
#                  font=("Helvetica", 9, "bold")).pack(anchor=tk.W, pady=(8, 0))

#         refs_scroll_container = tk.Frame(right, bg="#141414", height=120)
#         refs_scroll_container.pack(fill=tk.X)
#         refs_scroll_container.pack_propagate(False)

#         refs_canvas = tk.Canvas(refs_scroll_container, bg="#141414",
#                                 highlightthickness=1, highlightbackground="#444")
#         refs_sb = ttk.Scrollbar(refs_scroll_container, orient=tk.VERTICAL,
#                                 command=refs_canvas.yview)
#         refs_canvas.configure(yscrollcommand=refs_sb.set)
#         refs_sb.pack(side=tk.RIGHT, fill=tk.Y)
#         refs_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

#         self.refs_frame = tk.Frame(refs_canvas, bg="#141414")
#         refs_canvas.create_window((0, 0), window=self.refs_frame, anchor=tk.NW)
#         self.refs_frame.bind("<Configure>",
#                              lambda e: refs_canvas.configure(
#                                  scrollregion=refs_canvas.bbox("all")))

#     # ── Capture / Inference thread ────────────────────────────────────────────
#     def _open_cap(self, source):
#         try:
#             src = int(source)
#         except ValueError:
#             src = source
#         cap = cv2.VideoCapture(src)
#         return cap

#     def _capture_loop(self):
#         self.cap = self._open_cap(self.source)
#         if not self.cap.isOpened():
#             print(f"[ERROR] Cannot open: {self.source}")
#             return

#         w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

#         if self.save:
#             fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#             self.writer = cv2.VideoWriter(self.output, fourcc, fps, (w, h))
#             print(f"[INFO] Saving to: {self.output}")

#         prev = time.time()

#         while self.running:
#             # ── Handle source switch request ──────────────────────────────
#             if self._switch_requested:
#                 self._switch_requested = False
#                 self.cap.release()
#                 self.source = self._next_source
#                 self.cap = self._open_cap(self.source)
#                 if not self.cap.isOpened():
#                     print(f"[ERROR] Cannot open: {self.source}")
#                     # Reset UI on main thread
#                     self.after(0, lambda: self.lbl_source.config(
#                         text="[ERROR] cannot open source"))
#                     continue
#                 # Warm up: discard a few frames so camera gives real image
#                 if self._using_camera:
#                     for _ in range(5):
#                         self.cap.read()
#                 print(f"[INFO] Switched source to: {self.source}")

#             if self.paused:
#                 time.sleep(0.05)
#                 continue

#             ret, frame = self.cap.read()
#             if not ret:
#                 if self._using_camera:
#                     # Camera hiccup — just retry next iteration
#                     time.sleep(0.03)
#                     continue
#                 # End of file — loop back
#                 self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
#                 continue

#             # ── Inference ────────────────────────────────────────────────────
#             results = self.model(frame, conf=self.conf,
#                                  device=self.device, verbose=False)
#             boxes = results[0].boxes

#             # ── Annotate + recognise ─────────────────────────────────────────
#             annotated = frame.copy()
#             crops = []
#             alerts = []          # names of matched people this frame
#             self._frame_idx += 1
#             run_recog = (self._frame_idx % RECOG_EVERY == 0
#                          and bool(self.ref_embeddings))

#             for box in boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 conf_val = float(box.conf[0])
#                 face_crop = crop_face(frame, box)
#                 crops.append(face_crop)

#                 # Recognition (every N frames to stay real-time)
#                 matched_name = None
#                 if run_recog and face_crop.size > 0:
#                     matched_name, score = self._match_face(face_crop)
#                     if matched_name:
#                         alerts.append(f"{matched_name} ({score:.0%})")

#                 color = ALERT_COLOR if matched_name else BOX_COLOR
#                 thick = 3 if matched_name else 2
#                 cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thick)
#                 label = matched_name if matched_name else f"{conf_val:.2f}"
#                 (tw, th), _ = cv2.getTextSize(
#                     label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
#                 cv2.rectangle(annotated,
#                               (x1, y1 - th - 6), (x1 + tw + 4, y1),
#                               color, -1)
#                 cv2.putText(annotated, label, (x1 + 2, y1 - 4),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
#                             cv2.LINE_AA)

#             if alerts:
#                 self.after(0, self._show_alert, alerts)

#             # ── FPS ──────────────────────────────────────────────────────────
#             now = time.time()
#             self.fps_val = 1.0 / max(now - prev, 1e-6)
#             self.face_count = len(boxes)
#             prev = now

#             cv2.putText(annotated,
#                         f"FPS: {self.fps_val:.1f}  Faces: {self.face_count}",
#                         (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
#                         (0, 255, 0), 2, cv2.LINE_AA)

#             if self.writer:
#                 self.writer.write(annotated)

#             self._last_frame = annotated

#             # Push updates to GUI on the main thread
#             self.after(0, self._update_video, annotated)
#             if crops:
#                 self.after(0, self._add_crops, crops)
#             self.after(0, self._update_stats)

#         self.cap.release()
#         if self.writer:
#             self.writer.release()
#         print("[INFO] Capture stopped.")

#     # ── GUI update callbacks (main thread) ───────────────────────────────────
#     def _update_video(self, frame):
#         cw = self.canvas.winfo_width()
#         ch = self.canvas.winfo_height()
#         if cw < 2 or ch < 2:
#             return
#         photo = cv2_to_tk(frame, cw, ch)
#         self.canvas.delete("all")
#         self.canvas.create_image(
#             cw // 2, ch // 2, anchor=tk.CENTER, image=photo)
#         self.canvas._photo = photo      # prevent GC

#     def _show_alert(self, names):
#         """Show a centered modal-style alert dialog for each matched face."""
#         # Also keep the top banner visible
#         msg = "🚨  MATCH FOUND: " + ",  ".join(names)
#         self.alert_lbl.config(text=msg)
#         if not self.alert_bar.winfo_ismapped():
#             self.alert_bar.pack(fill=tk.X)
#             self.alert_bar.lift()
#         self._alert_until = time.time() + 3.0
#         self.after(3100, self._hide_alert)

#         # Centered overlay dialog
#         dlg = tk.Toplevel(self)
#         dlg.overrideredirect(True)          # no OS title bar
#         dlg.configure(bg="#cc0000")
#         dlg.attributes("-topmost", True)

#         # Content frame with padding / rounded feel
#         inner = tk.Frame(dlg, bg="#1a0000", padx=30, pady=24)
#         inner.pack(padx=3, pady=3)

#         tk.Label(inner, text="🚨  FACE DETECTED",
#                  fg="#ff4444", bg="#1a0000",
#                  font=("Helvetica", 20, "bold")).pack(pady=(0, 10))

#         for name_score in names:
#             tk.Label(inner, text=name_score,
#                      fg="white", bg="#1a0000",
#                      font=("Helvetica", 16)).pack()

#         tk.Label(inner, text="Match found in live feed",
#                  fg="#aaa", bg="#1a0000",
#                  font=("Helvetica", 11)).pack(pady=(8, 14))

#         tk.Button(inner, text="  Dismiss  ", command=dlg.destroy,
#                   bg="#cc0000", fg="white", relief=tk.FLAT,
#                   font=("Helvetica", 12, "bold"),
#                   activebackground="#ff2222",
#                   cursor="hand2").pack()

#         # Center on screen
#         self.update_idletasks()
#         dlg.update_idletasks()
#         sw = self.winfo_screenwidth()
#         sh = self.winfo_screenheight()
#         dw = dlg.winfo_reqwidth()
#         dh = dlg.winfo_reqheight()
#         dlg.geometry(f"+{(sw - dw) // 2}+{(sh - dh) // 2}")

#         # Auto-close after 5 seconds
#         dlg.after(5000, lambda: dlg.destroy() if dlg.winfo_exists() else None)

#     def _hide_alert(self):
#         if time.time() >= self._alert_until:
#             self.alert_bar.pack_forget()

#     def _add_crops(self, crops):
#         for crop in crops:
#             if crop.size == 0:
#                 continue
#             thumb_rgb = cv2.cvtColor(
#                 cv2.resize(crop, THUMB_SIZE), cv2.COLOR_BGR2RGB)
#             photo = ImageTk.PhotoImage(Image.fromarray(thumb_rgb))

#             lbl = tk.Label(self.faces_frame, image=photo,
#                            bg="#141414", relief=tk.FLAT, bd=2,
#                            highlightbackground="#333",
#                            highlightthickness=1)
#             lbl.image = photo           # prevent GC
#             lbl.grid(row=self._thumb_row, column=self._thumb_col,
#                      padx=3, pady=3)

#             self._thumb_col += 1
#             if self._thumb_col >= THUMBS_PER_ROW:
#                 self._thumb_col = 0
#                 self._thumb_row += 1

#             # Drop oldest crop when over limit
#             children = self.faces_frame.winfo_children()
#             if len(children) > MAX_THUMBS:
#                 children[0].destroy()

#         # Auto-scroll to bottom
#         self.faces_canvas.yview_moveto(1.0)

#     def _update_stats(self):
#         self.lbl_fps.config(text=f"FPS: {self.fps_val:.1f}")
#         self.lbl_count.config(text=f"Faces: {self.face_count}")
#         total = len(self.faces_frame.winfo_children())
#         self.lbl_total.config(text=f"Crops saved: {total}")

#     # ── Controls ──────────────────────────────────────────────────────────────
#     # ── Reference face loading ────────────────────────────────────────────────
#     def _load_references(self):
#         """Pick one or more image files. Ask for a name for each one."""
#         paths = filedialog.askopenfilenames(
#             title="Select reference face image(s)",
#             filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
#                        ("All files", "*.*")])
#         if not paths:
#             return

#         loaded, failed = 0, 0
#         for img_path in paths:
#             default_name = os.path.splitext(os.path.basename(img_path))[0]
#             # Ask user to confirm / edit the name for this face
#             name = self._ask_name(img_path, default_name)
#             if name is None:        # user clicked Cancel for this image
#                 continue
#             name = name.strip() or default_name

#             print(f"[REF] Processing '{name}' from {img_path}...")
#             emb = self._get_embedding_from_file(img_path)
#             if emb is not None:
#                 self.ref_embeddings[name] = emb
#                 loaded += 1
#                 print(f"[REF] ✓ Loaded '{name}'")
#             else:
#                 failed += 1
#                 print(
#                     f"[REF] ✗ No face detected in '{os.path.basename(img_path)}'")

#         self.lbl_refs.config(text=f"Refs: {len(self.ref_embeddings)}")
#         self._refresh_refs_panel()
#         msg = f"{loaded} face(s) loaded successfully."
#         if failed:
#             msg += f"\n{failed} image(s) had no detectable face — try a clearer front-facing photo."
#         messagebox.showinfo("References loaded", msg)

#     def _ask_name(self, img_path, default_name):
#         """Show a small dialog with the image preview and a name entry."""
#         dialog = tk.Toplevel(self)
#         dialog.title("Name this person")
#         dialog.configure(bg="#2d2d2d")
#         dialog.resizable(False, False)
#         dialog.grab_set()

#         result = [None]

#         # Preview thumbnail
#         try:
#             img = Image.open(img_path).convert("RGB")
#             img.thumbnail((160, 160))
#             photo = ImageTk.PhotoImage(img)
#             lbl_img = tk.Label(dialog, image=photo, bg="#2d2d2d")
#             lbl_img.image = photo
#             lbl_img.pack(pady=(14, 4))
#         except Exception:
#             pass

#         tk.Label(dialog, text="Enter the person's name:",
#                  fg="#ccc", bg="#2d2d2d",
#                  font=("Helvetica", 11)).pack(pady=(4, 2))

#         entry = tk.Entry(dialog, font=("Helvetica", 12), width=22,
#                          bg="#444", fg="white", insertbackground="white")
#         entry.insert(0, default_name)
#         entry.select_range(0, tk.END)
#         entry.pack(padx=20, pady=4)
#         entry.focus_set()

#         def confirm(event=None):
#             result[0] = entry.get()
#             dialog.destroy()

#         def cancel():
#             result[0] = None
#             dialog.destroy()

#         btn_frame = tk.Frame(dialog, bg="#2d2d2d")
#         btn_frame.pack(pady=12)
#         tk.Button(btn_frame, text="✓  Add", command=confirm,
#                   bg="#1a6b3c", fg="white", relief=tk.FLAT,
#                   font=("Helvetica", 11), padx=12).pack(side=tk.LEFT, padx=6)
#         tk.Button(btn_frame, text="Skip", command=cancel,
#                   bg="#555", fg="white", relief=tk.FLAT,
#                   font=("Helvetica", 11), padx=12).pack(side=tk.LEFT, padx=6)
#         entry.bind("<Return>", confirm)

#         self.wait_window(dialog)
#         return result[0]

#     def _refresh_refs_panel(self):
#         """Rebuild the references list shown under the right panel."""
#         for w in self.refs_frame.winfo_children():
#             w.destroy()
#         for i, name in enumerate(self.ref_embeddings):
#             row = tk.Frame(self.refs_frame, bg="#1e1e1e")
#             row.pack(fill=tk.X, padx=4, pady=1)
#             tk.Label(row, text=f"● {name}", fg="#ff9944", bg="#1e1e1e",
#                      font=("Helvetica", 10)).pack(side=tk.LEFT)
#             tk.Button(row, text="✕", fg="#ff4444", bg="#1e1e1e",
#                       relief=tk.FLAT, font=("Helvetica", 9),
#                       command=lambda n=name: self._remove_ref(n)).pack(side=tk.RIGHT)

#     def _remove_ref(self, name):
#         self.ref_embeddings.pop(name, None)
#         self.lbl_refs.config(text=f"Refs: {len(self.ref_embeddings)}")
#         self._refresh_refs_panel()
#         print(f"[REF] Removed '{name}'")

#     def _get_embedding_from_file(self, path):
#         """Return a normalised DeepFace embedding for the first face in an image, or None."""
#         try:
#             from deepface import DeepFace
#             result = DeepFace.represent(img_path=path, model_name="SFace",
#                                         enforce_detection=False, detector_backend="opencv")
#             vec = np.array(result[0]["embedding"], dtype=np.float32)
#             return vec / (np.linalg.norm(vec) + 1e-6)
#         except Exception as e:
#             print(f"[REF] Embedding failed for {path}: {e}")
#             return None

#     def _get_embedding_from_crop(self, crop_bgr):
#         """Return a normalised embedding for a face crop (BGR numpy array), or None."""
#         try:
#             from deepface import DeepFace
#             crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
#             result = DeepFace.represent(img_path=crop_rgb, model_name="SFace",
#                                         enforce_detection=False, detector_backend="skip")
#             vec = np.array(result[0]["embedding"], dtype=np.float32)
#             return vec / (np.linalg.norm(vec) + 1e-6)
#         except Exception:
#             return None

#     def _match_face(self, crop_bgr):
#         """Return (name, score) if crop matches a reference, else (None, 0)."""
#         if not self.ref_embeddings:
#             return None, 0.0
#         emb = self._get_embedding_from_crop(crop_bgr)
#         if emb is None:
#             return None, 0.0
#         best_name, best_score = None, -1.0
#         for name, ref_emb in self.ref_embeddings.items():
#             # cosine similarity (both normalised vectors)
#             score = float(np.dot(emb, ref_emb))
#             if score > best_score:
#                 best_name, best_score = name, score
#         print(
#             f"[RECOG] best={best_name} score={best_score:.3f} thresh={RECOG_THRESH}")
#         if best_score >= RECOG_THRESH:
#             return best_name, best_score
#         return None, best_score

#     def _open_file(self):
#         """Open a file picker and switch to the chosen video file."""
#         path = filedialog.askopenfilename(
#             title="Select video file",
#             filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),
#                        ("All files", "*.*")])
#         if not path:
#             return
#         self._using_camera = False
#         self.file_source = path
#         self._next_source = path
#         self._switch_requested = True
#         self.btn_cam.config(text="📷  Switch to Camera", bg="#1a6b3c")
#         self.lbl_source.config(text=f"▶ {path.split('/')[-1]}")

#     def _toggle_camera(self):
#         if not self._using_camera:
#             # Switch to webcam
#             self._using_camera = True
#             self._next_source = "0"
#             self._switch_requested = True
#             self.btn_cam.config(text="🎬  Switch to File", bg="#6b1a1a")
#             self.lbl_source.config(text="▶ CAMERA")
#         else:
#             # Switch back to last file
#             self._using_camera = False
#             self._next_source = self.file_source
#             self._switch_requested = True
#             self.btn_cam.config(text="📷  Switch to Camera", bg="#1a6b3c")
#             self.lbl_source.config(
#                 text=f"▶ {self.file_source.split('/')[-1]}" if self.file_source else "▶ FILE")

#     def _toggle_pause(self):
#         self.paused = not self.paused

#     def _clear_crops(self):
#         for w in self.faces_frame.winfo_children():
#             w.destroy()
#         self._thumb_col = 0
#         self._thumb_row = 0

#     def _on_close(self):
#         self.running = False
#         self.destroy()


# # ── CLI ────────────────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="YOLOv8 face detection GUI")
#     parser.add_argument("--source",     default=None,
#                         help="Video file path or webcam index. Omit to pick via GUI dialog.")
#     parser.add_argument("--conf",       type=float, default=0.45,
#                         help="Confidence threshold (default: 0.45)")
#     parser.add_argument("--save",       action="store_true",
#                         help="Save annotated output video")
#     parser.add_argument("--output",     default="output_faces.mp4",
#                         help="Output path (default: output_faces.mp4)")
#     parser.add_argument("--model-size", default="n",
#                         choices=["n", "s", "m", "l", "x"],
#                         help="YOLOv8 size: n/s/m/l/x (default: n = fastest)")
#     args = parser.parse_args()

#     app = FaceDetectorApp(
#         source=args.source,
#         conf=args.conf,
#         model_size=args.model_size,
#         save=args.save,
#         output=args.output,
#     )
#     app.mainloop()
