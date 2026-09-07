"""
Muse 2 Standalone Online Testing Engine (Real-Time Feedback)
-------------------------------------------------------------
Upgraded with robust BCI decision logic, pre-flight signal quality checks,
explicit visual UI mapping, dynamic task routing (Math vs. Words),
corrected Beta/Alpha ratio feature mapping, OS-level window lock,
screenshot hotkeys, and automated evaluation report generation.
"""

import os
import sys
import time
import pygame
import joblib
import multiprocessing
import numpy as np
from scipy import signal
from collections import deque
from pylsl import StreamInlet, resolve_byprop
from muselsl import stream, list_muses
import random

# CONFIGURATION & DECISION LOGIC PARAMETERS
# Default lookup directory; will be verified against personal_info.txt
BASE_DATA_DIR = r"D:\MuseData"

SAMPLING_RATE = 256
EEG_CHANNELS = ["TP9", "AF7", "AF8", "TP10"]
EPOCH_LENGTH_SEC = 2.0
SAMPLES_PER_EPOCH = int(SAMPLING_RATE * EPOCH_LENGTH_SEC)

# 0.25s hop provides a snappy 4 Hz refresh rate on the UI
HOP_SAMPLES = int(SAMPLING_RATE * 0.25)
REFRESH_RATE_HZ = SAMPLING_RATE / HOP_SAMPLES

# Artifact & Quality Thresholds
ARTIFACT_THRESHOLD_UV = 350.0
FLATLINE_THRESHOLD_UV = 2.0  # Detects loose sensors

# THE 4-STEP DECISION RECIPE
SMOOTHING_WINDOW_SIZE = 8
CONFIDENCE_THRESHOLD = 0.80
DWELL_TARGET_HITS = 5
REFRACTORY_PERIOD_SEC = 3.0

# STRUCTURED TESTING PARADIGM SETTINGS
REST_PHASE_DURATION_SEC = 30
TARGET_COMMAND_COUNT = 5

# Dynamic Task Refresh Rates
TASK_REFRESH_MATH = 4
TASK_REFRESH_WORDS = 2

# UI Colors
BG_DARK = (20, 20, 25)
TEXT_LIGHT = (211, 211, 211)
COLOR_RELAX = (52, 152, 219)
COLOR_FOCUS = (46, 204, 113)
COLOR_WARN = (231, 76, 60)
COLOR_PROGRESS = (155, 89, 182)

# Global pool of distinct technical words for the ONLINE session
WORD_POOL = [
    'PETROLEUM', 'HYDROGEN', 'MOLECULE', 'POLYMER', 'CATALYST',
    'EFFICIENCY', 'ORGANIC', 'BIOMASS', 'METHANE', 'PIPELINE',
    'METRIC', 'STANDARD', 'ECOLOGY', 'CLIMATE', 'THERMAL',
    'TURBINE', 'SENSOR', 'CHANNEL', 'PARADIGM', 'COGNITIVE',
    'FRONTAL', 'VOLTAGE', 'SENSORY', 'DECODER', 'HEADSET',
    'CLASSIFIER', 'FUNCTION', 'VARIABLE', 'SCALER', 'TENSOR',
    'BOOLEAN', 'PRECISION', 'ACCURACY', 'BASELINE', 'CHEMISTRY',
    'HARDWARE', 'SOFTWARE', 'NETWORK', 'SUSTAIN', 'STRATEGY'
]
random.shuffle(WORD_POOL)


# PROCESS 1: BACKGROUND HARDWARE STREAMER
def start_stream(mac_address):
    """Handles the Bluetooth BLE connection and data broadcasting."""
    if mac_address and mac_address != "UNKNOWN":
        print(f"\n🔗 Establishing direct connection to Muse MAC: {mac_address}...")
        stream(mac_address)
    else:
        print("🔍 Scanning for Muse devices...")
        muses = list_muses()
        if not muses:
            print("❌ No Muse found. Ensure Bluetooth is ON and headset is in pairing mode.")
            sys.exit(1)
        sys.exit(0)


# TKINTER EXISTING USER REGISTRATION
def run_pre_registration():
    import tkinter as tk
    from tkinter import messagebox
    root = tk.Tk()
    root.title("Operator Login & Protocol Settings")
    root.geometry("380x250")

    tk.Label(root, text="3-Digit Subject ID (e.g., 001):").pack(pady=5)
    entry_id = tk.Entry(root, font=("Arial", 14), justify="center")
    entry_id.pack(pady=5)

    tk.Label(root, text="Select Evaluation Task:").pack(pady=5)
    task_var = tk.StringVar(value="math")
    tk.Radiobutton(root, text="Complex Mental Math", variable=task_var, value="math").pack()
    tk.Radiobutton(root, text="Missing Letter (Cloze)", variable=task_var, value="words").pack()

    save_dir_path = []
    subject_id_val = []
    selected_task = []
    target_mac_val = []
    base_data_dir_val = []

    def submit_data():
        reg_id = entry_id.get().strip()
        if not reg_id or len(reg_id) != 3 or not reg_id.isdigit():
            messagebox.showerror("Error", "Please enter a valid 3-digit Registration ID.")
            return

        folder_path = os.path.join(BASE_DATA_DIR, f"ID_{reg_id}")
        online_path = os.path.join(folder_path, "online")
        info_path = os.path.join(folder_path, "personal_info.txt")

        if not os.path.exists(folder_path):
            messagebox.showerror("Error", f"Profile ID_{reg_id} does not exist.")
            return

        if not os.path.exists(online_path):
            messagebox.showerror("Error",
                                 f"No 'online' data folder found for ID_{reg_id}. Please ensure models are trained and saved.")
            return

        if not os.path.exists(info_path):
            messagebox.showerror("Error",
                                 f"'personal_info.txt' not found for ID_{reg_id}. Cannot extract hardware metadata.")
            return

        # Directly parse MAC Address and Base Directory from personal_info.txt
        extracted_mac = "UNKNOWN"
        extracted_base = BASE_DATA_DIR

        with open(info_path, "r") as f:
            for line in f:
                if line.startswith("Hardware MAC:"):
                    extracted_mac = line.split(":", 1)[1].strip()
                elif line.startswith("Base Directory:"):
                    extracted_base = line.split(":", 1)[1].strip()

        save_dir_path.append(online_path)
        subject_id_val.append(reg_id)
        selected_task.append(task_var.get())
        target_mac_val.append(extracted_mac)
        base_data_dir_val.append(extracted_base)
        root.destroy()

    tk.Button(root, text="Load Profile & Launch System", command=submit_data, bg="#3498db", fg="white",
              font=("Arial", 10, "bold")).pack(pady=15)
    root.mainloop()

    if not save_dir_path:
        sys.exit(0)

    return subject_id_val[0], save_dir_path[0], selected_task[0], target_mac_val[0], base_data_dir_val[0]


# SHARED PIPELINE FUNCTIONS
def load_subject_bundle(subject_id, target_dir, task_type):
    print("=" * 60)
    print(f"🧠 Standalone BCI Online Testing Engine ({task_type.upper()} TASK)")
    print("=" * 60)

    model_path = os.path.join(target_dir, f"bci_live_production_{task_type}_lda_online.joblib")

    if not os.path.exists(model_path):
        fallback_path = os.path.join(target_dir, "bci_live_production_online_training.joblib")
        if os.path.exists(fallback_path):
            model_path = fallback_path
        else:
            print(f"❌ Error: Model bundle not found at:\n   {model_path}")
            sys.exit(1)

    print(f"\n📂 Loading production bundle for Subject {subject_id}...")
    bundle = joblib.load(model_path)
    return bundle['model'], bundle['scaler'], bundle['feature_names']


def apply_causal_filters(eeg_data, fs):
    b_notch, a_notch = signal.iirnotch(w0=50.0, Q=30.0, fs=fs)
    filtered_data = signal.lfilter(b_notch, a_notch, eeg_data, axis=0)

    b_band, a_band = signal.butter(N=4, Wn=[1.0, 40.0], btype='bandpass', fs=fs)
    filtered_data = signal.lfilter(b_band, a_band, filtered_data, axis=0)
    return filtered_data


def extract_live_features(epoch_data, fs, required_features):
    features = {}
    for ch_idx, ch_name in enumerate(EEG_CHANNELS):
        channel_signal = epoch_data[:, ch_idx]
        freqs, psd = signal.welch(channel_signal, fs=fs, nperseg=fs * 2)

        # Alpha band power calculation (8-12 Hz)
        alpha_idx = np.logical_and(freqs >= 8, freqs <= 12)
        alpha_normalized = np.trapz(psd[alpha_idx], freqs[alpha_idx]) / (12 - 8)

        # Beta band power calculation (13-30 Hz)
        beta_idx = np.logical_and(freqs >= 13, freqs <= 30)
        beta_normalized = np.trapz(psd[beta_idx], freqs[beta_idx]) / (30 - 13)

        # Calculate Beta/Alpha ratio
        ratio = beta_normalized / (alpha_normalized + 1e-9)

        # Store using the exact naming convention the trained model expects
        feature_name = f"{ch_name}_Beta_Alpha_Ratio"
        features[feature_name] = ratio

    # Map the extracted features to exactly what the loaded model requires
    feature_vector = [features[name] for name in required_features]
    return np.array(feature_vector).reshape(1, -1)


# TASK GENERATORS
def generate_math_problem():
    """Generates simplified math problem format: (X * X) +/- X"""
    num1 = random.randint(1, 9)
    num2 = random.randint(1, 9)
    op2 = random.choice(['+', '-'])
    num3 = random.randint(1, 9)
    return f"( {num1} * {num2} ) {op2} {num3} = ?"


def generate_cloze_prompt():
    """Pulls a unique word, replaces a random character with an underscore."""
    global WORD_POOL
    if not WORD_POOL:
        WORD_POOL = [
            'PETROLEUM', 'HYDROGEN', 'MOLECULE', 'POLYMER', 'CATALYST',
            'EFFICIENCY', 'ORGANIC', 'BIOMASS', 'METHANE', 'PIPELINE',
            'METRIC', 'STANDARD', 'ECOLOGY', 'CLIMATE', 'THERMAL',
            'TURBINE', 'SENSOR', 'CHANNEL', 'PARADIGM', 'COGNITIVE',
            'FRONTAL', 'VOLTAGE', 'SENSORY', 'DECODER', 'HEADSET',
            'CLASSIFIER', 'FUNCTION', 'VARIABLE', 'SCALER', 'TENSOR',
            'BOOLEAN', 'PRECISION', 'ACCURACY', 'BASELINE', 'CHEMISTRY',
            'HARDWARE', 'SOFTWARE', 'NETWORK', 'SUSTAIN', 'STRATEGY'
        ]
        random.shuffle(WORD_POOL)

    target_word = WORD_POOL.pop()
    idx = random.randint(0, len(target_word) - 1)
    puzzle = target_word[:idx] + "_" + target_word[idx + 1:]
    return " ".join(puzzle)


# UI FUNCTIONS
def draw_signal_check_ui(screen, fonts, channel_status, stable_progress):
    """Pre-flight check to ensure the headset is worn properly before testing."""
    screen.fill(BG_DARK)
    title_font, main_font, small_font = fonts
    w, h = screen.get_size()

    title = title_font.render("PRE-FLIGHT SIGNAL CHECK", True, TEXT_LIGHT)
    screen.blit(title, title.get_rect(center=(w // 2, h // 4)))

    prompt = main_font.render("Please adjust your headset until all channels display OK.", True, COLOR_RELAX)
    screen.blit(prompt, prompt.get_rect(center=(w // 2, h // 4 + 60)))

    # Draw individual channel status
    start_y = h // 2 - 50
    for i, ch in enumerate(EEG_CHANNELS):
        status = channel_status[i]
        color = COLOR_FOCUS if status == "OK" else COLOR_WARN
        txt = small_font.render(f"{ch} Electrode:   {status}", True, color)
        screen.blit(txt, (w // 2 - 180, start_y + i * 40))

    # Stability Progress Bar (Needs 5 seconds)
    bar_w = 400
    bar_h = 30
    bar_x = (w - bar_w) // 2
    bar_y = start_y + 200

    pygame.draw.rect(screen, (50, 50, 50), (bar_x, bar_y, bar_w, bar_h), border_radius=5)

    ratio = min(1.0, stable_progress / (5.0 * REFRESH_RATE_HZ))
    if ratio > 0:
        pygame.draw.rect(screen, COLOR_PROGRESS, (bar_x, bar_y, int(bar_w * ratio), bar_h), border_radius=5)

    msg = small_font.render("Maintaining clear signal for 5 seconds to lock in calibration...", True, TEXT_LIGHT)
    screen.blit(msg, msg.get_rect(center=(w // 2, bar_y - 25)))

    pygame.display.flip()


def draw_instruction_screen(screen, task_type):
    """Halts the system to educate the participant dynamically based on task type."""
    screen.fill(BG_DARK)
    w, h = screen.get_size()

    title_font = pygame.font.SysFont('Arial', 40, bold=True)
    header_font = pygame.font.SysFont('Arial', 26, bold=True)
    body_font = pygame.font.SysFont('Arial', 22)

    if task_type == "math":
        task_prompt = "Solve the on-screen complex mental math to drive the confidence bar UP."
    else:
        task_prompt = "Identify the missing letter in the on-screen word to drive the confidence bar UP."

    lines = [
        (f"STANDALONE TESTING ENGINE: {task_type.upper()} TASK", (255, 255, 255), -320, title_font),

        ("YOUR MENTAL TASK:", COLOR_FOCUS, -250, header_font),
        (f"   ACTIVE PHASE: {task_prompt}", TEXT_LIGHT, -210, body_font),
        ("   REST PHASE: Clear your mind, soften your gaze, and relax to keep the bar DOWN.", TEXT_LIGHT, -170,
         body_font),

        ("THE 4-STEP DECISION ENGINE:", COLOR_RELAX, -90, header_font),
        ("1. Smoothing (The stabilizer)", COLOR_FOCUS, -40, header_font),
        ("   Calculates a rolling average over the last 8 windows (2.0s of history).", TEXT_LIGHT, -10, body_font),
        ("2. Decision threshold (The gate)", COLOR_FOCUS, 30, header_font),
        ("   The smoothed average must definitively cross the 80% confidence boundary.", TEXT_LIGHT, 60, body_font),
        ("3. Dwell time (The confirmation)", COLOR_WARN, 100, header_font),
        ("   Must remain above 80% for 5 hops (1.25s) to confirm intent and reject noise.", TEXT_LIGHT, 130, body_font),
        ("4. Refractory period (The cooldown)", COLOR_PROGRESS, 170, header_font),
        ("   Strict 3.0s cooldown enforced after triggering to prevent system overload.", TEXT_LIGHT, 200, body_font),

        ("Press SPACE to begin the protocol...", (255, 255, 255), 280, header_font)
    ]

    for text, color, y_offset, font in lines:
        surf = font.render(text, True, color)
        rect = surf.get_rect(center=(w // 2, h // 2 + y_offset))
        screen.blit(surf, rect)

    pygame.display.flip()

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                sys.exit(0)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                waiting = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                pygame.image.save(screen, f"BCI_Instructions_Screenshot_{timestamp}.png")
                print(f"📸 Screenshot saved: BCI_Instructions_Screenshot_{timestamp}.png")
        time.sleep(0.05)


def draw_dynamic_ui(screen, fonts, test_phase, prompt_text, system_msg, conf_score, dwell_progress, color, prob_history,
                    refract_remain):
    """Renders real-time visual feedback for the 4-phase decision engine."""
    screen.fill(BG_DARK)
    title_font, main_font, small_font = fonts
    w, h = screen.get_size()

    # --- TOP STATUS & PROMPTS ---
    title_surf = small_font.render(f"PHASE: {test_phase} (ESC to Abort)", True, (100, 100, 100))
    screen.blit(title_surf, title_surf.get_rect(center=(w // 2, 40)))

    prompt_surf = main_font.render(prompt_text, True, TEXT_LIGHT)
    screen.blit(prompt_surf, prompt_surf.get_rect(center=(w // 2, 100)))

    sys_surf = title_font.render(system_msg, True, color)
    screen.blit(sys_surf, sys_surf.get_rect(center=(w // 2, 160)))

    # --- 4-STEP DECISION ENGINE DASHBOARD ---
    panel_w = int(w * 0.75)
    panel_x = (w - panel_w) // 2
    start_y = 250
    y_step = 120

    # Phase 1: Smoothing (The Stabilizer)
    p1_y = start_y
    screen.blit(small_font.render("1. Smoothing (The stabilizer): 8-Window Raw Probability Deque", True, TEXT_LIGHT),
                (panel_x, p1_y))
    block_w = int((panel_w - (7 * 10)) / 8)
    for i in range(8):
        bx = panel_x + i * (block_w + 10)
        by = p1_y + 35
        pygame.draw.rect(screen, (50, 50, 50), (bx, by, block_w, 25), border_radius=4)
        if i < len(prob_history):
            val = prob_history[i]
            c = COLOR_FOCUS if val >= CONFIDENCE_THRESHOLD else COLOR_RELAX
            pygame.draw.rect(screen, c, (bx, by, int(block_w * val), 25), border_radius=4)

    # Phase 2: Decision Threshold (The Gate)
    p2_y = start_y + y_step
    screen.blit(small_font.render(f"2. Decision threshold (The gate): Smoothed Average = {conf_score * 100:.1f}%", True,
                                  TEXT_LIGHT), (panel_x, p2_y))
    bar_y = p2_y + 35
    pygame.draw.rect(screen, (50, 50, 50), (panel_x, bar_y, panel_w, 35), border_radius=6)
    bar_c = COLOR_FOCUS if conf_score >= CONFIDENCE_THRESHOLD else COLOR_RELAX
    if conf_score > 0:
        pygame.draw.rect(screen, bar_c, (panel_x, bar_y, int(panel_w * conf_score), 35), border_radius=6)

    # 80% Threshold Marker
    t_x = panel_x + int(panel_w * CONFIDENCE_THRESHOLD)
    pygame.draw.line(screen, COLOR_WARN, (t_x, bar_y - 10), (t_x, bar_y + 45), 4)

    # Phase 3: Dwell Time (The Confirmation)
    p3_y = start_y + y_step * 2
    screen.blit(
        small_font.render(f"3. Dwell time (The confirmation): {dwell_progress} / {DWELL_TARGET_HITS} Consecutive Hops",
                          True, TEXT_LIGHT), (panel_x, p3_y))
    circ_y = p3_y + 50
    circ_r = 18
    for i in range(DWELL_TARGET_HITS):
        cx = panel_x + 25 + (i * 60)
        if i < dwell_progress:
            pygame.draw.circle(screen, COLOR_PROGRESS, (cx, circ_y), circ_r)
            pygame.draw.circle(screen, TEXT_LIGHT, (cx, circ_y), circ_r, 2)
        else:
            pygame.draw.circle(screen, (50, 50, 50), (cx, circ_y), circ_r)
            pygame.draw.circle(screen, (100, 100, 100), (cx, circ_y), circ_r, 2)

    # Phase 4: Refractory Period (The Cooldown)
    p4_y = start_y + y_step * 3
    screen.blit(
        small_font.render(f"4. Refractory period (The cooldown): {refract_remain:.1f}s Remaining", True, TEXT_LIGHT),
        (panel_x, p4_y))
    r_bar_y = p4_y + 35
    refract_ratio = refract_remain / REFRACTORY_PERIOD_SEC
    pygame.draw.rect(screen, (50, 50, 50), (panel_x, r_bar_y, panel_w, 25), border_radius=4)
    if refract_ratio > 0:
        pygame.draw.rect(screen, COLOR_WARN, (panel_x, r_bar_y, int(panel_w * refract_ratio), 25), border_radius=4)

    pygame.display.flip()


def calculate_itr(n_classes, accuracy, time_per_command):
    if accuracy == 1.0:
        b = np.log2(n_classes)
    elif accuracy <= (1.0 / n_classes):
        b = 0.0
    else:
        b = (accuracy * np.log2(accuracy) + (1.0 - accuracy) * np.log2((1.0 - accuracy) / (n_classes - 1.0)) + np.log2(
            n_classes))
    return max(0.0, b * (60.0 / time_per_command)) if time_per_command > 0 else 0


def run_online_engine(model, scaler, feature_names, task_type):
    print("📊 Resolving LSL EEG Stream (Waiting up to 15s)...")
    streams = resolve_byprop('type', 'EEG', timeout=15.0)
    if not streams:
        print("❌ Error: No EEG stream found.")
        sys.exit(1)
    inlet = StreamInlet(streams[0])

    pygame.init()
    info = pygame.display.Info()
    # NOFRAME enables borderless window so screenshots do not crash the engine
    screen = pygame.display.set_mode((info.current_w, info.current_h), pygame.NOFRAME)
    pygame.display.set_caption("Live BCI Evaluator")
    pygame.mouse.set_visible(False)

    # --- OS-LEVEL PERMANENT FOREGROUND LOCK ---
    pygame.event.pump()
    if os.name == 'nt':
        import ctypes
        hwnd = pygame.display.get_wm_info()["window"]
        user32 = ctypes.windll.user32

        user32.keybd_event(0x12, 0, 0, 0)
        user32.keybd_event(0x12, 0, 2, 0)

        user32.ShowWindow(hwnd, 9)
        user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 3)
        user32.SetForegroundWindow(hwnd)

    time.sleep(0.1)

    fonts = (pygame.font.SysFont('Arial', 48, bold=True),
             pygame.font.SysFont('Arial', 36, bold=True),
             pygame.font.SysFont('Arial', 20))

    # METRICS TRACKING VARIABLES
    metrics = {
        'total_windows': 0,
        'artifact_windows': 0,
        'processing_latencies': [],
        'missed_deadlines': 0,
        'false_activations': 0,
        'command_times': [],
        'failed_tasks': 0
    }

    probability_buffer = deque(maxlen=SMOOTHING_WINDOW_SIZE)
    dwell_counter = 0
    refractory_end_time = 0

    # TASK STATE MACHINE
    test_phase = "SIGNAL_CHECK"
    phase_start_time = time.time()
    commands_completed = 0
    command_cue_time = 0
    stable_frames = 0
    last_task_time = 0
    current_task_prompt = ""

    task_refresh_rate = TASK_REFRESH_MATH if task_type == "math" else TASK_REFRESH_WORDS

    inlet.pull_chunk(timeout=1.0, max_samples=3000)
    eeg_buffer = []
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                pygame.image.save(screen, f"BCI_Dashboard_Screenshot_{timestamp}.png")
                print(f"📸 Screenshot saved: BCI_Dashboard_Screenshot_{timestamp}.png")

        if test_phase == "INSTRUCTIONS":
            draw_instruction_screen(screen, task_type)
            test_phase = "REST"
            phase_start_time = time.time()
            continue

        current_time = time.time()

        # Pull raw live data
        samples, _ = inlet.pull_chunk(timeout=0.0, max_samples=100)
        if samples:
            eeg_buffer.extend(samples)

        if len(eeg_buffer) >= SAMPLES_PER_EPOCH:
            metrics['total_windows'] += 1
            loop_start = time.time()

            raw_epoch = np.array(eeg_buffer[:SAMPLES_PER_EPOCH])[:, :4]
            ui_system_msg = ""
            ui_color = TEXT_LIGHT
            smoothed_score = 0.0

            # 1. PER-CHANNEL SIGNAL QUALITY CHECK
            channel_status = []
            for ch_idx in range(4):
                ch_data = raw_epoch[:, ch_idx]
                if np.ptp(ch_data) > ARTIFACT_THRESHOLD_UV:
                    channel_status.append("ARTIFACT (Blink/Jaw)")
                elif np.var(ch_data) < FLATLINE_THRESHOLD_UV:
                    channel_status.append("FLATLINE (Loose)")
                else:
                    channel_status.append("OK")

            # 2. STATE MACHINE ROUTING
            if test_phase == "SIGNAL_CHECK":
                if all(s == "OK" for s in channel_status):
                    stable_frames += 1
                else:
                    stable_frames = 0

                draw_signal_check_ui(screen, fonts, channel_status, stable_frames)

                if stable_frames >= (5.0 * REFRESH_RATE_HZ):
                    test_phase = "INSTRUCTIONS"

                # Slide window and continue
                eeg_buffer = eeg_buffer[HOP_SAMPLES:]
                continue

            # --- LIVE TESTING LOGIC ---
            if not all(s == "OK" for s in channel_status):
                metrics['artifact_windows'] += 1
                bad_reasons = [s for s in channel_status if s != "OK"]
                ui_system_msg = f"POOR SIGNAL: {bad_reasons[0]}"
                ui_color = COLOR_WARN
                dwell_counter = 0
                probability_buffer.clear()
            else:
                clean_epoch = apply_causal_filters(raw_epoch, SAMPLING_RATE)
                live_features = extract_live_features(clean_epoch, SAMPLING_RATE, feature_names)
                scaled_features = scaler.transform(live_features)

                if hasattr(model, "predict_proba"):
                    confidence = model.predict_proba(scaled_features)[0][1]
                else:
                    prediction = model.predict(scaled_features)[0]
                    confidence = 1.0 if prediction == 1 else 0.0

                probability_buffer.append(confidence)

            # 1. SMOOTHING
            if len(probability_buffer) > 0:
                smoothed_score = np.mean(probability_buffer)

            # 2. DECISION THRESHOLD & 3. DWELL TIME
            command_triggered = False

            if current_time > refractory_end_time:
                if ui_system_msg == "":
                    if smoothed_score >= CONFIDENCE_THRESHOLD:
                        dwell_counter += 1
                        ui_system_msg = "BUILDING COMMAND..."
                        ui_color = COLOR_PROGRESS

                        if dwell_counter >= DWELL_TARGET_HITS:
                            command_triggered = True
                            dwell_counter = 0
                            refractory_end_time = current_time + REFRACTORY_PERIOD_SEC
                            ui_system_msg = "COMMAND ISSUED! (Cooldown)"
                            ui_color = COLOR_FOCUS
                            probability_buffer.clear()
                    else:
                        dwell_counter = 0
                        ui_system_msg = "ANALYZING..."
                        ui_color = COLOR_RELAX
            elif ui_system_msg == "":
                ui_system_msg = "SYSTEM COOLDOWN"
                ui_color = (100, 100, 100)

            # LATENCY TRACKING
            loop_latency = time.time() - loop_start
            metrics['processing_latencies'].append(loop_latency)
            if loop_latency > (1.0 / REFRESH_RATE_HZ):
                metrics['missed_deadlines'] += 1

            # TEST PARADIGM LOGIC
            ui_prompt_text = ""

            if test_phase == "REST":
                ui_prompt_text = f"Resting Phase ({REST_PHASE_DURATION_SEC - int(current_time - phase_start_time)}s) - DO NOT TRIGGER"
                if command_triggered:
                    metrics['false_activations'] += 1
                if current_time - phase_start_time >= REST_PHASE_DURATION_SEC:
                    test_phase = "ACTIVE"
                    command_cue_time = current_time

            elif test_phase == "ACTIVE":
                if current_time - last_task_time > task_refresh_rate:
                    if task_type == "math":
                        current_task_prompt = generate_math_problem()
                    else:
                        current_task_prompt = generate_cloze_prompt()
                    last_task_time = current_time

                ui_prompt_text = f"ACTIVE PHASE: Solve {current_task_prompt} (Command {commands_completed + 1} of {TARGET_COMMAND_COUNT})"

                if command_triggered:
                    time_taken = current_time - command_cue_time
                    metrics['command_times'].append(time_taken)
                    commands_completed += 1
                    command_cue_time = current_time + REFRACTORY_PERIOD_SEC

                elif current_time - command_cue_time > 20.0:
                    metrics['failed_tasks'] += 1
                    commands_completed += 1
                    command_cue_time = current_time
                    dwell_counter = 0
                    probability_buffer.clear()

                if commands_completed >= TARGET_COMMAND_COUNT:
                    test_phase = "COMPLETE"

            elif test_phase == "COMPLETE":
                running = False

            # Slide window forward
            eeg_buffer = eeg_buffer[HOP_SAMPLES:]

            refract_remain = max(0.0, refractory_end_time - current_time)
            prob_history = list(probability_buffer)

            draw_dynamic_ui(screen, fonts, test_phase, ui_prompt_text, ui_system_msg, smoothed_score, dwell_counter,
                            ui_color, prob_history, refract_remain)

    pygame.quit()
    return metrics


def print_evaluation_report(metrics, target_dir, subject_id, task_type, target_mac, base_dir):
    # Task-Level Calculations
    total_commands = len(metrics['command_times'])
    avg_time = np.mean(metrics['command_times']) if total_commands > 0 else 0
    accuracy = total_commands / (total_commands + metrics['failed_tasks'] + metrics['false_activations']) if (
                                                                                                                     total_commands +
                                                                                                                     metrics[
                                                                                                                         'failed_tasks'] +
                                                                                                                     metrics[
                                                                                                                         'false_activations']) > 0 else 0

    itr = calculate_itr(2, accuracy, avg_time)

    # System-Level Calculations
    latencies = np.array(metrics['processing_latencies']) * 1000
    med_latency = np.median(latencies) if len(latencies) > 0 else 0
    worst_latency = np.max(latencies) if len(latencies) > 0 else 0
    rejection_rate = (metrics['artifact_windows'] / metrics['total_windows']) * 100 if metrics[
                                                                                           'total_windows'] > 0 else 0

    # Build the report string
    report_text = f"""============================================================
📈 SYSTEM EVALUATION & ONLINE METRICS REPORT (SUBJECT {subject_id} - {task_type.upper()})
Hardware Setup -> MAC: {target_mac} | Data Root: {base_dir}
============================================================

[ TASK-LEVEL METRICS ]
Task Completion Rate : {(total_commands / TARGET_COMMAND_COUNT) * 100:.1f}%
Avg Time per Command : {avg_time:.2f} seconds
Estimated ITR        : {itr:.2f} bits/min

[ SYSTEM-LEVEL METRICS ]
End-to-End Latency   : {med_latency:.1f}ms (Median) / {worst_latency:.1f}ms (Worst Case)
Missed Deadlines     : {metrics['missed_deadlines']} loops exceeded {1000 / REFRESH_RATE_HZ:.1f}ms budget
Artifact Rejection   : {rejection_rate:.1f}% ({metrics['artifact_windows']}/{metrics['total_windows']} windows refused)

[ CRITICAL UX METRICS ]
False Activations    : {metrics['false_activations']} (Commands fired during Rest Phase)
User Experience      : ⚠️ PENDING. Please survey participant on workload, fatigue, and control.
============================================================
"""

    # 1. Print to console
    print(report_text)

    # 2. Save to text file in the /online/ directory
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    filename = f"report_ONLINE_EVALUATION_{task_type}_{timestamp}.txt"
    report_path = os.path.join(target_dir, filename)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"📄 Evaluation report successfully saved to:\n   {report_path}\n")


if __name__ == "__main__":
    multiprocessing.freeze_support()

    subject_id, target_dir, task_type, target_mac, verified_base_dir = run_pre_registration()

    print(f"\n📡 Audit Trail - Hardware MAC Verified: {target_mac}")
    print(f"📁 Audit Trail - Base Directory Verified: {verified_base_dir}")

    model, scaler, feature_names = load_subject_bundle(subject_id, target_dir, task_type)

    stream_process = multiprocessing.Process(target=start_stream, args=(target_mac,))
    stream_process.daemon = True
    stream_process.start()
    time.sleep(5.0)

    try:
        results = run_online_engine(model, scaler, feature_names, task_type)
        print_evaluation_report(results, target_dir, subject_id, task_type, target_mac, verified_base_dir)
    except KeyboardInterrupt:
        pass
    finally:
        stream_process.terminate()
        print("\n🛑 Hardware disconnected.")