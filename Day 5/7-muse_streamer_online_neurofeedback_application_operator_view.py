"""
Autonomous BCI Closed-Loop System & Operator Diagnostics
--------------------------------------------------------
Process 1: Background Hardware Streamer (Connects to Muse & Broadcasts LSL)
Process 2: Operator View (Diagnostics, Latency Budget, Subject Replica)
Process 3: Main Task (Smart Energy Operator) loads the ML model & drives feedback.
Upgraded with unified task routing (Math/Words), 2-minute timed gamification,
automated outcome logging (Success vs. Timeout), dynamic stimulus generation,
and hardware metadata extraction from personal_info.txt.
"""

import os
import sys
import time
import joblib
import numpy as np
import pygame
import multiprocessing
import traceback
import random
import glob
from collections import deque
from scipy import signal
from pylsl import StreamInlet, resolve_byprop
from muselsl import stream

# ==============================================================================
# PIPELINE CONFIGURATION
# ==============================================================================
# Default lookup directory; will be verified against personal_info.txt
BASE_DATA_DIR = r"D:\MuseData"

SAMPLING_RATE = 256
EEG_CHANNELS = ["TP9", "AF7", "AF8", "TP10"]
EPOCH_LENGTH_SEC = 2.0
SAMPLES_PER_EPOCH = int(SAMPLING_RATE * EPOCH_LENGTH_SEC)
HOP_SAMPLES = int(SAMPLING_RATE * 0.25)

# --- STRICT DECISION LOGIC PARAMETERS ---
SMOOTHING_WINDOWS = 4
DECISION_THRESHOLD = 0.75
DWELL_TIME_SEC = 1.5
REFRACTORY_TIME_SEC = 3.0

# --- GAMIFICATION & TIMING ---
TARGET_SCORE = 10
SESSION_TIME_LIMIT_SEC = 120.0  # 2 Minutes

# --- OPERATOR VIEW PARAMETERS ---
DISPLAY_WINDOW_SEC = 3.0
SAMPLES_TO_SHOW = int(SAMPLING_RATE * DISPLAY_WINDOW_SEC)
ARTIFACT_THRESHOLD_UV = 350.0
FLATLINE_THRESHOLD_UV = 2.0

# Dynamic Task Refresh Rates
TASK_REFRESH_MATH = 4
TASK_REFRESH_WORDS = 2

# --- SHARED STATE MAPPING ---
STATE_MAP = {
    -1: "⚠️ CRITICAL: LSL BLUETOOTH STREAM LOST",
    0: "WAITING FOR DATA",
    1: "SYSTEM HALTED BY OPERATOR",
    2: "COOLDOWN (REFRACTORY)",
    3: "DWELLING...",
    4: "COMMAND ISSUED!",
    5: "ACTIVE (ECO-MODE)",
    6: "MISSION ACCOMPLISHED! 🎉",
    7: "TIME EXPIRED (MISSION FAILED) ⏱️"
}

# Global pool of distinct technical words for the ONLINE session
WORD_POOL = [
    'EMISSION', 'FOOTPRINT', 'PROTOCOL', 'COMPLIANCE', 'REFINERY',
    'AUDITING', 'EFFLUENT', 'DISCHARGE', 'ABSTRACTION', 'LEGISLATION',
    'MECHANISM', 'GREENHOUSE', 'INVENTORY', 'VERIFICATION', 'NEURAL',
    'SYNAPSE', 'CORTICAL', 'ACQUISITION', 'PROCESSING', 'SPATIAL',
    'TEMPORAL', 'FREQUENCY', 'ARTIFACT', 'HACKATHON', 'ALGORITHM',
    'INTERFACE', 'ELECTRODE', 'IMPEDANCE', 'AMPLIFIER', 'EXECUTIVE',
    'LEADERSHIP', 'REGISTER', 'SYNERGY', 'WORKSHOP', 'IMMERSIVE',
    'CURRICULUM', 'AWARENESS', 'MITIGATION', 'OPERATIONAL', 'ALLOCATION'
]
random.shuffle(WORD_POOL)


# ==============================================================================
# SHARED PIPELINE FUNCTIONS
# ==============================================================================
def apply_causal_filters(eeg_data, fs):
    b_notch, a_notch = signal.iirnotch(w0=50.0, Q=30.0, fs=fs)
    filtered_data = signal.lfilter(b_notch, a_notch, eeg_data, axis=0)
    b_band, a_band = signal.butter(N=4, Wn=[1.0, 40.0], btype='bandpass', fs=fs)
    filtered_data = signal.lfilter(b_band, a_band, filtered_data, axis=0)
    return filtered_data


def extract_live_features(epoch_data, fs, required_features):
    """Normalized PSD extraction and Beta/Alpha Ratio calculation matching the ML model pipeline."""
    features = {}
    alpha_powers, beta_powers = [], []

    for ch_idx, ch_name in enumerate(EEG_CHANNELS):
        channel_signal = epoch_data[:, ch_idx]
        freqs, psd = signal.welch(channel_signal, fs=fs, nperseg=fs * 2)

        # Alpha band power calculation (8-12 Hz)
        alpha_idx = np.logical_and(freqs >= 8, freqs <= 12)
        alpha_normalized = np.trapz(psd[alpha_idx], freqs[alpha_idx]) / (12 - 8)
        alpha_powers.append(alpha_normalized)

        # Beta band power calculation (13-30 Hz)
        beta_idx = np.logical_and(freqs >= 13, freqs <= 30)
        beta_normalized = np.trapz(psd[beta_idx], freqs[beta_idx]) / (30 - 13)
        beta_powers.append(beta_normalized)

        # Calculate Beta/Alpha ratio
        ratio = beta_normalized / (alpha_normalized + 1e-9)

        feature_name = f"{ch_name}_Beta_Alpha_Ratio"
        features[feature_name] = ratio

    feature_vector = [features[name] for name in required_features]
    return np.array(feature_vector).reshape(1, -1), np.mean(alpha_powers), np.mean(beta_powers)


# TASK GENERATORS
def generate_math_problem():
    num1 = random.randint(1, 9)
    num2 = random.randint(1, 9)
    op2 = random.choice(['+', '-'])
    num3 = random.randint(1, 9)
    return f"( {num1} * {num2} ) {op2} {num3} = ?"


def generate_cloze_prompt():
    global WORD_POOL
    if not WORD_POOL:
        WORD_POOL = [
            'EMISSION', 'FOOTPRINT', 'PROTOCOL', 'COMPLIANCE', 'REFINERY',
            'AUDITING', 'EFFLUENT', 'DISCHARGE', 'ABSTRACTION', 'LEGISLATION',
            'MECHANISM', 'GREENHOUSE', 'INVENTORY', 'VERIFICATION', 'NEURAL',
            'SYNAPSE', 'CORTICAL', 'ACQUISITION', 'PROCESSING', 'SPATIAL',
            'TEMPORAL', 'FREQUENCY', 'ARTIFACT', 'HACKATHON', 'ALGORITHM',
            'INTERFACE', 'ELECTRODE', 'IMPEDANCE', 'AMPLIFIER', 'EXECUTIVE',
            'LEADERSHIP', 'REGISTER', 'SYNERGY', 'WORKSHOP', 'IMMERSIVE',
            'CURRICULUM', 'AWARENESS', 'MITIGATION', 'OPERATIONAL', 'ALLOCATION'
        ]
        random.shuffle(WORD_POOL)

    target_word = WORD_POOL.pop()
    idx = random.randint(0, len(target_word) - 1)
    puzzle = target_word[:idx] + "_" + target_word[idx + 1:]
    return " ".join(puzzle)


# ==============================================================================
# PROCESS 1: BACKGROUND HARDWARE STREAMER
# ==============================================================================
def start_stream(mac_address):
    """Initiates the Muse LSL stream using the extracted hardware MAC address."""
    if mac_address and mac_address != "UNKNOWN":
        print(f"\n🔗 Establishing direct connection to Muse MAC: {mac_address}...")
        stream(mac_address)
    else:
        print("❌ Error: Invalid MAC Address provided. Cannot establish stream.")
        sys.exit(0)


# ==============================================================================
# TKINTER EXISTING USER REGISTRATION
# ==============================================================================
def run_pre_registration():
    import tkinter as tk
    from tkinter import messagebox
    root = tk.Tk()
    root.title("Operator Login (Closed-Loop System)")
    root.geometry("380x250")

    tk.Label(root, text="3-Digit Subject ID (e.g., 001):", font=("Arial", 10, "bold")).pack(pady=(10, 5))
    entry_id = tk.Entry(root, font=("Arial", 14), justify="center", width=10)
    entry_id.pack(pady=5)

    # Task Toggle
    tk.Label(root, text="Select Cognitive Task:", font=("Arial", 10, "bold")).pack(pady=(10, 0))
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

        if not os.path.exists(folder_path) or not os.path.exists(online_path):
            messagebox.showerror("Error", f"Profile ID_{reg_id} or 'online' folder does not exist.")
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

    tk.Button(root, text="Load Profile & Launch Game", command=submit_data, bg="#3498db", fg="white",
              font=("Arial", 10, "bold")).pack(pady=15)
    root.mainloop()

    if not save_dir_path:
        sys.exit(0)

    return subject_id_val[0], save_dir_path[0], selected_task[0], target_mac_val[0], base_data_dir_val[0]


# ==============================================================================
# PROCESS 2: OPERATOR VIEW TERMINAL
# ==============================================================================
class OperatorViewTerminal:
    def __init__(self, pause_event, exit_event, sys_state, sys_prob, sys_alpha, sys_beta, sys_angle, sys_eff, sys_eco,
                 sys_lat_prep, sys_lat_inf, sys_dwell, sys_refract, sys_score, sys_session_time):
        self.pause_event = pause_event
        self.exit_event = exit_event
        self.sys_state = sys_state
        self.sys_prob = sys_prob
        self.sys_alpha = sys_alpha
        self.sys_beta = sys_beta
        self.sys_angle = sys_angle
        self.sys_eff = sys_eff
        self.sys_eco = sys_eco
        self.sys_lat_prep = sys_lat_prep
        self.sys_lat_inf = sys_lat_inf
        self.sys_dwell = sys_dwell
        self.sys_refract = sys_refract
        self.sys_score = sys_score
        self.sys_session_time = sys_session_time

        pygame.display.init()
        num_displays = pygame.display.get_num_displays()
        pygame.display.quit()

        if num_displays == 1:
            os.environ['SDL_VIDEO_WINDOW_POS'] = "20,50"
        elif 'SDL_VIDEO_WINDOW_POS' in os.environ:
            del os.environ['SDL_VIDEO_WINDOW_POS']

        pygame.init()
        if num_displays > 1:
            self.screen = pygame.display.set_mode((0, 0), pygame.NOFRAME, display=1)
        else:
            self.screen = pygame.display.set_mode((950, 750), pygame.NOFRAME)

        self.width, self.height = self.screen.get_size()
        pygame.display.set_caption("Operator View Terminal")
        self.clock = pygame.time.Clock()

        pygame.event.pump()
        if os.name == 'nt':
            import ctypes
            hwnd = pygame.display.get_wm_info()["window"]
            user32 = ctypes.windll.user32
            user32.keybd_event(0x12, 0, 0, 0)
            user32.keybd_event(0x12, 0, 2, 0)
            user32.ShowWindow(hwnd, 9)
            user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 3)

        self.BG_COLOR = (15, 15, 15)
        self.GRID_COLOR = (40, 40, 40)
        self.TEXT_COLOR = (220, 220, 220)
        self.CH_COLORS = [(46, 204, 113), (52, 152, 219), (155, 89, 182), (241, 196, 15)]
        self.FLAG_OK = (46, 204, 113)
        self.FLAG_WARN = (231, 76, 60)

        self.font_small = pygame.font.SysFont("Courier New", 14)
        self.font_large = pygame.font.SysFont("Courier New", 20, bold=True)
        self.font_title = pygame.font.SysFont("Arial", 28, bold=True)

        self.eeg_buffer = deque(maxlen=SAMPLES_TO_SHOW)
        for _ in range(SAMPLES_TO_SHOW):
            self.eeg_buffer.append([0.0, 0.0, 0.0, 0.0])

        self.quality_flags = ["OK", "OK", "OK", "OK"]

        self.panel_x = int(self.width * 0.72)
        self.btn_w = int(self.width * 0.23)
        self.btn_h = 70
        self.btn_pause_rect = pygame.Rect(self.panel_x, self.height - 200, self.btn_w, self.btn_h)
        self.btn_exit_rect = pygame.Rect(self.panel_x, self.height - 100, self.btn_w, self.btn_h)

    def assess_signal_quality(self, channel_data, ch_idx):
        peak_to_peak = np.ptp(channel_data)
        if peak_to_peak > ARTIFACT_THRESHOLD_UV:
            self.quality_flags[ch_idx] = "ARTIFACT (CLENCH/BLINK)"
            return self.FLAG_WARN
        elif np.var(channel_data) < FLATLINE_THRESHOLD_UV:
            self.quality_flags[ch_idx] = "FLATLINE (LOOSE SENSOR)"
            return self.FLAG_WARN
        else:
            self.quality_flags[ch_idx] = "OK"
            return self.FLAG_OK

    def draw_oscilloscope(self, data_array):
        graph_x, graph_w = int(self.width * 0.05), int(self.width * 0.62)
        graph_y, graph_h = int(self.height * 0.05), int(self.height * 0.40)

        pygame.draw.rect(self.screen, (20, 20, 20), (graph_x, graph_y, graph_w, graph_h))
        pygame.draw.rect(self.screen, (100, 100, 100), (graph_x, graph_y, graph_w, graph_h), 2)

        for i in range(1, 4):
            y_line = graph_y + (graph_h // 4) * i
            pygame.draw.line(self.screen, self.GRID_COLOR, (graph_x, y_line), (graph_x + graph_w, y_line))

        x_step = graph_w / SAMPLES_TO_SHOW
        ch_height = graph_h // 4

        for ch_idx in range(4):
            ch_data = data_array[:, ch_idx]
            ch_y_offset = graph_y + (ch_idx * ch_height) + (ch_height // 2)
            status_color = self.assess_signal_quality(ch_data, ch_idx)

            points = []
            for i, val in enumerate(ch_data):
                scaled_val = max(min(val * (ch_height / 150.0), ch_height // 2 - 5), -(ch_height // 2 - 5))
                points.append((graph_x + int(i * x_step), ch_y_offset - int(scaled_val)))

            if len(points) > 1:
                pygame.draw.lines(self.screen, self.CH_COLORS[ch_idx], False, points, 2)
            self.screen.blit(self.font_large.render(EEG_CHANNELS[ch_idx], True, status_color),
                             (graph_x + 10, graph_y + (ch_idx * ch_height) + 10))

    def draw_subject_replica(self):
        rep_x, rep_w = int(self.width * 0.05), int(self.width * 0.62)
        rep_y, rep_h = int(self.height * 0.48), int(self.height * 0.25)

        self.screen.blit(self.font_large.render("SUBJECT VIEW REPLICA", True, self.TEXT_COLOR), (rep_x, rep_y - 25))

        pygame.draw.rect(self.screen, (25, 28, 35), (rep_x, rep_y, rep_w, rep_h), border_radius=8)
        pygame.draw.rect(self.screen, self.GRID_COLOR, (rep_x, rep_y, rep_w, rep_h), 2, border_radius=8)

        state_idx = self.sys_state.value
        current_color = self.TEXT_COLOR
        if self.pause_event.is_set() or state_idx in [1, 2, -1, 7]:
            current_color = self.FLAG_WARN
        elif state_idx == 3:
            current_color = (230, 126, 34)
        elif state_idx in [4, 6]:
            current_color = self.FLAG_OK
        elif state_idx == 5:
            current_color = (41, 128, 185)

        t_center_x, t_center_y = rep_x + int(rep_w * 0.25), rep_y + int(rep_h * 0.5)
        t_radius = int(rep_h * 0.40)

        pygame.draw.circle(self.screen, (30, 37, 48), (t_center_x, t_center_y), t_radius + 10, 0)
        pygame.draw.circle(self.screen, current_color, (t_center_x, t_center_y), t_radius, 2)
        for i in range(3):
            rad_angle = np.radians(self.sys_angle.value + (i * 120))
            end_x = t_center_x + int(t_radius * np.cos(rad_angle))
            end_y = t_center_y + int(t_radius * np.sin(rad_angle))
            pygame.draw.line(self.screen, current_color, (t_center_x, t_center_y), (end_x, end_y), 4)
        pygame.draw.circle(self.screen, self.TEXT_COLOR, (t_center_x, t_center_y), 6)

        text_start_x = rep_x + int(rep_w * 0.52)
        text_start_y = rep_y + int(rep_h * 0.10)

        rem_time = max(0, int(self.sys_session_time.value))
        mins, secs = divmod(rem_time, 60)
        time_str = f"{mins:02d}:{secs:02d}"

        lines = [
            ("TIME REMAINING :", time_str, (241, 196, 15) if rem_time > 20 else self.FLAG_WARN),
            ("SCORE PROGRESS:", f"{self.sys_score.value} / {TARGET_SCORE}", self.FLAG_OK),
            ("EFFICIENCY    :", f"{self.sys_eff.value:.1f}%", self.FLAG_OK),
            ("CONFIDENCE    :", f"{self.sys_prob.value * 100:.1f}%", self.TEXT_COLOR)
        ]

        for i, (label, val, color) in enumerate(lines):
            self.screen.blit(self.font_large.render(label, True, self.TEXT_COLOR),
                             (text_start_x, text_start_y + (i * 38)))
            self.screen.blit(self.font_title.render(val, True, color),
                             (text_start_x + 190, text_start_y + (i * 38) - 2))

    def draw_latency_and_decision_logic(self):
        startX = int(self.width * 0.05)
        startY = int(self.height * 0.77)

        self.screen.blit(self.font_large.render("4-STEP DECISION LOGIC PIPELINE", True, self.TEXT_COLOR),
                         (startX, startY))

        # 1 & 2: Smooth and Threshold
        prob_val = self.sys_prob.value
        pygame.draw.rect(self.screen, self.GRID_COLOR, (startX, startY + 25, 200, 15))
        pygame.draw.rect(self.screen, (52, 152, 219), (startX, startY + 25, int(200 * prob_val), 15))
        pygame.draw.line(self.screen, self.FLAG_WARN, (startX + int(200 * DECISION_THRESHOLD), startY + 20),
                         (startX + int(200 * DECISION_THRESHOLD), startY + 45), 2)
        self.screen.blit(
            self.font_small.render(f"1. Smooth & 2. Threshold ({prob_val * 100:.1f}%)", True, self.TEXT_COLOR),
            (startX + 210, startY + 25))

        # 3. Dwell Time
        dwell_ratio = min(1.0, self.sys_dwell.value / DWELL_TIME_SEC)
        pygame.draw.rect(self.screen, self.GRID_COLOR, (startX, startY + 45, 200, 15))
        pygame.draw.rect(self.screen, (230, 126, 34), (startX, startY + 45, int(200 * dwell_ratio), 15))
        self.screen.blit(self.font_small.render(f"3. Dwell Debounce (1.5s)", True, self.TEXT_COLOR),
                         (startX + 210, startY + 45))

        # 4. Refractory Cooldown
        refract_ratio = min(1.0, self.sys_refract.value / REFRACTORY_TIME_SEC)
        pygame.draw.rect(self.screen, self.GRID_COLOR, (startX, startY + 65, 200, 15))
        pygame.draw.rect(self.screen, self.FLAG_WARN, (startX, startY + 65, int(200 * refract_ratio), 15))
        self.screen.blit(self.font_small.render(f"4. Refractory Cooldown (3.0s)", True, self.TEXT_COLOR),
                         (startX + 210, startY + 65))

        # Latency Budget Panel
        latX = int(self.width * 0.38)
        self.screen.blit(self.font_large.render("LATENCY BUDGET (Worst Case Outliers)", True, self.TEXT_COLOR),
                         (latX, startY))

        total_latency = 25.0 + 250.0 + self.sys_lat_prep.value + self.sys_lat_inf.value + 16.7

        self.screen.blit(self.font_small.render(f"Hardware/LSL Link : 25.0 ms", True, self.TEXT_COLOR),
                         (latX, startY + 25))
        self.screen.blit(self.font_small.render(f"Window Wait (Hop) : 250.0 ms", True, self.TEXT_COLOR),
                         (latX, startY + 40))
        self.screen.blit(
            self.font_small.render(f"DSP Extraction    : {self.sys_lat_prep.value:.1f} ms", True, self.TEXT_COLOR),
            (latX, startY + 55))
        self.screen.blit(
            self.font_small.render(f"Model Inference   : {self.sys_lat_inf.value:.1f} ms", True, self.TEXT_COLOR),
            (latX, startY + 70))
        self.screen.blit(self.font_small.render(f"UI Game Render    : 16.7 ms", True, self.TEXT_COLOR),
                         (latX, startY + 85))

        tot_color = self.FLAG_OK if total_latency < 350 else self.FLAG_WARN
        self.screen.blit(self.font_large.render(f"TOTAL ROUND-TRIP: {total_latency:.1f} ms", True, tot_color),
                         (latX, startY + 110))

    def draw_diagnostics_panel(self):
        panel_y = int(self.height * 0.05)
        state_idx = self.sys_state.value
        state_text = STATE_MAP.get(state_idx, "UNKNOWN")

        state_color = self.TEXT_COLOR
        if state_idx in [1, 2, -1, 7]:
            state_color = self.FLAG_WARN
        elif state_idx == 3:
            state_color = (230, 126, 34)
        elif state_idx in [4, 6]:
            state_color = self.FLAG_OK
        elif state_idx == 5:
            state_color = (41, 128, 185)

        self.screen.blit(self.font_large.render("PARTICIPANT STATE", True, self.TEXT_COLOR), (self.panel_x, panel_y))

        font_to_use = self.font_title if state_idx not in [-1, 6, 7] else self.font_small
        self.screen.blit(font_to_use.render(state_text, True, state_color), (self.panel_x, panel_y + 25))

        y_offset = panel_y + 110
        self.screen.blit(self.font_large.render("COGNITIVE LOAD (BETA/ALPHA)", True, self.TEXT_COLOR),
                         (self.panel_x, y_offset))

        total_pwr = self.sys_alpha.value + self.sys_beta.value + 0.001
        a_ratio = self.sys_alpha.value / total_pwr
        b_ratio = self.sys_beta.value / total_pwr

        pygame.draw.rect(self.screen, self.GRID_COLOR, (self.panel_x, y_offset + 30, self.btn_w, 15))
        pygame.draw.rect(self.screen, (41, 128, 185), (self.panel_x, y_offset + 30, int(self.btn_w * a_ratio), 15))
        self.screen.blit(self.font_small.render("Alpha (Relaxation)", True, self.TEXT_COLOR),
                         (self.panel_x, y_offset + 48))

        pygame.draw.rect(self.screen, self.GRID_COLOR, (self.panel_x, y_offset + 70, self.btn_w, 15))
        pygame.draw.rect(self.screen, (241, 196, 15), (self.panel_x, y_offset + 70, int(self.btn_w * b_ratio), 15))
        self.screen.blit(self.font_small.render("Beta (Active Focus)", True, self.TEXT_COLOR),
                         (self.panel_x, y_offset + 88))

        y_offset += 140
        self.screen.blit(self.font_large.render("LIVE QUALITY FLAGS", True, self.TEXT_COLOR), (self.panel_x, y_offset))
        for ch_idx in range(4):
            status = self.quality_flags[ch_idx]
            color = self.FLAG_WARN if "OK" not in status else self.FLAG_OK
            self.screen.blit(self.font_small.render(f"{EEG_CHANNELS[ch_idx]}: {status}", True, color),
                             (self.panel_x, y_offset + 25 + (ch_idx * 20)))

        is_paused = self.pause_event.is_set()
        pause_color = self.FLAG_WARN if is_paused else (80, 80, 80)
        pause_text = "RESUME SYSTEM" if is_paused else "EMERGENCY STOP (PAUSE)"

        pygame.draw.rect(self.screen, pause_color, self.btn_pause_rect, border_radius=8)
        pygame.draw.rect(self.screen, self.TEXT_COLOR, self.btn_pause_rect, width=2, border_radius=8)
        p_txt = self.font_large.render(pause_text, True, self.TEXT_COLOR)
        self.screen.blit(p_txt, p_txt.get_rect(center=self.btn_pause_rect.center))

        pygame.draw.rect(self.screen, (150, 40, 40), self.btn_exit_rect, border_radius=8)
        pygame.draw.rect(self.screen, self.TEXT_COLOR, self.btn_exit_rect, width=2, border_radius=8)
        e_txt = self.font_large.render("EXIT FULL SYSTEM", True, self.TEXT_COLOR)
        self.screen.blit(e_txt, e_txt.get_rect(center=self.btn_exit_rect.center))

    def run(self):
        streams = []
        wait_start = time.time()

        while time.time() - wait_start < 25.0:
            elapsed = int(time.time() - wait_start)
            self.screen.fill(self.BG_COLOR)
            self.screen.blit(
                self.font_title.render(f"SEARCHING FOR HEADSET LSL STREAM... ({25 - elapsed}s)", True, self.TEXT_COLOR),
                (50, 50))
            pygame.display.flip()

            streams = resolve_byprop('type', 'EEG', timeout=1.0)
            if streams: break
            for event in pygame.event.get():
                if event.type == pygame.QUIT: sys.exit(0)

        if not streams:
            print("❌ Operator View Error: Bluetooth timeout. No LSL stream detected.")
            sys.exit(1)

        inlet = StreamInlet(streams[0])
        running = True
        last_data_time = time.time()

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.exit_event.set()
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        if self.btn_pause_rect.collidepoint(event.pos):
                            self.pause_event.clear() if self.pause_event.is_set() else self.pause_event.set()
                        elif self.btn_exit_rect.collidepoint(event.pos):
                            self.exit_event.set()
                            running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.pause_event.clear() if self.pause_event.is_set() else self.pause_event.set()
                    elif event.key == pygame.K_ESCAPE:
                        self.exit_event.set()
                        running = False
                    elif event.key == pygame.K_p:
                        timestamp = time.strftime('%Y%m%d_%H%M%S')
                        pygame.image.save(self.screen, f"Operator_Diagnostic_Screenshot_{timestamp}.png")
                        print(f"📸 Operator Screenshot saved: Operator_Diagnostic_Screenshot_{timestamp}.png")

            if self.exit_event.is_set(): running = False

            chunk, _ = inlet.pull_chunk(timeout=0.0, max_samples=100)
            if chunk:
                last_data_time = time.time()
                for sample in chunk: self.eeg_buffer.append(sample[:4])

            if time.time() - last_data_time > 2.0:
                self.sys_state.value = -1

            self.screen.fill(self.BG_COLOR)
            self.screen.blit(self.font_title.render("SMART ENERGY OPERATOR: MASTER DIAGNOSTICS", True, self.TEXT_COLOR),
                             (int(self.width * 0.05), 10))

            vis_data = np.array(self.eeg_buffer) - np.mean(self.eeg_buffer, axis=0)
            self.draw_oscilloscope(vis_data)
            self.draw_subject_replica()
            self.draw_latency_and_decision_logic()
            self.draw_diagnostics_panel()

            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit()


# ==============================================================================
# PROCESS 3: GAME ENVIRONMENT ENGINE (PARTICIPANT VIEW)
# ==============================================================================
class SmartEnergyOperatorGame:
    def __init__(self, subject_id, target_dir, model_path, task_type, pause_event, exit_event, sys_state, sys_prob,
                 sys_alpha, sys_beta, sys_angle, sys_eff, sys_eco, sys_lat_prep, sys_lat_inf, sys_dwell, sys_refract,
                 sys_score, sys_session_time):
        self.subject_id = subject_id
        self.target_dir = target_dir
        self.model_path = model_path
        self.task_type = task_type
        self.pause_event = pause_event
        self.exit_event = exit_event
        self.sys_state = sys_state
        self.sys_prob = sys_prob
        self.sys_alpha = sys_alpha
        self.sys_beta = sys_beta
        self.sys_angle = sys_angle
        self.sys_eff = sys_eff
        self.sys_eco = sys_eco
        self.sys_lat_prep = sys_lat_prep
        self.sys_lat_inf = sys_lat_inf
        self.sys_dwell = sys_dwell
        self.sys_refract = sys_refract
        self.sys_score = sys_score
        self.sys_session_time = sys_session_time

        pygame.display.init()
        num_displays = pygame.display.get_num_displays()
        pygame.display.quit()

        if num_displays == 1:
            os.environ['SDL_VIDEO_WINDOW_POS'] = "980,50"
        elif 'SDL_VIDEO_WINDOW_POS' in os.environ:
            del os.environ['SDL_VIDEO_WINDOW_POS']

        pygame.init()
        if num_displays > 1:
            self.screen = pygame.display.set_mode((0, 0), pygame.NOFRAME, display=0)
        else:
            self.screen = pygame.display.set_mode((900, 650), pygame.NOFRAME)

        self.width, self.height = self.screen.get_size()
        pygame.display.set_caption("Smart Energy Operator")
        pygame.mouse.set_visible(False)
        self.clock = pygame.time.Clock()

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

        self.BG_DARK = (18, 22, 28)
        self.PANEL_GRAY = (30, 37, 48)
        self.TEXT_WHITE = (240, 244, 248)
        self.FOCUS_GREEN = (46, 204, 113)
        self.RELAX_BLUE = (41, 128, 185)
        self.ALERT_ORANGE = (230, 126, 34)
        self.WARNING_RED = (231, 76, 60)

        self.font_title = pygame.font.SysFont("Arial", 28, bold=True)
        self.font_data = pygame.font.SysFont("Courier New", 20, bold=True)
        self.font_label = pygame.font.SysFont("Arial", 16)
        self.font_large = pygame.font.SysFont("Arial", 42, bold=True)
        self.font_body = pygame.font.SysFont("Arial", 28)

        self.system_status_text = "WAITING FOR DATA"
        self.sys_state.value = 0
        self.prob_history = deque(maxlen=SMOOTHING_WINDOWS)
        self.dwell_start_time = None
        self.refractory_end_time = 0
        self.eeg_buffer = []
        self.target_spin_speed = 0.0

        self.task_refresh_rate = TASK_REFRESH_MATH if task_type == "math" else TASK_REFRESH_WORDS

        self.load_classifier_bundle()

    def load_classifier_bundle(self):
        print(f"🛡️ Loading personalized model bundle: {os.path.basename(self.model_path)}")
        try:
            artifact = joblib.load(self.model_path)
            self.scaler = artifact['scaler']
            self.model = artifact['model']
            self.feature_names = artifact['feature_names']
        except Exception as e:
            print(f"❌ Initialization Error: {e}")
            sys.exit(1)

    def draw_instruction_screen(self):
        self.screen.fill(self.BG_DARK)
        w, h = self.width, self.height

        if self.task_type == "math":
            prompt = "Solve the complex mental math equation."
        else:
            prompt = "Mentally identify the missing letter in the word."

        lines = [
            ("OPERATOR MANUAL: SMART ENERGY TERMINAL", self.TEXT_WHITE, -260, True),
            (f"GOAL: Power turbine to full capacity {TARGET_SCORE} times in 2 MINUTES.", self.FOCUS_GREEN, -180, True),
            ("To INCREASE Production (Spin Faster):", self.FOCUS_GREEN, -100, True),
            (prompt, self.TEXT_WHITE, -50, False),
            ("Push the confidence bar past the red line and hold for 1.5 seconds.", self.TEXT_WHITE, -10, False),
            ("To DECREASE Production (Eco-Mode):", self.RELAX_BLUE, 70, True),
            ("Stop your mental task. Soften your gaze, drop your shoulders, and relax.", self.TEXT_WHITE, 120, False),
            ("The operator will initiate the task shortly.", self.ALERT_ORANGE, 250, True)
        ]
        for text, color, y_offset, is_bold in lines:
            font = self.font_large if is_bold else self.font_body
            surface = font.render(text, True, color)
            rect = surface.get_rect(center=(w // 2, h // 2 + y_offset))
            self.screen.blit(surface, rect)
        pygame.display.flip()

        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    self.exit_event.set()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                    timestamp = time.strftime('%Y%m%d_%H%M%S')
                    pygame.image.save(self.screen, f"Game_Instructions_Screenshot_{timestamp}.png")
                    print(f"📸 Game Screenshot saved: Game_Instructions_Screenshot_{timestamp}.png")

            if self.exit_event.is_set(): sys.exit()
            if not self.pause_event.is_set(): waiting = False
            time.sleep(0.1)

    def save_performance(self, outcome_status):
        """Automated logger to track gamified performance across sessions."""
        log_file = os.path.join(self.target_dir, f"performance_log_{self.task_type}.txt")
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        time_taken = SESSION_TIME_LIMIT_SEC - self.sys_session_time.value
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(
                f"[{timestamp}] Subject: ID_{self.subject_id} | Task: {self.task_type.upper()} | "
                f"Outcome: {outcome_status} | Score: {self.sys_score.value}/{TARGET_SCORE} | "
                f"Time Taken: {time_taken:.1f}s / {SESSION_TIME_LIMIT_SEC:.0f}s\n"
            )
        print(f"\n💾 Session performance appended to: {log_file}")

    def draw_turbine(self, center_x, center_y, radius, angle, color):
        pygame.draw.circle(self.screen, self.PANEL_GRAY, (center_x, center_y), radius + 20, 0)
        pygame.draw.circle(self.screen, color, (center_x, center_y), radius, 3)
        for i in range(3):
            rad_angle = np.radians(angle + (i * 120))
            end_x = center_x + int(radius * np.cos(rad_angle))
            end_y = center_y + int(radius * np.sin(rad_angle))
            pygame.draw.line(self.screen, color, (center_x, center_y), (end_x, end_y), 6)
        pygame.draw.circle(self.screen, self.TEXT_WHITE, (center_x, center_y), 12)

    def draw_dashboard_ui(self, current_task_prompt):
        self.screen.fill(self.BG_DARK)
        w, h = self.width, self.height
        is_paused = self.pause_event.is_set()

        header_rect = pygame.Rect(w * 0.05, h * 0.05, w * 0.9, h * 0.1)
        pygame.draw.rect(self.screen, self.PANEL_GRAY, header_rect, border_radius=8)

        state_idx = self.sys_state.value
        if state_idx == -1:
            self.system_status_text = STATE_MAP[-1]

        title_text = self.font_title.render(f"SMART ENERGY OPERATOR - STATUS: {self.system_status_text}", True,
                                            self.TEXT_WHITE)
        self.screen.blit(title_text, title_text.get_rect(center=header_rect.center))

        current_color = self.TEXT_WHITE
        if is_paused or state_idx in [-1, 7]:
            current_color = self.WARNING_RED
        elif self.system_status_text == "COMMAND ISSUED!" or state_idx == 6:
            current_color = self.FOCUS_GREEN
        elif self.system_status_text == "COOLDOWN (REFRACTORY)":
            current_color = self.WARNING_RED
        elif self.system_status_text == "DWELLING...":
            current_color = self.ALERT_ORANGE
        elif self.system_status_text == "ACTIVE (ECO-MODE)":
            current_color = self.RELAX_BLUE

        t_center_x, t_center_y = int(w * 0.3), int(h * 0.45)
        t_radius = int(h * 0.22)

        prompt_color = self.TEXT_WHITE
        if "RELAX" in current_task_prompt:
            prompt_color = self.RELAX_BLUE
        elif "WAITING" in current_task_prompt or "LOST" in current_task_prompt or "EXPIRED" in current_task_prompt:
            prompt_color = self.WARNING_RED
        elif state_idx == 6:
            prompt_color = self.FOCUS_GREEN
        else:
            prompt_color = self.FOCUS_GREEN

        task_label = self.font_large.render(current_task_prompt, True, prompt_color)
        self.screen.blit(task_label, task_label.get_rect(center=(t_center_x, t_center_y - t_radius - 50)))

        self.draw_turbine(t_center_x, t_center_y, t_radius, self.sys_angle.value, current_color)

        p_width, p_height = int(w * 0.4), int(h * 0.5)
        p_x, p_y = int(w * 0.55), int(h * 0.20)
        panel_rect = pygame.Rect(p_x, p_y, p_width, p_height)
        pygame.draw.rect(self.screen, self.PANEL_GRAY, panel_rect, border_radius=12)

        rem_time = max(0, int(self.sys_session_time.value))
        mins, secs = divmod(rem_time, 60)
        time_str = f"{mins:02d}:{secs:02d}"

        telemetry_lines = [
            ("TIME REMAINING    :", time_str, (241, 196, 15) if rem_time > 20 else self.WARNING_RED),
            ("CLASSIFIER CONFID :", f"{self.sys_prob.value * 100:.1f}%", self.TEXT_WHITE),
            ("PROD EFFICIENCY   :", f"{self.sys_eff.value:.1f}%", self.FOCUS_GREEN),
            ("SESSION SCORE     :", f"{self.sys_score.value} / {TARGET_SCORE}", self.FOCUS_GREEN)
        ]

        start_y = p_y + int(h * 0.05)
        for label, val, color in telemetry_lines:
            self.screen.blit(self.font_label.render(label, True, self.TEXT_WHITE), (p_x + int(w * 0.05), start_y))
            self.screen.blit(self.font_data.render(val, True, color), (p_x + int(w * 0.05), start_y + 30))
            start_y += int(h * 0.09)

        bar_x, bar_y = p_x + int(w * 0.05), start_y + int(h * 0.01)
        bar_w, bar_h = int(p_width * 0.8), int(h * 0.04)
        pygame.draw.rect(self.screen, self.BG_DARK, (bar_x, bar_y, bar_w, bar_h))
        pygame.draw.rect(self.screen, current_color, (bar_x, bar_y, int(bar_w * self.sys_prob.value), bar_h))
        thresh_x = bar_x + int(bar_w * DECISION_THRESHOLD)
        pygame.draw.line(self.screen, self.WARNING_RED, (thresh_x, bar_y - 8), (thresh_x, bar_y + bar_h + 8), 4)

    def run_loop(self):
        self.pause_event.set()
        self.draw_instruction_screen()

        streams = resolve_byprop('type', 'EEG', timeout=15.0)
        if not streams: sys.exit(1)

        inlet = StreamInlet(streams[0])
        inlet.pull_chunk(timeout=1.0, max_samples=3000)
        running = True
        last_data_time = time.time()

        session_start_time = time.time()
        last_task_time = 0
        current_task_prompt = "WAITING ON OPERATOR"
        outcome_status = "ABORTED"

        while running:
            if self.exit_event.is_set(): break
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                    timestamp = time.strftime('%Y%m%d_%H%M%S')
                    pygame.image.save(self.screen, f"Game_Dashboard_Screenshot_{timestamp}.png")
                    print(f"📸 Game Screenshot saved: Game_Dashboard_Screenshot_{timestamp}.png")

            # Check Termination States
            if self.sys_state.value == 6:
                self.draw_dashboard_ui("MISSION ACCOMPLISHED!")
                pygame.display.flip()
                time.sleep(3.0)
                outcome_status = "SUCCESS"
                running = False
                continue

            if self.sys_state.value == 7:
                self.draw_dashboard_ui("TIME EXPIRED - GAME OVER")
                pygame.display.flip()
                time.sleep(3.0)
                outcome_status = "TIMEOUT"
                running = False
                continue

            # Update Session Timer
            if not self.pause_event.is_set():
                elapsed_session_time = time.time() - session_start_time
                remaining_session_time = max(0.0, SESSION_TIME_LIMIT_SEC - elapsed_session_time)
                self.sys_session_time.value = remaining_session_time

                if remaining_session_time <= 0.0 and self.sys_score.value < TARGET_SCORE:
                    self.system_status_text = "TIME EXPIRED!"
                    self.sys_state.value = 7
                    continue
            else:
                # Pause timer offset
                session_start_time = time.time() - (SESSION_TIME_LIMIT_SEC - self.sys_session_time.value)

            samples, _ = inlet.pull_chunk(timeout=0.0, max_samples=SAMPLES_PER_EPOCH)
            if samples:
                self.eeg_buffer.extend(samples)
                last_data_time = time.time()

            if len(self.eeg_buffer) >= SAMPLES_PER_EPOCH:
                current_time = time.time()
                epoch_data = np.array(self.eeg_buffer[:SAMPLES_PER_EPOCH])[:, :4]

                t0_prep = time.perf_counter()
                clean_epoch = apply_causal_filters(epoch_data, SAMPLING_RATE)
                features, a_pwr, b_pwr = extract_live_features(clean_epoch, SAMPLING_RATE, self.feature_names)
                t1_prep = time.perf_counter()
                self.sys_lat_prep.value = (t1_prep - t0_prep) * 1000.0

                self.sys_alpha.value = a_pwr
                self.sys_beta.value = b_pwr

                scaled_features = self.scaler.transform(features)

                t0_inf = time.perf_counter()
                raw_prob = self.model.predict_proba(scaled_features)[0][1]
                t1_inf = time.perf_counter()
                self.sys_lat_inf.value = (t1_inf - t0_inf) * 1000.0

                self.prob_history.append(raw_prob)
                self.sys_prob.value = np.mean(self.prob_history)

                # DECOUPLED STIMULUS GENERATION & GAME LOGIC
                if self.pause_event.is_set():
                    self.system_status_text = "SYSTEM HALTED BY OPERATOR"
                    self.sys_state.value = 1
                    self.dwell_start_time = None
                    self.sys_eff.value = 0.0
                    self.target_spin_speed = 0.0
                    current_task_prompt = "WAITING ON OPERATOR"

                elif current_time < self.refractory_end_time:
                    self.system_status_text = "COOLDOWN (REFRACTORY)"
                    self.sys_state.value = 2
                    self.dwell_start_time = None
                    self.sys_eff.value = max(10.0, self.sys_eff.value - 0.5)
                    self.sys_eco.value = min(100.0, self.sys_eco.value + 1.0)
                    self.target_spin_speed = 8.0
                    current_task_prompt = "RELAX (COOLDOWN)"

                else:
                    if current_time - last_task_time > self.task_refresh_rate:
                        if self.task_type == "math":
                            current_task_prompt = "SOLVE: " + generate_math_problem()
                        else:
                            current_task_prompt = "COMPLETE: " + generate_cloze_prompt()
                        last_task_time = current_time

                    if self.sys_prob.value >= DECISION_THRESHOLD:
                        if self.dwell_start_time is None:
                            self.dwell_start_time = current_time
                            self.system_status_text = "DWELLING..."
                            self.sys_state.value = 3
                            self.target_spin_speed = 4.0
                        elif (current_time - self.dwell_start_time) >= DWELL_TIME_SEC:

                            self.sys_score.value += 1

                            if self.sys_score.value >= TARGET_SCORE:
                                self.system_status_text = "MISSION ACCOMPLISHED!"
                                self.sys_state.value = 6
                                self.target_spin_speed = 30.0
                            else:
                                self.system_status_text = "COMMAND ISSUED!"
                                self.sys_state.value = 4
                                self.refractory_end_time = current_time + REFRACTORY_TIME_SEC
                                self.dwell_start_time = None
                                self.sys_eff.value = min(100.0, self.sys_eff.value + 15.0)
                                self.sys_eco.value = max(35.0, self.sys_eco.value - 10.0)
                                self.target_spin_speed = 25.0
                    else:
                        self.system_status_text = "ACTIVE (ECO-MODE)"
                        self.sys_state.value = 5
                        self.dwell_start_time = None
                        self.sys_eff.value = max(10.0, self.sys_eff.value - 0.2)
                        self.sys_eco.value = min(100.0, self.sys_eco.value + 0.5)
                        self.target_spin_speed = 2.0

                # Metric Tracking
                if self.dwell_start_time is not None:
                    self.sys_dwell.value = min(current_time - self.dwell_start_time, DWELL_TIME_SEC)
                else:
                    self.sys_dwell.value = 0.0

                if self.refractory_end_time > current_time:
                    self.sys_refract.value = self.refractory_end_time - current_time
                else:
                    self.sys_refract.value = 0.0

                self.eeg_buffer = self.eeg_buffer[HOP_SAMPLES:]

            if time.time() - last_data_time > 2.0:
                self.sys_state.value = -1
                self.target_spin_speed = 0.0
                current_task_prompt = "CONNECTION LOST"

            self.sys_angle.value += self.target_spin_speed
            self.sys_angle.value %= 360.0

            self.draw_dashboard_ui(current_task_prompt)
            pygame.display.flip()
            self.clock.tick(60)

        self.save_performance(outcome_status)
        pygame.quit()


# ==============================================================================
# MULTIPROCESSING LAUNCHER
# ==============================================================================
def run_operator_view(pause, exit, state, prob, alpha, beta, angle, eff, eco, lat_prep, lat_inf, dwell, refract, score,
                      time_lim):
    try:
        OperatorViewTerminal(pause, exit, state, prob, alpha, beta, angle, eff, eco, lat_prep, lat_inf, dwell,
                             refract, score, time_lim).run()
    except Exception as e:
        with open("operator_crash_log.txt", "w") as f:
            f.write(traceback.format_exc())


if __name__ == "__main__":
    multiprocessing.freeze_support()

    print("=" * 60)
    print("🧠 Smart Energy Closed-Loop Terminal")
    print("=" * 60)

    # Receive Task type dynamically from the UI
    subject_id, target_dir, task_type, target_mac, verified_base_dir = run_pre_registration()

    print(f"\n📡 Audit Trail - Hardware MAC Verified: {target_mac}")
    print(f"📁 Audit Trail - Base Directory Verified: {verified_base_dir}")

    # Automatically map to the single model generated in the /online/ directory for that specific task
    search_pattern = os.path.join(target_dir, f"*{task_type}*.joblib")
    model_files = glob.glob(search_pattern)

    if not model_files:
        fallback_pattern = os.path.join(target_dir, "*.joblib")
        model_files = glob.glob(fallback_pattern)

    if not model_files:
        print(f"❌ Error: No model bundle found in {target_dir}")
        sys.exit(1)

    model_path = model_files[0]

    sys_pause_event = multiprocessing.Event()
    sys_exit_event = multiprocessing.Event()
    system_state_val = multiprocessing.Value('i', 0)
    sys_prob = multiprocessing.Value('d', 0.0)
    sys_alpha = multiprocessing.Value('d', 0.0)
    sys_beta = multiprocessing.Value('d', 0.0)
    sys_angle = multiprocessing.Value('d', 0.0)
    sys_eff = multiprocessing.Value('d', 10.0)
    sys_eco = multiprocessing.Value('d', 100.0)

    sys_lat_prep = multiprocessing.Value('d', 0.0)
    sys_lat_inf = multiprocessing.Value('d', 0.0)
    sys_dwell = multiprocessing.Value('d', 0.0)
    sys_refract = multiprocessing.Value('d', 0.0)

    # Gamification Values
    sys_score_val = multiprocessing.Value('i', 0)
    sys_session_time_val = multiprocessing.Value('d', SESSION_TIME_LIMIT_SEC)

    stream_process = multiprocessing.Process(target=start_stream, args=(target_mac,))
    stream_process.daemon = True
    stream_process.start()
    time.sleep(3.0)

    operator_process = multiprocessing.Process(
        target=run_operator_view,
        args=(sys_pause_event, sys_exit_event, system_state_val, sys_prob, sys_alpha, sys_beta, sys_angle, sys_eff,
              sys_eco, sys_lat_prep, sys_lat_inf, sys_dwell, sys_refract, sys_score_val, sys_session_time_val)
    )
    operator_process.start()
    time.sleep(1.0)

    try:
        game = SmartEnergyOperatorGame(subject_id, target_dir, model_path, task_type, sys_pause_event, sys_exit_event,
                                       system_state_val, sys_prob, sys_alpha, sys_beta, sys_angle, sys_eff, sys_eco,
                                       sys_lat_prep, sys_lat_inf, sys_dwell, sys_refract, sys_score_val,
                                       sys_session_time_val)
        game.run_loop()
    except Exception as e:
        with open("protocol_crash_log.txt", "w") as f:
            f.write(traceback.format_exc())

    operator_process.terminate()
    stream_process.terminate()
    print("\n🛑 System Terminated Safely.")