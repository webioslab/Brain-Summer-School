"""
Online BCI Calibration & Data Collection Paradigm
Task: Unified Cognitive Stimulus (Math / Words) - Co-Adaptive
--------------------------------------------------
Phase 1: Pre-Registration UI (Existing Subjects + Trial Count + Task Selection)
Phase 2: Dual-Screen Execution (Multiprocessing IPC)
    - Screen 0 (Participant): NOFRAME visual protocol, ERD baselining.
    - Screen 1 (Operator): NOFRAME live diagnostics & synced timers.
Phase 3: Automated Co-Adaptive Post-Processing
    - Collects original offline data + ALL historical online data + new online data.
    - Applies MNE ICA, extracts Beta/Alpha ratios, trains LDA,
      generates a full performance report, and saves the model inside /online/.
"""

import os
import sys
import time
import csv
import random
import threading
import multiprocessing
import numpy as np
import pandas as pd
import pygame
import winsound
import joblib
import glob
from collections import deque
from scipy import signal
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, cohen_kappa_score,
    f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
)
from pylsl import StreamInlet, resolve_byprop
from muselsl import stream
import mne

# CONFIGURATION & PROTOCOL SETTINGS
# Default lookup directory; will be verified against personal_info.txt
BASE_DATA_DIR = r"D:\MuseData"

SAMPLING_RATE = 256
EEG_CHANNELS = ["TP9", "AF7", "AF8", "TP10"]
EPOCH_LENGTH_SEC = 2.0
SAMPLES_PER_EPOCH = int(SAMPLING_RATE * EPOCH_LENGTH_SEC)
HOP_SAMPLES = int(SAMPLING_RATE * 0.25)

ARTIFACT_THRESHOLD_UV = 350.0
FLATLINE_THRESHOLD_UV = 2.0
DISPLAY_WINDOW_SEC = 3.0
SAMPLES_TO_SHOW = int(SAMPLING_RATE * DISPLAY_WINDOW_SEC)

DUR_BASELINE_OPEN = 30
DUR_BASELINE_CLOSED = 30
DUR_RELAX = 60
DUR_TASK = 60

# Task Refresh Rates
TASK_REFRESH_MATH = 4
TASK_REFRESH_WORDS = 2

MARKER_IDLE = 0
MARKER_BASE_OPEN = 1
MARKER_BASE_CLOSED = 2
MARKER_RELAX = 3
MARKER_TASK = 4

STATE_MAP = {
    0: "IDLE (WAITING FOR START)",
    1: "BASELINE (EYES OPEN)",
    2: "BASELINE (EYES CLOSED)",
    3: "RESTING PHASE",
    4: "ACTIVE COGNITIVE TASK",
    5: "PROTOCOL COMPLETE"
}

# Global pool of high-frequency words for working memory without vocabulary strain
WORD_POOL = [
    'WATER', 'EARTH', 'PLANT', 'SOLAR', 'POWER', 'LIGHT', 'WIND',
    'OCEAN', 'RIVER', 'TREES', 'GREEN', 'NATURE', 'WORLD', 'CLEAN',
    'HOUSE', 'GLASS', 'PAPER', 'METAL', 'CHAIR', 'TABLE', 'BRAIN',
    'HEART', 'BLOOD', 'SENSE', 'SOUND', 'VOICE', 'MUSIC', 'NOISE',
    'SMART', 'THINK', 'LEARN', 'STUDY', 'FOCUS', 'LOGIC', 'SPEED',
    'FORCE', 'MOTION', 'SPACE', 'TRACK', 'PHASE'
]
random.shuffle(WORD_POOL)

# Keep MNE from flooding the console with logs
mne.set_log_level('ERROR')

# SHARED PIPELINE FUNCTIONS
def apply_causal_filters(eeg_data, fs):
    """Fast linear filters for live data visualization."""
    b_notch, a_notch = signal.iirnotch(w0=50.0, Q=30.0, fs=fs)
    filtered_data = signal.lfilter(b_notch, a_notch, eeg_data, axis=0)
    b_band, a_band = signal.butter(N=4, Wn=[1.0, 40.0], btype='bandpass', fs=fs)
    filtered_data = signal.lfilter(b_band, a_band, filtered_data, axis=0)
    return filtered_data

def extract_live_features(epoch_data, fs):
    """Calculates normalized Alpha and Beta powers to mirror the offline pipeline."""
    features = []
    for ch_idx in range(4):
        freqs, psd = signal.welch(epoch_data[:, ch_idx], fs=fs, nperseg=fs * 2)

        alpha_idx = np.logical_and(freqs >= 8, freqs <= 12)
        alpha_pwr = np.trapz(psd[alpha_idx], freqs[alpha_idx]) / (12 - 8)

        beta_idx = np.logical_and(freqs >= 13, freqs <= 30)
        beta_pwr = np.trapz(psd[beta_idx], freqs[beta_idx]) / (30 - 13)

        features.extend([alpha_pwr, beta_pwr])
    return np.array(features).reshape(1, -1)

def generate_cloze_prompt():
    """Pulls a unique word, replaces a random character with an underscore."""
    global WORD_POOL
    if not WORD_POOL:
        WORD_POOL = [
            'WATER', 'EARTH', 'PLANT', 'SOLAR', 'POWER', 'LIGHT', 'WIND',
            'OCEAN', 'RIVER', 'TREES', 'GREEN', 'NATURE', 'WORLD', 'CLEAN',
            'HOUSE', 'GLASS', 'PAPER', 'METAL', 'CHAIR', 'TABLE', 'BRAIN',
            'HEART', 'BLOOD', 'SENSE', 'SOUND', 'VOICE', 'MUSIC', 'NOISE',
            'SMART', 'THINK', 'LEARN', 'STUDY', 'FOCUS', 'LOGIC', 'SPEED',
            'FORCE', 'MOTION', 'SPACE', 'TRACK', 'PHASE'
        ]
        random.shuffle(WORD_POOL)

    target_word = WORD_POOL.pop()
    idx = random.randint(0, len(target_word) - 1)
    puzzle = target_word[:idx] + "_" + target_word[idx + 1:]
    return " ".join(puzzle)

def generate_math_problem():
    """Generates sequential addition/subtraction to test working memory, not memorization."""
    num1 = random.randint(1, 9)
    num2 = random.randint(1, 9)
    num3 = random.randint(1, 9)
    op1 = random.choice(['+', '-'])
    op2 = random.choice(['+', '-'])
    return f"{num1} {op1} {num2} {op2} {num3} = ?"

def draw_multiline_text(screen, text, font, color, center_x, center_y):
    lines = text.split('\n')
    line_height = font.get_linesize()
    start_y = center_y - (len(lines) * line_height) // 2
    for i, line in enumerate(lines):
        surface = font.render(line, True, color)
        rect = surface.get_rect(center=(center_x, start_y + (i * line_height)))
        screen.blit(surface, rect)

# PROCESS 1: STREAMER
def start_stream(mac_address):
    """Initiates the Muse LSL stream using the extracted hardware MAC address."""
    if mac_address and mac_address != "UNKNOWN":
        print(f"\n🔗 Establishing direct connection to Muse MAC: {mac_address}...")
        stream(mac_address)
    else:
        print("❌ Error: Invalid MAC Address provided. Cannot establish stream.")
        sys.exit(0)

# PROCESS 2: OPERATOR VIEW
class OperatorDiagnostics:
    def __init__(self, start_event, exit_event, sys_state, sys_time, base_alpha, sys_trial, sys_total_trials,
                 task_type):
        self.start_event = start_event
        self.exit_event = exit_event
        self.sys_state = sys_state
        self.sys_time = sys_time
        self.base_alpha = base_alpha
        self.sys_trial = sys_trial
        self.sys_total_trials = sys_total_trials
        self.task_type = task_type

        # Isolate monitor counting before locking environment
        pygame.display.init()
        num_displays = pygame.display.get_num_displays()
        pygame.display.quit()

        # Force Operator View to primary laptop screen (Display 0)
        os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"

        pygame.init()
        self.screen = pygame.display.set_mode((0, 0), pygame.NOFRAME, display=0)
        self.width, self.height = self.screen.get_size()
        pygame.display.set_caption("Operator View Terminal")
        self.clock = pygame.time.Clock()

        pygame.event.pump()
        if os.name == 'nt':
            import ctypes
            hwnd = pygame.display.get_wm_info()["window"]
            user32 = ctypes.windll.user32
            user32.ShowWindow(hwnd, 9)
            user32.SetForegroundWindow(hwnd)

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

        self.artifact_detected = False
        self.rhythm_flag_detected = False
        self.alpha_pwr, self.beta_pwr = 0.0, 0.0
        self.quality_flags = ["OK"] * 4

        self.panel_x = int(self.width * 0.72)
        self.btn_w = int(self.width * 0.23)
        self.btn_h = 70
        self.btn_start_rect = pygame.Rect(self.panel_x, self.height - 200, self.btn_w, self.btn_h)
        self.btn_exit_rect = pygame.Rect(self.panel_x, self.height - 100, self.btn_w, self.btn_h)

    def assess_signal_quality(self, channel_data, ch_idx):
        if np.ptp(channel_data) > ARTIFACT_THRESHOLD_UV:
            self.quality_flags[ch_idx] = "ARTIFACT (CLENCH/BLINK)"
            return self.FLAG_WARN
        elif np.var(channel_data) < FLATLINE_THRESHOLD_UV:
            self.quality_flags[ch_idx] = "FLATLINE (LOOSE SENSOR)"
            return self.FLAG_WARN
        self.quality_flags[ch_idx] = "OK"
        return self.FLAG_OK

    def run(self):
        self.screen.fill(self.BG_COLOR)
        self.screen.blit(self.font_title.render("SEARCHING FOR HEADSET STREAM (Up to 25s)...", True, self.TEXT_COLOR),
                         (50, 50))
        pygame.display.flip()

        streams = []
        wait_start = time.time()
        while time.time() - wait_start < 25.0:
            streams = resolve_byprop('type', 'EEG', timeout=1.0)
            if streams: break
            for event in pygame.event.get():
                if event.type == pygame.QUIT: sys.exit(0)

        if not streams: sys.exit(1)
        inlet = StreamInlet(streams[0])
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.exit_event.set()
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.btn_start_rect.collidepoint(event.pos):
                        self.start_event.set()
                    elif self.btn_exit_rect.collidepoint(event.pos):
                        self.exit_event.set()
                        running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.start_event.set()
                    elif event.key == pygame.K_ESCAPE:
                        self.exit_event.set()
                        running = False

            if self.exit_event.is_set(): running = False

            chunk, _ = inlet.pull_chunk(timeout=0.0, max_samples=100)
            if chunk:
                for sample in chunk: self.eeg_buffer.append(sample[:4])

            data_array = np.array(self.eeg_buffer)
            self.artifact_detected = False
            self.rhythm_flag_detected = False

            if len(data_array) >= int(SAMPLING_RATE * 0.5):
                if np.max(np.ptp(data_array[-int(SAMPLING_RATE * 0.5):], axis=0)) > ARTIFACT_THRESHOLD_UV:
                    self.artifact_detected = True

            if len(data_array) >= SAMPLES_PER_EPOCH:
                clean_epoch = apply_causal_filters(data_array[-SAMPLES_PER_EPOCH:], SAMPLING_RATE)
                try:
                    feats = extract_live_features(clean_epoch, SAMPLING_RATE)
                    self.alpha_pwr = np.mean(feats[0, [0, 2, 4, 6]])
                    self.beta_pwr = np.mean(feats[0, [1, 3, 5, 7]])

                    c_state = self.sys_state.value
                    if c_state == 4 and self.base_alpha.value > 0:
                        if self.alpha_pwr >= (self.base_alpha.value * 0.85):
                            self.rhythm_flag_detected = True
                except:
                    pass

            self.screen.fill(self.BG_COLOR)
            g_x, g_w, g_h, g_y = int(self.width * 0.05), int(self.width * 0.62), int(self.height * 0.8), int(
                self.height * 0.1)
            pygame.draw.rect(self.screen, (20, 20, 20), (g_x, g_y, g_w, g_h))
            pygame.draw.rect(self.screen, (100, 100, 100), (g_x, g_y, g_w, g_h), 2)

            for i in range(1, 4):
                pygame.draw.line(self.screen, self.GRID_COLOR, (g_x, g_y + (g_h // 4) * i),
                                 (g_x + g_w, g_y + (g_h // 4) * i))

            x_step, ch_h = g_w / SAMPLES_TO_SHOW, g_h // 4
            vis_data = data_array - np.mean(data_array, axis=0)

            for ch_idx in range(4):
                ch_d = vis_data[:, ch_idx]
                offset = g_y + (ch_idx * ch_h) + (ch_h // 2)
                status_color = self.assess_signal_quality(ch_d, ch_idx)

                pts = []
                for i, val in enumerate(ch_d):
                    scaled_val = max(min(val * (ch_h / 150.0), ch_h // 2 - 5), -(ch_h // 2 - 5))
                    pts.append((g_x + int(i * x_step), offset - int(scaled_val)))

                if len(pts) > 1: pygame.draw.lines(self.screen, self.CH_COLORS[ch_idx], False, pts, 2)
                self.screen.blit(self.font_large.render(EEG_CHANNELS[ch_idx], True, status_color),
                                 (g_x + 10, g_y + (ch_idx * ch_h) + 10))

            state_idx = self.sys_state.value
            state_text = STATE_MAP.get(state_idx, "UNKNOWN")

            if state_idx in [3, 4]:
                state_text += f" (TRIAL {self.sys_trial.value}/{self.sys_total_trials.value})"

            state_color = self.FLAG_WARN if state_idx == 0 else (52, 152, 219) if state_idx in [1, 2,
                                                                                                3] else self.FLAG_OK

            p_y = int(self.height * 0.1)
            self.screen.blit(self.font_large.render("PROTOCOL PHASE", True, self.TEXT_COLOR), (self.panel_x, p_y))
            self.screen.blit(self.font_title.render(state_text, True, state_color), (self.panel_x, p_y + 25))

            time_rem = self.sys_time.value
            time_str = f"TIME REMAINING: {time_rem}s" if state_idx in [1, 2, 3, 4] else "TIME REMAINING: --"
            self.screen.blit(self.font_large.render(time_str, True, (241, 196, 15)), (self.panel_x, p_y + 65))

            y_offset = p_y + 120
            art_color = (231, 76, 60) if self.artifact_detected else (40, 40, 40)
            pygame.draw.circle(self.screen, art_color, (self.panel_x + 20, y_offset + 10), 20)
            self.screen.blit(self.font_large.render("ARTIFACT WARNING", True, self.TEXT_COLOR),
                             (self.panel_x + 50, y_offset))

            rhy_color = (241, 196, 15) if self.rhythm_flag_detected else (40, 40, 40)
            pygame.draw.circle(self.screen, rhy_color, (self.panel_x + 20, y_offset + 50), 20)
            self.screen.blit(self.font_large.render("RHYTHM ERD MISMATCH", True, self.TEXT_COLOR),
                             (self.panel_x + 50, y_offset + 40))

            y_offset += 100
            self.screen.blit(self.font_large.render("COGNITIVE DENSITY (BETA/ALPHA)", True, self.TEXT_COLOR),
                             (self.panel_x, y_offset))
            total_pwr = self.alpha_pwr + self.beta_pwr + 0.001

            pygame.draw.rect(self.screen, self.GRID_COLOR, (self.panel_x, y_offset + 30, self.btn_w, 15))
            pygame.draw.rect(self.screen, (41, 128, 185),
                             (self.panel_x, y_offset + 30, int(self.btn_w * (self.alpha_pwr / total_pwr)), 15))
            self.screen.blit(self.font_small.render("Alpha (Relaxation)", True, self.TEXT_COLOR),
                             (self.panel_x, y_offset + 48))

            pygame.draw.rect(self.screen, self.GRID_COLOR, (self.panel_x, y_offset + 70, self.btn_w, 15))
            pygame.draw.rect(self.screen, (241, 196, 15),
                             (self.panel_x, y_offset + 70, int(self.btn_w * (self.beta_pwr / total_pwr)), 15))

            beta_label = "Beta (Mental Math Focus)" if self.task_type == "math" else "Beta (Missing Letter Focus)"
            self.screen.blit(self.font_small.render(beta_label, True, self.TEXT_COLOR),
                             (self.panel_x, y_offset + 88))

            has_started = self.start_event.is_set()
            start_color = (80, 80, 80) if has_started else (46, 204, 113)
            pygame.draw.rect(self.screen, start_color, self.btn_start_rect, border_radius=8)
            pygame.draw.rect(self.screen, self.TEXT_COLOR, self.btn_start_rect, width=2, border_radius=8)
            s_surf = self.font_large.render("PROTOCOL RUNNING" if has_started else "START PROTOCOL", True,
                                            self.TEXT_COLOR)
            self.screen.blit(s_surf, s_surf.get_rect(center=self.btn_start_rect.center))

            pygame.draw.rect(self.screen, (150, 40, 40), self.btn_exit_rect, border_radius=8)
            pygame.draw.rect(self.screen, self.TEXT_COLOR, self.btn_exit_rect, width=2, border_radius=8)
            e_surf = self.font_large.render("EXIT FULL SYSTEM", True, self.TEXT_COLOR)
            self.screen.blit(e_surf, e_surf.get_rect(center=self.btn_exit_rect.center))

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()

# PROCESS 3: PARTICIPANT PROTOCOL
class CalibrationProtocol:
    def __init__(self, save_dir, start_event, exit_event, sys_state, sys_time, base_alpha, sys_trial, sys_total_trials,
                 task_type):
        self.start_event = start_event
        self.exit_event = exit_event
        self.sys_state = sys_state
        self.sys_time = sys_time
        self.base_alpha = base_alpha
        self.sys_trial = sys_trial
        self.sys_total_trials = sys_total_trials
        self.task_type = task_type

        # Isolate monitor counting before locking environment
        pygame.display.init()
        num_displays = pygame.display.get_num_displays()
        pygame.display.quit()

        if num_displays == 1:
            os.environ['SDL_VIDEO_WINDOW_POS'] = "950,50"  # Push Participant View to Top-Right
        elif 'SDL_VIDEO_WINDOW_POS' in os.environ:
            del os.environ['SDL_VIDEO_WINDOW_POS']

        pygame.init()
        if num_displays > 1:
            # Bind to the extended display explicitly (Display 1)
            self.screen = pygame.display.set_mode((0, 0), pygame.NOFRAME, display=1)
        else:
            self.screen = pygame.display.set_mode((900, 650))

        self.width, self.height = self.screen.get_size()
        pygame.mouse.set_visible(False)
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("Courier New", 64, bold=True)
        self.prompt_font = pygame.font.SysFont("Arial", 42, bold=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')

        self.csv_path = os.path.join(save_dir, f"Live_Calibration_data_{self.task_type}_{timestamp}.csv")

        self.sys_state.value = 0
        self.sys_time.value = 0
        self.sys_trial.value = 1
        self.marker = MARKER_IDLE
        self.display_text = "WAITING FOR OPERATOR\n\nPlease keep your head still and relax."
        self.task_prompt = ""
        self.phase_start_time = 0
        self.last_task_time = 0

    def trigger_beep(self, frequency, duration):
        threading.Thread(target=winsound.Beep, args=(frequency, duration), daemon=True).start()

    def run(self):
        self.screen.fill((18, 22, 28))
        draw_multiline_text(self.screen, "CONNECTING TO HEADSET...\n\nSearching for LSL Stream (Up to 25s).",
                            self.prompt_font,
                            (240, 244, 248), self.width // 2, self.height // 2)
        pygame.display.flip()

        streams = []
        wait_start = time.time()
        while time.time() - wait_start < 25.0:
            streams = resolve_byprop('type', 'EEG', timeout=1.0)
            if streams: break
            for event in pygame.event.get():
                if event.type == pygame.QUIT: sys.exit(0)

        if not streams: sys.exit(1)

        csv_file = open(self.csv_path, 'w', newline='')
        writer = csv.writer(csv_file)
        writer.writerow(["Timestamp", "TP9", "AF7", "AF8", "TP10", "Marker", "Artifact_Flag", "Rhythm_Flag"])

        inlet = StreamInlet(streams[0])
        running = True
        recent_chunk_artifact = deque(maxlen=int(SAMPLING_RATE * 0.5))
        recent_chunk_rhythm = deque(maxlen=SAMPLES_PER_EPOCH)

        baseline_alpha_history = []
        rhythm_hop_counter = 0

        while running:
            if self.exit_event.is_set(): break
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False

            current_time = time.time()
            elapsed = current_time - self.phase_start_time
            current_state = self.sys_state.value

            if current_state == 0 and self.start_event.is_set():
                self.sys_state.value = 1
                self.marker = MARKER_BASE_OPEN
                self.phase_start_time = time.time()
                elapsed = 0
                self.trigger_beep(1000, 500)

            elif current_state == 1:
                self.sys_time.value = int(DUR_BASELINE_OPEN - elapsed)
                self.display_text = f"BASELINE (EYES OPEN)\n\nStare at the crosshair and relax."
                if elapsed > DUR_BASELINE_OPEN:
                    self.sys_state.value = 2
                    self.marker = MARKER_BASE_CLOSED
                    self.phase_start_time = current_time
                    self.trigger_beep(1000, 500)

            elif current_state == 2:
                self.sys_time.value = int(DUR_BASELINE_CLOSED - elapsed)
                self.display_text = "BASELINE (EYES CLOSED)\n\nClose your eyes and relax completely."
                if elapsed > DUR_BASELINE_CLOSED:
                    self.sys_state.value = 3
                    self.marker = MARKER_RELAX
                    self.phase_start_time = current_time
                    self.trigger_beep(1200, 500)

            elif current_state == 3:
                self.sys_time.value = int(DUR_RELAX - elapsed)
                self.display_text = f"RESTING PHASE (Trial {self.sys_trial.value}/{self.sys_total_trials.value})\n\nKeep your eyes open and relax."
                if elapsed > DUR_RELAX:
                    self.sys_state.value = 4
                    self.marker = MARKER_TASK
                    self.phase_start_time = current_time
                    self.last_task_time = 0
                    self.trigger_beep(1500, 800)

            elif current_state == 4:
                self.sys_time.value = int(DUR_TASK - elapsed)

                task_refresh_limit = TASK_REFRESH_MATH if self.task_type == "math" else TASK_REFRESH_WORDS

                if current_time - self.last_task_time >= task_refresh_limit:
                    if self.task_type == "math":
                        self.task_prompt = generate_math_problem()
                    else:
                        self.task_prompt = generate_cloze_prompt()
                    self.last_task_time = current_time

                if self.task_type == "math":
                    self.display_text = f"ACTIVE FOCUS (Trial {self.sys_trial.value}/{self.sys_total_trials.value})\n\nSolve Mentally:\n\n{self.task_prompt}"
                else:
                    self.display_text = f"ACTIVE FOCUS (Trial {self.sys_trial.value}/{self.sys_total_trials.value})\n\nMentally identify the missing letter:\n\n{self.task_prompt}"

                if elapsed > DUR_TASK:
                    if self.sys_trial.value < self.sys_total_trials.value:
                        self.sys_trial.value += 1
                        self.sys_state.value = 3
                        self.marker = MARKER_RELAX
                        self.phase_start_time = current_time
                        self.trigger_beep(1200, 500)
                    else:
                        self.sys_state.value = 5
                        self.marker = MARKER_IDLE
                        self.display_text = "PROTOCOL COMPLETE\n\nSaving data..."
                        self.phase_start_time = current_time

            elif current_state == 5:
                self.sys_time.value = 0
                if elapsed > 3:
                    self.exit_event.set()
                    running = False

            chunk, timestamps = inlet.pull_chunk(timeout=0.0)
            if chunk:
                for i, sample in enumerate(chunk):
                    s_data = sample[:4]
                    recent_chunk_artifact.append(s_data)
                    recent_chunk_rhythm.append(s_data)
                    rhythm_hop_counter += 1

                    artifact_flag = 0
                    if len(recent_chunk_artifact) == int(SAMPLING_RATE * 0.5):
                        if np.max(np.ptp(recent_chunk_artifact, axis=0)) > ARTIFACT_THRESHOLD_UV:
                            artifact_flag = 1

                    current_rhythm_flag = 0
                    if rhythm_hop_counter >= HOP_SAMPLES:
                        rhythm_hop_counter = 0
                        if len(recent_chunk_rhythm) == SAMPLES_PER_EPOCH:
                            clean_epoch = apply_causal_filters(np.array(recent_chunk_rhythm), SAMPLING_RATE)
                            try:
                                feats = extract_live_features(clean_epoch, SAMPLING_RATE)
                                a_pwr = np.mean(feats[0, [0, 2, 4, 6]])
                                b_pwr = np.mean(feats[0, [1, 3, 5, 7]])

                                if current_state == 3:
                                    baseline_alpha_history.append(a_pwr)
                                    self.base_alpha.value = np.mean(baseline_alpha_history)

                                elif current_state == 4 and self.base_alpha.value > 0:
                                    if a_pwr >= (self.base_alpha.value * 0.85):
                                        current_rhythm_flag = 1
                            except:
                                pass

                    writer.writerow([timestamps[i]] + s_data + [self.marker, artifact_flag, current_rhythm_flag])

            self.screen.fill((18, 22, 28))
            draw_multiline_text(self.screen, self.display_text, self.font, (240, 244, 248), self.width // 2,
                                self.height // 2)
            pygame.display.flip()
            self.clock.tick(60)

        csv_file.close()
        pygame.quit()

# TKINTER EXISTING USER REGISTRATION
def run_pre_registration():
    import tkinter as tk
    from tkinter import messagebox
    root = tk.Tk()
    root.title("Operator Login & Protocol Settings")
    root.geometry("400x300")

    tk.Label(root, text="3-Digit Subject ID (e.g., 001):").pack(pady=5)
    entry_id = tk.Entry(root, font=("Arial", 14), justify="center")
    entry_id.pack(pady=5)

    tk.Label(root, text="Number of Task Trials (e.g., 3):").pack(pady=5)
    entry_trials = tk.Entry(root, font=("Arial", 14), justify="center")
    entry_trials.insert(0, "1")
    entry_trials.pack(pady=5)

    tk.Label(root, text="Select Cognitive Task:").pack(pady=5)
    task_var = tk.StringVar(value="math")
    tk.Radiobutton(root, text="Complex Mental Math", variable=task_var, value="math").pack()
    tk.Radiobutton(root, text="Missing Letter (Cloze)", variable=task_var, value="words").pack()

    save_dir_path = []
    num_trials_val = []
    selected_task = []
    target_mac_val = []
    base_data_dir_val = []

    def submit_data():
        reg_id = entry_id.get().strip()
        trials = entry_trials.get().strip()

        if not reg_id or len(reg_id) != 3 or not reg_id.isdigit():
            messagebox.showerror("Error", "Please enter a valid 3-digit Registration ID.")
            return

        if not trials.isdigit() or int(trials) < 1:
            messagebox.showerror("Error", "Please enter a valid number of trials (minimum 1).")
            return

        folder_path = os.path.join(BASE_DATA_DIR, f"ID_{reg_id}")
        online_path = os.path.join(folder_path, "online")
        info_path = os.path.join(folder_path, "personal_info.txt")

        if not os.path.exists(folder_path):
            messagebox.showerror("Error", f"Profile ID_{reg_id} does not exist. Please register offline first.")
            return

        if not os.path.exists(info_path):
            messagebox.showerror("Error",
                                 f"'personal_info.txt' not found for ID_{reg_id}. Cannot extract hardware metadata.")
            return

        # Extract Hardware MAC and Base Directory from personal_info.txt
        extracted_mac = "UNKNOWN"
        extracted_base = BASE_DATA_DIR

        with open(info_path, "r") as f:
            for line in f:
                if line.startswith("Hardware MAC:"):
                    extracted_mac = line.split(":", 1)[1].strip()
                elif line.startswith("Base Directory:"):
                    extracted_base = line.split(":", 1)[1].strip()

        os.makedirs(online_path, exist_ok=True)

        save_dir_path.append(online_path)
        num_trials_val.append(int(trials))
        selected_task.append(task_var.get())
        target_mac_val.append(extracted_mac)
        base_data_dir_val.append(extracted_base)
        root.destroy()

    tk.Button(root, text="Load Profile & Launch System", command=submit_data, bg="#3498db", fg="white",
              font=("Arial", 10, "bold")).pack(pady=15)
    root.mainloop()

    if not save_dir_path:
        sys.exit(0)

    return save_dir_path[0], num_trials_val[0], selected_task[0], target_mac_val[0], base_data_dir_val[0]

# POST-PROCESSING: AUTOMATED MNE ICA & MODEL TRAINING
def calculate_itr(n_classes, accuracy, trial_duration_sec):
    """Calculates Information Transfer Rate in bits per minute."""
    if accuracy == 1.0:
        b = np.log2(n_classes)
    elif accuracy <= (1.0 / n_classes):
        b = 0.0
    else:
        b = (accuracy * np.log2(accuracy) +
             (1.0 - accuracy) * np.log2((1.0 - accuracy) / (n_classes - 1.0)) +
             np.log2(n_classes))
    return max(0.0, b * (60.0 / trial_duration_sec))

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Handles evaluation across folds and testing data."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accuracies = []
    for train_idx, val_idx in skf.split(X_train, y_train):
        model.fit(X_train[train_idx], y_train[train_idx])
        preds = model.predict(X_train[val_idx])
        cv_accuracies.append(accuracy_score(y_train[val_idx], preds))

    cv_mean = np.mean(cv_accuracies)
    cv_std = np.std(cv_accuracies)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]

    metrics = {
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'acc': accuracy_score(y_test, y_pred),
        'bal_acc': balanced_accuracy_score(y_test, y_pred),
        'kappa': cohen_kappa_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_probs),
        'cm': confusion_matrix(y_test, y_pred),
        'itr': calculate_itr(n_classes=2, accuracy=accuracy_score(y_test, y_pred), trial_duration_sec=2.0)
    }
    return metrics

def generate_report_text(model_name, features, metrics, target_mac, base_dir):
    """Formats the performance text report for a specific model."""
    return f"""==================================================
BCI CLASSIFIER LIVE CALIBRATION PERFORMANCE: {model_name}
Hardware Setup -> MAC: {target_mac} | Data Root: {base_dir}
Utilized Features ({len(features)}) : {", ".join(features)}

1. SYSTEM ROBUSTNESS (5-Fold CV)
Mean CV Accuracy       : {metrics['cv_mean'] * 100:.2f}%
CV Accuracy Spread     : ± {metrics['cv_std'] * 100:.2f}%

2. HOLDOUT PERFORMANCE METRICS
Raw Accuracy           : {metrics['acc'] * 100:.2f}%
Balanced Accuracy      : {metrics['bal_acc'] * 100:.2f}%
Cohen's Kappa          : {metrics['kappa']:.4f}
ROC-AUC Score          : {metrics['auc']:.4f}
F1-Score               : {metrics['f1']:.4f}
Information Transfer   : {metrics['itr']:.2f} bits/min

3. CONFUSION MATRIX
                Pred Relax | Pred Focus
Actual Relax :    {metrics['cm'][0, 0]:<8} |  {metrics['cm'][0, 1]:<8}
Actual Focus :    {metrics['cm'][1, 0]:<8} |  {metrics['cm'][1, 1]:<8}
==================================================
"""

def apply_mne_pipeline_offline(df, fs):
    eeg_data = df[EEG_CHANNELS].values.T
    info = mne.create_info(ch_names=EEG_CHANNELS, sfreq=fs, ch_types='eeg')
    raw = mne.io.RawArray(eeg_data, info)

    raw.notch_filter(freqs=50.0)
    raw.filter(l_freq=1.0, h_freq=40.0)

    try:
        ica = mne.preprocessing.ICA(n_components=3, random_state=42, max_iter='auto')
        ica.fit(raw)
        eog_indices, _ = ica.find_bads_eog(raw, ch_name='AF7')
        ica.exclude = eog_indices
        raw = ica.apply(raw)
    except Exception:
        pass
    return raw.get_data().T

def extract_frequency_features_offline(epoch_data, fs):
    features = {}
    for ch_idx, ch_name in enumerate(EEG_CHANNELS):
        channel_signal = epoch_data[:, ch_idx]
        freqs, psd = signal.welch(channel_signal, fs=fs, nperseg=fs * 2)

        alpha_idx = np.logical_and(freqs >= 8, freqs <= 12)
        alpha_normalized = np.trapz(psd[alpha_idx], freqs[alpha_idx]) / (12 - 8)

        beta_idx = np.logical_and(freqs >= 13, freqs <= 30)
        beta_normalized = np.trapz(psd[beta_idx], freqs[beta_idx]) / (30 - 13)

        ratio = beta_normalized / (alpha_normalized + 1e-9)
        features[f"{ch_name}_Beta_Alpha_Ratio"] = ratio
    return features

def train_live_model(csv_path, target_dir, task_type, target_mac, base_dir):
    print("\n" + "=" * 60)
    print("🧠 PHASE 3: CO-ADAPTIVE LEARNING (OFFLINE + HISTORICAL ONLINE + CURRENT DATA)")
    print("=" * 60)

    session_features = []

    # 1. Locate Offline and Online Files
    parent_dir = os.path.dirname(target_dir)
    offline_dir = os.path.join(parent_dir, "offline")

    # Start with the newly recorded Online file
    files_to_process = [csv_path]

    # Collect relevant raw CSV files from the Offline directory
    if os.path.exists(offline_dir):
        offline_files = glob.glob(os.path.join(offline_dir, "*.csv"))
        for f in offline_files:
            # We specifically target the selected task to match this script's paradigm
            if task_type in os.path.basename(f).lower() and "extracted_offline_features" not in os.path.basename(f):
                files_to_process.append(f)

    # Collect previous relevant raw CSV files from the Online directory
    if os.path.exists(target_dir):
        online_files = glob.glob(os.path.join(target_dir, "*.csv"))
        for f in online_files:
            if task_type in os.path.basename(f).lower() and os.path.abspath(f) != os.path.abspath(csv_path):
                files_to_process.append(f)

    print(f"🔍 Found {len(files_to_process)} total '{task_type}' session files for co-adaptive training.")

    # 2. Process all collected files
    for file in files_to_process:
        filename = os.path.basename(file)
        print(f"  -> Processing: {filename}")
        try:
            df = pd.read_csv(file)
        except Exception as e:
            print(f"     ❌ Error reading file: {e}")
            continue

        # Clean flags
        if 'Artifact_Flag' in df.columns:
            df = df[df['Artifact_Flag'] == 0]
        if 'Rhythm_Flag' in df.columns:
            df = df[df['Rhythm_Flag'] == 0]

        # Apply MNE ICA
        print("     ⚙️ Applying MNE Pipeline & ICA Artifact Scrubbing...")
        clean_eeg = apply_mne_pipeline_offline(df, SAMPLING_RATE)
        df.loc[:, EEG_CHANNELS] = clean_eeg

        # Slicing
        df = df[df['Marker'].isin([3, 4])].copy()
        if df.empty:
            print("     ⚠️ No valid target marker data (3 or 4) found after cleaning. Skipping.")
            continue

        # Epoching
        print("     🧩 Extracting Beta/Alpha Ratio Features...")
        samples_per_epoch = int(SAMPLING_RATE * EPOCH_LENGTH_SEC)

        for marker_val in [3, 4]:
            state_data = df[df['Marker'] == marker_val]
            num_epochs = len(state_data) // samples_per_epoch

            for i in range(num_epochs):
                start_idx = i * samples_per_epoch
                end_idx = start_idx + samples_per_epoch
                epoch_slice = state_data[EEG_CHANNELS].iloc[start_idx:end_idx].values

                features = extract_frequency_features_offline(epoch_slice, SAMPLING_RATE)
                features['Target_Class'] = 0 if marker_val == 3 else 1
                session_features.append(features)

    unpruned_df = pd.DataFrame(session_features)
    X = unpruned_df.drop('Target_Class', axis=1)
    y = unpruned_df['Target_Class'].values

    if len(np.unique(y)) < 2:
        print("❌ Error: Only one class found in combined data. Cannot train model.")
        return

    # Feature Selection on the combined dataset
    print("\n📈 Selecting top 2 optimal features via ANOVA on pooled data...")
    num_features_to_select = min(2, X.shape[1])
    selector = SelectKBest(score_func=f_classif, k=num_features_to_select)
    selector.fit(X, y)

    top_feature_indices = selector.get_support(indices=True)
    top_feature_names = X.columns[top_feature_indices].tolist()
    X_selected = X[top_feature_names].values

    # Train and Evaluate LDA
    print("🛡️ Evaluating Regularized LDA Classification Model...")
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.20, stratify=y, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lda_model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    metrics = evaluate_model(lda_model, X_train_scaled, y_train, X_test_scaled, y_test)

    # Generate and Save Report inside the /online/ directory
    report_text = generate_report_text("Regularized LDA (Co-Adaptive)", top_feature_names, metrics, target_mac,
                                       base_dir)
    report_path = os.path.join(target_dir, f"report_LIVE_CALIBRATION_{task_type}_LDA_metrics.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n📄 Co-adaptive performance report saved to: {report_path}")

    # Retrain on ALL data for max live production robustness
    print("🚀 Retraining model on 100% of pooled calibration data for live production...")
    final_scaler = StandardScaler()
    X_all_scaled = final_scaler.fit_transform(X_selected)
    final_model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    final_model.fit(X_all_scaled, y)

    # Serialize & Save inside the /online/ directory
    model_path = os.path.join(target_dir, f"bci_live_production_{task_type}_lda_online.joblib")
    print(f"💾 Saving Co-Adaptive Production Model to: {model_path}")
    joblib.dump({'scaler': final_scaler, 'model': final_model, 'feature_names': top_feature_names}, model_path)
    print("✅ Full calibration pipeline complete and ready for live streaming.")

# MULTIPROCESSING LAUNCHER
def run_operator_view(start_event, exit_event, sys_state, sys_time, base_alpha, sys_trial, sys_total_trials, task_type):
    OperatorDiagnostics(start_event, exit_event, sys_state, sys_time, base_alpha, sys_trial, sys_total_trials,
                        task_type).run()

if __name__ == "__main__":
    multiprocessing.freeze_support()

    # Phase 1: Registration
    session_folder, total_trials, task_type, target_mac, verified_base_dir = run_pre_registration()

    print(f"\n📡 Audit Trail - Hardware MAC Verified: {target_mac}")
    print(f"📁 Audit Trail - Base Directory Verified: {verified_base_dir}")

    sys_start_event = multiprocessing.Event()
    sys_exit_event = multiprocessing.Event()
    sys_state_val = multiprocessing.Value('i', 0)
    sys_time_val = multiprocessing.Value('i', 0)
    sys_base_alpha = multiprocessing.Value('d', -1.0)
    sys_trial_val = multiprocessing.Value('i', 1)
    sys_total_trials_val = multiprocessing.Value('i', total_trials)

    stream_process = multiprocessing.Process(target=start_stream, args=(target_mac,))
    stream_process.daemon = True
    stream_process.start()
    time.sleep(3.0)

    operator_process = multiprocessing.Process(
        target=run_operator_view,
        args=(sys_start_event, sys_exit_event, sys_state_val, sys_time_val, sys_base_alpha, sys_trial_val,
              sys_total_trials_val, task_type)
    )
    operator_process.start()
    time.sleep(1.0)

    # Phase 2: Protocol Execution
    protocol = CalibrationProtocol(session_folder, sys_start_event, sys_exit_event, sys_state_val, sys_time_val,
                                   sys_base_alpha, sys_trial_val, sys_total_trials_val, task_type)
    protocol.run()

    # Shut down collection processes securely to free memory
    operator_process.terminate()
    stream_process.terminate()

    # Phase 3: Automated Post-Processing & Model Training
    saved_csv_path = protocol.csv_path
    if os.path.exists(saved_csv_path):
        train_live_model(saved_csv_path, session_folder, task_type, target_mac, verified_base_dir)
    else:
        print(f"❌ Error: Could not locate the saved CSV file at {saved_csv_path}")