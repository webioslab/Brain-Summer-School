"""
Offline BCI Calibration & Data Collection Paradigm
Task: Unified Cognitive Stimulus (Math / Words)
--------------------------------------------------
Phase 1: Pre-Registration UI (New vs Existing Subject + Task Selection)
Phase 2: Dual-Screen Execution (Multiprocessing IPC)
    - Screen 0 (Participant): NOFRAME visual protocol and baselining.
    - Screen 1 (Operator): NOFRAME live diagnostics & synced timers.
    - Automatically routes saved data to the /offline/ subfolder.
"""

import os
import sys
import time
import csv
import random
import threading
import multiprocessing
import numpy as np
import pygame
import winsound
from collections import deque
from scipy import signal
from pylsl import StreamInlet, resolve_byprop
from muselsl import stream

# CONFIGURATION & PROTOCOL SETTINGS
TARGET_MAC_ADDRESS = "00:55:DA:B7:51:03"
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

# SHARED PIPELINE FUNCTIONS
def apply_causal_filters(eeg_data, fs):
    b_notch, a_notch = signal.iirnotch(w0=50.0, Q=30.0, fs=fs)
    filtered_data = signal.lfilter(b_notch, a_notch, eeg_data, axis=0)
    b_band, a_band = signal.butter(N=4, Wn=[1.0, 40.0], btype='bandpass', fs=fs)
    filtered_data = signal.lfilter(b_band, a_band, filtered_data, axis=0)
    return filtered_data

def extract_live_features(epoch_data, fs):
    features = []
    bands = {'Alpha': (8, 12), 'Beta': (13, 30)}
    for ch_idx in range(4):
        freqs, psd = signal.welch(epoch_data[:, ch_idx], fs=fs, nperseg=fs * 2)
        for low, high in bands.values():
            idx = np.logical_and(freqs >= low, freqs <= high)
            features.append(np.trapz(psd[idx], freqs[idx]))
    return np.array(features).reshape(1, -1)

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

def generate_cloze_prompt():
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
def start_stream():
    if TARGET_MAC_ADDRESS:
        stream(TARGET_MAC_ADDRESS)
    else:
        sys.exit(0)

# PROCESS 2: OPERATOR VIEW
class OperatorDiagnostics:
    def __init__(self, start_event, exit_event, sys_state, sys_time, base_alpha, task_type):
        self.start_event = start_event
        self.exit_event = exit_event
        self.sys_state = sys_state
        self.sys_time = sys_time
        self.base_alpha = base_alpha
        self.task_type = task_type

        pygame.display.init()
        num_displays = pygame.display.get_num_displays()
        pygame.display.quit()

        # Force Operator View to primary laptop screen
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
        self.screen.blit(self.font_title.render("SEARCHING FOR HEADSET STREAM (Up to 25s)...", True, self.TEXT_COLOR), (50, 50))
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
                    self.alpha_pwr = np.mean(feats[0, [1, 4, 7, 10]]) / 4.0
                    self.beta_pwr = np.mean(feats[0, [2, 5, 8, 11]]) / 17.0

                    c_state = self.sys_state.value
                    if c_state == 4 and self.base_alpha.value > 0:
                        if self.alpha_pwr >= (self.base_alpha.value * 0.85):
                            self.rhythm_flag_detected = True
                except:
                    pass

            self.screen.fill(self.BG_COLOR)
            g_x, g_w, g_h, g_y = int(self.width * 0.05), int(self.width * 0.62), int(self.height * 0.8), int(self.height * 0.1)
            pygame.draw.rect(self.screen, (20, 20, 20), (g_x, g_y, g_w, g_h))
            pygame.draw.rect(self.screen, (100, 100, 100), (g_x, g_y, g_w, g_h), 2)

            for i in range(1, 4):
                pygame.draw.line(self.screen, self.GRID_COLOR, (g_x, g_y + (g_h // 4) * i), (g_x + g_w, g_y + (g_h // 4) * i))

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
                self.screen.blit(self.font_large.render(EEG_CHANNELS[ch_idx], True, status_color), (g_x + 10, g_y + (ch_idx * ch_h) + 10))

            state_idx = self.sys_state.value
            state_text = STATE_MAP.get(state_idx, "UNKNOWN")
            state_color = self.FLAG_WARN if state_idx == 0 else (52, 152, 219) if state_idx in [1, 2, 3] else self.FLAG_OK

            p_y = int(self.height * 0.1)
            self.screen.blit(self.font_large.render("PROTOCOL PHASE", True, self.TEXT_COLOR), (self.panel_x, p_y))
            self.screen.blit(self.font_title.render(state_text, True, state_color), (self.panel_x, p_y + 25))

            time_rem = self.sys_time.value
            time_str = f"TIME REMAINING: {time_rem}s" if state_idx in [1, 2, 3, 4] else "TIME REMAINING: --"
            self.screen.blit(self.font_large.render(time_str, True, (241, 196, 15)), (self.panel_x, p_y + 65))

            y_offset = p_y + 120
            art_color = (231, 76, 60) if self.artifact_detected else (40, 40, 40)
            pygame.draw.circle(self.screen, art_color, (self.panel_x + 20, y_offset + 10), 20)
            self.screen.blit(self.font_large.render("ARTIFACT WARNING", True, self.TEXT_COLOR), (self.panel_x + 50, y_offset))

            rhy_color = (241, 196, 15) if self.rhythm_flag_detected else (40, 40, 40)
            pygame.draw.circle(self.screen, rhy_color, (self.panel_x + 20, y_offset + 50), 20)
            self.screen.blit(self.font_large.render("RHYTHM ERD MISMATCH", True, self.TEXT_COLOR), (self.panel_x + 50, y_offset + 40))

            y_offset += 100
            self.screen.blit(self.font_large.render("COGNITIVE DENSITY (BETA/ALPHA)", True, self.TEXT_COLOR), (self.panel_x, y_offset))
            total_pwr = self.alpha_pwr + self.beta_pwr + 0.001

            pygame.draw.rect(self.screen, self.GRID_COLOR, (self.panel_x, y_offset + 30, self.btn_w, 15))
            pygame.draw.rect(self.screen, (41, 128, 185), (self.panel_x, y_offset + 30, int(self.btn_w * (self.alpha_pwr / total_pwr)), 15))
            self.screen.blit(self.font_small.render("Alpha (Relaxation)", True, self.TEXT_COLOR), (self.panel_x, y_offset + 48))

            pygame.draw.rect(self.screen, self.GRID_COLOR, (self.panel_x, y_offset + 70, self.btn_w, 15))
            pygame.draw.rect(self.screen, (241, 196, 15), (self.panel_x, y_offset + 70, int(self.btn_w * (self.beta_pwr / total_pwr)), 15))

            beta_label = "Beta (Mental Math Focus)" if self.task_type == "math" else "Beta (Missing Letter Focus)"
            self.screen.blit(self.font_small.render(beta_label, True, self.TEXT_COLOR), (self.panel_x, y_offset + 88))

            has_started = self.start_event.is_set()
            start_color = (80, 80, 80) if has_started else (46, 204, 113)
            pygame.draw.rect(self.screen, start_color, self.btn_start_rect, border_radius=8)
            pygame.draw.rect(self.screen, self.TEXT_COLOR, self.btn_start_rect, width=2, border_radius=8)
            s_surf = self.font_large.render("PROTOCOL RUNNING" if has_started else "START PROTOCOL", True, self.TEXT_COLOR)
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
    def __init__(self, save_dir, start_event, exit_event, sys_state, sys_time, base_alpha, task_type):
        self.start_event = start_event
        self.exit_event = exit_event
        self.sys_state = sys_state
        self.sys_time = sys_time
        self.base_alpha = base_alpha
        self.task_type = task_type

        pygame.display.init()
        num_displays = pygame.display.get_num_displays()
        pygame.display.quit()

        if num_displays == 1:
            os.environ['SDL_VIDEO_WINDOW_POS'] = "950,50"
        elif 'SDL_VIDEO_WINDOW_POS' in os.environ:
            del os.environ['SDL_VIDEO_WINDOW_POS']

        pygame.init()
        if num_displays > 1:
            # Bind to the extended display explicitly
            self.screen = pygame.display.set_mode((0, 0), pygame.NOFRAME, display=1)
        else:
            self.screen = pygame.display.set_mode((900, 650))

        self.width, self.height = self.screen.get_size()
        pygame.mouse.set_visible(False)
        self.clock = pygame.time.Clock()

        pygame.event.pump()
        if os.name == 'nt':
            import ctypes
            hwnd = pygame.display.get_wm_info()["window"]
            user32 = ctypes.windll.user32
            user32.ShowWindow(hwnd, 9)
            user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 3)
            user32.SetForegroundWindow(hwnd)
            user32.SetWindowPos(hwnd, -2, 0, 0, 0, 0, 3)

        self.font = pygame.font.SysFont("Arial", 48, bold=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.csv_path = os.path.join(save_dir, f"Offline_data_{self.task_type}_{timestamp}.csv")

        self.sys_state.value = 0
        self.sys_time.value = 0
        self.marker = MARKER_IDLE
        self.display_text = "WAITING FOR OPERATOR\n\nPlease keep your head still and relax."
        self.task_prompt = ""
        self.phase_start_time = 0
        self.last_task_time = 0

    def trigger_beep(self, frequency, duration):
        """Native windows beep mapped to a background thread to prevent UI freezing."""
        threading.Thread(target=winsound.Beep, args=(frequency, duration), daemon=True).start()

    def trigger_double_beep(self):
        """Distinct auditory cue to unambiguously signal the end of the eyes-closed phase."""
        def _beeps():
            winsound.Beep(1200, 400)
            time.sleep(0.1)
            winsound.Beep(1200, 400)
        threading.Thread(target=_beeps, daemon=True).start()

    def run(self):
        self.screen.fill((18, 22, 28))
        draw_multiline_text(self.screen, "CONNECTING TO HEADSET...\n\nSearching for LSL Stream (Up to 25s).", self.font, (240, 244, 248), self.width // 2, self.height // 2)
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
                    # The double beep fires exactly here
                    self.trigger_double_beep()

            elif current_state == 3:
                self.sys_time.value = int(DUR_RELAX - elapsed)
                self.display_text = f"RESTING PHASE\n\nKeep your eyes open and relax."
                if elapsed > DUR_RELAX:
                    self.sys_state.value = 4
                    self.marker = MARKER_TASK
                    self.phase_start_time = current_time
                    self.last_task_time = 0
                    self.trigger_beep(1500, 800)

            elif current_state == 4:
                self.sys_time.value = int(DUR_TASK - elapsed)

                # Use dynamic refresh rate based on selected task
                task_refresh_limit = TASK_REFRESH_MATH if self.task_type == "math" else TASK_REFRESH_WORDS

                if current_time - self.last_task_time > task_refresh_limit:
                    if self.task_type == "math":
                        self.task_prompt = generate_math_problem()
                    else:
                        self.task_prompt = generate_cloze_prompt()
                    self.last_task_time = current_time

                if self.task_type == "math":
                    self.display_text = f"ACTIVE FOCUS\n\nSolve Mentally:\n{self.task_prompt}"
                else:
                    self.display_text = f"ACTIVE FOCUS\n\nMentally identify the missing letter:\n\n{self.task_prompt}"

                if elapsed > DUR_TASK:
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
                                a_pwr = np.mean(feats[0, [1, 4, 7, 10]]) / 4.0
                                b_pwr = np.mean(feats[0, [2, 5, 8, 11]]) / 17.0

                                # Phase 3: Learn Baseline
                                if current_state == 3:
                                    baseline_alpha_history.append(a_pwr)
                                    self.base_alpha.value = np.mean(baseline_alpha_history)

                                # Phase 4: Enforce BASELINE logic
                                elif current_state == 4 and self.base_alpha.value > 0:
                                    if a_pwr >= (self.base_alpha.value * 0.85):
                                        current_rhythm_flag = 1
                            except:
                                pass

                    writer.writerow([timestamps[i]] + s_data + [self.marker, artifact_flag, current_rhythm_flag])

            self.screen.fill((18, 22, 28))
            draw_multiline_text(self.screen, self.display_text, self.font, (240, 244, 248), self.width // 2, self.height // 2)
            pygame.display.flip()
            self.clock.tick(60)

        csv_file.close()
        pygame.quit()

# TKINTER PRE-REGISTRATION FORM
def run_pre_registration():
    import tkinter as tk
    from tkinter import messagebox
    root = tk.Tk()
    root.title("Participant Registration")
    root.geometry("450x620")
    mode_var = tk.StringVar(value="new")

    def toggle_mode():
        state = 'disabled' if mode_var.get() == "existing" else 'normal'
        if mode_var.get() == "existing":
            entry_name.config(state='disabled')
        else:
            entry_name.config(state='disabled' if anon_var.get() else 'normal')

        entry_age.config(state=state)
        for rb in healthy_radios + handed_radios + anon_checkboxes + gender_radios:
            rb.config(state=state)

    tk.Radiobutton(root, text="New Subject", variable=mode_var, value="new", command=toggle_mode).pack(pady=5)
    tk.Radiobutton(root, text="Existing Subject", variable=mode_var, value="existing", command=toggle_mode).pack(pady=5)

    tk.Label(root, text="3-Digit Reg ID (e.g., 001):").pack(pady=2)
    entry_id = tk.Entry(root)
    entry_id.pack(pady=2)

    tk.Label(root, text="Name:").pack(pady=2)
    frame_name = tk.Frame(root)
    frame_name.pack(pady=2)

    entry_name = tk.Entry(frame_name)
    entry_name.pack(side=tk.LEFT, padx=5)

    anon_var = tk.BooleanVar(value=False)

    def toggle_anon():
        if anon_var.get():
            entry_name.delete(0, tk.END)
            entry_name.insert(0, "Anonymous")
            entry_name.config(state='disabled')
        else:
            entry_name.config(state='normal')
            entry_name.delete(0, tk.END)

    cb_anon = tk.Checkbutton(frame_name, text="Anonymous", variable=anon_var, command=toggle_anon)
    cb_anon.pack(side=tk.LEFT)
    anon_checkboxes = [cb_anon]

    tk.Label(root, text="Age:").pack(pady=2)
    entry_age = tk.Entry(root)
    entry_age.pack(pady=2)

    tk.Label(root, text="Gender:").pack(pady=2)
    frame_gender = tk.Frame(root)
    frame_gender.pack(pady=2)
    gender_var = tk.StringVar(value="Male")
    rb_gender_m = tk.Radiobutton(frame_gender, text="Male", variable=gender_var, value="Male")
    rb_gender_m.pack(side=tk.LEFT)
    rb_gender_f = tk.Radiobutton(frame_gender, text="Female", variable=gender_var, value="Female")
    rb_gender_f.pack(side=tk.LEFT)
    gender_radios = [rb_gender_m, rb_gender_f]

    healthy_var = tk.StringVar(value="Yes")
    handed_var = tk.StringVar(value="Yes")

    tk.Label(root, text="Are you healthy controlled?").pack(pady=2)
    frame_healthy = tk.Frame(root)
    frame_healthy.pack(pady=2)
    rb_healthy_yes = tk.Radiobutton(frame_healthy, text="Yes", variable=healthy_var, value="Yes")
    rb_healthy_yes.pack(side=tk.LEFT)
    rb_healthy_no = tk.Radiobutton(frame_healthy, text="No", variable=healthy_var, value="No")
    rb_healthy_no.pack(side=tk.LEFT)

    tk.Label(root, text="Are you right handed?").pack(pady=2)
    frame_handed = tk.Frame(root)
    frame_handed.pack(pady=2)
    rb_handed_yes = tk.Radiobutton(frame_handed, text="Yes", variable=handed_var, value="Yes")
    rb_handed_yes.pack(side=tk.LEFT)
    rb_handed_no = tk.Radiobutton(frame_handed, text="No", variable=handed_var, value="No")
    rb_handed_no.pack(side=tk.LEFT)

    healthy_radios = [rb_healthy_yes, rb_healthy_no]
    handed_radios = [rb_handed_yes, rb_handed_no]

    # Add Task Selection
    tk.Label(root, text="Select Cognitive Task:").pack(pady=(15, 2))
    task_var = tk.StringVar(value="math")
    tk.Radiobutton(root, text="Complex Mental Math", variable=task_var, value="math").pack()
    tk.Radiobutton(root, text="Missing Letter (Cloze)", variable=task_var, value="words").pack()

    save_dir_path = []
    selected_task = []

    def submit_data():
        reg_id = entry_id.get().strip()
        if not reg_id or len(reg_id) != 3 or not reg_id.isdigit():
            messagebox.showerror("Error", "Please enter a valid 3-digit Registration ID.")
            return
        folder_path = os.path.join(BASE_DATA_DIR, f"ID_{reg_id}")
        if mode_var.get() == "existing" and not os.path.exists(folder_path):
            messagebox.showerror("Error", f"Folder ID_{reg_id} does not exist. Select 'New Subject'.")
            return
        elif mode_var.get() == "new" and os.path.exists(folder_path):
            messagebox.showerror("Error", f"Folder ID_{reg_id} already exists. Select 'Existing Subject'.")
            return

        if mode_var.get() == "new":
            os.makedirs(folder_path, exist_ok=True)
            with open(os.path.join(folder_path, "personal_info.txt"), "w") as f:
                final_name = "Anonymous" if anon_var.get() or not entry_name.get().strip() else entry_name.get()
                f.write(f"ID: {reg_id}\nName: {final_name}\nAge: {entry_age.get()}\nGender: {gender_var.get()}\n")
                f.write(f"Healthy Controlled: {healthy_var.get()}\n")
                f.write(f"Right Handed: {handed_var.get()}\n")
                f.write(f"Hardware MAC: {TARGET_MAC_ADDRESS}\n")
                f.write(f"Base Directory: {BASE_DATA_DIR}\n")
                f.write(f"Date Created: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        offline_folder_path = os.path.join(folder_path, "offline")
        os.makedirs(offline_folder_path, exist_ok=True)

        save_dir_path.append(offline_folder_path)
        selected_task.append(task_var.get())
        root.destroy()

    tk.Button(root, text="Confirm & Launch System", command=submit_data, bg="#2ecc71").pack(pady=20)
    root.mainloop()
    if not save_dir_path: sys.exit(0)
    return save_dir_path[0], selected_task[0]

# MULTIPROCESSING LAUNCHER
def run_operator_view(start_event, exit_event, sys_state, sys_time, base_alpha, task_type):
    OperatorDiagnostics(start_event, exit_event, sys_state, sys_time, base_alpha, task_type).run()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    session_folder, task_type = run_pre_registration()

    sys_start_event = multiprocessing.Event()
    sys_exit_event = multiprocessing.Event()
    sys_state_val = multiprocessing.Value('i', 0)
    sys_time_val = multiprocessing.Value('i', 0)
    sys_base_alpha = multiprocessing.Value('d', -1.0)

    stream_process = multiprocessing.Process(target=start_stream)
    stream_process.daemon = True
    stream_process.start()
    time.sleep(3.0)

    operator_process = multiprocessing.Process(
        target=run_operator_view,
        args=(sys_start_event, sys_exit_event, sys_state_val, sys_time_val, sys_base_alpha, task_type)
    )
    operator_process.start()
    time.sleep(1.0)

    CalibrationProtocol(session_folder, sys_start_event, sys_exit_event, sys_state_val, sys_time_val, sys_base_alpha, task_type).run()

    operator_process.terminate()
    stream_process.terminate()