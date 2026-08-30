"""
Muse 2 LSL Streamer with Automated Pre-Flight Hardware Checks
-------------------------------------------------------------
Connects to the Muse via hardcoded MAC, broadcasts to LSL, and runs
a live, auto-refreshing terminal dashboard to evaluate electrode
contact quality using calibrated dry-sensor amplitude thresholds.
Auto-launches the viewer once the signal stabilizes.
"""

import os
import asyncio
import sys
import time
import threading
import multiprocessing
import numpy as np
from pylsl import StreamInlet, resolve_byprop
from muselsl import stream, list_muses, view

# ---------------------------------------------------------
# ASYNC INFRASTRUCTURE SAFETY NET
# Because we isolate the hardware connection into a background
# process, it may wake up without an initialized event loop.
# This block hooks into an existing loop or builds a new one
# so the Bluetooth Low Energy (BLE) adapter doesn't crash.
# ---------------------------------------------------------
try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Hardcoded to bypass the slow Bluetooth scanning phase
TARGET_MAC_ADDRESS = "00:55:DA:B7:51:03"

def start_stream():
    """Handles the Bluetooth BLE connection and data broadcasting."""
    if TARGET_MAC_ADDRESS:
        print(f"🔗 Attempting direct connection to Muse with MAC: {TARGET_MAC_ADDRESS}...")
        stream(TARGET_MAC_ADDRESS)
    else:
        print("🔍 Scanning for Muse devices (make sure they are ON)...")
        muses = list_muses()
        if not muses:
            print("❌ No Muse found. Ensure Bluetooth is ON and headset is in pairing mode.")
            sys.exit(1)
        print("\n📋 Available Headsets Found:")
        for m in muses:
            print(f"   Name: {m['name']} | MAC Address: {m['address']}")
        sys.exit(0)

# Calibrated for dry-sensor realities
MAX_NOISE_THRESHOLD_UV = 350.0
MIN_FLATLINE_THRESHOLD_UV = 2.0

def run_preflight_check():
    """
    Evaluates the live LSL stream for signal quality using a continuous loop.
    Provides live feedback while the operator adjusts the headset.
    """
    print("\n⏳ Searching for local LSL Network Stream...")

    try:
        streams = resolve_byprop('type', 'EEG', timeout=15.0)
        if not streams:
            print("❌ Error: Could not find EEG stream on the network.")
            return False
        inlet = StreamInlet(streams[0])
    except Exception as e:
        print(f"❌ Network Error: {e}")
        return False

    print("\n🛠️ LIVE HARDWARE DIAGNOSTIC INITIATED")
    print("Adjust the headset until all channels are stable. (Checking up to 30 times)\n")

    # Flush stale connection data
    inlet.pull_chunk(timeout=1.0, max_samples=2000)
    time.sleep(1.0)

    channels = ["TP9 (Left Ear)", "AF7 (Left Forehead)", "AF8 (Right Forehead)", "TP10 (Right Ear)"]
    max_attempts = 30

    for attempt in range(max_attempts):
        time.sleep(1.0)  # Wait for 1 second of fresh data
        samples, _ = inlet.pull_chunk(timeout=1.0, max_samples=1000)

        if not samples:
            continue

        data = np.array(samples)
        all_clear = True

        sys.stdout.write(f"\r--- Check {attempt + 1}/{max_attempts} ---\n")

        for i in range(4):
            chan_data = data[:, i]
            peak_to_peak = np.ptp(chan_data)

            if peak_to_peak < MIN_FLATLINE_THRESHOLD_UV:
                status = "❌ FLATLINE (Check Connection)"
                all_clear = False
            elif peak_to_peak > MAX_NOISE_THRESHOLD_UV:
                status = "⚠️ NOISY   (Adjust Electrode) "
                all_clear = False
            else:
                status = "✅ GOOD                      "

            sys.stdout.write(f"{channels[i]:<22}: {status} (P2P: {peak_to_peak:>5.1f}µV)\n")

        if all_clear:
            print("-" * 50)
            print("✅ ALL CHANNELS STABLE. Headset is ready for data collection.")
            return True

        sys.stdout.write("\033[5A")

    sys.stdout.write("\033[5B")
    print("-" * 50)
    print("⚠️ Maximum attempts reached. Signal is still noisy.")
    return False

def force_window_to_front():
    time.sleep(2.0)  # Wait for muselsl to initialize its graphical backend
    if os.name == 'nt':
        import ctypes
        user32 = ctypes.windll.user32

        # 1. ALT Key Trick to bypass Windows Foreground Lock Timeout
        user32.keybd_event(0x12, 0, 0, 0)
        user32.keybd_event(0x12, 0, 2, 0)

        # 2. Iterate through all windows owned by this Python process ID
        pid = os.getpid()

        def enum_proc(hwnd, lParam):
            pid_out = ctypes.c_uint(0)
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid_out))

            # Check if it belongs to our script
            if pid_out.value == pid:
                # Filter out invisible dummy windows and windows with no title
                if user32.IsWindowVisible(hwnd) and user32.GetWindowTextLengthW(hwnd) > 0:
                    # Apply the Punch and Release override
                    user32.ShowWindow(hwnd, 9)
                    user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 3)
                    user32.SetForegroundWindow(hwnd)
                    user32.SetWindowPos(hwnd, -2, 0, 0, 0, 0, 3)
            return True

        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
        user32.EnumWindows(WNDENUMPROC(enum_proc), 0)

def start_viewer():
    """Acts as an LSL receiver and visualizer."""
    print("📊 Launching real-time EEG viewer...")
    threading.Thread(target=force_window_to_front, daemon=True).start()
    view(window=5, scale=150, version=2)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        if TARGET_MAC_ADDRESS:
            stream_process = multiprocessing.Process(target=start_stream)
            stream_process.daemon = True
            stream_process.start()

            time.sleep(3.0)

            diagnostic_passed = run_preflight_check()

            if not diagnostic_passed:
                print("🛑 WARNING: Hardware check failed.")
                print("You can view the raw signals to debug. Opening viewer...")
                time.sleep(2)

            start_viewer()
            stream_process.terminate()
            print("\n🛑 Viewer closed. Stream stopped safely.")
        else:
            start_stream()

    except KeyboardInterrupt:
        print("\n🛑 System forcefully stopped.")
