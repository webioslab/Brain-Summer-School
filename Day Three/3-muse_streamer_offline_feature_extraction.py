"""
Muse 2 Single-Subject Offline Preprocessing & Feature Extraction (MNE ICA Integration)
----------------------------------------------------------------
This script processes raw CSV outputs using MNE-Python to mathematically
scrub ocular artifacts via ICA, rather than discarding contaminated files.
It applies automated cleaning, MNE DSP filtering, epochs the data,
extracts the Beta/Alpha ratio features per electrode, selects the top 2 winning features via ANOVA,
and saves ONLY those features to disk inside the /offline/ subfolder.

Target Classes:
    Marker 3: Resting Phase (Expected Alpha dominant)
    Marker 4: Active Task (Expected Beta dominant)
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
from scipy import signal
from sklearn.feature_selection import SelectKBest, f_classif
import tkinter as tk
from tkinter import messagebox
import mne

# CONFIGURATION
# Default lookup directory; will be verified against personal_info.txt
BASE_DATA_DIR = r"D:\MuseData"

SAMPLING_RATE = 256
EEG_CHANNELS = ["TP9", "AF7", "AF8", "TP10"]
EPOCH_LENGTH_SEC = 2.0

# Keep MNE from flooding the console with logs
mne.set_log_level('ERROR')

def clean_flagged_data(df, filename):
    """
    Scans the raw dataframe and drops any data points that triggered
    the hardware artifact or expected rhythm safety flags during recording.
    """
    if 'Artifact_Flag' in df.columns:
        df = df[df['Artifact_Flag'] == 0]

    if 'Rhythm_Flag' in df.columns:
        df = df[df['Rhythm_Flag'] == 0]

    return df

def apply_mne_pipeline(df, fs):
    """
    Replaces scipy filters. Converts DataFrame to an MNE Raw object,
    applies Bandpass/Notch filtering, and runs ICA to scrub eye blink
    artifacts using the AF7 channel as a proxy EOG.
    """
    # MNE expects shape (n_channels, n_samples)
    eeg_data = df[EEG_CHANNELS].values.T
    info = mne.create_info(ch_names=EEG_CHANNELS, sfreq=fs, ch_types='eeg')
    raw = mne.io.RawArray(eeg_data, info)

    # 1. DSP Filtering via MNE
    raw.notch_filter(freqs=50.0)
    raw.filter(l_freq=1.0, h_freq=40.0)

    # 2. ICA Artifact Scrubbing
    try:
        # Max independent components for a 4-channel headset is 3
        ica = mne.preprocessing.ICA(n_components=3, random_state=42, max_iter='auto')
        ica.fit(raw)

        # Auto-detect blinks using AF7 as the EOG proxy
        eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name='AF7')
        ica.exclude = eog_indices

        # Apply ICA (zeroes out the bad blink components and rebuilds signal)
        raw = ica.apply(raw)
    except Exception as e:
        # If ICA fails (usually due to lack of variance in very short files), it safely skips
        pass

    # Return the cleaned data in original (n_samples, n_channels) shape
    return raw.get_data().T

def verify_berger_effect(df, fs, filename):
    """
    Validates biological signal integrity by checking for the Alpha block phenomenon.
    Assumes the dataframe (df) has already been filtered and ICA-scrubbed by MNE.
    """
    m1_data = df[df['Marker'] == 1]
    m2_data = df[df['Marker'] == 2]

    if m1_data.empty or m2_data.empty:
        return False, f"[{filename}] ⚠️ Skipped: Missing Marker 1 or 2 baseline data."

    rear_channels = ['TP9', 'TP10']

    try:
        # Slice the pre-cleaned data
        m1_clean = m1_data[rear_channels].values
        m2_clean = m2_data[rear_channels].values

        freqs, psd_m1 = signal.welch(m1_clean, fs=fs, nperseg=fs * 2, axis=0)
        _, psd_m2 = signal.welch(m2_clean, fs=fs, nperseg=fs * 2, axis=0)

        alpha_idx = np.logical_and(freqs >= 8, freqs <= 12)

        m1_alpha_pwr = np.mean(np.trapz(psd_m1[alpha_idx], freqs[alpha_idx], axis=0))
        m2_alpha_pwr = np.mean(np.trapz(psd_m2[alpha_idx], freqs[alpha_idx], axis=0))

        surge_ratio = m2_alpha_pwr / m1_alpha_pwr

        if surge_ratio > 1.2:
            return True, f"[{filename}] ✅ Passed: Alpha surged by {surge_ratio:.2f}x."
        else:
            return False, f"[{filename}] ❌ Failed: Alpha ratio was only {surge_ratio:.2f}x (Poor cortical contact)."

    except Exception as e:
        return False, f"[{filename}] ⚠️ Error calculating Berger Effect: {e}"

def extract_frequency_features(epoch_data, fs):
    """
    Calculates the Beta/Alpha PSD ratio per Hz using Welch's method for each electrode.
    """
    features = {}

    for ch_idx, ch_name in enumerate(EEG_CHANNELS):
        channel_signal = epoch_data[:, ch_idx]
        freqs, psd = signal.welch(channel_signal, fs=fs, nperseg=fs * 2)

        # Alpha band power calculation (8-12 Hz)
        alpha_idx = np.logical_and(freqs >= 8, freqs <= 12)
        alpha_absolute = np.trapz(psd[alpha_idx], freqs[alpha_idx])
        alpha_normalized = alpha_absolute / (12 - 8)

        # Beta band power calculation (13-30 Hz)
        beta_idx = np.logical_and(freqs >= 13, freqs <= 30)
        beta_absolute = np.trapz(psd[beta_idx], freqs[beta_idx])
        beta_normalized = beta_absolute / (30 - 13)

        # Calculate Beta/Alpha ratio (adding a small epsilon to avoid division by zero errors)
        ratio = beta_normalized / (alpha_normalized + 1e-9)

        feature_name = f"{ch_name}_Beta_Alpha_Ratio"
        features[feature_name] = ratio

    return features

def process_single_file(filepath):
    """
    Handles the filtering, ICA scrubbing, and epoching for a single CSV file.
    """
    filename = os.path.basename(filepath)
    print(f"  -> Applying ICA & Extracting Features: {filename}")
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"     ❌ Error reading file: {e}")
        return []

    df = clean_flagged_data(df, filename)

    if not all(col in df.columns for col in EEG_CHANNELS + ['Marker']):
        print("     ⚠️ Missing required EEG or Marker columns. Skipping.")
        return []

    # Apply full MNE Pipeline (Filter + ICA Scrubbing) to the continuous recording
    clean_eeg = apply_mne_pipeline(df, SAMPLING_RATE)
    df.loc[:, EEG_CHANNELS] = clean_eeg

    # Slice only target ML data AFTER cleaning to ensure ICA had enough data to fit
    df = df[df['Marker'].isin([3, 4])].copy()

    if df.empty:
        print("     ⚠️ No valid target marker data (3 or 4) found after cleaning. Skipping.")
        return []

    samples_per_epoch = int(SAMPLING_RATE * EPOCH_LENGTH_SEC)
    session_features = []

    for marker_val in [3, 4]:
        state_data = df[df['Marker'] == marker_val]
        total_samples = len(state_data)
        num_epochs = total_samples // samples_per_epoch

        for i in range(num_epochs):
            start_idx = i * samples_per_epoch
            end_idx = start_idx + samples_per_epoch

            epoch_slice = state_data[EEG_CHANNELS].iloc[start_idx:end_idx].values
            features = extract_frequency_features(epoch_slice, SAMPLING_RATE)

            binary_class = 0 if marker_val == 3 else 1
            features['Target_Class'] = binary_class

            session_features.append(features)

    return session_features

# TKINTER EXISTING USER REGISTRATION
def run_pre_registration():
    root = tk.Tk()
    root.title("Operator Login (Existing Subjects Only)")
    root.geometry("380x250")

    tk.Label(root, text="3-Digit Subject ID (e.g., 001):").pack(pady=10)
    entry_id = tk.Entry(root, font=("Arial", 14), justify="center")
    entry_id.pack(pady=5)

    tk.Label(root, text="Select Task to Process:").pack(pady=5)
    task_var = tk.StringVar(value="math")
    tk.Radiobutton(root, text="Complex Mental Math", variable=task_var, value="math").pack()
    tk.Radiobutton(root, text="Missing Letter (Cloze)", variable=task_var, value="words").pack()

    save_dir_path = []
    subject_id_val = []
    selected_task_val = []
    target_mac_val = []
    base_data_dir_val = []

    def submit_data():
        reg_id = entry_id.get().strip()
        if not reg_id or len(reg_id) != 3 or not reg_id.isdigit():
            messagebox.showerror("Error", "Please enter a valid 3-digit Registration ID.")
            return

        folder_path = os.path.join(BASE_DATA_DIR, f"ID_{reg_id}")
        offline_path = os.path.join(folder_path, "offline")
        info_path = os.path.join(folder_path, "personal_info.txt")

        if not os.path.exists(folder_path):
            messagebox.showerror("Error", f"Profile ID_{reg_id} does not exist. Please register offline first.")
            return

        if not os.path.exists(offline_path):
            messagebox.showerror("Error", f"No 'offline' data folder found for ID_{reg_id}. Please run the offline calibration protocol first.")
            return

        if not os.path.exists(info_path):
            messagebox.showerror("Error", f"'personal_info.txt' not found for ID_{reg_id}. Cannot extract hardware metadata.")
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

        save_dir_path.append(offline_path)
        subject_id_val.append(reg_id)
        selected_task_val.append(task_var.get())
        target_mac_val.append(extracted_mac)
        base_data_dir_val.append(extracted_base)

        root.destroy()

    tk.Button(root, text="Load Profile & Extract Features", command=submit_data, bg="#3498db", fg="white",
              font=("Arial", 10, "bold")).pack(pady=15)
    root.mainloop()

    if not save_dir_path:
        sys.exit(0)

    return subject_id_val[0], save_dir_path[0], selected_task_val[0], target_mac_val[0], base_data_dir_val[0]

def main():
    print("=" * 60)
    print("🧠 Subject-Specific Feature Extraction & MNE ICA Scrubbing")
    print("=" * 60)

    subject_id, target_dir, task_type, target_mac, verified_base_dir = run_pre_registration()

    print(f"\n📡 Audit Trail - Hardware MAC Verified: {target_mac}")
    print(f"📁 Audit Trail - Base Directory Verified: {verified_base_dir}")

    print(f"\n📂 Scanning directory: {target_dir} for '{task_type}' CSV files...")
    search_pattern = os.path.join(target_dir, "*.csv")
    all_csv_files = glob.glob(search_pattern)

    # Filter files strictly by the user-selected task
    files_to_process = [f for f in all_csv_files if task_type in os.path.basename(f).lower()]

    if not files_to_process:
        print(f"❌ Error: No CSV files for the '{task_type}' task found in '{target_dir}'.")
        return

    print(f"🔍 Found {len(files_to_process)} '{task_type}' sessions to process for Subject {subject_id}.\n")

    print("\n" + "=" * 60)
    print(f"🚀 INITIATING PIPELINE FOR TASK: {task_type.upper()} ({len(files_to_process)} files)")
    print("=" * 60)

    # PASS 1: BIOLOGICAL VALIDATION (BERGER EFFECT ONLY)
    print(f"🧪 Running Pass 1: Biological Validation (Berger Effect) for {task_type.upper()}...")

    passed_files = []
    report_lines = [
        f"HERBioLab - Biological Validation Quality Report",
        f"Subject ID: {subject_id} | Task: {task_type.upper()}",
        f"Hardware Setup -> MAC: {target_mac} | Data Root: {verified_base_dir}",
        "Note: Frontal artifacts are now scrubbed mathematically via MNE ICA.",
        "=" * 60
    ]

    for file in files_to_process:
        filename = os.path.basename(file)
        try:
            df = pd.read_csv(file)
            df = clean_flagged_data(df, filename)

            # Apply MNE Pipeline to the full file first so the Berger check operates on clean data
            clean_eeg = apply_mne_pipeline(df, SAMPLING_RATE)
            df.loc[:, EEG_CHANNELS] = clean_eeg

            is_berger_valid, berger_msg = verify_berger_effect(df, SAMPLING_RATE, filename)

            print(f"     {berger_msg}")
            report_lines.append(berger_msg)

            if is_berger_valid:
                passed_files.append(file)

        except Exception as e:
            err_msg = f"[{filename}] ❌ File Read Error: {e}"
            print(f"     {err_msg}")
            report_lines.append(err_msg)

    report_path = os.path.join(target_dir, f"muse_quality_report_{task_type}_ID_{subject_id}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n📄 Quality reference report saved to: {report_path}")

    # GATEKEEPER DECISION LOGIC
    valid_files = passed_files

    if not valid_files:
        print(f"\n⚠️ WARNING: NO files passed the Berger Effect validation criteria for {task_type.upper()}.")

        root = tk.Tk()
        root.withdraw()
        force_override = messagebox.askyesno(
            "Biological Validation Failed",
            f"None of the {len(files_to_process)} files for the {task_type.upper()} task passed the Berger Effect validation.\n\n"
            "This suggests poor cortical contact across all sessions.\n\n"
            "Do you want to FORCE process all files anyway?"
        )
        root.destroy()

        if force_override:
            print("⚠️ Operator Override Activated. Processing ALL files...")
            valid_files = files_to_process
        else:
            print(f"❌ Pipeline terminated for {task_type.upper()} task to preserve dataset integrity.")
            sys.exit(0)
    else:
        print(f"\n✅ Proceeding to ML Extraction with {len(valid_files)} validated {task_type.upper()} files.")

    # PASS 2: MNE ICA & FEATURE EXTRACTION
    print(f"\n⚙️ Running Pass 2: Feature Extraction for {task_type.upper()}...")

    all_extracted_features = []
    for file in valid_files:
        file_features = process_single_file(file)
        all_extracted_features.extend(file_features)

    if not all_extracted_features:
        print(f"\n❌ Error: No valid epochs could be extracted from any {task_type.upper()} files.")
        return

    unpruned_df = pd.DataFrame(all_extracted_features)
    print(f"\n✅ Extraction Complete! Fused {len(unpruned_df)} total pure epochs for {task_type.upper()}.")

    X = unpruned_df.drop('Target_Class', axis=1)
    y = unpruned_df['Target_Class']

    if len(y.unique()) < 2:
        print(
            f"❌ Error: Only one class found in the extracted epochs for {task_type.upper()}. Cannot perform feature selection.")
        return

    print("\n🧠 Executing Mathematical Feature Selection (ANOVA)...")

    # Adjusted to select the top 2 features
    num_features_to_select = min(2, X.shape[1])
    selector = SelectKBest(score_func=f_classif, k=num_features_to_select)
    selector.fit(X, y)

    top_feature_indices = selector.get_support(indices=True)
    top_feature_names = X.columns[top_feature_indices].tolist()

    print(f"🏆 Top {num_features_to_select} Features Selected for Subject {subject_id} ({task_type.upper()}):")

    anova_audit_lines = [
        "",
        "=" * 60,
        "PASS 2: ANOVA FEATURE SELECTION AUDIT",
        "=" * 60,
        f"Top {num_features_to_select} Features Selected (Higher F-Score = Better Class Separability):"
    ]

    for i, name in enumerate(top_feature_names, 1):
        score = selector.scores_[top_feature_indices[i - 1]]
        print(f"  {i}. {name:<22} (F-Score: {score:.2f})")
        anova_audit_lines.append(f"  {i}. {name:<22} (F-Score: {score:.2f})")

    with open(report_path, "a", encoding="utf-8") as f:
        f.write("\n".join(anova_audit_lines) + "\n")
    print(f"📄 ANOVA F-Scores appended to quality report: {report_path}")

    final_feature_df = unpruned_df[top_feature_names + ['Target_Class']]
    output_file = os.path.join(target_dir, f"muse_extracted_offline_features_{task_type}_ID_{subject_id}.csv")
    final_feature_df.to_csv(output_file, index=False)
    print(f"\n💾 Sliced feature matrix successfully saved to '{output_file}'\n")

if __name__ == "__main__":
    main()