"""
BCI Offline Classifier: Subject-Specific LDA vs. SVM Comparison
------------------------------------------------------------------------
This pipeline prompts for a Subject ID and a specific Cognitive Task
(Math vs. Words) via a GUI. It reads the corresponding feature matrix
from the /offline/ subfolder, and independently trains both a Regularized
LDA and an RBF Kernel SVM. It saves models and performance reports
directly back into the /offline/ directory using task-specific naming.
"""

import os
import sys
import glob
import joblib
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, cohen_kappa_score,
    f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
)

# DIRECTORY CONFIGURATION
BASE_DATA_DIR = r"D:\MuseData"

def calculate_itr(n_classes, accuracy, trial_duration_sec):
    """Calculates the Information Transfer Rate (ITR) in bits per minute."""
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
    """Handles CV and Holdout evaluation for a given model."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accuracies = []

    # Cross-Validation
    for train_idx, val_idx in skf.split(X_train, y_train):
        model.fit(X_train[train_idx], y_train[train_idx])
        preds = model.predict(X_train[val_idx])
        cv_accuracies.append(accuracy_score(y_train[val_idx], preds))

    cv_mean = np.mean(cv_accuracies)
    cv_std = np.std(cv_accuracies)

    # Final Fit & Holdout
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
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_probs),
        'cm': confusion_matrix(y_test, y_pred),
        'itr': calculate_itr(n_classes=2, accuracy=accuracy_score(y_test, y_pred), trial_duration_sec=2.0)
    }
    return metrics

def generate_report_text(model_name, task_type, features, metrics, target_mac, base_dir):
    """Formats the performance text report for a specific model."""
    return f"""==================================================
BCI CLASSIFIER OFFLINE PERFORMANCE: {model_name} ({task_type.upper()})
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

# TKINTER EXISTING USER REGISTRATION
def run_pre_registration():
    root = tk.Tk()
    root.title("Operator Login (Existing Subjects Only)")
    root.geometry("380x250")

    tk.Label(root, text="3-Digit Subject ID (e.g., 001):").pack(pady=10)
    entry_id = tk.Entry(root, font=("Arial", 14), justify="center")
    entry_id.pack(pady=5)

    tk.Label(root, text="Select Task to Train Models For:").pack(pady=5)
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
            messagebox.showerror("Error",
                                 f"No 'offline' data folder found for ID_{reg_id}. Please run the offline extraction pipeline first.")
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

        # Target the offline subfolder specifically
        save_dir_path.append(offline_path)
        subject_id_val.append(reg_id)
        selected_task_val.append(task_var.get())
        target_mac_val.append(extracted_mac)
        base_data_dir_val.append(extracted_base)
        root.destroy()

    tk.Button(root, text="Load Profile & Train Models", command=submit_data, bg="#3498db", fg="white",
              font=("Arial", 10, "bold")).pack(pady=15)
    root.mainloop()

    if not save_dir_path:
        sys.exit(0)

    return subject_id_val[0], save_dir_path[0], selected_task_val[0], target_mac_val[0], base_data_dir_val[0]

def main():
    print("=" * 60)
    print("🤖 Subject-Specific Classifier Training (LDA vs SVM)")
    print("=" * 60)

    # 1. Prompt for Subject ID and Task via UI
    subject_id, target_dir, task_type, target_mac, verified_base_dir = run_pre_registration()

    print(f"\n📡 Audit Trail - Hardware MAC Verified: {target_mac}")
    print(f"📁 Audit Trail - Base Directory Verified: {verified_base_dir}")

    # Locate the specific extracted feature matrix for the requested task
    feature_file_path = os.path.join(target_dir, f"muse_extracted_offline_features_{task_type}_ID_{subject_id}.csv")

    # Fallback to catch the generic name if the extraction script hasn't been re-run yet
    if not os.path.exists(feature_file_path):
        generic_path = os.path.join(target_dir, f"muse_extracted_offline_features_ID_{subject_id}.csv")
        if os.path.exists(generic_path):
            feature_file_path = generic_path
            print(f"⚠️ Specific '{task_type}' feature file not found. Falling back to generic file.")
        else:
            print(f"❌ Error: No feature file found for '{task_type}' in {target_dir}. Run feature extraction first.")
            return

    filename = os.path.basename(feature_file_path).lower()

    print("\n" + "=" * 60)
    print(f"🚀 INITIATING MODEL TRAINING FOR TASK: {task_type.upper()}")
    print("=" * 60)

    # 2. Load Data
    print(f"\n📂 Loading {task_type.upper()} features from Subject {subject_id}...")
    df = pd.read_csv(feature_file_path)

    if 'Target_Class' not in df.columns:
        print(f"❌ Error: 'Target_Class' missing in {filename}. Exiting.")
        return

    X = df.drop('Target_Class', axis=1).values
    feature_names = df.drop('Target_Class', axis=1).columns.tolist()
    y = df['Target_Class'].values

    # 3. Split and Scale (CRITICAL for SVM)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Initialize Models
    print("🛡️ Training Regularized LDA...")
    lda_model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    lda_metrics = evaluate_model(lda_model, X_train_scaled, y_train, X_test_scaled, y_test)

    print("🧠 Training Kernel SVM (RBF)...")
    # probability=True is required to calculate the ROC-AUC score for SVM
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    svm_metrics = evaluate_model(svm_model, X_train_scaled, y_train, X_test_scaled, y_test)

    # 5. Save Model Output Names
    lda_out_name = f"bci_live_production_{task_type}_lda.joblib"
    svm_out_name = f"bci_live_production_{task_type}_svm.joblib"

    print("\n💾 Serializing subject-specific production models...")
    joblib.dump({'scaler': scaler, 'model': lda_model, 'feature_names': feature_names},
                os.path.join(target_dir, lda_out_name))
    joblib.dump({'scaler': scaler, 'model': svm_model, 'feature_names': feature_names},
                os.path.join(target_dir, svm_out_name))

    # 6. Write Individual Reports
    lda_report_name = f"report_{task_type}_LDA_metrics.txt"
    svm_report_name = f"report_{task_type}_SVM_metrics.txt"

    with open(os.path.join(target_dir, lda_report_name), "w", encoding="utf-8") as f:
        f.write(generate_report_text("Regularized LDA", task_type, feature_names, lda_metrics, target_mac,
                                     verified_base_dir))

    with open(os.path.join(target_dir, svm_report_name), "w", encoding="utf-8") as f:
        f.write(generate_report_text("RBF Kernel SVM", task_type, feature_names, svm_metrics, target_mac,
                                     verified_base_dir))

    # 7. Write Head-to-Head Comparison Report
    comp_text = f"""==================================================
HEAD-TO-HEAD BCI CLASSIFIER COMPARISON (SUBJECT {subject_id} - {task_type.upper()})
Hardware Setup -> MAC: {target_mac} | Data Root: {verified_base_dir}
==================================================
Target Directory: {target_dir}

METRIC                 | REGULARIZED LDA      | RBF KERNEL SVM
------------------------------------------------------------------
CV Mean Accuracy       | {lda_metrics['cv_mean'] * 100:>6.2f}% (±{lda_metrics['cv_std'] * 100:.1f}%) 
                       | {svm_metrics['cv_mean'] * 100:>6.2f}% (±{svm_metrics['cv_std'] * 100:.1f}%)
Holdout Accuracy       | {lda_metrics['acc'] * 100:>13.2f}% | {svm_metrics['acc'] * 100:>13.2f}%
Balanced Accuracy      | {lda_metrics['bal_acc'] * 100:>13.2f}% | {svm_metrics['bal_acc'] * 100:>13.2f}%
ROC-AUC Score          | {lda_metrics['auc']:>14.4f} | {svm_metrics['auc']:>14.4f}
Cohen's Kappa          | {lda_metrics['kappa']:>14.4f} | {svm_metrics['kappa']:>14.4f}
F1-Score               | {lda_metrics['f1']:>14.4f} | {svm_metrics['f1']:>14.4f}
ITR (bits/min)         | {lda_metrics['itr']:>14.2f} | {svm_metrics['itr']:>14.2f}

CONCLUSION DIAGNOSTIC:
- If SVM Holdout significantly outperforms LDA, your EEG features possess non-linear relationships.
- If LDA matches or beats SVM, keep LDA. It is faster, less prone to overfitting, and better suited for real-time BCI.
==================================================
"""
    comp_name = f"report_{task_type}_COMPARISON_matrix.txt"
    with open(os.path.join(target_dir, comp_name), "w", encoding="utf-8") as f:
        f.write(comp_text)

    print(f"✅ Pipeline complete for {task_type.upper()}! 5 new files generated in {target_dir}:")
    print(f"   - {lda_out_name}")
    print(f"   - {svm_out_name}")
    print(f"   - {lda_report_name}")
    print(f"   - {svm_report_name}")
    print(f"   - {comp_name}")

if __name__ == "__main__":
    main()