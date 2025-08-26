import neurokit2 as nk
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import os

# อ่านข้อมูล EDA
eda_signal = pd.read_csv(r"D:\DataCenter\Dek66\kk\Emotibit-10min\S08\RAW\Warm Pain\2025-08-23_14-50-39-730070_EA.csv")
rateOfSample = 15  # ค่า sampling rate ของ EDA

# User inputs
age = "21"
gender_input = "Male"
bmi_str = "BMI=29.7"
painLevel = input("Enter pain level (e.g., None, Mild, Moderate, Severe): ")

filename = "S08.warmpain_eda"
save_folder = r"D:\DataCenter\Dek66\kk\Emotibit-10min\S08\RAW\Warm Pain"

def min_to_sampling(minute, sr):
    return int(minute * 60 * sr)

rest = min_to_sampling(2, rateOfSample)
stop = min_to_sampling(12, rateOfSample)

# ทำความสะอาดสัญญาณ
eda_cleaned = nk.eda_clean(eda_signal['EA'], sampling_rate=rateOfSample)

# แยก tonic และ phasic
eda_decomposed = nk.eda_phasic(eda_cleaned, sampling_rate=rateOfSample, method='cvxeda')

# ช่วง baseline และ body
head = eda_decomposed[0:rest]
body = eda_decomposed[rest:stop]

baselineTonic = float(np.mean(head["EDA_Tonic"]))
debaseTonic = body["EDA_Tonic"].values - baselineTonic
time = np.arange(len(body)) / rateOfSample

y_min_tonic = float(np.min(debaseTonic))
y_max_tonic = float(np.max(debaseTonic))
y_min_phasic = float(np.min(body["EDA_Phasic"].values))
y_max_phasic = float(np.max(body["EDA_Phasic"].values))

y_min = float(np.floor(min(y_min_tonic, y_min_phasic)))
y_max = float(np.ceil(max(y_max_tonic, y_max_phasic)))

# หา peak ของ phasic
peaks_info = nk.eda_findpeaks(body["EDA_Phasic"].values, sampling_rate=rateOfSample, method="neurokit", amplitude_min=0.05)
peaks = np.max(peaks_info['SCR_Height']) if len(peaks_info['SCR_Height']) > 0 else None

# วาดกราฟ tonic และ phasic ช่วง 2-12 นาที
plt.figure(figsize=(12, 6))
plt.plot(time, debaseTonic, label="Tonic Component", linewidth=2)
plt.plot(time, body["EDA_Phasic"].values, label="Phasic Component", linewidth=1)
plt.xlabel("Time (seconds)")
plt.ylabel("EDA Signal")
plt.ylim(y_min, y_max)

step = 0.25
tick_values = []
current = y_min
while current <= y_max:
    tick_values.append(current)
    current += step
plt.yticks(tick_values)

plt.legend()
plt.title(f"EDA - {filename}\nAge: {age}, Gender: {gender_input}, BMI: {bmi_str}, Pain Level: {painLevel}")
plt.grid(True, alpha=0.3)
plt.text(0.02, 0.98, f'Y-axis range: [{y_min:.2f}, {y_max:.2f}]', transform=plt.gca().transAxes,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# บันทึกรูปภาพ
image_path_abs = os.path.join(save_folder, f"{filename}.png")
plt.savefig(image_path_abs, dpi=300)
plt.close()
print(f"Saved graph: {image_path_abs}")

# บันทึกข้อมูล decomposition ทั้ง baseline+body เป็น csv
csv_path = os.path.join(save_folder, f"{filename}.csv")
eda_decomposed.to_csv(csv_path, index=False)
print(f"Saved CSV data: {csv_path}")

# แสดงสถิติ tonic
print("\nTonic Component Stats:")
print(f"Min: {np.min(debaseTonic):.3f}")
print(f"Max: {np.max(debaseTonic):.3f}")
print(f"Mean: {np.mean(debaseTonic):.3f}")
print(f"Median: {np.median(debaseTonic):.3f}")
print(f"Standard Deviation: {np.std(debaseTonic):.3f}")

# แสดงสถิติ phasic
print("\nPhasic Component Stats:")
print(f"Min: {np.min(body['EDA_Phasic']):.3f}")
print(f"Max: {np.max(body['EDA_Phasic']):.3f}")
print(f"Mean: {np.mean(body['EDA_Phasic']):.3f}")
print(f"Median: {np.median(body['EDA_Phasic']):.3f}")
print(f"Standard Deviation: {np.std(body['EDA_Phasic']):.3f}")

# บันทึกสถิติ tonic และ phasic ลง CSV ชื่อ eda.csv
eda_stats = pd.DataFrame({
    "Component": ["Tonic", "Phasic"],
    "Min": [np.min(debaseTonic), np.min(body['EDA_Phasic'])],
    "Max": [np.max(debaseTonic), np.max(body['EDA_Phasic'])],
    "Mean": [np.mean(debaseTonic), np.mean(body['EDA_Phasic'])],
    "Median": [np.median(debaseTonic), np.median(body['EDA_Phasic'])],
    "Standard Deviation": [np.std(debaseTonic), np.std(body['EDA_Phasic'])]
})
stats_csv_path = os.path.join(save_folder, "eda.csv")
eda_stats.to_csv(stats_csv_path, index=False)
print(f"Saved EDA stats to CSV: {stats_csv_path}")

# แบ่งช่วงเวลาช่วงละ 3 นาทีและบันทึกภาพ+csv
time_ranges = [
    (2, 5),
    (5, 8),
    (8, 12)
]

for start_min, end_min in time_ranges:
    start_idx = min_to_sampling(start_min, rateOfSample)
    end_idx = min_to_sampling(end_min, rateOfSample)

    segment = eda_decomposed[start_idx:end_idx]

    debase_tonic_segment = segment["EDA_Tonic"].values - baselineTonic
    time_segment = np.arange(len(segment)) / rateOfSample

    y_min_tonic_seg = float(np.min(debase_tonic_segment))
    y_max_tonic_seg = float(np.max(debase_tonic_segment))
    y_min_phasic_seg = float(np.min(segment["EDA_Phasic"].values))
    y_max_phasic_seg = float(np.max(segment["EDA_Phasic"].values))

    y_min_seg = float(np.floor(min(y_min_tonic_seg, y_min_phasic_seg)))
    y_max_seg = float(np.ceil(max(y_max_tonic_seg, y_max_phasic_seg)))

    peaks_info_seg = nk.eda_findpeaks(segment["EDA_Phasic"].values, sampling_rate=rateOfSample, method="neurokit", amplitude_min=0.05)
    peaks_seg = np.max(peaks_info_seg['SCR_Height']) if len(peaks_info_seg['SCR_Height']) > 0 else None

    plt.figure(figsize=(12, 6))
    plt.plot(time_segment, debase_tonic_segment, label="Tonic Component", linewidth=2)
    plt.plot(time_segment, segment["EDA_Phasic"].values, label="Phasic Component", linewidth=1)
    plt.xlabel("Time (seconds)")
    plt.ylabel("EDA Signal")
    plt.ylim(y_min_seg, y_max_seg)

    step = 0.25
    tick_values = []
    current = y_min_seg
    while current <= y_max_seg:
        tick_values.append(current)
        current += step
    plt.yticks(tick_values)

    plt.legend()
    plt.title(f"EDA - {filename}_{start_min}to{end_min}min\nAge: {age}, Gender: {gender_input}, BMI: {bmi_str}, Pain Level: {painLevel}")
    plt.grid(True, alpha=0.3)
    plt.text(0.02, 0.98, f'Y-axis range: [{y_min_seg:.2f}, {y_max_seg:.2f}]',
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    image_path_seg = os.path.join(save_folder, f"{filename}_{start_min}to{end_min}min.png")
    plt.savefig(image_path_seg, dpi=300)
    plt.close()
    print(f"Saved graph: {image_path_seg}")

    csv_path_segment = os.path.join(save_folder, f"{filename}_{start_min}to{end_min}min.csv")
    segment.to_csv(csv_path_segment, index=False)
    print(f"Saved CSV data: {csv_path_segment}")
