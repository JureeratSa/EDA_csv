import pandas as pd
import neurokit2 as nk
import numpy as np
from pyhrv.frequency_domain import welch_psd
import matplotlib.pyplot as plt
import os

# ------------------ ฟังก์ชันเตรียม HRV จาก PPG ------------------
def prepare_ppg_data_EmotiBit(ppg_df, window_size, sampling_rate):
    window_samples = window_size * sampling_rate
    ppg_df['Timestamp'] = pd.to_datetime(ppg_df['LocalTimestamp'], unit='s').dt.tz_localize('UTC').dt.tz_convert('Asia/Bangkok')
    ppg_cleaned = nk.ppg_clean(ppg_df['PG'], sampling_rate=sampling_rate)
    hrv_results = []

    for start in range(0, len(ppg_cleaned) - window_samples, sampling_rate):
        end = start + window_samples
        window = ppg_cleaned[start:end]
        start_time = ppg_df['Timestamp'].iloc[start]

        if start % (sampling_rate * 30) == 0:
            print(f"Start time: {start_time} - End time: {ppg_df['Timestamp'].iloc[end]}")

        signals, info = nk.ppg_peaks(window, sampling_rate=sampling_rate)
        peaks = info['PPG_Peaks']
        nni = np.diff(peaks) * 1000 / sampling_rate

        if len(nni) > 0:
            freq_results = welch_psd(nni=nni, show=False)

            print(freq_results)
            hrv_dict = {
                "Timestamp": start_time,
                "LF/HF_ratio": freq_results.as_dict()['fft_ratio'],
                "LF_n": freq_results.as_dict()['fft_norm'][0],
                "HF_n": freq_results.as_dict()['fft_norm'][1],
                "LF_abs": freq_results.as_dict()['fft_abs'][1],
                "HF_abs": freq_results.as_dict()['fft_abs'][2],
                "Total": freq_results.as_dict()['fft_total'],
            }
            hrv_results.append(hrv_dict)

    hrv_df = pd.DataFrame(hrv_results)
    hrv_df['Timestamp'] = pd.to_datetime(hrv_df['Timestamp'])
    hrv_df = hrv_df.set_index('Timestamp')
    extracted_df = hrv_df.resample("1s").max().reset_index()
    return extracted_df


# ------------------ เริ่มประมวลผล ------------------
print("Processing EmotiBit PPG data...")
df_EmotiBit = pd.read_csv(
    r"D:\DataCenter\Dek66\kk\Emotibit-10min\S08\RAW\Warm Pain\2025-08-23_14-50-39-730070_PG.csv"
)

hrv_ppg_EB = prepare_ppg_data_EmotiBit(df_EmotiBit, window_size=60 * 5, sampling_rate=100)

# hrv_ppg_EB.to_csv(r"C:\Users\Booklab\Desktop\kk\Emotibit-10min\S01 - sm\RAW\Normal\S01.normal_ppg.csv")


# รวมข้อมูล HRV
lfhf_df = hrv_ppg_EB[['Timestamp', 'LF/HF_ratio']].dropna()
hf_df = hrv_ppg_EB[['Timestamp', 'HF_n', 'HF_abs']].dropna()
lf_df = hrv_ppg_EB[['Timestamp', 'LF_n', 'LF_abs']].dropna()
total_df = hrv_ppg_EB[['Timestamp', 'Total']].dropna()

hf_df = hf_df.sort_values('Timestamp')
lf_df = lf_df.sort_values('Timestamp')
lfhf_df = lfhf_df.sort_values('Timestamp')
total_df = total_df.sort_values('Timestamp')

merged_df = pd.merge_asof(hf_df, lf_df, on='Timestamp', direction='nearest')
final_merged_df_ppg_EB = pd.merge_asof(merged_df, lfhf_df, on='Timestamp', direction='nearest')
final_merged_df_ppg_EB = pd.merge_asof(final_merged_df_ppg_EB, total_df, on='Timestamp', direction='nearest')

final_merged_df_ppg_EB = final_merged_df_ppg_EB.rename(columns={
    'HF_n': 'HF_n_PG_EB',
    'LF_n': 'LF_n_PG_EB',
    'LF/HF_ratio': 'LFHF_ratio_PG_EB',
    'LF_abs': 'LF_abs_PG_EB',
    'HF_abs': 'HF_abs_PG_EB',
    'Total': 'Total_PG_EB',
})

final_merged_df_ppg_EB['Timestamp'] = final_merged_df_ppg_EB['Timestamp'].dt.tz_convert('Asia/Bangkok')

# Save CSV
output_path = r"D:\DataCenter\Dek66\kk\Emotibit-10min\S08\RAW\Warm Pain\S08.warmpain_ppg.csv"
final_merged_df_ppg_EB.to_csv(output_path, index=False)
print(f"Saved merged HRV data: {output_path}")
#

# ------------------ Plot Results (ทั้งช่วง) ------------------
filename = "S08.warmpain_ppg"
age = 20
gender_input = "Male"
bmi_str = "BMI= 29.7"
painLevel = input("Enter pain level (e.g., None, Mild, Moderate, Severe): ")
save_folder = r"D:\DataCenter\Dek66\kk\Emotibit-10min\S08\RAW\Warm Pain"

# Absolute Power
plt.figure(figsize=(16, 8))
plt.plot(final_merged_df_ppg_EB["Timestamp"], final_merged_df_ppg_EB["HF_abs_PG_EB"], label="HF_abs", alpha=0.7)
plt.plot(final_merged_df_ppg_EB["Timestamp"], final_merged_df_ppg_EB["LF_abs_PG_EB"], label="LF_abs", alpha=0.7)
plt.plot(final_merged_df_ppg_EB["Timestamp"], final_merged_df_ppg_EB["LFHF_ratio_PG_EB"], label="LF/HF Ratio", alpha=0.7)
plt.title(f"HRV Absolute Power - {filename}\nAge: {age}, Gender: {gender_input}, {bmi_str}, Pain Level: {painLevel}")
plt.xlabel("Time")
plt.ylabel("Power (ms²) / Ratio")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

image_path_abs = os.path.join(save_folder, f"{filename}_abs.png")
plt.savefig(image_path_abs, dpi=300)
print(f"Saved graph: {image_path_abs}")


# Normalized Power
plt.figure(figsize=(16, 8))
plt.plot(final_merged_df_ppg_EB["Timestamp"], final_merged_df_ppg_EB["HF_n_PG_EB"], label="HF_n (Normalized)", alpha=0.7)
plt.plot(final_merged_df_ppg_EB["Timestamp"], final_merged_df_ppg_EB["LF_n_PG_EB"], label="LF_n (Normalized)", alpha=0.7)
plt.plot(final_merged_df_ppg_EB["Timestamp"], final_merged_df_ppg_EB["LFHF_ratio_PG_EB"], label="LF/HF Ratio", alpha=0.7)
plt.title(f"HRV Normalized Power - {filename}\nAge: {age}, Gender: {gender_input}, {bmi_str}, Pain Level: {painLevel}")
plt.xlabel("Time")
plt.ylabel("Normalized Power / Ratio")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

image_path_norm = os.path.join(save_folder, f"{filename}_norm.png")
plt.savefig(image_path_norm, dpi=300)
print(f"Saved graph: {image_path_norm}")



# ------------------ Save CSV และ Plot แยกตามช่วงเวลาที่ต้องการ ------------------

time_ranges = [
    (2, 5),
    (5, 7)
]

for start_min, end_min in time_ranges:
    start_time = final_merged_df_ppg_EB["Timestamp"].min() + pd.Timedelta(minutes=start_min)
    end_time = final_merged_df_ppg_EB["Timestamp"].min() + pd.Timedelta(minutes=end_min)

    # กรองข้อมูลในช่วงเวลานี้
    segment_df = final_merged_df_ppg_EB[
        (final_merged_df_ppg_EB["Timestamp"] >= start_time) &
        (final_merged_df_ppg_EB["Timestamp"] < end_time)
    ].copy()

    # บันทึก CSV แยกสำหรับช่วงนี้
    csv_segment_path = os.path.join(save_folder, f"{filename}_{start_min}to{end_min}min.csv")
    segment_df.to_csv(csv_segment_path, index=False)
    print(f"Saved CSV segment: {csv_segment_path}")

    # Plot Absolute Power
    plt.figure(figsize=(12, 6))
    plt.plot(segment_df["Timestamp"], segment_df["HF_abs_PG_EB"], label="HF_abs", alpha=0.7)
    plt.plot(segment_df["Timestamp"], segment_df["LF_abs_PG_EB"], label="LF_abs", alpha=0.7)
    plt.plot(segment_df["Timestamp"], segment_df["LFHF_ratio_PG_EB"], label="LF/HF Ratio", alpha=0.7)
    plt.title(f"HRV Absolute Power - {filename} {start_min}-{end_min} min\nAge: {age}, Gender: {gender_input}, {bmi_str}, Pain Level: {painLevel}")
    plt.xlabel("Time")
    plt.ylabel("Power (ms²) / Ratio")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    img_abs_path = os.path.join(save_folder, f"{filename}_{start_min}to{end_min}min_abs.png")
    plt.savefig(img_abs_path, dpi=300)
    plt.close()
    print(f"Saved graph: {img_abs_path}")

    # Plot Normalized Power
    plt.figure(figsize=(12, 6))
    plt.plot(segment_df["Timestamp"], segment_df["HF_n_PG_EB"], label="HF_n (Normalized)", alpha=0.7)
    plt.plot(segment_df["Timestamp"], segment_df["LF_n_PG_EB"], label="LF_n (Normalized)", alpha=0.7)
    plt.plot(segment_df["Timestamp"], segment_df["LFHF_ratio_PG_EB"], label="LF/HF Ratio", alpha=0.7)
    plt.title(f"HRV Normalized Power - {filename} {start_min}-{end_min} min\nAge: {age}, Gender: {gender_input}, {bmi_str}, Pain Level: {painLevel}")
    plt.xlabel("Time")
    plt.ylabel("Normalized Power / Ratio")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    img_norm_path = os.path.join(save_folder, f"{filename}_{start_min}to{end_min}min_norm.png")
    plt.savefig(img_norm_path, dpi=300)
    plt.close()
    print(f"Saved graph: {img_norm_path}")
