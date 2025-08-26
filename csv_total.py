import pandas as pd
import numpy as np
import os

# ค่าคงที่สำหรับข้อมูลส่วนบุคคล
age = "21"
gender_input = "1"
bmi_str = "29.7"  # เอาแค่ตัวเลข
painLevel = 1

# ไฟล์ที่ต้องการรวม 3 ไฟล์
eda_file = r"D:\DataCenter\Dek66\kk\Emotibit-10min\S02\RAW\Cold Pain\S02.coldpain_eda.csv"
ppg_file = r"D:\DataCenter\Dek66\kk\Emotibit-10min\S02\RAW\Cold Pain\S02.coldpain_ppg.csv"
temp_file = r"D:\DataCenter\Dek66\kk\Emotibit-10min\S02\RAW\Cold Pain\2025-08-10_11-28-29-952627_T1.csv"

# ไฟล์ผลลัพธ์
output_folder = r"D:\DataCenter\Dek66\kk\Emotibit-10min\S02\RAW\Cold Pain"
output_filename = "S02_EAPG.csv"

try:
    # อ่านไฟล์ EDA
    print("Reading EDA file...")
    eda_data = pd.read_csv(eda_file)
    print(f"EDA data shape: {eda_data.shape}")
    print(f"EDA columns: {list(eda_data.columns)}")

    # อ่านไฟล์ PPG
    print("\nReading PPG file...")
    ppg_data = pd.read_csv(ppg_file)
    print(f"PPG data shape: {ppg_data.shape}")
    print(f"PPG columns: {list(ppg_data.columns)}")

    # อ่านไฟล์ Temperature
    print("\nReading Temperature file...")
    temp_data = pd.read_csv(temp_file)
    print(f"Temperature data shape: {temp_data.shape}")
    print(f"Temperature columns: {list(temp_data.columns)}")

    # ตรวจสอบจำนวนแถวและปรับให้เท่ากัน
    min_rows = min(len(eda_data), len(ppg_data), len(temp_data))
    print(f"\nAdjusting to minimum rows: {min_rows}")

    eda_data = eda_data.iloc[:min_rows].reset_index(drop=True)
    ppg_data = ppg_data.iloc[:min_rows].reset_index(drop=True)
    temp_data = temp_data.iloc[:min_rows].reset_index(drop=True)

    # สร้าง DataFrame ผลลัพธ์
    merged_data = pd.DataFrame()

    # เพิ่มข้อมูล EDA
    if 'EDA_Tonic' in eda_data.columns:
        merged_data['EDA_Tonic'] = eda_data['EDA_Tonic']
        print("Added EDA_Tonic")
    else:
        print("Warning: EDA_Tonic column not found in EDA file")
        merged_data['EDA_Tonic'] = np.nan

    if 'EDA_Phasic' in eda_data.columns:
        merged_data['EDA_Phasic'] = eda_data['EDA_Phasic']
        print("Added EDA_Phasic")
    else:
        print("Warning: EDA_Phasic column not found in EDA file")
        merged_data['EDA_Phasic'] = np.nan

    # เพิ่มข้อมูล PPG (ตามชื่อ columns จริงจากภาพ)
    ppg_columns_map = {
        'HF_n_PG': ['HF_n_PG_', 'HF_n_PG'],
        'LF_n_PG': ['LF_n_PG_f', 'LF_n_PG'],
        'LFHF_ratrio': ['LFHF_ratio', 'LFHF_ratrio'],
        'Total_PG_EB': ['Total_PG_EB']
    }

    for target_col, possible_names in ppg_columns_map.items():
        found = False
        for col_name in possible_names:
            if col_name in ppg_data.columns:
                merged_data[target_col] = ppg_data[col_name]
                found = True
                print(f"Found {col_name} for {target_col}")
                break
        if not found:
            print(f"Warning: {target_col} column not found in PPG file")
            merged_data[target_col] = np.nan

    # เพิ่มข้อมูล Temperature (Skintemp จาก column T1)
    if 'T1' in temp_data.columns:
        merged_data['Skintemp'] = temp_data['T1']
        print("Found T1 for Skintemp")
    else:
        print("Warning: T1 column not found in temperature file")
        # ลองหา columns อื่นที่เป็นไปได้
        temp_columns_possible = ['Temp', 'Temperature', 'Skintemp', 'Skin_Temperature']
        temp_found = False
        for col_name in temp_columns_possible:
            if col_name in temp_data.columns:
                merged_data['Skintemp'] = temp_data[col_name]
                temp_found = True
                print(f"Found {col_name} for Skintemp instead")
                break

        if not temp_found:
            print("Warning: No temperature column found, using NaN")
            merged_data['Skintemp'] = np.nan

    # เพิ่มข้อมูลส่วนบุคคล (ค่าคงที่)
    merged_data['Gender'] = gender_input
    merged_data['BMI'] = bmi_str
    merged_data['Age'] = age
    merged_data['PainLevel'] = painLevel

    # จัดลำดับ columns ตามที่ต้องการ
    desired_columns = [
        'EDA_Tonic', 'EDA_Phasic', 'HF_n_PG', 'LF_n_PG',
        'LFHF_ratrio', 'Total_PG_EB', 'Skintemp', 'Gender',
        'BMI', 'Age', 'PainLevel'
    ]

    # เรียงลำดับ columns
    merged_data = merged_data[desired_columns]

    # บันทึกไฟล์
    output_path = os.path.join(output_folder, output_filename)
    merged_data.to_csv(output_path, index=False)

    print(f"\nSuccessfully created: {output_path}")
    print(f"Final data shape: {merged_data.shape}")
    print(f"Final columns: {list(merged_data.columns)}")

except FileNotFoundError as e:
    print(f"File not found: {e}")

except Exception as e:
    print(f"Error occurred: {e}")