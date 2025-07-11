import pandas as pd
import numpy as np
import os
from qrisk3_male import cvd_male_raw
from qrisk3_female import cvd_female_raw

def calculate_qrisk3_combined():
    # Define base path for data folder relative to this script
    base_path = os.path.join(os.path.dirname(__file__), 'data')

    # Load datasets with error handling
    try:
        predictions_df = pd.read_csv(os.path.join(base_path, "combined_sbp_cholesterol_predictions.csv"))
        patients_df = pd.read_csv(os.path.join(base_path, "full_patients.csv"))
        conditions_df = pd.read_csv(os.path.join(base_path, "subject_condition_presence.csv"))
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return

    # Merge predictions with patient demographics
    merged_df = pd.merge(predictions_df, patients_df[['subject_id', 'gender', 'anchor_age']], on='subject_id', how='left')

    # Merge with medical conditions
    merged_df = pd.merge(merged_df, conditions_df, on='subject_id', how='left')

    # Rename columns to match QRISK3 function parameters
    merged_df.rename(columns={
        'anchor_age': 'age',
        'BMI': 'bmi',
        'predicted_sbp': 'sbp',
        'predicted_cholesterol_ratio': 'rati'
    }, inplace=True)

    # Convert relevant columns to numeric, coerce errors to NaN
    for col in ['age', 'bmi', 'sbp', 'rati']:
        merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

    # Cap BMI at 70
    merged_df['bmi'] = merged_df['bmi'].apply(lambda x: min(x, 70.0) if pd.notna(x) else x)

    # Initialize QRISK3_Score column
    merged_df['QRISK3_Score'] = np.nan

    # Mapping condition names to QRISK3 parameters
    condition_to_qrisk_param = {
        'atrialfibrillation': 'b_AF',
        'chronic kidney disease': 'b_renal',
        'rheumatoidarthritis': 'b_ra',
        'sle': 'b_sle',
        'type 1 diabetes': 'b_type1',
        'type 2 diabetes': 'b_type2',
        'migraines': 'b_migraine',
        'steroidtablets': 'b_corticosteroids',
        'erectiledysfunction': 'b_impotence2',
        'atypicalantipsychotics': 'b_atypicalantipsy',
        'bloodpressuretreatment': 'b_treatedhyp',
        'severe mental illness': 'b_semi',
    }

    # Iterate through each patient row
    for index, row in merged_df.iterrows():
        gender = row['gender']
        age = row['age']
        bmi = row['bmi']
        sbp = row['sbp']
        rati = row['rati']

        # Skip if critical inputs missing
        if pd.isna(age) or pd.isna(bmi) or pd.isna(sbp) or pd.isna(rati) or pd.isna(gender):
            continue

        # Fixed values
        smoke_cat = 0
        surv_val = 10

        # Default QRISK3 parameters
        qrisk_params = {param: 0 for param in [
            'b_AF', 'b_atypicalantipsy', 'b_corticosteroids', 'b_impotence2',
            'b_migraine', 'b_ra', 'b_renal', 'b_semi', 'b_sle', 'b_treatedhyp',
            'b_type1', 'b_type2', 'fh_cvd', 'sbps5', 'town'
        ]}
        qrisk_params['surv'] = surv_val

        # Fill in parameters from conditions if present
        for condition, param in condition_to_qrisk_param.items():
            if condition in row and pd.notna(row[condition]):
                qrisk_params[param] = row[condition]

        # Calculate score based on gender
        try:
            if gender == 'M':
                score = cvd_male_raw(
                    age=age,
                    b_AF=qrisk_params['b_AF'],
                    b_atypicalantipsy=qrisk_params['b_atypicalantipsy'],
                    b_corticosteroids=qrisk_params['b_corticosteroids'],
                    b_impotence2=qrisk_params['b_impotence2'],
                    b_migraine=qrisk_params['b_migraine'],
                    b_ra=qrisk_params['b_ra'],
                    b_renal=qrisk_params['b_renal'],
                    b_semi=qrisk_params['b_semi'],
                    b_sle=qrisk_params['b_sle'],
                    b_treatedhyp=qrisk_params['b_treatedhyp'],
                    b_type1=qrisk_params['b_type1'],
                    b_type2=qrisk_params['b_type2'],
                    bmi=bmi,
                    ethrisk=0,
                    fh_cvd=qrisk_params['fh_cvd'],
                    rati=rati,
                    sbp=sbp,
                    sbps5=qrisk_params['sbps5'],
                    smoke_cat=smoke_cat,
                    surv=qrisk_params['surv'],
                    town=qrisk_params['town']
                )
            elif gender == 'F':
                score = cvd_female_raw(
                    age=age,
                    b_AF=qrisk_params['b_AF'],
                    b_atypicalantipsy=qrisk_params['b_atypicalantipsy'],
                    b_corticosteroids=qrisk_params['b_corticosteroids'],
                    b_migraine=qrisk_params['b_migraine'],
                    b_ra=qrisk_params['b_ra'],
                    b_renal=qrisk_params['b_renal'],
                    b_semi=qrisk_params['b_semi'],
                    b_sle=qrisk_params['b_sle'],
                    b_treatedhyp=qrisk_params['b_treatedhyp'],
                    b_type1=qrisk_params['b_type1'],
                    b_type2=qrisk_params['b_type2'],
                    bmi=bmi,
                    ethrisk=0,
                    fh_cvd=qrisk_params['fh_cvd'],
                    rati=rati,
                    sbp=sbp,
                    sbps5=qrisk_params['sbps5'],
                    smoke_cat=smoke_cat,
                    surv=qrisk_params['surv'],
                    town=qrisk_params['town']
                )
            else:
                score = np.nan

            merged_df.at[index, 'QRISK3_Score'] = score

        except Exception as e:
            print(f"Error calculating QRISK3 for subject {row['subject_id']}: {e}")
            merged_df.at[index, 'QRISK3_Score'] = np.nan

    # Save output
    output_df = merged_df[['subject_id', 'mace_status', 'QRISK3_Score']]
    output_file_path = os.path.join(base_path, 'qrisk3_scores_combined.csv')
    output_df.to_csv(output_file_path, index=False)
    print(f"QRISK3 scores saved to {output_file_path}")


if __name__ == "__main__":
    calculate_qrisk3_combined()

