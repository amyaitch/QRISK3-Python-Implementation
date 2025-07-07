
import pandas as pd
import numpy as np
from qrisk3_male import cvd_male_raw
from qrisk3_female import cvd_female_raw

def calculate_qrisk3_combined():
    # Load the datasets
    try:
        predictions_df = pd.read_csv("/Users/amyaitchison/Desktop/MSc/Project/Practice/combined_sbp_cholesterol_predictions.csv")
        patients_df = pd.read_csv("/Users/amyaitchison/Desktop/MSc/Project/Practice/full_patients.csv")
        conditions_df = pd.read_csv("/Users/amyaitchison/Desktop/MSc/Project/Practice/subject_condition_presence.csv")
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

    # Explicitly convert relevant columns to numeric, coercing errors to NaN
    merged_df['age'] = pd.to_numeric(merged_df['age'], errors='coerce')
    merged_df['bmi'] = pd.to_numeric(merged_df['bmi'], errors='coerce')
    merged_df['sbp'] = pd.to_numeric(merged_df['sbp'], errors='coerce')
    merged_df['rati'] = pd.to_numeric(merged_df['rati'], errors='coerce')

    # Cap BMI values to a reasonable maximum (e.g., 70)
    merged_df['bmi'] = merged_df['bmi'].apply(lambda x: min(x, 70.0) if pd.notna(x) else x)

    # Initialize QRISK3 Score column with NaN
    merged_df['QRISK3_Score'] = np.nan

    # Define a mapping from cleaned disease names to QRISK3 function parameters
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

    # Iterate through each row to calculate QRISK3 score
    for index, row in merged_df.iterrows():
        subject_id = row['subject_id']
        gender = row['gender']
        age = row['age']
        bmi = row['bmi']
        sbp = row['sbp']
        rati = row['rati']

        # Check for NaN in critical input parameters for QRISK3
        if pd.isna(age) or pd.isna(bmi) or pd.isna(sbp) or pd.isna(rati) or pd.isna(gender):
            merged_df.loc[index, 'QRISK3_Score'] = np.nan
            continue

        # Smoking: set to 0 (non-smoker) as requested in original script
        smoke_cat = 0

        # Set surv to a fixed value (e.g., 10) - placeholder from original script
        surv_val = 10

        # Handle binary conditions: if missing, set to 0
        qrisk_params = {
            'b_AF': 0,
            'b_atypicalantipsy': 0,
            'b_corticosteroids': 0,
            'b_impotence2': 0,
            'b_migraine': 0,
            'b_ra': 0,
            'b_renal': 0,
            'b_semi': 0,
            'b_sle': 0,
            'b_treatedhyp': 0,
            'b_type1': 0,
            'b_type2': 0,
            'fh_cvd': 0, # Family history of CVD - not available in conditions_df, set to 0
            'sbps5': 0, # SBP variability - not available, set to 0
            'surv': surv_val,
            'town': 0 # Town/deprivation - not available, set to 0
        }

        # Populate qrisk_params from conditions_df
        for condition, param_name in condition_to_qrisk_param.items():
            if condition in row and pd.notna(row[condition]):
                qrisk_params[param_name] = row[condition]

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
                    ethrisk=0, # Ethnicity - not available, set to 0
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
                    ethrisk=0, # Ethnicity - not available, set to 0
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

            merged_df.loc[index, 'QRISK3_Score'] = score

        except Exception as e:
            merged_df.loc[index, 'QRISK3_Score'] = np.nan
            print(f"Error calculating QRISK3 for subject {subject_id}: {e}")

    # Select and save the results
    output_df = merged_df[['subject_id', 'mace_status', 'QRISK3_Score']]
    output_file_path = 'qrisk3_scores_combined.csv'
    output_df.to_csv(output_file_path, index=False)

    print(f"QRISK3 scores saved to {output_file_path}")

if __name__ == "__main__":
    calculate_qrisk3_combined()
