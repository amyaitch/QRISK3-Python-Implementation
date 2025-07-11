import pandas as pd
import os

def filter_prescriptions_by_non_mace_subjects(
    non_mace_subjects_file,
    full_prescriptions_file,
    output_filtered_file,
    chunksize=100000
):
    # Load the subject IDs from the non-MACE subjects CSV
    non_mace_subjects_df = pd.read_csv(non_mace_subjects_file)
    non_mace_subject_ids = set(non_mace_subjects_df['subject_id'].unique())

    # Initialize flag for writing header once
    header_written = False

    # Process the large prescriptions file in chunks
    for chunk in pd.read_csv(full_prescriptions_file, chunksize=chunksize):
        filtered_chunk = chunk[chunk['subject_id'].isin(non_mace_subject_ids)]

        if not header_written:
            filtered_chunk.to_csv(output_filtered_file, mode='w', index=False, header=True)
            header_written = True
        else:
            filtered_chunk.to_csv(output_filtered_file, mode='a', index=False, header=False)

    print(f"Filtered prescriptions saved to {output_filtered_file}")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(__file__)
    DATA_DIR = os.path.join(BASE_DIR, 'data')

    # Define file paths relative to script location
    non_mace_subjects_file = os.path.join(DATA_DIR, 'predictions_for_non_mace_subjects_full_omr.csv')
    full_prescriptions_file = os.path.join(DATA_DIR, 'full_prescriptions.csv')
    output_filtered_file = os.path.join(DATA_DIR, 'filtered_prescriptions_non_mace.csv')

    filter_prescriptions_by_non_mace_subjects(
        non_mace_subjects_file,
        full_prescriptions_file,
        output_filtered_file
    )

