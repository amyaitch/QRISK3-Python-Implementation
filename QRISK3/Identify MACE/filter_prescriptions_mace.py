import pandas as pd
import os

def filter_prescriptions_by_mace_subjects(
    mace_subjects_file="data/predictions_for_mace_subjects_full_omr.csv",
    prescriptions_file="data/full_prescriptions.csv",
    output_file="data/filtered_prescriptions_mace.csv",
    chunksize=100000
):
    # Check if input files exist
    if not os.path.exists(mace_subjects_file):
        raise FileNotFoundError(f"Missing input file: {mace_subjects_file}")
    if not os.path.exists(prescriptions_file):
        raise FileNotFoundError(f"Missing input file: {prescriptions_file}")

    # Load the subject IDs
    mace_subjects_df = pd.read_csv(mace_subjects_file)
    mace_subject_ids = set(mace_subjects_df['subject_id'].unique())

    # Filter in chunks and save
    header_written = False
    for chunk in pd.read_csv(prescriptions_file, chunksize=chunksize):
        filtered_chunk = chunk[chunk['subject_id'].isin(mace_subject_ids)]

        filtered_chunk.to_csv(
            output_file,
            mode='w' if not header_written else 'a',
            index=False,
            header=not header_written
        )

        header_written = True

    print(f"Filtered prescriptions saved to {output_file}")

if __name__ == "__main__":
    filter_prescriptions_by_mace_subjects()


