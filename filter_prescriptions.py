
import pandas as pd

def filter_prescriptions_by_non_mace_subjects():
    # Load the subject IDs from predictions_for_non_mace_subjects_full_omr.csv
    non_mace_subjects_df = pd.read_csv("/Users/amyaitchison/Desktop/MSc/Project/Practice/predictions_for_non_mace_subjects_full_omr.csv")
    non_mace_subject_ids = set(non_mace_subjects_df['subject_id'].unique())

    # Define the input and output file paths
    input_prescriptions_path = "/Users/amyaitchison/Desktop/MSc/Project/Practice/full_prescriptions.csv"
    output_filtered_prescriptions_path = "/Users/amyaitchison/Desktop/MSc/Project/Practice/filtered_prescriptions_non_mace.csv"

    # Set chunk size for reading large CSV
    chunksize = 100000  # Adjust as needed based on memory

    # Initialize a flag to write header only once
    header_written = False

    # Read full_prescriptions.csv in chunks and filter
    for chunk in pd.read_csv(input_prescriptions_path, chunksize=chunksize):
        # Filter the chunk based on subject_id
        filtered_chunk = chunk[chunk['subject_id'].isin(non_mace_subject_ids)]

        # Write the filtered chunk to the output file
        if not header_written:
            filtered_chunk.to_csv(output_filtered_prescriptions_path, mode='w', index=False, header=True)
            header_written = True
        else:
            filtered_chunk.to_csv(output_filtered_prescriptions_path, mode='a', index=False, header=False)

    print(f"Filtered prescriptions saved to {output_filtered_prescriptions_path}")

if __name__ == "__main__":
    filter_prescriptions_by_non_mace_subjects()
