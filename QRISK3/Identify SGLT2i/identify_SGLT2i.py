
import pandas as pd
import os

def identify_sglt2i_prescriptions(chunk):
    """
    Identifies prescriptions for SGLT2 inhibitors in a given chunk of data.

    Args:
        chunk (pd.DataFrame): A chunk of the prescriptions dataframe.

    Returns:
        pd.DataFrame: A dataframe containing only the SGLT2i prescriptions.
    """
    sglt2i_drugs = ["Canagliflozin", "Dapagliflozin", "Empagliflozin", "Ertugliflozin"]
    return chunk[chunk['drug'].str.contains('|'.join(sglt2i_drugs), case=False, na=False)]

if __name__ == "__main__":
    # Define input and output paths using relative directories
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    input_filename = os.path.join(data_dir, 'full_prescriptions.csv')
    output_filename = os.path.join(data_dir, 'sglt2i_prescriptions.csv')

    # Process the large CSV file in chunks
    chunksize = 10 ** 6  # 1 million rows at a time
    first_chunk = True
    for chunk in pd.read_csv(input_filename, chunksize=chunksize):
        sglt2i_chunk = identify_sglt2i_prescriptions(chunk)
        if not sglt2i_chunk.empty:
            sglt2i_chunk.to_csv(output_filename, index=False, mode='w' if first_chunk else 'a', header=first_chunk)
            first_chunk = False

    print(f"SGLT2i prescriptions saved to {output_filename}")
