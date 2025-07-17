
import pandas as pd

def identify_sglt2i_prescriptions(chunk):
    """
    Identifies prescriptions for SGLT2 inhibitors in a given chunk of data.

    Args:
        chunk (pd.DataFrame): A chunk of the prescriptions dataframe.

    Returns:
        pd.DataFrame: A dataframe containing only the SGLT2i prescriptions.
    """
    sglt2i_drugs = ["Canagliflozin", "Dapagliflozin", "Empagliflozin", "Ertugliflozin"]
    # The drug name is in the 'drug' column. 
    # The regular expression below will look for any of the drugs in the sglt2i_drugs list, case-insensitive.
    return chunk[chunk['drug'].str.contains('|'.join(sglt2i_drugs), case=False, na=False)]

# Create a new csv file to store the results
output_filename = '/Users/amyaitchison/Desktop/MSc/Project/Practice/sglt2i_prescriptions.csv'

# Process the large CSV file in chunks
chunksize = 10 ** 6  # 1 million rows at a time
first_chunk = True
for chunk in pd.read_csv('/Users/amyaitchison/Desktop/MSc/Project/Practice/full_prescriptions.csv', chunksize=chunksize):
    sglt2i_chunk = identify_sglt2i_prescriptions(chunk)
    if not sglt2i_chunk.empty:
        if first_chunk:
            # For the first chunk, write with header
            sglt2i_chunk.to_csv(output_filename, index=False, mode='w', header=True)
            first_chunk = False
        else:
            # For subsequent chunks, append without header
            sglt2i_chunk.to_csv(output_filename, index=False, mode='a', header=False)

print(f"SGLT2i prescriptions saved to {output_filename}")
