This project provides a Python-based implementation of the QRISK3 cardiovascular disease score calculator. It enables users to perform QRISK3 assessments directly in Python. This is ideal for researchers working with large-scale patient-level datasets, without relying on the online tool.

Setup:
Place your input files in the data folder within the project directory or update the input file paths in the function to point to your desired locations before running the code.

Featuress include:
- QRISK3 score calculation: Fully automated risk score computation using the official QRISK3 algorithm.
- Clinical information extraction: Identifies relevant conditions and risk factors from prescription data (e.g. diabetes, SLE, corticosteroid use).
- Multiple imputation: Implements multiple imputation techniques to estimate missing values for systolic blood pressure and cholesterol ratios based on BMI.
- MACE event identification: Uses ICD codes to identify Major Adverse Cardiovascular Events (MACE) for study eligibility.

Licenses:
This work uses and depends on the QRISK3 algorithm, which is protected by Copyright 2017 ClinRisk Ltd. You must adhere to the licensing of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. This repository is provided for research and academic purposes only.

Tools:
Google Gemini was used in the processing of this repository. The software was used strictly for internal development purposes in accordance with its licensing terms, without redistribution or integration of any Gemini-generated software components into distributed tools or outputs.

Getting Started
To run the QRISK3 calculation, ensure your input CSV files are placed in the data/ folder inside the project directory:
- combined_sbp_cholesterol_predictions.csv
- full_patients.csv
- subject_condition_presence.csv
  
Alternatively, modify the file paths in the calculate_qrisk3_combined() function if your files are located elsewhere.
