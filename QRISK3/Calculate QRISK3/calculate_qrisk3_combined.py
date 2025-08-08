import pandas as pd
import numpy as np
import math
from typing import Union

def cvd_male_vectorized(df: pd.DataFrame) -> pd.Series:
    """
    Vectorized QRISK3 calculation for males
    """
    # Survivor function value for 10 years
    survivor = 0.977268040180206
    
    # Ethnicity risk multipliers
    Iethrisk = np.array([
        0, 0, 0.2771924876030827900000000, 0.4744636071493126800000000,
        0.5296172991968937100000000, 0.0351001591862990170000000,
        -0.3580789966932791900000000, -0.4005648523216514000000000,
        -0.4152279288983017300000000, -0.2632134813474996700000000
    ])
    
    # Smoking risk multipliers
    Ismoke = np.array([
        0, 0.1912822286338898300000000, 0.5524158819264555200000000,
        0.6383505302750607200000000, 0.7898381988185801900000000
    ])
    
    # Extract and process variables
    age = df['age'].values
    bmi = df['bmi'].values
    sbp = df['sbp'].values
    rati = df['cholesterol_ratio'].values
    ethrisk = df['ethnicity'].values
    smoke_cat = df['smoking_status'].values
    
    # Fractional polynomial transforms
    dage = age / 10
    age_1 = np.power(dage, -1) - 0.234766781330109
    age_2 = np.power(dage, 3) - 77.284080505371094
    
    dbmi = bmi / 10
    bmi_1 = np.power(dbmi, -2) - 0.149176135659218
    bmi_2 = np.power(dbmi, -2) * np.log(dbmi) - 0.141913309693336
    
    # Center continuous variables
    rati_c = rati - 4.300998687744141
    sbp_c = sbp - 128.571578979492190
    sbps5_c = df['sbp_variability'].fillna(0).values - 8.756621360778809
    town_c = df['townsend_score'].fillna(0).values - 0.526304900646210
    
    # Initialize sum
    a = np.zeros(len(df))
    
    # Add ethnicity and smoking effects
    a += Iethrisk[ethrisk.astype(int)]
    a += Ismoke[smoke_cat.astype(int)]
    
    # Continuous variables
    a += age_1 * -17.8397816660055750000000000
    a += age_2 * 0.0022964880605765492000000
    a += bmi_1 * 2.4562776660536358000000000
    a += bmi_2 * -8.3011122314711354000000000
    a += rati_c * 0.1734019685632711100000000
    a += sbp_c * 0.0129101265425533050000000
    a += sbps5_c * 0.0102519142912904560000000
    a += town_c * 0.0332682012772872950000000
    
    # Boolean conditions
    a += df['atrial_fibrillation'].values * 0.8820923692805465700000000
    a += df['atypical_antipsychotics'].fillna(0).values * 0.1304687985517351300000000
    a += df['corticosteroids'].fillna(0).values * 0.4548539975044554300000000
    a += df['erectile_dysfunction'].fillna(0).values * 0.2225185908670538300000000
    a += df['migraine'].fillna(0).values * 0.2558417807415991300000000
    a += df['rheumatoid_arthritis'].values * 0.2097065801395656700000000
    a += df['chronic_kidney_disease'].values * 0.7185326128827438400000000
    a += df['severe_mental_illness'].values * 0.1213303988204716400000000
    a += df['systemic_lupus_erythematosus'].values * 0.4401572174457522000000000
    a += df['treated_hypertension'].values * 0.5165987108269547400000000
    a += df['type1_diabetes'].values * 1.2343425521675175000000000
    a += df['type2_diabetes'].values * 0.8594207143093222100000000
    a += df['family_history_cvd'].values * 0.5405546900939015600000000
    
    # Interaction terms (major ones)
    smoke1 = (smoke_cat == 1).astype(float)
    smoke2 = (smoke_cat == 2).astype(float)
    smoke3 = (smoke_cat == 3).astype(float)
    smoke4 = (smoke_cat == 4).astype(float)
    
    a += age_1 * smoke1 * -0.2101113393351634600000000
    a += age_1 * smoke2 * 0.7526867644750319100000000
    a += age_1 * smoke3 * 0.9931588755640579100000000
    a += age_1 * smoke4 * 2.1331163414389076000000000
    a += age_1 * df['atrial_fibrillation'].values * 3.4896675530623207000000000
    a += age_1 * df['corticosteroids'].fillna(0).values * 1.1708133653489108000000000
    a += age_1 * df['erectile_dysfunction'].fillna(0).values * -1.5064009857454310000000000
    a += age_1 * df['migraine'].fillna(0).values * 2.3491159871402441000000000
    a += age_1 * df['chronic_kidney_disease'].values * -0.5065671632722369400000000
    a += age_1 * df['treated_hypertension'].values * 6.5114581098532671000000000
    a += age_1 * df['type1_diabetes'].values * 5.3379864878006531000000000
    a += age_1 * df['type2_diabetes'].values * 3.6461817406221311000000000
    a += age_1 * bmi_1 * 31.0049529560338860000000000
    a += age_1 * bmi_2 * -111.2915718439164300000000000
    a += age_1 * df['family_history_cvd'].values * 2.7808628508531887000000000
    a += age_1 * sbp_c * 0.0188585244698658530000000
    a += age_1 * town_c * -0.1007554870063731000000000
    
    # Age_2 interactions (complete set)
    a += age_2 * smoke1 * -0.0004985487027532612100000
    a += age_2 * smoke2 * -0.0007987563331738541400000
    a += age_2 * smoke3 * -0.0008370618426625129600000
    a += age_2 * smoke4 * -0.0007840031915563728900000
    a += age_2 * df['atrial_fibrillation'].values * -0.0003499560834063604900000
    a += age_2 * df['corticosteroids'].fillna(0).values * -0.0002496045095297166000000
    a += age_2 * df['erectile_dysfunction'].fillna(0).values * -0.0011058218441227373000000
    a += age_2 * df['migraine'].fillna(0).values * 0.0001989644604147863100000
    a += age_2 * df['chronic_kidney_disease'].values * -0.0018325930166498813000000
    a += age_2 * df['treated_hypertension'].values * 0.0006383805310416501300000
    a += age_2 * df['type1_diabetes'].values * 0.0006409780808752897000000
    a += age_2 * df['type2_diabetes'].values * -0.0002469569558886831500000
    a += age_2 * bmi_1 * 0.0050380102356322029000000
    a += age_2 * bmi_2 * -0.0130744830025243190000000
    a += age_2 * df['family_history_cvd'].values * -0.0002479180990739603700000
    a += age_2 * sbp_c * -0.0000127187419158845700000
    a += age_2 * town_c * -0.0000932996423232728880000
    
    # Calculate final score
    score = 100.0 * (1 - np.power(survivor, np.exp(a)))
    return pd.Series(score, index=df.index)


def cvd_female_vectorized(df: pd.DataFrame) -> pd.Series:
    """
    Vectorized QRISK3 calculation for females
    """
    # Survivor function value for 10 years
    survivor = 0.988876402378082
    
    # Ethnicity risk multipliers
    Iethrisk = np.array([
        0, 0, 0.2804031433299542500000000, 0.5629899414207539800000000,
        0.2959000085111651600000000, 0.0727853798779825450000000,
        -0.1707213550885731700000000, -0.3937104331487497100000000,
        -0.3263249528353027200000000, -0.1712705688324178400000000
    ])
    
    # Smoking risk multipliers
    Ismoke = np.array([
        0, 0.1338683378654626200000000, 0.5620085801243853700000000,
        0.6674959337750254700000000, 0.8494817764483084700000000
    ])
    
    # Extract and process variables
    age = df['age'].values
    bmi = df['bmi'].values
    sbp = df['sbp'].values
    rati = df['cholesterol_ratio'].values
    ethrisk = df['ethnicity'].values
    smoke_cat = df['smoking_status'].values
    
    # Fractional polynomial transforms
    dage = age / 10
    age_1 = np.power(dage, -2) - 0.053274843841791
    age_2 = dage - 4.332503318786621
    
    dbmi = bmi / 10
    bmi_1 = np.power(dbmi, -2) - 0.154946178197861
    bmi_2 = np.power(dbmi, -2) * np.log(dbmi) - 0.144462317228317
    
    # Center continuous variables
    rati_c = rati - 3.476326465606690
    sbp_c = sbp - 123.130012512207030
    sbps5_c = df['sbp_variability'].fillna(0).values - 9.002537727355957
    town_c = df['townsend_score'].fillna(0).values - 0.392308831214905
    
    # Initialize sum
    a = np.zeros(len(df))
    
    # Add ethnicity and smoking effects
    a += Iethrisk[ethrisk.astype(int)]
    a += Ismoke[smoke_cat.astype(int)]
    
    # Continuous variables
    a += age_1 * -8.1388109247726188000000000
    a += age_2 * 0.7973337668969909800000000
    a += bmi_1 * 0.2923609227546005200000000
    a += bmi_2 * -4.1513300213837665000000000
    a += rati_c * 0.1533803582080255400000000
    a += sbp_c * 0.0131314884071034240000000
    a += sbps5_c * 0.0078894541014586095000000
    a += town_c * 0.0772237905885901080000000
    
    # Boolean conditions
    a += df['atrial_fibrillation'].values * 1.5923354969269663000000000
    a += df['atypical_antipsychotics'].fillna(0).values * 0.2523764207011555700000000
    a += df['corticosteroids'].fillna(0).values * 0.5952072530460185100000000
    a += df['migraine'].fillna(0).values * 0.3012672608703450000000000
    a += df['rheumatoid_arthritis'].values * 0.2136480343518194200000000
    a += df['chronic_kidney_disease'].values * 0.6519456949384583300000000
    a += df['severe_mental_illness'].values * 0.1255530805882017800000000
    a += df['systemic_lupus_erythematosus'].values * 0.7588093865426769300000000
    a += df['treated_hypertension'].values * 0.5093159368342300400000000
    a += df['type1_diabetes'].values * 1.7267977510537347000000000
    a += df['type2_diabetes'].values * 1.0688773244615468000000000
    a += df['family_history_cvd'].values * 0.4544531902089621300000000
    
    # Key interaction terms
    smoke1 = (smoke_cat == 1).astype(float)
    smoke2 = (smoke_cat == 2).astype(float)
    smoke3 = (smoke_cat == 3).astype(float)
    smoke4 = (smoke_cat == 4).astype(float)
    
    a += age_1 * smoke1 * -4.7057161785851891000000000
    a += age_1 * smoke2 * -2.7430383403573337000000000
    a += age_1 * smoke3 * -0.8660808882939218200000000
    a += age_1 * smoke4 * 0.9024156236971064800000000
    a += age_1 * df['atrial_fibrillation'].values * 19.9380348895465610000000000
    a += age_1 * df['corticosteroids'].fillna(0).values * -0.9840804523593628100000000
    a += age_1 * df['migraine'].fillna(0).values * 1.7634979587872999000000000
    a += age_1 * df['chronic_kidney_disease'].values * -3.5874047731694114000000000
    a += age_1 * df['systemic_lupus_erythematosus'].values * 19.6903037386382920000000000
    a += age_1 * df['treated_hypertension'].values * 11.8728097339218120000000000
    a += age_1 * df['type1_diabetes'].values * -1.2444332714320747000000000
    a += age_1 * df['type2_diabetes'].values * 6.8652342000009599000000000
    a += age_1 * bmi_1 * 23.8026234121417420000000000
    a += age_1 * bmi_2 * -71.1849476920870070000000000
    a += age_1 * df['family_history_cvd'].values * 0.9946780794043512700000000
    a += age_1 * sbp_c * 0.0341318423386154850000000
    a += age_1 * town_c * -1.0301180802035639000000000
    
    # Age_2 interactions (complete set)
    a += age_2 * smoke1 * -0.0755892446431930260000000
    a += age_2 * smoke2 * -0.1195119287486707400000000
    a += age_2 * smoke3 * -0.1036630639757192300000000
    a += age_2 * smoke4 * -0.1399185359171838900000000
    a += age_2 * df['atrial_fibrillation'].values * -0.0761826510111625050000000
    a += age_2 * df['corticosteroids'].fillna(0).values * -0.1200536494674247200000000
    a += age_2 * df['migraine'].fillna(0).values * -0.0655869178986998590000000
    a += age_2 * df['chronic_kidney_disease'].values * -0.2268887308644250700000000
    a += age_2 * df['systemic_lupus_erythematosus'].values * 0.0773479496790162730000000
    a += age_2 * df['treated_hypertension'].values * 0.0009685782358817443600000
    a += age_2 * df['type1_diabetes'].values * -0.2872406462448894900000000
    a += age_2 * df['type2_diabetes'].values * -0.0971122525906954890000000
    a += age_2 * bmi_1 * 0.5236995893366442900000000
    a += age_2 * bmi_2 * 0.0457441901223237590000000
    a += age_2 * df['family_history_cvd'].values * -0.0768850516984230380000000
    a += age_2 * sbp_c * -0.0015082501423272358000000
    a += age_2 * town_c * -0.0315934146749623290000000
    
    # Calculate final score
    score = 100.0 * (1 - np.power(survivor, np.exp(a)))
    return pd.Series(score, index=df.index)


def calculate_qrisk3_population(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate QRISK3 scores for entire population dataset
    """
    # Validate required columns
    required_cols = [
        'subject_id', 'age', 'gender', 'bmi', 'sbp', 'cholesterol_ratio',
        'ethnicity', 'smoking_status', 'family_history_cvd', 'atrial_fibrillation',
        'rheumatoid_arthritis', 'chronic_kidney_disease', 'severe_mental_illness',
        'systemic_lupus_erythematosus', 'treated_hypertension', 'type1_diabetes',
        'type2_diabetes'
    ]
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Cap BMI at 70 (QRISK3 validation range)
    df = df.copy()
    df['bmi'] = df['bmi'].clip(upper=70.0)
    
    # Split by gender
    males = df[df['gender'] == 'M'].copy()
    females = df[df['gender'] == 'F'].copy()
    
    results = []
    
    # Calculate scores for males
    if len(males) > 0:
        males['qrisk3_score'] = cvd_male_vectorized(males)
        results.append(males[['subject_id', 'qrisk3_score']])
    
    # Calculate scores for females  
    if len(females) > 0:
        females['qrisk3_score'] = cvd_female_vectorized(females)
        results.append(females[['subject_id', 'qrisk3_score']])
    
    # Combine results
    if results:
        final_results = pd.concat(results, ignore_index=True)
        return final_results.sort_values('subject_id')
    else:
        return pd.DataFrame(columns=['subject_id', 'qrisk3_score'])
