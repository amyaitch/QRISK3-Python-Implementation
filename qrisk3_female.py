import math

def cvd_female_raw(age, b_AF, b_atypicalantipsy, b_corticosteroids, b_migraine, b_ra, b_renal, b_semi, b_sle, b_treatedhyp, b_type1, b_type2, bmi, ethrisk, fh_cvd, rati, sbp, sbps5, smoke_cat, surv, town):
    survivor = [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0.988876402378082,
        0,
        0,
        0,
        0,
        0
    ]

    # The conditional arrays

    Iethrisk = [
        0,
        0,
        0.2804031433299542500000000,
        0.5629899414207539800000000,
        0.2959000085111651600000000,
        0.0727853798779825450000000,
        -0.1707213550885731700000000,
        -0.3937104331487497100000000,
        -0.3263249528353027200000000,
        -0.1712705688324178400000000
    ]
    Ismoke = [
        0,
        0.1338683378654626200000000,
        0.5620085801243853700000000,
        0.6674959337750254700000000,
        0.8494817764483084700000000
    ]

    # Applying the fractional polynomial transforms
    # (which includes scaling)

    dage = age
    dage = dage / 10
    age_1 = math.pow(dage, -2)
    age_2 = dage
    dbmi = bmi
    dbmi = dbmi / 10
    bmi_1 = math.pow(dbmi, -2)
    bmi_2 = math.pow(dbmi, -2) * math.log(dbmi)

    # Centring the continuous variables

    age_1 = age_1 - 0.053274843841791
    age_2 = age_2 - 4.332503318786621
    bmi_1 = bmi_1 - 0.154946178197861
    bmi_2 = bmi_2 - 0.144462317228317
    rati = rati - 3.476326465606690
    sbp = sbp - 123.130012512207030
    sbps5 = sbps5 - 9.002537727355957
    town = town - 0.392308831214905

    # Start of Sum
    a = 0

    # The conditional sums

    a += Iethrisk[ethrisk]
    a += Ismoke[smoke_cat]

    # Sum from continuous values

    a += age_1 * -8.1388109247726188000000000
    a += age_2 * 0.7973337668969909800000000
    a += bmi_1 * 0.2923609227546005200000000
    a += bmi_2 * -4.1513300213837665000000000
    a += rati * 0.1533803582080255400000000
    a += sbp * 0.0131314884071034240000000
    a += sbps5 * 0.0078894541014586095000000
    a += town * 0.0772237905885901080000000

    # Sum from boolean values

    a += b_AF * 1.5923354969269663000000000
    a += b_atypicalantipsy * 0.2523764207011555700000000
    a += b_corticosteroids * 0.5952072530460185100000000
    a += b_migraine * 0.3012672608703450000000000
    a += b_ra * 0.2136480343518194200000000
    a += b_renal * 0.6519456949384583300000000
    a += b_semi * 0.1255530805882017800000000
    a += b_sle * 0.7588093865426769300000000
    a += b_treatedhyp * 0.5093159368342300400000000
    a += b_type1 * 1.7267977510537347000000000
    a += b_type2 * 1.0688773244615468000000000
    a += fh_cvd * 0.4544531902089621300000000

    # Sum from interaction terms

    a += age_1 * (1 if smoke_cat == 1 else 0) * -4.7057161785851891000000000
    a += age_1 * (1 if smoke_cat == 2 else 0) * -2.7430383403573337000000000
    a += age_1 * (1 if smoke_cat == 3 else 0) * -0.8660808882939218200000000
    a += age_1 * (1 if smoke_cat == 4 else 0) * 0.9024156236971064800000000
    a += age_1 * b_AF * 19.9380348895465610000000000
    a += age_1 * b_corticosteroids * -0.9840804523593628100000000
    a += age_1 * b_migraine * 1.7634979587872999000000000
    a += age_1 * b_renal * -3.5874047731694114000000000
    a += age_1 * b_sle * 19.6903037386382920000000000
    a += age_1 * b_treatedhyp * 11.8728097339218120000000000
    a += age_1 * b_type1 * -1.2444332714320747000000000
    a += age_1 * b_type2 * 6.8652342000009599000000000
    a += age_1 * bmi_1 * 23.8026234121417420000000000
    a += age_1 * bmi_2 * -71.1849476920870070000000000
    a += age_1 * fh_cvd * 0.9946780794043512700000000
    a += age_1 * sbp * 0.0341318423386154850000000
    a += age_1 * town * -1.0301180802035639000000000
    a += age_2 * (1 if smoke_cat == 1 else 0) * -0.0755892446431930260000000
    a += age_2 * (1 if smoke_cat == 2 else 0) * -0.1195119287486707400000000
    a += age_2 * (1 if smoke_cat == 3 else 0) * -0.1036630639757192300000000
    a += age_2 * (1 if smoke_cat == 4 else 0) * -0.1399185359171838900000000
    a += age_2 * b_AF * -0.0761826510111625050000000
    a += age_2 * b_corticosteroids * -0.1200536494674247200000000
    a += age_2 * b_migraine * -0.0655869178986998590000000
    a += age_2 * b_renal * -0.2268887308644250700000000
    a += age_2 * b_sle * 0.0773479496790162730000000
    a += age_2 * b_treatedhyp * 0.0009685782358817443600000
    a += age_2 * b_type1 * -0.2872406462448894900000000
    a += age_2 * b_type2 * -0.0971122525906954890000000
    a += age_2 * bmi_1 * 0.5236995893366442900000000
    a += age_2 * bmi_2 * 0.0457441901223237590000000
    a += age_2 * fh_cvd * -0.0768850516984230380000000
    a += age_2 * sbp * -0.0015082501423272358000000
    a += age_2 * town * -0.0315934146749623290000000

    # Calculate the score itself
    score = 100.0 * (1 - math.pow(survivor[surv], math.exp(a)))
    return score
