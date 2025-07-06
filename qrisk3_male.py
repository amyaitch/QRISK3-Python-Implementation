
import math

def cvd_male_raw(age, b_AF, b_atypicalantipsy, b_corticosteroids, b_impotence2, b_migraine, b_ra, b_renal, b_semi, b_sle, b_treatedhyp, b_type1, b_type2, bmi, ethrisk, fh_cvd, rati, sbp, sbps5, smoke_cat, surv, town):
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
        0.977268040180206,
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
        0.2771924876030827900000000,
        0.4744636071493126800000000,
        0.5296172991968937100000000,
        0.0351001591862990170000000,
        -0.3580789966932791900000000,
        -0.4005648523216514000000000,
        -0.4152279288983017300000000,
        -0.2632134813474996700000000
    ]
    
    Ismoke = [
        0,
        0.1912822286338898300000000,
        0.5524158819264555200000000,
        0.6383505302750607200000000,
        0.7898381988185801900000000
    ]

    # Applying the fractional polynomial transforms
    # (which includes scaling)

    dage = age
    dage = dage / 10
    age_1 = math.pow(dage, -1)
    age_2 = math.pow(dage, 3)
    dbmi = bmi
    dbmi = dbmi / 10
    bmi_2 = math.pow(dbmi, -2) * math.log(dbmi)
    bmi_1 = math.pow(dbmi, -2)

    # Centring the continuous variables

    age_1 = age_1 - 0.234766781330109
    age_2 = age_2 - 77.284080505371094
    bmi_1 = bmi_1 - 0.149176135659218
    bmi_2 = bmi_2 - 0.141913309693336
    rati = rati - 4.300998687744141
    sbp = sbp - 128.571578979492190
    sbps5 = sbps5 - 8.756621360778809
    town = town - 0.526304900646210

    # Start of Sum
    a = 0

    # The conditional sums

    a += Iethrisk[ethrisk]
    a += Ismoke[smoke_cat]

    # Sum from continuous values

    a += age_1 * -17.8397816660055750000000000
    a += age_2 * 0.0022964880605765492000000
    a += bmi_1 * 2.4562776660536358000000000
    a += bmi_2 * -8.3011122314711354000000000
    a += rati * 0.1734019685632711100000000
    a += sbp * 0.0129101265425533050000000
    a += sbps5 * 0.0102519142912904560000000
    a += town * 0.0332682012772872950000000

    # Sum from boolean values

    a += b_AF * 0.8820923692805465700000000
    a += b_atypicalantipsy * 0.1304687985517351300000000
    a += b_corticosteroids * 0.4548539975044554300000000
    a += b_impotence2 * 0.2225185908670538300000000
    a += b_migraine * 0.2558417807415991300000000
    a += b_ra * 0.2097065801395656700000000
    a += b_renal * 0.7185326128827438400000000
    a += b_semi * 0.1213303988204716400000000
    a += b_sle * 0.4401572174457522000000000
    a += b_treatedhyp * 0.5165987108269547400000000
    a += b_type1 * 1.2343425521675175000000000
    a += b_type2 * 0.8594207143093222100000000
    a += fh_cvd * 0.5405546900939015600000000

    # Sum from interaction terms

    a += age_1 * (1 if smoke_cat == 1 else 0) * -0.2101113393351634600000000
    a += age_1 * (1 if smoke_cat == 2 else 0) * 0.7526867644750319100000000
    a += age_1 * (1 if smoke_cat == 3 else 0) * 0.9931588755640579100000000
    a += age_1 * (1 if smoke_cat == 4 else 0) * 2.1331163414389076000000000
    a += age_1 * b_AF * 3.4896675530623207000000000
    a += age_1 * b_corticosteroids * 1.1708133653489108000000000
    a += age_1 * b_impotence2 * -1.5064009857454310000000000
    a += age_1 * b_migraine * 2.3491159871402441000000000
    a += age_1 * b_renal * -0.5065671632722369400000000
    a += age_1 * b_treatedhyp * 6.5114581098532671000000000
    a += age_1 * b_type1 * 5.3379864878006531000000000
    a += age_1 * b_type2 * 3.6461817406221311000000000
    a += age_1 * bmi_1 * 31.0049529560338860000000000
    a += age_1 * bmi_2 * -111.2915718439164300000000000
    a += age_1 * fh_cvd * 2.7808628508531887000000000
    a += age_1 * sbp * 0.0188585244698658530000000
    a += age_1 * town * -0.1007554870063731000000000
    a += age_2 * (1 if smoke_cat == 1 else 0) * -0.0004985487027532612100000
    a += age_2 * (1 if smoke_cat == 2 else 0) * -0.0007987563331738541400000
    a += age_2 * (1 if smoke_cat == 3 else 0) * -0.0008370618426625129600000
    a += age_2 * (1 if smoke_cat == 4 else 0) * -0.0007840031915563728900000
    a += age_2 * b_AF * -0.0003499560834063604900000
    a += age_2 * b_corticosteroids * -0.0002496045095297166000000
    a += age_2 * b_impotence2 * -0.0011058218441227373000000
    a += age_2 * b_migraine * 0.0001989644604147863100000
    a += age_2 * b_renal * -0.0018325930166498813000000
    a += age_2 * b_treatedhyp * 0.0006383805310416501300000
    a += age_2 * b_type1 * 0.0006409780808752897000000
    a += age_2 * b_type2 * -0.0002469569558886831500000
    a += age_2 * bmi_1 * 0.0050380102356322029000000
    a += age_2 * bmi_2 * -0.0130744830025243190000000
    a += age_2 * fh_cvd * -0.0002479180990739603700000
    a += age_2 * sbp * -0.0000127187419158845700000
    a += age_2 * town * -0.0000932996423232728880000

    # Calculate the score itself
    score = 100.0 * (1 - math.pow(survivor[surv], math.exp(a)))
    return score

