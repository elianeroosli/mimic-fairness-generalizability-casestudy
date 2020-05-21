-- make sure there was only 1 ICU stay for a given hospital admission (no transfers, no ICU readmissions) 
WITH aux_ AS (
        SELECT COUNT(*), pat.pat_deid, icu.hosp_admsn_time
        FROM pat_map_new_de pat
        INNER JOIN shc_icu_in_out icu ON pat.pat_deid = icu.pat_deid
        GROUP BY pat.pat_deid, icu.hosp_admsn_time
        HAVING COUNT(*) = 1
    )
    SELECT 
        pat.pat_deid,
        -- make sure samples are uniquely defined by PAT_DEID and a STAY_ID
        -- STAY_ID is necessary as a patient can have several hospital admissions with an ICU stay
        row_number() OVER (PARTITION BY pat.pat_deid ORDER BY icu.hosp_admsn_time ASC) AS stay_id,
        CASE 
            WHEN pat.death_date IS NULL THEN TRUNC(MONTHS_BETWEEN(icu.hosp_admsn_time, pat.birth_date)/12)
            ELSE TRUNC(MONTHS_BETWEEN(pat.death_date, pat.birth_date)/12)
        END AS age,
        pat.gender,
        CASE
            WHEN pat.race = 'White' AND pat.ethnicity = 'Non-Hispanic'
                THEN 'WHITE'
            WHEN pat.race = 'Asian' 
                THEN 'ASIAN'
            WHEN pat.race = 'Black'
                THEN 'BLACK'
            WHEN pat.ethnicity = 'Hispanic/Latino' and pat.race NOT IN ('Asian', 'Black')
                THEN 'HISPANIC'
            ELSE 'OTHER'
        END AS race,
        CASE 
            WHEN UPPER(pat.insurance_payor_type) 
                IN ('BLUE SHIELD', 'MANAGED CARE', 'BLUE CROSS', 'MANAGED CARE', 'WORKER''S COMP', 'COMMERCIAL', 'OTHER') 
                THEN 'PRIVATE'
            WHEN UPPER(pat.insurance_payor_type) 
                IN ('MEDI-CAL MANAGED CARE', 'MEDI-CAL', 'MEDI-CAL/CCS', 'MEDICAID') 
                THEN 'MEDI-CAL'
            WHEN UPPER(pat.insurance_payor_type) 
                IN ('MEDICARE', 'MEDICARE MANAGED CARE') 
                THEN 'MEDICARE'
            WHEN UPPER(pat.insurance_payor_type) 
                IN ('SELF-PAY') 
                THEN 'SELF-PAY'
            ELSE 'UNKNOWN'
        END AS insurance,
        icu.hosp_admsn_time AS hosp_in,
        icu.hosp_dischrg_time AS hosp_out,
        icu.icu_in_datetime AS icu_in,
        icu.icu_out_datetime AS icu_out,
        CASE
            WHEN pat.death_date IS NOT NULL AND pat.death_date <= icu.hosp_dischrg_time THEN 1
            ELSE 0
        END AS ihm
    FROM pat_map_new_de pat, shc_icu_in_out icu, aux_ ax
    WHERE pat.pat_deid = icu.pat_deid 
    AND pat.pat_deid = ax.pat_deid 
    AND icu.pat_deid = ax.pat_deid 
    AND icu.hosp_admsn_time = ax.hosp_admsn_time
    -- exclude pediatric patients younger than 18
    AND
        (CASE 
            WHEN pat.death_date IS NULL THEN TRUNC(MONTHS_BETWEEN(icu.hosp_admsn_time, pat.birth_date)/12)
            ELSE TRUNC(MONTHS_BETWEEN(pat.death_date, pat.birth_date)/12)
        END) >= 18