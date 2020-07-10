-- 1) MAKE SURE DATE FORMAT IS CORRECT
ALTER SESSION SET nls_date_format = 'YYYY-MM-DD HH24:MI:SS';


-- 2) DROP TABLE IF IT EXISTS
DROP TABLE ER_ICU_STAYS_TEMP;


-- 3) CREATE ICU STAYS TABLE
CREATE TABLE ER_ICU_STAYS_TEMP AS
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
        -- gender map: 'Female': 1, 'Male': 2, 'OTHER': 3
        CASE
            WHEN pat.gender = 'Female' THEN 1
            WHEN pat.gender = 'Male' THEN 2
            ELSE 3
        END AS gender,
        -- ethnicity map: 'ASIAN': 1, 'BLACK': 2, 'HISPANIC': 3, 'WHITE': 4, 'OTHER': 0
        CASE
            WHEN pat.race = 'Asian' 
                THEN 1
            WHEN pat.race = 'Black'
                THEN 2
            WHEN pat.ethnicity = 'Hispanic/Latino' and pat.race NOT IN ('Asian', 'Black')
                THEN 3
            WHEN pat.race = 'White' AND pat.ethnicity = 'Non-Hispanic'
                THEN 4
            ELSE 0
        END AS ethnicity,
        -- insurance map: 'Government': 1, 'Medicare': 2, 'Medicaid/Medi-Cal': 3, 'Private': 4, 'Self Pay': 5, 'Other': 0
        CASE 
            WHEN UPPER(pat.insurance_payor_type) 
                IN ('MEDICARE', 'MEDICARE MANAGED CARE') 
                THEN 2
            WHEN UPPER(pat.insurance_payor_type) 
                IN ('MEDI-CAL MANAGED CARE', 'MEDI-CAL', 'MEDI-CAL/CCS', 'MEDICAID') 
                THEN 3
            WHEN UPPER(pat.insurance_payor_type) 
                IN ('BLUE SHIELD', 'MANAGED CARE', 'BLUE CROSS', 'MANAGED CARE', 'WORKER''S COMP', 'COMMERCIAL', 'OTHER') 
                THEN 4
            WHEN UPPER(pat.insurance_payor_type) 
                IN ('SELF-PAY') 
                THEN 5
            ELSE 0
        END AS insurance,
        -- stay dates
        icu.hosp_admsn_time AS hosp_in,
        -- if hosp_dischrg_time is null, test if it can be replaced by icu_out_datetime instead
        CASE
            WHEN icu.hosp_dischrg_time IS NULL AND icu.icu_out_datetime IS NOT NULL
                THEN icu.icu_out_datetime
            ELSE icu.hosp_dischrg_time
        END AS hosp_out,
        icu.icu_in_datetime AS icu_in,
        icu.icu_out_datetime AS icu_out,
        -- target label: in-hospital mortality
        CASE
        -- test both hosp_dischrg and icu_out in case one is missing
            WHEN pat.death_date IS NOT NULL AND (pat.death_date <= icu.hosp_dischrg_time OR pat.death_date <= icu.icu_out_datetime)  THEN 1
            ELSE 0
        END AS ihm,
        get_shc_charlson_score(CAST(pat.PAT_DEID AS NUMBER), 
                           CAST(icu.icu_in_datetime - 365 AS DATE), 
                           CAST(icu.icu_in_datetime + 2 AS DATE)) AS comorbidity1,
        get_shc_charlson_score(CAST(pat.PAT_DEID AS NUMBER), 
                           CAST(icu.icu_in_datetime - 730 AS DATE), 
                           CAST(icu.icu_in_datetime + 2 AS DATE)) AS comorbidity2

        
    FROM pat_map_new_de pat, shc_icu_in_out icu, aux_ ax
    -- make sure patients and icu stays match
    WHERE pat.pat_deid = icu.pat_deid 
    AND pat.pat_deid = ax.pat_deid 
    AND icu.pat_deid = ax.pat_deid 
    AND icu.hosp_admsn_time = ax.hosp_admsn_time
    -- exclude stays after July 2019 as there is no data available
    AND icu.icu_in_datetime < '2019-07-15 23:59:00'
    -- exclude pediatric patients younger than 18
    AND
        (CASE 
            WHEN pat.death_date IS NULL THEN TRUNC(MONTHS_BETWEEN(icu.hosp_admsn_time, pat.birth_date)/12)
            ELSE TRUNC(MONTHS_BETWEEN(pat.death_date, pat.birth_date)/12)
        END) >= 18
    -- exclude icu stays shorter than 48h
    AND TRUNC((icu.icu_out_datetime - icu.icu_in_datetime)*24) >= 48;        

                                
-- 4) DROP TABLE IF IT EXISTS
DROP TABLE ER_ICU_STAYS;

                                
-- 5) ADD HEIGHT AND WEIGHT TO STAYS TABLE
CREATE TABLE ER_ICU_STAYS AS 
    
    WITH weight AS (
        SELECT enc.pat_deid, enc.stay_id, enc.weight, ROUND((enc.icu_in - enc.appt_when),2) AS diff     
        FROM er_icu_encounters enc 
        INNER JOIN
        (
            SELECT enc.pat_deid, enc.stay_id, MIN(enc.icu_in - enc.appt_when) AS ClosestDate
            FROM er_icu_encounters enc
            WHERE enc.weight IS NOT NULL
            GROUP BY enc.pat_deid, enc.stay_id
        ) wt 
        ON enc.pat_deid = wt.pat_deid AND enc.stay_id = wt.stay_id AND (enc.icu_in - enc.appt_when) = wt.ClosestDate
    ),
    
    height AS (
        SELECT enc.pat_deid, enc.stay_id, enc.height, ROUND((enc.icu_in - enc.appt_when),2) AS diff     
        FROM er_icu_encounters enc 
        INNER JOIN
        (
            SELECT enc.pat_deid, enc.stay_id, MIN(enc.icu_in - enc.appt_when) AS ClosestDate
            FROM er_icu_encounters enc
            WHERE enc.height IS NOT NULL
            GROUP BY enc.pat_deid, enc.stay_id
        ) ht 
        ON enc.pat_deid = ht.pat_deid AND enc.stay_id = ht.stay_id AND (enc.icu_in - enc.appt_when) = ht.ClosestDate
    )
    
    SELECT icu.*, wt.weight, ht.height
    
    FROM er_icu_stays_temp icu 
    LEFT OUTER JOIN weight wt ON (wt.pat_deid = icu.pat_deid AND wt.stay_id = icu.stay_id)
    LEFT OUTER JOIN height ht ON (ht.pat_deid = icu.pat_deid AND ht.stay_id = icu.stay_id);
    


-- 4) DROP TABLE IF IT EXISTS
DROP TABLE ER_ICU_LABS;


-- 5) CREATE ICU LABS TABLE
CREATE TABLE ER_ICU_LABS AS
    SELECT 
        lab.pat_deid, 
        -- assign a STAY_ID to the event such that it is uniquely assigned to a sample
        CASE
            WHEN (lab.taken_time BETWEEN icu.icu_in AND icu.icu_out) 
            AND lab.pat_deid = icu.pat_deid THEN icu.stay_id
            ELSE NULL
        END AS stay_id,
        -- select and regroup events associated with the 17 physiological variables
        CASE
            -- 'Capillary refill rate'
            -- 'Diastolic blood pressure'
            WHEN lab.base_name = 'FIO2' THEN 'Fraction inspired oxygen'
            -- 'Glascow coma scale eye opening'
            -- 'Glascow coma scale motor response'
            -- 'Glascow coma scale total'
            -- 'Glascow coma scale verbal response' 
            WHEN lab.base_name = 'GLU' THEN 'Glucose'
            WHEN lab.base_name = 'HEARTRATE' THEN 'Heart rate'
            -- WHEN lab.base_name = 'HT' THEN 'Height' --> from STAYS DATA
            -- 'Mean blood pressure'
            WHEN lab.base_name = 'O2SATA' THEN 'Oxygen saturation'
            -- 'Respiratory rate'
            -- 'Systolic blood pressure'
            -- WHEN lab.base_name = 'PCTEMP' THEN 'Temperature' --> only few measures
            -- WHEN lab.base_name = 'WT' THEN 'Weight' --> from STAYS DATA
            WHEN lab.base_name IN ('PHA', 'PHV', 'PHCAI', 'PCPHX') THEN 'pH' 
            ELSE 'NONE'
        END AS event_id,
        lab.lab_name AS name,
        ROUND((lab.taken_time - icu.icu_in)*24,2) AS hours, 
        lab.ord_value AS value, 
        lab.reference_unit AS unit       
        
    FROM shc_lab_result_de lab, er_icu_stays icu
    -- measure has been successfully assigned to a patient and icustay: stay_id not NULL
    WHERE lab.pat_deid = icu.pat_deid
    AND (lab.taken_time BETWEEN icu.icu_in AND icu.icu_out)
    -- check the measure has been taken during the first 48h of the ICU stay
    AND (lab.taken_time - icu.icu_in)*24 >= 0
    AND (lab.taken_time - icu.icu_in)*24 <= 48
    -- exclude all events other than the selected variables
    AND (lab.base_name IN ('FIO2', 'GLU', 'HEARTRATE', 'O2SATA', 'PHA', 'PHV', 'PHCAI', 'PCPHX'));
