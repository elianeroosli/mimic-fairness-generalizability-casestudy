SELECT 
    lab.pat_deid, 
    -- assign a STAY_ID to the event such that it is uniquely assigned to a sample
    CASE
        WHEN (lab.taken_time BETWEEN icu.icu_in_datetime AND icu.icu_out_datetime) 
        AND lab.pat_deid = icu.pat_deid THEN icu.stay_id
        ELSE NULL
    END AS stay_id,
    -- select and regroup events associated with the 17 physiological variables
    CASE
        WHEN lab.base_name = 'FIO2' THEN 'FIO2'
        WHEN lab.base_name = 'GLU' THEN 'GLU'
        WHEN lab.base_name = 'HEARTRATE' THEN 'HR'
        WHEN lab.base_name = 'HT' THEN 'HT'
        WHEN lab.base_name = 'O2SATA' THEN 'O2S'
        WHEN lab.base_name = 'PCTEMP' THEN 'TP'
        WHEN lab.base_name = 'WT' THEN 'WT'
        WHEN lab.base_name IN ('PH', 'PHA', 'PHV') THEN 'PH'
        ELSE 'NONE'
    END AS event_id,
    lab.taken_time, 
    lab.ord_value AS value, 
    lab.reference_unit AS unit      
        
FROM shc_lab_result_de lab, er_icu_stays icu
WHERE lab.pat_deid = icu.pat_deid
-- check the measure has been taken during the ICU stay
AND (lab.taken_time BETWEEN icu.icu_in_datetime AND icu.icu_out_datetime)
-- exclude all events other than the 17 variables
AND (lab.base_name IN ('FIO2', 'GLU', 'HEARTRATE', 'HT', 'O2SATA', 'PCTEMP', 'WT', 'PHA', 'PHV', 'PH'))
