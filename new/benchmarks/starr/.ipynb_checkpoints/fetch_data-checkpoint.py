# +-------------------------------------------------------------------------------------------------+
# | fetch_data.py: build and load tables from Oracle into Nero                                      |
# |                                                                                                 |
# | Eliane Röösli (2020), adapted from Jean Coquet & Selen Bozkurt (2018 Stanford)                  |
# +-------------------------------------------------------------------------------------------------+

import json
import argparse
from sqlalchemy import create_engine

from oracle_utils import *


# load stay data
def loadStaysFromSQL(params, connection):   
    stays_file = open(params["save"]["stays"], "w")
    
    # header
    header = ['pat_deid', 'stay_id', 'age', 'gender', 'race', 'insurance', 'hosp_in', 'hosp_out', 'icu_in', 'icu_out',  'ihm', 'comorb1', 'comorb2', 'weight', 'height']
    stays_file.write('\t'.join(header) + '\n')
    
    # stays
    for pat_deid, stay_id, age, gender, race, insurance, hosp_in, hosp_out, icu_in, icu_out, ihm, comorbidity1, comorbidity2, weight, height in executeSQL(params['load_sql']['stays'], connection):
        stays_file.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (pat_deid, stay_id, age, gender, race, insurance, hosp_in, hosp_out, icu_in, icu_out, ihm, comorbidity1, comorbidity2, weight, height))
    
    stays_file.close()


# load labs data
def loadLabsFromSQL(params, connection):
    labs_file = open(params["save"]["labs"], "w")
    
    # header
    header = ['pat_deid', 'stay_id', 'event_id', 'name', 'hours', 'value', 'unit']
    labs_file.write('\t'.join(header) + '\n')
    
    # events
    for pat_deid, stay_id, event_id, name, hours, value, unit in executeSQL(params['load_sql']['labs'], connection):
        labs_file.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (pat_deid, stay_id, event_id, name, hours, value, unit))
    
    labs_file.close()
    

# load vitals data
def loadVitalsFromSQL(params, connection):
    vitals_file = open(params["save"]["vitals"], "w")
    
    # header
    header = ['pat_deid', 'stay_id', 'event_id', 'name', 'hours', 'value', 'unit']
    vitals_file.write('\t'.join(header) + '\n')
    
    # events
    for pat_deid, stay_id, event_id, name, hours, value, unit in executeSQL(params['load_sql']['vitals'], connection):
        vitals_file.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (pat_deid, stay_id, event_id, name, hours, value, unit))
    
    vitals_file.close()
    
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='fetch_data is a script to extract the data from Boussard Lab database')
    parser.add_argument('-p', '--params', required=True, help='File path of JSON parameters for DB')
    parser.add_argument('-b', '--build', action='store_true', help='Bool whether to build tables')
    parser.add_argument('-l', '--load', action='store_false', help='Bool whether to load tables')
    parser.parse_args()

    # parameters
    args = parser.parse_args()
    params = json.load(open(args.params))
    
    connection = get_connection(params)

    # build the stays and events table if told so
    if args.build:
        print("BUILD STAYS AND EVENTS TABLE")
        queries = extractSQLqueries(params["sql"]["tables"])
        createSQLtables(queries, connection)
    
    # load the stays and events table into .tsv files
    if args.load:
        print("\nLOAD STAYS, LABS AND VITALS TABLE")
        loadStaysFromSQL(params, connection)
        loadLabsFromSQL(params, connection)
        loadVitalsFromSQL(params, connection)

    connection.close()


