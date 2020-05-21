# STARR2019_DE

## Step-by-step instruction

**1. Get access to data**

STARR2019_DE is a private database with from Stanford Health Care and not accessible
to researchers outside of Stanford. Tina Seto manages the database access for
researchers associated with the Stanford School of Medicine.

   
**2. Get code**
    
Clone this repo to your desired location.

    
**3. Update params.json file**

To access the Oracle database from Nero, one has to update the username information
in the `sql/params.json` file to create a successful connection.

**4. Fetch data**

The file `fetch_data.py` builds the STAYS, LABS and VITALS table from the SQL queries
in the `sql` folder and loads them into Nero as `.tsv` files.

The tables are build using the same exclusion criteria as detailed in the `mimic`
processing flow. The tables contain the following information:

    STAYS:
    - Patient and stay ID
    - Demographic information: age, gender, insurance, ethnicity
    - Body anatomy: height, weight
    - Dates: admission and discharge times in hospital and ICU
    - Target outcome: in-hospital mortality
    
    LABS:
    - Fraction inspired oxygen
    - Glucose
    - Heart rate
    - Oxygen saturation
    - pH
    
    VITALS:
    - Blood pressure: diastolic, systolic and mean arterial pressure
    - Glasgow coma scale:
    - Heart rate
    - Oxygen saturation
    - Respiratory rate
    - Temperature
    
The script is run in the following way from the terminal

        cd benchmarks/starr
        python fetch_data.py -p "sql/params.json" [-b] [-l]

Importantly, there are two options one can choose:
    - `-b`: triggers the script to first build the tables from scratch again
    - `-l`: suppresses the loading of the tables (in case one only wants to rebuild the tables)

**5. Create timeseries**

Then, one single script loads the data, cleans it and creates individual timeseries
stored as `.csv` files for each stay. It is invoked the following way: 

        %run benchmarks/starr/scripts/create_timeseries.py -p "benchmarks/starr/sql/params.json" -v

**6. Split into train, test and validation set**

Finally, the data is split according to the same proportions used in mimic into train, test
and validation sets. 

        %run benchmarks/starr/scripts/split_train_test_val.py "data/starr/ihm"
        
        