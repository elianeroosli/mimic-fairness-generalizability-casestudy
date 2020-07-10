# +-------------------------------------------------------------------------------------------------+
# | oracle_utils.py: link from Oracle database to nero                                              |
# |                                                                                                 |
# | Eliane Röösli (2020), adapted from Jean Coquet (2019, Stanford)                                 |
# +-------------------------------------------------------------------------------------------------+

from sqlalchemy import create_engine
import getpass, re


# build connection to oracle database
def get_connection(params):
    pwd = getpass.getpass(prompt="Password: ")
    engine = create_engine("oracle+cx_oracle://{username}:{pwd}@{tnsname}".format(
        username=params["database"]["username"],
        pwd=pwd,
        tnsname=params["database"]["tnsname"]))
    return engine.connect()


#------------- execute a query from a file to load table ----------------------
def executeSQLFile(SQLFilePath, connection):
    query = ""
    with open(SQLFilePath) as f:
        query = f.read()
    return executeSQL(query, connection)

def executeSQL(query, connection):
    if query == '' or query == '\n':
        return None
    print('\tExecuting new query')
    return connection.execute(query)


#------------- execute queries from a file to build tables -------------------
def createSQLtables(queries, connection):
    for query in queries:
        executeSQL(query, connection)

def extractSQLqueries(queries_file):
    new_query = True
    queries = []
    query = ""
    with open(str(queries_file), "r") as f:
        for line in f:
            if re.search(r'.*select\n', line.lower()) and new_query:
                line = re.sub(r'^([\n \t]+)', '', line)
                query = line
                new_query = False
            else:
                query += line

            if re.search(r';', line):
                query = re.sub(r';([\n \t]*)$', '', query)
                queries.append(query)
                query = ""
                new_query = True

    return queries
