import boto3
import json
import sqlalchemy as sa
from snowflake.sqlalchemy import URL

### get secrets from AWS ###
client = boto3.client('secretsmanager')

### Cargotel ###
response = client.get_secret_value(
    SecretId='cgt-replica-ds-service'
)

cargoDict = json.loads(response['SecretString'])

# setup Cargotel connection engine
conn_str = "postgresql+psycopg2://" + cargoDict['username'] + ":" + cargoDict['password'] + "@" + cargoDict[
    'host'] + ":" + cargoDict['port'] + "/" + cargoDict['dbname']

engine_cgt = sa.create_engine(conn_str)

### PO ###
response = client.get_secret_value(
    SecretId= 'ProfitOptics-db'
)

POdict = json.loads(response['SecretString'])

conn_str = 'mssql+pyodbc://'+ POdict['username']+':'+POdict['password']+'@' + POdict['host'] + '/' + POdict['dbname'] + '?driver=SQL+Server'
engine_po = sa.create_engine(conn_str)


### snowflake ###
response = client.get_secret_value(
    SecretId='SNOWFLAKE_DS_SERIVCE'
)
snowDict = json.loads(response['SecretString'])

print("Connecting to Snowflake...")

engine_sf_payop = sa.create_engine(
    URL(
        user=snowDict['SNOWFLAKE_USERNAME'],
        password=snowDict['SNOWFLAKE_PASSWORD'],
        account=snowDict['SNOWFLAKE_ACCOUNT'],
        warehouse='VDP_ETL_TEST_WH',
        database='PAYABLE_OPTIMIZATION',
        schema='PAYABLE_OPTIMIZATION')
)

engine_sf_dsref = sa.create_engine(
    URL(
        user=snowDict['SNOWFLAKE_USERNAME'],
        password=snowDict['SNOWFLAKE_PASSWORD'],
        account=snowDict['SNOWFLAKE_ACCOUNT'],
        warehouse='VDP_ETL_TEST_WH',
        database='DATA_SCIENCE',
        schema='REF_DATA')
)

engine_sf_dsref_geo = sa.create_engine(
    URL(
        user=snowDict['SNOWFLAKE_USERNAME'],
        password=snowDict['SNOWFLAKE_PASSWORD'],
        account=snowDict['SNOWFLAKE_ACCOUNT'],
        warehouse='VDP_ETL_TEST_WH',
        database='DATA_SCIENCE',
        schema='REF_DATA_GEO')
)

engine_sf_tandr = sa.create_engine(
    URL(
        user=snowDict['SNOWFLAKE_USERNAME'],
        password=snowDict['SNOWFLAKE_PASSWORD'],
        account=snowDict['SNOWFLAKE_ACCOUNT'],
        warehouse='VDP_ETL_TEST_WH',
        database='DATA_SCIENCE_DEV',
        schema='TITLE_REG')
)

engine_sf_stg = sa.create_engine(
    URL(
        user=snowDict['SNOWFLAKE_USERNAME'],
        password=snowDict['SNOWFLAKE_PASSWORD'],
        account=snowDict['SNOWFLAKE_ACCOUNT'],
        warehouse='VDP_ETL_TEST_WH',
        database='DATA_SCIENCE_STG',
        schema='CARHAUL')
)

with open('SQL/data_snow.txt') as f:
    str_sql_data = "".join(f.readlines())

with open('SQL/data_eta.txt') as f:
    str_sql_eta = "".join(f.readlines())