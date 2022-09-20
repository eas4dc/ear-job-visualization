# pip install mysql-connector
import os
import mysql.connector
import pandas as pd

job_id = 1536577
node_id = 'fcn23'


ear_etc_path = os.getenv("EAR_ETC")
ear_conf_file = os.path.join(ear_etc_path, "ear/ear.conf")

# Get DB connections info (!! check config of other cluster)
db = {}
for line in open(ear_conf_file):
    li = line.strip()
    if li.startswith("DBIp"):
        db['host'] = li.split("=")[1]
    if li.startswith("DBCommandsUser"):
        db['user'] = li.split("=")[1]
    if li.startswith("DBCommandsPassw"):
        db['password'] = li.split("=")[1]
    if li.startswith("DBDatabase"):
        db['database'] = li.split("=")[1]
    if li.startswith("DBPort"):
        db['port'] = li.split("=")[1]

# Connect to MariaDB
conn = mysql.connector.connect( user=db['user'], password=db['password'],
        host=db['host'], database=db['database'],port=db['port'])
cursor = conn.cursor()

query = "SELECT job_id, node_id, timestamp, event_type FROM Events WHERE job_id = %s AND node_id = %s"

cursor.execute(query, (job_id, node_id))
values = cursor.fetchall()
columns = [col[0] for col in cursor.description]
df = pd.DataFrame(values, columns=columns)
df.dropna(inplace=True)

# Close DB connection
cursor.close()
conn.close()
