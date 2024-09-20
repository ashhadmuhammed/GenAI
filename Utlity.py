import json
from sqlalchemy import create_engine, inspect
from chromadb.config import Settings
import chromadb
import Configs
# Define connection parameters
DATABASE_URI = 'mysql+mysqlconnector://root:root@localhost/emp'


# Define connetion parameters
username = Configs.username
password = Configs.password
host = Configs.host
port =Configs.port   # Default MySQL port
database = Configs.database


# Construct the DATABASE_URI for SAP HANA using hdbcli

DATABASE_URI = f'hana+hdbcli://{username}:{password}@{host}:{port}/?databaseName={database}'


client = chromadb.PersistentClient(path='./chroma')
import os
print(os.getcwd()) 
try:
    # Try to retrieve the collection
    collection = client.get_collection("chunk_collection")
    print("Collection loaded.")
except Exception as e:
    # If the collection doesn't exist, create it
    collection = client.create_collection("chunk_collection")
    print("New collection created.")


print("connected.............")
engine = create_engine(DATABASE_URI)
inspector = inspect(engine)

schema_info = []

# Extract schema information
tables = inspector.get_table_names()
tables=[table for table in tables if table.startswith('cw_itm_wn')]
for table_name in tables:
    print(table_name)
    table_info = {
        "table_name": table_name,
        "columns": inspector.get_columns(table_name),
        "primary_keys": inspector.get_pk_constraint(table_name)["constrained_columns"],
        "indexes": inspector.get_indexes(table_name),
        "foreign_keys": inspector.get_foreign_keys(table_name),
    }
    schema_info.append(table_info)


chunks = []
for table in schema_info:
    print(table)
    chunk_text = f"Table: {table['table_name']}\n"
    chunk_text += "Columns:\n"
    for col in table['columns']:
        chunk_text += f"  - {col['name']} ({col['type']})\n"
    chunk_text += f"Primary Keys: {', '.join(table['primary_keys'])}\n"
    if table['foreign_keys']:
        chunk_text += "Foreign Keys:\n"
        for fk in table['foreign_keys']:
            chunk_text += f"  - {fk['constrained_columns']} references {fk['referred_table']}.{fk['referred_columns']}\n"
    chunks.append(chunk_text)


relationship_chunks = []
for table in schema_info:   
    for fk in table['foreign_keys']:
        chunk_text = f"Foreign Key Relationship:\n"
        chunk_text += f"  - {fk['constrained_columns']} in {table['table_name']} references {fk['referred_table']}.{fk['referred_columns']}\n"
        relationship_chunks.append(chunk_text)
print(relationship_chunks)

metadata = [{"chunk": chunk, "table": schema_info[i]['table_name']} for i, chunk in enumerate(chunks)]


print("Meta Data:::::::::",metadata)
# # Store the schema info in a JSON file
# with open("schema_info.json", "w") as f:
#     json.dump(schema_info, f, indent=4)