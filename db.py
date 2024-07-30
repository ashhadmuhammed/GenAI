from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.orm import sessionmaker

# Define connection parameters
DATABASE_URI = 'mysql+mysqlconnector://root:root@localhost/emp'

# Create an engine to connect to the MySQL database
engine = create_engine(DATABASE_URI)

# Create a session
Session = sessionmaker(bind=engine)
session = Session()

# Create a metadata object to reflect tables
metadata = MetaData()
metadata.reflect(bind=engine)

# Access the Employees table
employees_table = metadata.tables['employees']

def fetch_employees():
    """Fetch all records from the Employees table"""
    with engine.connect() as connection:
        result = connection.execute(employees_table.select())
        for row in result:
            print(row)

def main():
    # Fetch and print employee records
    fetch_employees()
    # Close the session
    session.close()

if __name__ == "__main__":
    main()
