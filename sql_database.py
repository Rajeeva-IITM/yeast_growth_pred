#adapted from https://www.geeksforgeeks.org/connecting-to-sql-database-using-sqlalchemy-in-python/
# To store Optuna Results

from sqlalchemy import create_engine

#Define database credentials
user="rajeeva"
password="password"
host="localhost"
port=1234
database="test"

def get_connection():
    return create_engine(
        url=f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
    )
    
if __name__ == "__main__":
    
    try:
        
        #Get connection
        engine = get_connection()
        
        print(f"Connection established for host: {host} and user {user}")
        
    except Exception as ex:
        print("Connection not established due to:", ex)