import pandas as pd
import psycopg2
from datetime import datetime
import ollama



# open a file and return paragraphs
def parse_file(filename):
    with open(filename, encoding="utf-8-sig") as f:
        paragraphs = []
        buffer = []
        for line in f.readlines():
            line = line.strip()
            if line:
                buffer.append(line)
            elif len(buffer):
                paragraphs.append((" ").join(buffer))
                buffer = []
        if len(buffer):
            paragraphs.append((" ").join(buffer))
        return paragraphs


def create_data_table(tbl_name):

    #connect to the database

    try:
        connection = psycopg2.connect(
            dbname="django_ops",
            user="django_user",
            password="django_password",
            host="localhost",  # or your host
            port="5432"  # default port for PostgreSQL
        )
        cursor = connection.cursor()
        print("Connected to the database")


            # SQL query to create a table
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {tbl_name} (
            id SERIAL PRIMARY KEY,
            paragraph TEXT [],
            embeddings FLOAT[],
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """

        # Execute the query
        cursor.execute(create_table_query)

        # Commit the transaction
        connection.commit()

        print(f"Table {tbl_name} created successfully")




    except Exception as error:
        print(f"Error connecting to the database: {error} {tbl_name}")


    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()
        print("Table Creation Connection closed")





def save_data_embeddings(tbl_name,paragraph, embeddings):

    # connect to the database
    try:
    # Connect to your PostgreSQL database
        connection = psycopg2.connect(
            dbname="django_ops",
            user="django_user",
            password="django_password",
            host="localhost",  # or your host
            port="5432"  # default port for PostgreSQL
        )
        cursor = connection.cursor()
        print("Connected to the database")

        created_at = datetime.now()

        #Save the embeddings
        # SQL query to insert data


        insert_query = f"""
        INSERT INTO {tbl_name} (paragraph, embeddings, created_at)
        VALUES (%s, %s, %s)
        """

        # Insert the arrays into the table
        cursor.execute(insert_query, (paragraph, embeddings, created_at))

        # Commit the transaction
        connection.commit()
        print("Data inserted successfully")

    except Exception as error:
        print(f"Data Inserted Failed {tbl_name}")


    finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()
            print("Data Insertion Connection closed")





def create_embeddings_database(tbl_list, file_path_list):

    modelname="nomic-embed-text"
    #Loop in the tbl_list and create each of the tables, get the embeddings from the respective file and store the embedding data
    for tb,fl in zip(tbl_list, file_path_list):

        #create the table
        create_data_table(tb)
        paragraphs=parse_file(fl)

        for paragraph in paragraphs:

            #Get the embeddings
            embeddings = ollama.embeddings(model=modelname, prompt=paragraph)["embedding"]

           #save the embeddings in the respective app
            save_data_embeddings(tb,[paragraph], embeddings)


    print("Database embeddings of all the table created")

# create_embeddings_database(['Economy'], [r'C:\Users\Lenovo\Desktop\Evertything\LLAMA\application\data\Economy.txt'])

tbl_list=['Artificial_Intelligence', 'Economy', 'Q_computing']
file_path_list=[r'C:\Users\Lenovo\Desktop\Evertything\LLAMA\application\data\Artificial_Intelligence.txt',
                r'C:\Users\Lenovo\Desktop\Evertything\LLAMA\application\data\Economy.txt',
                r'C:\Users\Lenovo\Desktop\Evertything\LLAMA\application\data\Q_computing.txt', ]


create_embeddings_database(tbl_list, file_path_list)
