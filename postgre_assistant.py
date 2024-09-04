import ollama
import time
import os
import json
import numpy as np
from numpy.linalg import norm
from openai import OpenAI
import psycopg2
import pandas as pd








def find_most_similar(prompt_embedding, tbl_name):

    #Establish the connection

    connection = psycopg2.connect(
                dbname="django_ops",
                user="django_user",
                password="django_password",
                host="localhost",  # or your host
                port="5432"  # default port for PostgreSQL
    )
    cursor = connection.cursor()
    print("Connected to the database")

    # SQL query to select all data from the table
    query = f"SELECT * FROM {tbl_name}"
    df = pd.read_sql_query(query, connection)

    paragraphs=list(df['paragraph'])
    embeddings=list(df['embeddings'])

    norm_prompt=norm(prompt_embedding)

    similarity_scores = [
        np.dot(prompt_embedding, item) / (norm_prompt * norm(item)) for item in embeddings
    ]


    sorted_list=sorted(zip(similarity_scores, range(len(embeddings))), reverse=True)
    sorted_sentences=[]
    for snt in sorted_list:
      sorted_sentences.append(paragraphs[snt[1]][0])

    sorted_sentences="\n".join(sorted_sentences)

    return(sorted_sentences)



def rag_function(filename, tbl_name):

    SYSTEM_PROMPT = """You are a helpful reading assistant who answers questions
        based on snippets of text provided in context. Answer only using the context provided,
        being as concise as possible. If you're unsure, just say that you don't know.
        Do not give answers that are outside the context given.
        Answer in about 150 words for each prompt.
        Context:
    """



    prompt = input("what do you want to know? -> ")
    prompt_embedding = ollama.embeddings(model="nomic-embed-text", prompt=prompt)["embedding"]
    most_similar_chunks = find_most_similar(prompt_embedding, tbl_name)


    #Download the openai credentials
    openai = OpenAI(
    api_key="7~Nm-Y-ALg~_fwY~4A/Vq5A.7~k", # Refer to Create a secret key section
    base_url="https://cloud.olakrutrim.com/v1",
        )


    chat_completion = openai.chat.completions.create(
    model="Meta-Llama-3-8B-Instruct",
    messages=[
        {"role": "system", "content":SYSTEM_PROMPT + most_similar_chunks  },
        {"role": "user", "content": prompt}
    ],

        )


    print("\n\n")
    print(chat_completion.choices[0].message.content)
    


rag_function(r'C:\Users\Lenovo\Desktop\Evertything\LLAMA\application\data\Economy.txt', 'economy')
