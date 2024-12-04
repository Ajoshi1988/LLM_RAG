import dash
from dash import Dash, dcc, html, dash_table, Input, Output, State, callback, ctx, dash_table
import dash_bootstrap_components as dbc

import base64
import datetime
import io
import pandas as pd
from dash import callback_context
import numpy as np
import pandas as pd
import time

import ollama
import os
from numpy.linalg import norm
from openai import OpenAI
import psycopg2


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


#Function to extract the OCR from the snippet  image


###############################################################

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


############################################################################

def rag_function(filename, tbl_name, prompt):

    SYSTEM_PROMPT = """You are a helpful reading assistant who answers questions
        based on snippets of text provided in context. Answer only using the context provided,
        being as concise as possible. If you're unsure, just say that you don't know.
        Do not give answers that are outside the context given.
        Answer in about 150 words for each prompt.
        Context:
    """



    # prompt = input("what do you want to know? -> ")
    prompt_embedding = ollama.embeddings(model="nomic-embed-text", prompt=prompt)["embedding"]
    most_similar_chunks = find_most_similar(prompt_embedding, tbl_name)


    #Download the openai credentials
    openai = OpenAI(
    api_key="api_key", # Refer to Create a secret key section
    base_url="https://cloud.olakrutrim.com/v1",
        )


    chat_completion = openai.chat.completions.create(
    model="Meta-Llama-3-8B-Instruct",
    messages=[
        {"role": "system", "content":SYSTEM_PROMPT + most_similar_chunks  },
        {"role": "user", "content": prompt}
    ],

        )


    # print("\n\n")
    # print(chat_completion.choices[0].message.content)

    return(chat_completion.choices[0].message.content)



app = Dash(__name__, external_stylesheets=external_stylesheets)




app.layout = html.Div([

    html.H4(children='Chat with Documents', style={'textAlign':'center'}),
    html.Br(),

   html.Div([

    dbc.Select(
    id="select_the_doc",
    options=[
        {"label": "Economy", "value": "Economy"},
        {"label": "Artificial_Intelligence", "value": "Artificial_Intelligence"},
        {"label": "Q_computing", "value": "Q_computing", },
    ],

    style={'width':'1250px','justify':'center', 'border': '2px solid black' },

   ),

   ], style={
        'display': 'flex',
        'justify-content': 'center',
        'align-items': 'center',



    }),



    html.Br(),
    html.Div([
    dbc.Input(id="input_prompt", placeholder="Ask anything...", type="text",
     style={'textAlign':'center', 'width':'1250px','justify':'center', 'border': '2px solid black',}
     ),

     ],  style={
          'display': 'flex',
          'justify-content': 'center',
          'align-items': 'center',



      } ),

    html.Br(),


   html.Div([

     html.Button('Submit', id='gen_prompt', n_clicks=0, style={'color':'white', 'background-color':'blue', 'marginLeft':'700px'})

   ]),


    html.Br(),

    dcc.Loading(

    html.Div([




        dcc.Textarea(
            id='output_text_area',
            value='Prompt Answers',
            style={'width': '80%', 'height': 150, 'border': '2px solid black' },
        ),
        html.Div(id='textarea-example-output', style={'whiteSpace': 'pre-line'})



           ],style={ 'display': 'flex','justify-content': 'center','align-items': 'center',   }),


    )


  ])



@callback(
              Output('output_text_area', 'value'),
              State('select_the_doc', 'value'),
              State('input_prompt', 'value'),
              Input('gen_prompt', 'n_clicks'),
              prevent_initial_call=True,



              )
def generate_output(doc_selected, input_prompt, gen_clicks):

    if doc_selected=='Economy':
        filename=r'C:\Users\Lenovo\Desktop\Evertything\LLAMA\application\data\Economy.txt'
        tbl_name='economy'
    elif doc_selected=='Artificial_Intelligence':
        filename=r'C:\Users\Lenovo\Desktop\Evertything\LLAMA\application\data\Artificial_Intelligence.txt'
        tbl_name='artificial_intelligence'
    elif doc_selected=='Q_computing':
        filename=r'C:\Users\Lenovo\Desktop\Evertything\LLAMA\application\data\Q_computing.txt'
        tbl_name='q_computing'



    txt=rag_function(filename, tbl_name, input_prompt)
    return(txt)


if __name__ == '__main__':
    app.run(debug=True)
