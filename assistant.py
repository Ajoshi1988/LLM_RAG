import ollama
import chromadb
import numpy as np
import psycopg
from psycopg.rows import dict_row
import ast
from tqdm import tqdm
from colorama import Fore

client=chromadb.Client()
system_prompt=(

    'You are an AI assistant that has memory of every conversation you have ever had with this user.'
    'On every prompt from the user, the system has checked for any relevant messages you have done with user.'
    'If any embedded previous conversations are attached, use them for contex to responding to user,'
    'If the context is relevant and useful to responding. If the recalled conversation are irrelevant,'
    'disregrad speaking about them and respond normally as an AI assistant. Do not talk about recalling conversations.'
    'Just use any useful data from the previous conversations and respond normally as an intelligent AI assistant.'
)

# message_history=[
#      {
#          'id':1,
#          'prompt': 'What is my name?',
#          'response':' Your name is Aditya, also known as Hindu Hriday Samrat.'

         
#      },

#      {
#          'id':2,
#          'prompt': 'What is the cube root of 27?',
#          'response':'3'
         
#      },

#      {

#          'id':3,
#          'prompt': 'What kind of cats do I have?',
#          'response':'Anu and Meera, my lovely ones'


#      }



# ]

convo=[{'role':'system', 'content':system_prompt}]
DB_PARAMS={

'dbname':'memory_agent',
'user':'aditya',
'password': 'aditya',
'host': 'localhost',
'port':'5432'


}


def connect_db():
    conn=psycopg.connect(**DB_PARAMS)
    return conn


def fetch_conversations():
    conn=connect_db()
    with conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute('SELECT * FROM conversations')
        conversations=cursor.fetchall()
    conn.close()
    return conversations



def store_conversations(prompt, response):
    conn=connect_db()
    with conn.cursor() as cursor:
        cursor.execute(
            'INSERT INTO conversations (timestamp, prompt, response) VALUES (CURRENT_TIMESTAMP, %s, %s)',
            (prompt, response)
        )
        conn.commit()
    conn.close()


def remove_last_conversation():
    conn=connect_db()
    with conn.cursor() as cursor:
        cursor.execute('DELETE FROM conversations WHERE id= (SELECT MAX(id) FROM conversations)')
        cursor.commit()

    conn.close()



def stream_response(prompt):
    # convo.append({'role':'user', 'content':prompt})
    response=''
    stream=ollama.chat(model='qwen:1.8b', messages=convo, stream=True)
    print(Fore.LIGHTGREEN_EX +'\nAssistant:')

    for chunk in stream:
        content=chunk['message']['content']
        response+=content
        print(content, end='', flush=True)

    print('\n')
    store_conversations(prompt=prompt, response=response)
    convo.append({'role':'assistant', 'content':response})



def create_vector_db(conversations):
    vector_db_name='conversations'

    try:
        client.delete_collection(name=vector_db_name)
    except ValueError:

        pass

    vector_db=client.create_collection(name=vector_db_name)

    for c in conversations:
        serialized_convo= f"prompt : {c['prompt']} response: {c['response']}"
        response=ollama.embeddings(model='nomic-embed-text', prompt=serialized_convo)
        embedding=response['embedding']

        vector_db.add(
            ids=[str(c['id'])],
            embeddings=[embedding],
            documents=[serialized_convo]
        )




def retrieve_embeddings(queries, results_per_query=2):
    
    embeddings=set()

    for query in tqdm(queries, desc='Processing queries to vector database' ):
        response=ollama.embeddings(model='nomic-embed-text', prompt=query)
        query_embedding=response['embedding']

        vector_db=client.get_collection(name='conversations')
        results=vector_db.query(query_embeddings=[query_embedding], n_results=results_per_query)
        best_embeddings=results['documents'][0]

        for best in best_embeddings:
            if best not in embeddings:
                if 'yes' in classify_embedding(query=query, context=best):
                    embeddings.add(best)

    return embeddings





    # response=ollama.embeddings(model='nomic-embed-text', prompt=prompt)
    # prompt_embedding=response['embedding']

    # vector_db=client.get_collection(name='conversations')
    # results=vector_db.query(query_embeddings=[prompt_embedding], n_results=1)
    # best_embedding=results['documents'][0][0]

    # return best_embedding


def create_queries(prompt):
    query_msg=(

        'You are a first principle reasoning search query AI agent.'
        'Your list of search queries will be ran on an embedding database of all your conversations'
        'you have ever had with the user. With first principles create a Python list of queries to'
        'search the embedding database for any data that would be necessary to have access to in '
        'order to correctly respond to the prompt. Your response must be a Python list with no syntax errors. '
        'Do not explain anything and do not ever generate anything but a perfect syntax Python list'


    )

    query_convo=[
       {'role':'system', 'content': query_msg},
       {'role':'user', 'content': 'Write an email to insurance company and create a persuasive request for them to lower my monthly rate.'},
       {'role':'assistant', 'content': '["What is users name?", "Who is users current insurance provider?", "What is the monthly rate the user pays currently pays for the insurance company?"]'},
       {'role':'user', 'content':'How can I convert the speak function in my llama3 python voice assistant to use pyttsx3 insurance?'},
       {'role':'assistant', 'content': '["Llama3 voice assistant", "Python voice assistant", "OPenAI TTS", "openai speak"]'},
       
       {'role':'user', 'content':prompt}
    ]

    response=ollama.chat(model='qwen:1.8b', messages=query_convo)
    print(Fore.YELLOW + f'\nVector database queries: {response["message"]["content"]} \n')

    try:
        return ast.literal_eval(response['message']['content'])
    except:
        return [prompt]
    

def classify_embedding(query, context):
    classify_msg=(
        'Your are an embedding classification AI agent. Your input will be a prompt and one embedded chunk of text.'
        "You will not respond as an AI assistant. You will respond in 'yes' or 'no'. "
        'Determine whether the context contains data that directly is related to the search query. '
        'If the context is seemingly exactly what the search query needs, respond "yes" if it is anything but directly'
        'related respond "no". Do not respond "yes" unless the content is highly relevant to the search query.'
    )

    classify_convo=[
        {'role':'system', 'content':classify_msg},
        {'role':'user', 'content': 'SEARCH QUERY: What is the users name? \n\nEMBEDDED CONTEXT: You are aditya. How can I help you? '},
        {'role':'assistant', 'content':'yes'},
        {'role':'user', 'content':f'SEARCH QUERY: Llama3 Python voice Assistant \n\nEMBEDDED CONTEXT: Siri is a voice assistant'},
        {'role':'assistant', 'content':'no'},
        {'role':'user', 'content': f'SEARCH QUERY: {query} \n\nEMBEDDED CONTEXT: {context} '}
    ]

    response=ollama.chat(model='qwen:1.8b', messages=classify_convo)
    return response['message']['content'].strip().lower()


def recall(prompt):
    queries=create_queries(prompt=prompt)
    embeddings=retrieve_embeddings(queries=queries)
    convo.append({'role':'user', 'content': f'MEMORIES: {embeddings} \n\n USER PROMPT: {prompt}'})
    print(f'\n{len(embeddings)} message:response embeddings added for context.')



conversations=fetch_conversations()
create_vector_db(conversations=conversations)
# print(fetch_conversations())

while True:
    prompt=input(Fore.WHITE + 'USER: \n')
    recall(prompt=prompt)
    # context=retrieve_embeddings(prompt=prompt)
    # prompt=f'USER PROMPT: {prompt} \nCONTEXT FROM EMBEDDINGS: {context}'

    if prompt[:7].lower()=='/recall':
        prompt=prompt[8:]
        recall(prompt=prompt)
        stream_response(prompt=prompt)
    elif prompt[:7].lower()=='/forget':
        remove_last_conversation()
        convo=convo[:-2]
        print("\n")
    elif prompt[:9].lower()=='/memorize':
        prompt=prompt[10:]
        store_conversations(prompt=prompt, response='Memory Stored .')
        print("\n")

    else:
        convo.append({'role':'user', 'content':prompt})
        stream_response(prompt=prompt)


    stream_response(prompt=prompt)




