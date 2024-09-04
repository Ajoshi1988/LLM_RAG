import ollama
import time
import os
import json
import numpy as np
from numpy.linalg import norm
from openai import OpenAI


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



def save_embeddings(filename, embeddings):
    # create dir if it doesn't exist
    if not os.path.exists("embeddings"):
        os.makedirs("embeddings")
    # dump embeddings to json
    with open(f"embeddings/{filename}.json", "w") as f:
        json.dump(embeddings, f)


def load_embeddings(filename):
    # check if file exists
    if not os.path.exists(f"embeddings/{filename}.json"):
        return False
    # load embeddings from json
    with open(f"embeddings/{filename}.json", "r") as f:
        return json.load(f)


def get_embeddings(filename, modelname, chunks):
    # check if embeddings are already saved
    if (embeddings := load_embeddings(filename)) is not False:
        return embeddings
    # get embeddings from ollama
    embeddings = [
        ollama.embeddings(model=modelname, prompt=chunk)["embedding"]
        for chunk in chunks
    ]
    # save embeddings
    save_embeddings(filename, embeddings)
    return embeddings



def find_most_similar(needle, haystack):
    needle_norm = norm(needle)

    similarity_scores = [
        np.dot(needle, item) / (needle_norm * norm(item)) for item in haystack
    ]

    print(needle, haystack[0])
    print(np.dot(needle, haystack[0]))
    return sorted(zip(similarity_scores, range(len(haystack))), reverse=True)






def rag_function():

    SYSTEM_PROMPT = """You are a helpful reading assistant who answers questions
        based on snippets of text provided in context. Answer only using the context provided,
        being as concise as possible. If you're unsure, just say that you don't know.
        Do not give answers that are outside the context given.
        Answer in about 150 words for each prompt.
        Context:
    """

    filename = "data.txt"
    paragraphs = parse_file(filename)

    embeddings = get_embeddings(filename, "nomic-embed-text", paragraphs)



    prompt = input("what do you want to know? -> ")
    prompt_embedding = ollama.embeddings(model="nomic-embed-text", prompt=prompt)["embedding"]
    most_similar_chunks = find_most_similar(prompt_embedding, embeddings)[:5]


    # for item in most_similar_chunks:
    #     print(item)

    #     print(paragraphs[item[1]])
    #     print("\n")


    #Download the openai credentials
    openai = OpenAI(
    api_key="7~Nm-Y-ALg~_fwY~4A/Vq5A.7~k", # Refer to Create a secret key section
    base_url="https://cloud.olakrutrim.com/v1",
        )


    chat_completion = openai.chat.completions.create(
    model="Meta-Llama-3-8B-Instruct",
    messages=[
        {"role": "system", "content":SYSTEM_PROMPT + "\n".join(paragraphs[item[1]] for item in most_similar_chunks)  },
        {"role": "user", "content": prompt}
    ],

        )

    # response = ollama.chat(
    # model="qwen:1.8b",
    # messages=[
    #     {
    #         "role": "system",
    #         "content": SYSTEM_PROMPT
    #         + "\n".join(paragraphs[item[1]] for item in most_similar_chunks),
    #     },
    #     {"role": "user", "content": prompt},
    # ],
    # )

    print("\n\n")
    print(chat_completion.choices[0].message.content)


rag_function()
#####################
