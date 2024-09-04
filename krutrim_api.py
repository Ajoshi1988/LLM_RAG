

from openai import OpenAI


openai = OpenAI(
    api_key="7~Nm-Y-ALg~_fwY~4A/Vq5A.7~k", # Refer to Create a secret key section
    base_url="https://cloud.olakrutrim.com/v1",
)
  
  
SYSTEM_PROMPT = """You are a helpful reading assistant who answers questions 
        based on snippets of text provided in context. Answer only using the context provided, 
        being as concise as possible. If you're unsure, just say that you don't know.
        Answer in about 100 words for each prompt.
        Context:
        Chandra is talented data analyst, data researcher. He also has info on various topics of interest, loves travelling.
        
    """

chat_completion = openai.chat.completions.create(
    model="Meta-Llama-3-8B-Instruct",
    messages=[
        {"role": "system", "content":SYSTEM_PROMPT },
        {"role": "user", "content": "Tell me about Chandra"}
    ],
   
)
print(chat_completion.choices[0].message.content)

# for chunk in chat_completion:
#     print(chunk)
#     # Each chunk will be a dictionary with the content
    # print(chunk['choices'][0]['delta']['content'], end="")