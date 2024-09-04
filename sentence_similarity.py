# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer

# model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# sentence1 = "what are the career options at Walmart?"
# sentence2 = "Are there any job opprtunities at Walmart?"

# embeddings = model.encode([sentence1, sentence2])
# similarity = cosine_similarity([embeddings[0]], [embeddings[1]])

# print(f"Cosine Similarity: {similarity[0][0]}")


# from transformers import AutoTokenizer, AutoModel
# import torch
# from sklearn.metrics.pairwise import cosine_similarity


# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = AutoModel.from_pretrained("bert-base-uncased")

# sentences = ["How many days in a week", "How many holidays in a week?"]
# inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
# with torch.no_grad():
#     outputs = model(**inputs)

# embeddings = outputs.last_hidden_state.mean(dim=1)
# similarity = cosine_similarity(embeddings[0].numpy().reshape(1, -1), embeddings[1].numpy().reshape(1, -1))

# print(f"Similarity: {similarity[0][0]}")


# from sentence_transformers import SentenceTransformer, util

# model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# sentence1 = "What is your name?"
# sentence2 = "Who are you?."

# embeddings = model.encode([sentence1, sentence2])
# similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])

# print(f"SBERT Similarity: {similarity.item()}")
