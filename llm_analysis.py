import openai
import numpy as np
from scipy.spatial.distance import cosine
import seaborn as sns
import matplotlib.pyplot as plt
import os

try:
    openai_api_key = os.environ["OPENAI_API_KEY"]
except KeyError:
    raise ValueError("The environment variable 'OPENAI_API_KEY' must be set with your OpenAI API key.")


#Example 1
base_prompt = ""
sentences = ["What is AI?", "What's AI?", "What AI?"]


def get_response_embedding(prompt, temperature = 0.5, model_generation = "text-davinci-003", model_embedding ="text-embedding-ada-002"):
    completions = openai.Completion.create(
        engine=model_generation,
        prompt=prompt,
        max_tokens=50,
        n=1,
        stop=None,
        temperature=temperature,
    )
    response = completions.choices[0].text.strip()
    response_to_convert = openai.Embedding.create(input=[response], model= model_embedding)
    response_embedding = response_to_convert["data"][0]["embedding"]
    return response, response_embedding


def get_prompt_embedding(prompt, model_embedding ="text-embedding-ada-002"):
    prompt_to_convert = openai.Embedding.create(input=[prompt], model= model_embedding)
    prompt_embedding = prompt_to_convert["data"][0]["embedding"]
    return prompt_embedding
prompt_embeddings = []
response_embeddings = []
responses = []
prompts = []

for sentence in sentences:
    prompt = base_prompt + sentence
    prompt_embedding = get_prompt_embedding(prompt)    
    response, response_embedding = get_response_embedding(prompt)
    prompt_embeddings.append(prompt_embedding)
    response_embeddings.append(response_embedding)
    responses.append(response)
    prompts.append(prompt)
    print(prompt)
    print(response)


def similarity(embedding1, embedding2):
    return 1 - cosine(embedding1, embedding2)

n = len(sentences)
prompt_similarity_matrix = np.zeros((n, n))
response_similarity_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        prompt_similarity_matrix[i, j] = similarity(prompt_embeddings[i], prompt_embeddings[j])
        response_similarity_matrix[i, j] = similarity(response_embeddings[i], response_embeddings[j])


print("Prompt similarity matrix:")
print(prompt_similarity_matrix)
print("\nResponse similarity matrix:")
print(response_similarity_matrix)




# Compute correlation between prompt similarity matrix and response similarity matrix
correlation_matrix = np.corrcoef(prompt_similarity_matrix.flatten(), response_similarity_matrix.flatten())[0, 1]

# Visualize the correlation using heatmaps
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(prompt_similarity_matrix, annot=True, cmap='coolwarm', ax=axs[0])
axs[0].set_title('Prompt Similarity Matrix')

sns.heatmap(response_similarity_matrix, annot=True, cmap='coolwarm', ax=axs[1])
axs[1].set_title('Response Similarity Matrix')

plt.suptitle(f'Correlation between Matrices: {correlation_matrix:.2f}')
plt.show()


# Function to compute cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Generate prompt and response embeddings
prompt_embeddings = [get_prompt_embedding(prompt) for prompt in prompts]

# Assuming get_response_embedding() returns response text and its embedding
responses, response_embeddings = zip(*[get_response_embedding(response) for response in responses])


# Compute similarities for adjacent prompts and responses
prompt_similarities = [cosine_similarity(prompt_embeddings[i], prompt_embeddings[i + 1]) for i in range(len(prompts) - 1)]
response_similarities = [cosine_similarity(response_embeddings[i], response_embeddings[i + 1]) for i in range(len(responses) - 1)]

# Visualize the correlation between prompt similarities and response similarities
plt.scatter(prompt_similarities, response_similarities)
plt.xlabel('Adjacent Prompt Similarities')
plt.ylabel('Adjacent Response Similarities')
plt.title('Correlation between Adjacent Prompt and Response Similarities')
plt.show()

import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

prompt_embeddings = [get_prompt_embedding(prompt) for prompt in prompts]
responses, response_embeddings = zip(*[get_response_embedding(response) for response in responses])

prompt_similarities = [cosine_similarity(prompt_embeddings[i], prompt_embeddings[i + 1]) for i in range(len(prompts) - 1)]
response_similarities = [cosine_similarity(response_embeddings[i], response_embeddings[i + 1]) for i in range(len(responses) - 1)]

average_prompt_similarity = np.mean(prompt_similarities)
average_response_similarity = np.mean(response_similarities)

print(f"Average similarity between adjacent prompts: {average_prompt_similarity}")
print(f"Average similarity between adjacent responses: {average_response_similarity}")


import numpy as np


def track_similarity_change(temperatures, prompts):
    avg_prompt_similarities = []
    avg_response_similarities = []

    for temperature in temperatures:
        prompt_embeddings = [get_prompt_embedding(prompt) for prompt in prompts]
        responses, response_embeddings = zip(*[get_response_embedding(prompt, temperature) for prompt in prompts])

        prompt_similarities = [cosine_similarity(prompt_embeddings[i], prompt_embeddings[i + 1]) for i in range(len(prompts) - 1)]
        response_similarities = [cosine_similarity(response_embeddings[i], response_embeddings[i + 1]) for i in range(len(responses) - 1)]

        avg_prompt_similarity = np.mean(prompt_similarities)
        avg_response_similarity = np.mean(response_similarities)

        avg_prompt_similarities.append(avg_prompt_similarity)
        avg_response_similarities.append(avg_response_similarity)

    return avg_prompt_similarities, avg_response_similarities

temperatures = [0.5, 1.0, 1.5, 2.0]

avg_prompt_similarities, avg_response_similarities = track_similarity_change(temperatures, prompts)

# Plot the results
import matplotlib.pyplot as plt

plt.plot(temperatures, avg_prompt_similarities, label="Average Prompt Similarity")
plt.plot(temperatures, avg_response_similarities, label="Average Response Similarity")
plt.xlabel("Temperature")
plt.ylabel("Similarity")
plt.legend()
plt.show()
