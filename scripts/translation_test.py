from utils import cohere_utils

def ask_aya(prompt):
    """Generate response from Aya."""
    response = cohere_utils.cohere_generate(model="Aya", prompt=prompt, max_tokens=100)
    return response.generations[0].text.strip()

# Example question in French
question = "Quelle est la capitale du Canada?"

# Direct response in French
response_direct = ask_aya(question)

# Translated question to English
translated_question = "What is the capital of Canada?"
response_translated = ask_aya(translated_question)

print(f"Direct response: {response_direct}")
print(f"Response after translation: {response_translated}")