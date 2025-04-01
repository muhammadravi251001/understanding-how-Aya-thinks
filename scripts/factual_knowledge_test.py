from utils import cohere_utils

def get_factual_answer(question, language):
    """Ask Aya a factual question in different languages."""
    translated_question = f"Translate this question into {language} and answer factually: {question}"
    response = cohere_utils.cohere_generate(model="Aya", prompt=translated_question, max_tokens=100)
    return response.generations[0].text.strip()

# Example factual question
question = "Who discovered gravity?"

# Languages to test
languages = ["English", "French", "Spanish", "German", "Chinese"]

# Run experiment
results = {lang: get_factual_answer(question, lang) for lang in languages}

# Display results
for lang, answer in results.items():
    print(f"Answer in {lang}: {answer}")
    print("-" * 50)