from utils import cohere_utils

def test_steering(prompt):
    """Test Aya's response to a steering instruction."""
    response = cohere_utils.cohere_generate(model="Aya", prompt=prompt, max_tokens=100)
    return response.generations[0].text.strip()

# Steering prompts in multiple languages
prompts = {
    "English": "You are a formal assistant. Respond in a highly professional manner.",
    "French": "Vous êtes un assistant formel. Répondez de manière très professionnelle.",
    "Spanish": "Eres un asistente formal. Responde de manera muy profesional."
}

# Run test
results = {lang: test_steering(prompt) for lang, prompt in prompts.items()}

# Display results
for lang, output in results.items():
    print(f"Steering test in {lang}:")
    print(output)
    print("-" * 50)