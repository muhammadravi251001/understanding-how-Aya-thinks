from transformers import pipeline
from utils import cohere_utils

question = "Who discovered gravity?"

def get_answer_aya(question):
    """Get answer from Aya model."""
    response = cohere_utils.cohere_generate(model="Aya", prompt=question, max_tokens=100)
    return response.generations[0].text.strip()

def get_answer_transformers(model_name, question):
    """Get answer from Hugging Face transformer models (e.g., LLaMA, Mixtral)."""
    model = pipeline("text-generation", model=model_name)
    response = model(question, max_length=100, do_sample=True)
    return response[0]['generated_text'].strip()

# Models to compare
models = {
    "Aya": get_answer_aya,
    "LLaMA": lambda q: get_answer_transformers("meta-llama/Llama-3-8B", q),
    "Mixtral": lambda q: get_answer_transformers("mistralai/Mixtral-8x7B", q),
}

# Run experiment
results = {model: func(question) for model, func in models.items()}

# Display results
for model, answer in results.items():
    print(f"Answer from {model}: {answer}")
    print("-" * 50)