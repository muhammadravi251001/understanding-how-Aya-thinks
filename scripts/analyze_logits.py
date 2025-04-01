from utils import cohere_utils

def analyze_logits(prompt, target_language):
    """Check if Aya generates English tokens before switching to the target language."""
    response = cohere_utils.cohere_generate(model="Aya", prompt=prompt, max_tokens=50, return_likelihoods="ALL")
    
    tokens = response.generations[0].token_likelihoods
    first_tokens = [token.token for token in tokens[:5]]  # Get first 5 tokens
    
    return first_tokens

# Example: Ask Aya in French, but check first tokens
prompt = "Quelle est la capitale de la France?"
tokens_generated = analyze_logits(prompt, "French")

print("First generated tokens:", tokens_generated)