import cohere

co = cohere.Client("YOUR_COHERE_API_KEY")

def cohere_generate(model, prompt, max_tokens, return_likelihoods=None):
    return co.generate(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        return_likelihoods=return_likelihoods
    )