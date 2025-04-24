import pandas as pd
from datasets import load_dataset
from langdetect import detect
import cohere
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import time

# Initialize Cohere Client
co = cohere.Client("YOUR_COHERE_API")  # Replace with your API key

def get_token_probabilities(text: str, max_tokens: int = 20):
    """
    Sends text to Aya model and returns tokens with their likelihoods.
    """
    response = co.generate(
        model="c4ai-aya-expanse-8b",
        prompt=text,
        max_tokens=max_tokens,
        return_likelihoods="GENERATION"
    )
    tokens = response.generations[0].token_likelihoods
    token_probs = [(t.token, t.likelihood) for t in tokens]
    return token_probs

def calculate_cosine_similarity(text1: str, text2: str, max_tokens: int = 10):
    """
    Calculate cosine similarity between the likelihood distributions of two texts.
    """
    english_tokens = get_token_probabilities(text1, max_tokens)
    target_tokens = get_token_probabilities(text2, max_tokens)

    # Token extraction and its probability
    english_probs = [prob for token, prob in english_tokens]
    target_probs = [prob for token, prob in target_tokens]

    # Pad the shorter vector with zeros to ensure the vector dimensions are the same.
    max_len = max(len(english_probs), len(target_probs))
    english_probs = english_probs + [0] * (max_len - len(english_probs))
    target_probs = target_probs + [0] * (max_len - len(target_probs))

    # Cosine similarity between probability distributions
    cos_sim = cosine_similarity([english_probs], [target_probs])
    return cos_sim[0][0]

def analyze_cosine_similarity(df, max_tokens=10):
    """
    Analyze cosine similarity for the first n tokens from all text columns in the dataframe.
    """
    all_results = []

    for column in tqdm(df.columns, desc="Processing columns", unit="column", dynamic_ncols=True, mininterval=0.1):
        if 'eng_Latn' in column:
            for idx, row in tqdm(df.iterrows(), desc=f"Processing rows in {column}", unit="row", total=len(df), dynamic_ncols=True, mininterval=0.1):

                source_text = row[column]
                if not source_text or not isinstance(source_text, str):
                    continue

                try:
                    # For example, we compare each English column with every other column.
                    for target_column in df.columns:
                        target_text = row[target_column]
                        if target_text and isinstance(target_text, str):
                            cos_sim = calculate_cosine_similarity(source_text, target_text, max_tokens)
                            
                            # Add results to the list
                            all_results.append({
                                "row_index": idx,
                                "source_lang": column.split("_")[1],
                                "source_script": column.split("_")[2],
                                "source_text": source_text,
                                "target_lang": target_column.split("_")[1],
                                "target_script": target_column.split("_")[2],
                                "target_text": target_text,
                                "cosine_similarity": cos_sim
                            })
                except Exception as e:
                    print(f"Error at row {idx}, column {column}: {e}")
                    continue  # Skip this row and move to the next

    return pd.DataFrame(all_results)

def main():
    # Load FLORES dataset from Hugging Face
    flores_ds = load_dataset("muhammadravi251001/restructured-flores200")["test"]
    flores_df = pd.DataFrame(flores_ds)

    # Select a subset for testing
    flores_df = flores_df[:25]  # Change the number of subset you like

    result_df = analyze_cosine_similarity(flores_df, max_tokens=20)

    # Save to CSV
    result_df.to_csv("../results/flores200_likelihood_cosine_similarity_results.csv", index=False)
    print("Cosine similarity results saved to '../results/flores200_likelihood_cosine_similarity_results.csv'")

if __name__ == "__main__":
    main()