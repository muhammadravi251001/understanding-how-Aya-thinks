import pandas as pd
from datasets import load_dataset
from langdetect import detect
import cohere
import time
from tqdm import tqdm
import pycountry

# Initialize Cohere Client
co = cohere.Client("YOUR_COHERE_API")  # Replace with your API key

def convert_lang_code_2_to_3(code_2: str) -> str:
    """
    Convert 2-letter ISO 639-1 language code to 3-letter ISO 639-2 code.
    Returns None if not found.
    """
    try:
        return pycountry.languages.get(alpha_2=code_2).alpha_3
    except:
        return None

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

def create_lang_script_mapping(columns):
    """
    Create a mapping from 3-letter language code to script based on column names.
    """
    mapping = {}
    for col in columns:
        try:
            parts = col.split("_")
            if len(parts) == 3 and parts[0] == "text":
                lang_2 = parts[1]
                script = parts[2]
                lang_3 = convert_lang_code_2_to_3(lang_2)
                if lang_3:
                    mapping[lang_3] = script
        except:
            continue
    return mapping

def analyze_first_n_tokens(df, lang_script_mapping, max_tokens=20, n_tokens=20):
    """
    Analyze the first n tokens from all text columns in the dataframe.
    """
    all_results = []

    for column in tqdm(df.columns, desc="Processing columns", unit="column"):
        for idx, row in tqdm(df.iterrows(), desc=f"Processing rows in {column}", unit="row", total=len(df)):
            text = row[column]
            if not text or not isinstance(text, str):
                continue

            try:
                token_probs = get_token_probabilities(text, max_tokens)
                if not token_probs:
                    continue

                sentence = "".join([token for token, _ in token_probs[:n_tokens]])

                try:
                    detected_lang = detect(sentence)
                    is_english = detected_lang == 'en'
                except:
                    detected_lang = None
                    is_english = False

                detected_lang_3 = convert_lang_code_2_to_3(detected_lang)
                detected_script = lang_script_mapping.get(detected_lang_3, "UNKNOWN")

                all_results.append({
                    "row_index": idx,
                    "original_lang": column.split("_")[1],
                    "original_script": column.split("_")[2],
                    "original_sentence": text,
                    "generated_first_n_tokens_sentence": sentence,
                    "detected_lang": detected_lang_3,
                    "detected_script": detected_script,
                    "is_english": is_english
                })
            except Exception as e:
                print(f"Error at row {idx}, column {column}: {e}")
                continue

    return pd.DataFrame(all_results)

def main():
    # Load FLORES dataset from Hugging Face
    flores_ds = load_dataset("muhammadravi251001/restructured-flores200")["test"]
    flores_df = pd.DataFrame(flores_ds)

    # Select a subset for testing
    flores_df = flores_df[:25]  # Change the number of subset you like

    # Build mapping of language (3-letter) to script from column names
    lang_script_mapping = create_lang_script_mapping(flores_df.columns)

    # Analyze tokens and enrich with detection and mapping
    result_df = analyze_first_n_tokens(flores_df, lang_script_mapping, max_tokens=20, n_tokens=20)

    # Save to CSV
    result_df.to_csv("../results/flores200_first_n_tokens_results.csv", index=False)
    print("Results saved to '../results/flores200_first_n_tokens_results.csv'")

if __name__ == "__main__":
    main()