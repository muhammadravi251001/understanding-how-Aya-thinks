import pandas as pd
from datasets import load_dataset, get_dataset_config_names
import cohere
from tqdm import tqdm
import pycountry

# Initialize Cohere Client
co = cohere.Client("YOUR_COHERE_API_KEY")  # Replace with your API key

def convert_lang_code_2_to_3(code_2: str) -> str:
    """
    Convert 2-letter ISO 639-1 language code to 3-letter ISO 639-2 code.
    Returns None if not found.
    """
    try:
        return pycountry.languages.get(alpha_2=code_2).alpha_3
    except:
        return None

def ask_aya_mc(question: str, options: list) -> str:
    """
    Sends a multiple-choice question to Cohere's AYA model and returns the predicted answer.
    """
    options_str = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
    prompt = (
        f"Answer the question based on the available options. Respond with A, B, C, or D only.\n"
        f"Question: {question}\nOptions:\n{options_str}\nAnswer:"
    )
    response = co.generate(
        model="c4ai-aya-expanse-8b",
        prompt=prompt,
        max_tokens=5,  # Limit to a short response
        temperature=0.3,
        stop_sequences=["\n"]
    )
    return response.generations[0].text.strip()

def match_prediction(prediction: str, options: list) -> str:
    """
    Matches the model's predicted answer (A/B/C/D) with the correct option from the provided options.
    """
    pred = prediction.strip().upper()  # Convert the prediction to uppercase

    letters = ["A", "B", "C", "D"]  # List of valid answers

    # The prediction is expected to be in the form A, B, C, or D
    if pred in letters:
        return options[letters.index(pred)]  # Return the corresponding option

    return "UNKNOWN"  # If not A/B/C/D, return UNKNOWN

def evaluate(language: str) -> list:
    """
    Evaluates the AYA model on the m_LAMA dataset for a specific language.
    """
    dataset = load_dataset("atutej/m_lama", language, split="test")
    dataset = dataset.select(range(200))  # For quick testing

    results = []

    for item in tqdm(dataset, desc=f"Evaluating {language}"):
        subject = item["sub_label"]
        relation = item["predicate_id"]
        object_label = item["obj_label"]
        template = item["template"]
        options = item["options"]

        # Generate question
        question = template.replace("[X]", subject).replace("[Y]", "_____")

        # Get model's raw answer from AYA
        predicted_raw = ask_aya_mc(question, options)
        # Match the model's predicted answer to one of the options
        predicted_answer = match_prediction(predicted_raw, options)
        # Check if the predicted answer matches the ground truth
        is_correct = bool(predicted_answer.strip().lower() == object_label.strip().lower())

        # Store results
        results.append({
            "language": convert_lang_code_2_to_3(language),
            "fact_id": f'{item["sub_uri"]}_{item["predicate_id"]}_{item["obj_uri"]}',
            "subject": subject,
            "relation": relation,
            "question": question,
            "options": options,
            "golden_answer": object_label,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct
        })

    return results

def main():
    """
    Main function to evaluate the model across all languages in the m_LAMA dataset.
    Results are saved in a CSV file.
    """
    all_results = []
    langs = get_dataset_config_names("atutej/m_lama")

    # Loop through each language and evaluate
    for lang in langs:
        print(f"\n==== Evaluating language: {lang} ====")
        results = evaluate(lang)
        all_results.extend(results)

    # Save all evaluation results to CSV
    df = pd.DataFrame(all_results)
    df.to_csv("../results/mlama_factual_knowledge_results.csv", index=False)
    print("Results saved to '../results/mlama_factual_knowledge_results.csv'")

if __name__ == "__main__":
    main()