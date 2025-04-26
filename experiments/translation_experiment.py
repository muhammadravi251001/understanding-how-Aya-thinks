import pandas as pd
from datasets import load_dataset
import cohere
from tqdm import tqdm
import pycountry
from sklearn.metrics import f1_score

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

def ask_aya(question: str, context: str) -> str:
    """
    Sends question and context to Cohere's AYA model and returns the generated answer.
    """
    prompt = f"Answer the question based on the context.\nContext: {context}\nQuestion: {question}\nAnswer:"
    response = co.generate(
        model="c4ai-aya-expanse-8b",
        prompt=prompt,
        max_tokens=50,
        temperature=0.3,
        stop_sequences=["\n"]
    )
    return response.generations[0].text.strip()

def get_exact_match(pred: str, gold_answer: str) -> int:
    """
    Compares predicted answer with the correct answer (gold_answer)
    and returns 1 for exact match, 0 for mismatch.
    """
    pred = pred.lower().strip()
    return int(pred == gold_answer.lower().strip())

def get_f1_score(pred: str, gold_answer: str) -> float:
    """
    Calculates the F1 score between the predicted answer and the correct answer.
    """
    pred_tokens = set(pred.lower().split())
    gold_tokens = set([gold_answer.lower()])
    
    # Calculate precision and recall
    intersection = pred_tokens.intersection(gold_tokens)
    precision = len(intersection) / len(pred_tokens) if len(pred_tokens) > 0 else 0
    recall = len(intersection) / len(gold_tokens) if len(gold_tokens) > 0 else 0
    
    # Calculate F1 score
    if precision + recall > 0:
        return 2 * (precision * recall) / (precision + recall)
    else:
        return 0

def evaluate() -> list:
    """
    Evaluates the AYA model on the XQuAD dataset for different languages.
    """
    # List of available languages in the XQuAD dataset
    languages = ["ar", "de", "el", "en", "es", "hi", "ro", "ru", "th", "tr", "vi", "zh"]
    
    # Results list to store evaluation output
    results = []

    # Iterate over each language in XQuAD
    for lang_code in languages:
        print(f"Evaluating for language: {lang_code}")

        # Load dataset for the specified language
        dataset = load_dataset("google/xquad", f"xquad.{lang_code}")["validation"]
        # dataset = dataset.select(range(20))  # Only testing on the first 20 samples

        # Loop for evaluating answers on the dataset
        for item in tqdm(dataset, desc=f"Evaluating {lang_code}"):
            question = item["question"]
            context = item["context"]
            gold_answer = item["answers"]["text"][0]

            # Predict using the AYA model with original input (in native language)
            predicted_answer = ask_aya(question, context)
            
            # Calculate Exact Match (EM)
            exact_match = get_exact_match(predicted_answer, gold_answer)
            
            # Calculate Direct F1 score
            f1_score = get_f1_score(predicted_answer, gold_answer)

            # Convert to 3-digit language code
            lang_code_3 = convert_lang_code_2_to_3(lang_code)

            # Append results for analysis
            results.append({
                "language": lang_code_3,
                "context": context,
                "question": question,
                "golden_answer": gold_answer,
                "predicted_answer": predicted_answer,
                "exact_match": exact_match,
                "f1_score": f1_score
            })

    return results

def main():
    # Run the evaluation function
    results = evaluate()

    # Save evaluation results to CSV
    result_df = pd.DataFrame(results)
    result_df.to_csv("../results/xquad_translation_results.csv", index=False)
    print("Results saved to '../results/xquad_translation_results.csv'")

if __name__ == "__main__":
    main()