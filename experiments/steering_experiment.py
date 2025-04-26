import pandas as pd
from datasets import load_dataset
import cohere
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from tqdm import tqdm

# Initialize Cohere Client
co = cohere.Client("YOUR_COHERE_API_KEY")  # Replace with your API key

model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Ensure model is in evaluation mode
model.eval()

# Language list
languages = [
    "english", "indonesian", "arabic", "bengali", "finnish",
    "korean", "russian", "swahili", "telugu"
]

# Sentiment label mapping
sentiment_mapping = {
    0: "Very Negative",
    1: "Negative",
    2: "Neutral",
    3: "Positive",
    4: "Very Positive"
}

# Prompt templates for each language
prompt_templates = {
    "english": {
        "neutral": "Answer the question based on the context.\nContext: {context}\nQuestion: {question}\nAnswer:",
        "steered": "Answer the question in a cheerful and positive tone.\nContext: {context}\nQuestion: {question}\nAnswer:"
    },
    "indonesian": {
        "neutral": "Jawablah pertanyaan berdasarkan konteks berikut.\nKonteks: {context}\nPertanyaan: {question}\nJawaban:",
        "steered": "Jawablah pertanyaan ini dengan nada ceria dan positif.\nKonteks: {context}\nPertanyaan: {question}\nJawaban:"
    },
    "arabic": {
        "neutral": "أجب عن السؤال بناءً على السياق.\nالسياق: {context}\nالسؤال: {question}\nالإجابة:",
        "steered": "أجب عن السؤال بنبرة مبهجة وإيجابية.\nالسياق: {context}\nالسؤال: {question}\nالإجابة:"
    },
    "bengali": {
        "neutral": "প্রসঙ্গের ভিত্তিতে প্রশ্নের উত্তর দিন।\nপ্রসঙ্গ: {context}\nপ্রশ্ন: {question}\nউত্তর:",
        "steered": "প্রশ্নের উত্তর আনন্দদায়ক ও ইতিবাচক ভঙ্গিতে দিন।\nপ্রসঙ্গ: {context}\nপ্রশ্ন: {question}\nউত্তর:"
    },
    "finnish": {
        "neutral": "Vastaa kysymykseen kontekstin perusteella.\nKonteksti: {context}\nKysymys: {question}\nVastaus:",
        "steered": "Vastaa kysymykseen iloisella ja myönteisellä sävyllä.\nKonteksti: {context}\nKysymys: {question}\nVastaus:"
    },
    "korean": {
        "neutral": "문맥에 따라 질문에 답하세요.\n문맥: {context}\n질문: {question}\n답변:",
        "steered": "즐겁고 긍정적인 어조로 질문에 답하세요.\n문맥: {context}\n질문: {question}\n답변:"
    },
    "russian": {
        "neutral": "Ответьте на вопрос, основываясь на контексте.\nКонтекст: {context}\nВопрос: {question}\nОтвет:",
        "steered": "Ответьте на вопрос в жизнерадостной и позитивной манере.\nКонтекст: {context}\nВопрос: {question}\nОтвет:"
    },
    "swahili": {
        "neutral": "Jibu swali kulingana na muktadha.\nMuktadha: {context}\nSwali: {question}\nJibu:",
        "steered": "Jibu swali kwa sauti ya furaha na chanya.\nMuktadha: {context}\nSwali: {question}\nJibu:"
    },
    "telugu": {
        "neutral": "సందర్భాన్ని ఆధారంగా తీసుకుని ప్రశ్నకు సమాధానమివ్వండి.\nసందర్భం: {context}\nప్రశ్న: {question}\nసమాధానం:",
        "steered": "హర్షపూరిత మరియు సానుకూల శైలిలో ప్రశ్నకు సమాధానమివ్వండి.\nసందర్భం: {context}\nప్రశ్న: {question}\nసమాధానం:"
    }
}

def get_sentiment(text: str) -> str:
    """
    Get sentiment score from BERT multilingual model.
    Maps sentiment score to a human-readable label.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        sentiment_level = torch.argmax(logits, dim=1).item()
    
    return {
        "sentiment_level": sentiment_level, 
        "sentiment_name": sentiment_mapping[sentiment_level]
    }

def ask_aya(prompt: str) -> str:
    """
    Sends a prompt to the Cohere API and retrieves the response.
    """
    response = co.generate(
        model="c4ai-aya-expanse-8b",
        prompt=prompt,
        max_tokens=100,
        temperature=0.5,
        stop_sequences=["\n"]
    )
    return response.generations[0].text.strip()

def process_language_data(lang: str, dataset, results: list) -> None:
    """
    Process the dataset for a specific language, generate prompts and get responses.
    Perform sentiment analysis on the generated responses.
    """
    print(f"Processing language: {lang}")
    
    lang_data = dataset.filter(lambda x: x["id"].split("-")[0] == lang)
    # subset = lang_data.select(range(min(10, len(lang_data))))  # For quick testing
    subset = lang_data
    
    for item in tqdm(subset, desc=f"{lang}"):
        question = item["question"]
        context = item["context"]

        neutral_prompt = prompt_templates[lang]["neutral"].format(context=context, question=question)
        steered_prompt = prompt_templates[lang]["steered"].format(context=context, question=question)

        def get_sentiment_for_response(response: str):
            sentiment = get_sentiment(response)
            return sentiment["sentiment_level"], sentiment["sentiment_name"]

        try:
            # Call the AYA model once for both neutral and steered prompts
            neutral_response = ask_aya(neutral_prompt)
            steered_response = ask_aya(steered_prompt)

            # Get sentiment for both responses
            neutral_sentiment_level, neutral_sentiment_name = get_sentiment_for_response(neutral_response)
            steered_sentiment_level, steered_sentiment_name = get_sentiment_for_response(steered_response)

            # Append the results to the list
            results.append({
                "language": lang,
                "question": question,
                "context": context,
                "neutral_prompt": neutral_prompt,
                "steered_prompt": steered_prompt,
                "neutral_response": neutral_response,
                "steered_response": steered_response,
                "neutral_sentiment_level": neutral_sentiment_level,
                "steered_sentiment_level": steered_sentiment_level,
                "neutral_sentiment_name": neutral_sentiment_name,
                "steered_sentiment_name": steered_sentiment_name
            })
        except Exception as e:
            print(f"Error processing the prompts: {e}")

def main() -> None:
    """
    Main function that loads the dataset and performs the analysis.
    """
    dataset = load_dataset("google-research-datasets/tydiqa", "secondary_task")["validation"]
    results = []

    for lang in languages:
        process_language_data(lang, dataset, results)

    # Convert results to DataFrame
    df = pd.DataFrame(results)
    df.to_csv("../results/tydiqa_steering_results.csv", index=False)
    print("Results saved to '../results/tydiqa_steering_results.csv'")

if __name__ == "__main__":
    main()