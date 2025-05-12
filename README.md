# 🧠 Understanding How Aya "Thinks"

> **Exploring how multilingual language models (specifically the Aya model) process language internally.**

---

<details>
<summary>📚 Table of Contents</summary>

- [Understanding How Aya "Thinks"](#🧠-understanding-how-aya-thinks)
- [🧭 Project Context](#🧭-project-context)
- [🚀 Model Clarification](#🚀-model-clarification)
- [✨ Summary Conclusions](#✨-summary-conclusions)
  - [1. Aya's English-First Bias (English-First Ratio)](#1-ayas-english-first-bias-english-first-ratio)
  - [2. Aya's Token Likelihoods (Cosine Similarity)](#2-ayas-token-likelihoods-cosine-similarity)
  - [3. Aya’s QA Performance (Translation Impact)](#3-ayas-qa-performance-translation-impact)
  - [4. Aya’s Sentiment Steering Effectiveness](#4-ayas-sentiment-steering-effectiveness)
  - [5. Aya's Factual Consistency Across Languages](#5-ayas-factual-consistency-across-languages)
- [✨ Final Summary Conclusion](#✨-final-summary-conclusion)
- [🗂️ Repository Structure](#🗂️-repository-structure)

</details>

---

## 🧭 Project Context

This research is part of the **Cohere Expedition: Aya Program**, an initiative to explore and better understand the capabilities, behaviors, and multilingual reasoning of the Aya model.  
We investigate how Aya internally processes information across languages and whether English-centric reasoning is prevalent.

You can read our [📄 Research Ideas and Framework here](https://docs.google.com/document/d/1F5JfcpT1whHLKkwHnCnAWsi6gJ1dTF_rdRq1q8bH_js/edit?usp=sharing) and [📊 Slides](https://docs.google.com/presentation/d/1o1uNmBi6_8UBuPOvolVvwMJ8cr2A4cgcjIcx17PxX1E/edit?usp=sharing).

---

## 🚀 Model Clarification

All experiments in this project use the **[Aya Expanse](https://docs.cohere.com/docs/aya-expanse)** (`c4ai-aya-expanse-8b`) model developed by Cohere.

Aya Expanse is described as a **highly performant multilingual model** capable of working with **23 languages**, providing strong capabilities in generation, reasoning, and understanding across diverse linguistic inputs.

---

## ✨ Summary Conclusions

### 1. Aya's English-First Bias (English-First Ratio)
- Aya shows **low overall English-centric bias** (English-First Ratio = **15%**).
- Certain languages like **English** and **Santhali** still exhibit strong English-first processing.

> 🎯 **Main Answer**: ❌ Aya does **not consistently "think" in English first**, *despite a small fraction of languages showing strong English-first processing*.

---

### 2. Aya's Token Likelihoods (Cosine Similarity)
- **Average cosine similarity** is **0.244**, showing moderate English similarity.
- **Only English** itself exhibits near-perfect similarity (cosine = **0.803**).

> 🎯 **Main Answer**: ❌ Aya **does not** consistently process languages with English-like token distributions.

---

### 3. Aya’s QA Performance (Translation Impact)
- **Performance strongest in English**.
- Aya's multilingual capabilities might still require more **adaptation** and **refinement** for non-English languages.

> 🎯 **Main Answer**: ✅ **Yes, translation improves Aya's QA performance** in the sense that **Aya performs better on English inputs**.

---

### 4. Aya’s Sentiment Steering Effectiveness
- **Stronger sentiment shifts in English**.
- Languages like **Telugu**, **Swahili** show poor steerability.

> 🎯 **Main Answer**: ✅ Aya is **much more steerable in English**.

---

### 5. Aya's Factual Consistency Across Languages

Aya's factual knowledge **is not fully neutral** across languages. **Factual consistency** varies significantly, with an overall consistency rate of just **34.5%**. Additionally, **semantic similarity** between answers is **moderate**, with an average embedding similarity of **0.674**, but still shows **notable variation** across languages.

> 🎯 **Main Answer**: ❌ Aya **does not** store factual knowledge in a fully language-neutral way.

---

Here’s the summary in well-structured English:

---

## ✨ Final Summary Conclusion

Aya exhibits **mixed English-first tendencies**, but overall, it does **not consistently rely on English** across all languages. While **English and some related languages** show stronger English-first processing, Aya demonstrates **adequate multilingual capabilities** for other languages.

* **English-first bias**: Aya shows a **low overall English-centric bias** (English-First Ratio = **15%**), with certain languages like **English** and **Santhali** displaying stronger English-first processing.
* **Token likelihoods**: The **average cosine similarity** is **0.244**, indicating moderate similarity to English. **Only English** itself exhibits near-perfect similarity (**0.803**).
* **QA performance**: Aya performs **best in English**, and while it shows promising multilingual capabilities, it still requires more **adaptation and refinement** for non-English languages.
* **Sentiment steering**: Aya is much **more steerable in English**, with stronger sentiment shifts. Languages like **Telugu** and **Swahili** show weaker steerability.
* **Factual consistency**: Aya's **factual knowledge is not fully neutral** across languages. **Factual consistency** varies, with an overall consistency rate of **34.5%**. Additionally, **semantic similarity** between answers is moderate (average similarity of **0.674**), though there is **notable variation** across languages.

> 🧠 **Final Verdict**: Aya **does not consistently think in English first**, though certain languages show English-centric behavior. While it handles many languages reasonably well, Aya’s **internal knowledge and behavior are not fully language-neutral**, with **notable performance and consistency differences across languages**.

---

## 🗂️ Repository Structure

````python
📁 understanding-how-Aya-thinks
 ├── 📁 experiments/              # Python scripts to run each experiment
 │    ├── factual_knowledge_experiment.py
 │    ├── first_n_tokens_experiment.py
 │    ├── likelihood_cosine_similarity_experiment.py
 │    ├── steering_experiment.py
 │    └── translation_experiment.py
 │
 ├── 📁 notebooks/                # Jupyter notebooks for analysis and visualization
 │    ├── factual_knowledge_analysis.ipynb
 │    ├── first_n_tokens_analysis.ipynb
 │    ├── likelihood_cosine_similarity_analysis.ipynb
 │    ├── steering_analysis.ipynb
 │    └── translation_analysis.ipynb
 │
 ├── 📁 results/                  # Output data generated from experiments
 │    ├── flores200_first_n_tokens_results.csv
 │    ├── flores200_likelihood_cosine_similarity_results.csv
 │    ├── mlama_factual_knowledge_results.csv
 │    ├── tydiqa_steering_results.csv
 │    └── xquad_translation_results.csv
 │
 ├── .env                         # Environment variables (API keys, config)
 ├── .gitignore                   # Specifies intentionally untracked files to ignore
 ├── LICENSE                      # Project open-source license
 ├── README.md                    # Main documentation file
 ├── requirements.txt             # Python package requirements
````

---

<br>

<p align="center">
  Made with ❤️ during <strong>Cohere Expedition: Aya Program</strong>
</p>