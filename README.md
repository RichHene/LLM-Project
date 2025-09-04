# 📚 Semantic and Emotion-Aware Book Recommendation System

![dashboard](/images/Capture.PNG)
![dashboard](/images/Capture2.PNG)

This project builds an **LLM-powered book recommendation system** that combines semantic similarity, category inference, and emotional tone analysis. The final system is deployed through an interactive **Gradio dashboard** where users can enter a description and receive tailored book recommendations.

---

## 🚀 Features

- **Semantic Search**  
  Uses OpenAI embeddings + Chroma vector database to retrieve books similar to a user’s query.

- **Category Inference**

  - Manual category mapping (Fiction, Nonfiction, Children’s Fiction/Nonfiction, etc.)
  - Zero-shot classification with `facebook/bart-large-mnli` to fill missing categories.

- **Emotion Analysis**

  - Sentence-level classification with `j-hartmann/emotion-english-distilroberta-base`.
  - Aggregates emotion scores (joy, sadness, anger, fear, surprise, disgust, neutral) at the book level.
  - Enables filtering recommendations by emotional tone.

- **Interactive Dashboard**
  - Built with **Gradio**.
  - User can filter by **category** and **emotional tone**.
  - Results shown as a **gallery of book covers with captions**.

---

## 🛠️ Tech Stack

- **Data**: [7k Books with Metadata (Kaggle)](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata)
- **Vector Database**: Chroma
- **Embeddings**: OpenAI embeddings
- **NLP Models**:
  - `facebook/bart-large-mnli` for zero-shot classification
  - `j-hartmann/emotion-english-distilroberta-base` for emotion tagging
- **Dashboard**: Gradio with `Glass` theme

---

## 📂 Workflow Overview

1. **Data Preparation**

   - Clean raw metadata → `books_cleaned.csv`
   - Add simplified categories → `books_with_categories.csv`
   - Add emotional tagging → `books_with_emotions.csv`

2. **Vector Database**

   - Export descriptions to `tagged_description.txt`
   - Load with LangChain `TextLoader`
   - Chunk with `CharacterTextSplitter`
   - Embed with `OpenAIEmbeddings` and store in Chroma

3. **Recommendation Function**

   - Semantic similarity search
   - Filtering by category (Fiction, Nonfiction, Children’s, etc.)
   - Sorting by emotional tone (Joy, Sad, Angry, Fear, Surprise)

4. **Dashboard**
   - Input: description query, category filter, tone filter
   - Output: Gallery with book covers and truncated captions

---

## 📊 Key Observations

- Many books lacked **descriptions, categories, or cover images**, requiring preprocessing and enrichment.
- **Description length** is a useful heuristic — books with fewer than ~25 words often lacked meaningful content.
- **Zero-shot classification** successfully recovered missing categories with reasonable accuracy (>80% on test samples).
- **Emotion tagging** revealed that most book descriptions leaned toward **neutral or joy**, with fewer strongly negative tones (anger, fear, sadness).
- Some categories (e.g., _Romance, Sci-Fi, Fantasy_) had too few samples for reliable classification in the dataset.
- Combining **semantic search + emotion filtering** provided more **personalized and nuanced recommendations** than either alone.

---

## 🖼️ Example Dashboard

Users can enter a description such as:

> _“A story about forgiveness and second chances.”_

Then select:

- Category: _Fiction_
- Tone: _Happy_

And receive a curated gallery of 16 recommendations with book covers and author details.
