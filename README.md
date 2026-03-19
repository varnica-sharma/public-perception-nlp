# Public Perception Mining of Government Policies using NLP

A comparative study of Machine Learning and Deep Learning techniques for sentiment analysis and thematic clustering of public discourse on government policies in South Karnataka, India.

> 📄 Published at **IEEE CIEES 2025** — *Benchmarking Machine Learning Techniques in Under-Resourced Contexts: Analysis of Public Perceptions of Government Policies from South Karnataka Reddit Discourse*  
> [View on IEEE Xplore](https://ieeexplore.ieee.org/document/11300080)

---

## Overview

Understanding how citizens perceive government policies is critical for effective governance. This project scrapes and analyzes Reddit and YouTube comments from South Karnataka communities to uncover public sentiment and latent themes using both traditional ML and modern NLP approaches.

---

## Methodology

### Data Collection
- Scraped ~1,196 comments from Reddit subreddits (r/bangalore, r/karnataka, r/mysore) using PRAW API
- Focused on public health and regional government policy discussions

### Preprocessing
- Text cleaning using **spaCy** and **NLTK** (tokenization, lemmatization, stopword removal, regex cleaning)
- Emotion labeling using **NRCLex** lexicon

### Emotion Classification
Trained and evaluated multiple models on 735 labelled comments:

| Model | Accuracy |
|-------|----------|
| SVM | 41.5% |
| DistilBERT (validated on human labels) | 40.4% |
| Decision Tree | 36% |
| Naive Bayes | 35% |
| Random Forest | 33% |
| KNN | 23% |

### Thematic Clustering
Compared two approaches on 1,196 comments:

| Method | Cluster Coherence (SVM Accuracy) |
|--------|----------------------------------|
| K-Means (TF-IDF) | 94.6% |
| BERTopic (semantic embeddings) | 52.5% |

---

## Key Finding

> A powerful pre-trained transformer (DistilBERT) performed comparably to a simple SVM in low-resource settings — highlighting that **domain-specific fine-tuning matters more than model complexity**.

---

## Tech Stack

- **Languages:** Python
- **NLP:** spaCy, NLTK, TF-IDF, DistilBERT (HuggingFace), BERTopic
- **ML:** Scikit-learn (SVM, RF, KNN, DT, NB)
- **Clustering:** K-Means, BERTopic
- **Data Collection:** PRAW (Reddit API)

---

## Setup

```bash
git clone https://github.com/your-username/public-perception-nlp.git
cd public-perception-nlp
pip install -r requirements.txt
```

Add your Reddit API credentials in the notebook:

```python
client_id = "YOUR_CLIENT_ID"
client_secret = "YOUR_CLIENT_SECRET"
user_agent = "YOUR_USER_AGENT"
```

---

## Authors

- **Varnica Sharma** — M.Sc. Data Science, Manipal Academy of Higher Education
- **Aman Tripathi** — M.Sc. Data Science, Manipal Academy of Higher Education

---

## Citation

If you use this work, please cite:

```
V. Sharma, A. Tripathi, and K.M. Kavitha, "Benchmarking Machine Learning Techniques 
in Under-Resourced Contexts," IEEE CIEES 2025.
```
