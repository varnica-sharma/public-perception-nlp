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
| SVM (best) | 41.5% |
| DistilBERT | 40.4% |
| Random Forest | — |
| Decision Tree | — |
| Naive Bayes | — |
| KNN | — |

### Thematic Clustering
Compared two approaches on 1,196 comments:
| Method | Result |
|--------|--------|
| K-Means (TF-IDF) | 94.6% cluster coherence |
| BERTopic (semantic embeddings) | Richer but overlapping topics |

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
