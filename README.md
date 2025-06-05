# Kannada Debate Topic Modeling

This project performs topic modeling on 1000 YouTube comments from the CNN News video titled "Language War Escalates". The goal is to identify key topics discussed in the comment section using NLP techniques.

## Project Description

- Used Python, spaCy, and scikit-learn libraries.
- Preprocessed comments by cleaning, lowercasing, and lemmatizing.
- Removed stopwords (including some custom ones relevant to the data).
- Vectorized text data using TF-IDF with unigrams and bigrams.
- Applied Latent Dirichlet Allocation (LDA) to extract 4 main topics.
- Output shows the top 10 keywords per topic.

## Results

🧠 **Topic #1:**  
know | kannada | english | language | speak | hindi | customer | hai | idiot | service

🧠 **Topic #2:**  
language | bank | speak | sbi | people | manager | regional | state | hindi | local

🧠 **Topic #3:**  
language | learn | hindi | people | kannada | speak | local | karnataka | local language | state

🧠 **Topic #4:**  
india | language | right | common | sense | hindi | state | north | common sense | problem

## How to Run

1. Place `comments.txt` file in the same directory.

2. Run the script using:  
   ```bash
   python topic_modeling.py
