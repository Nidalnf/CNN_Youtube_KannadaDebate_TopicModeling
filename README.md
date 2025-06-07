# Kannada Debate Topic Modeling

This project performs topic modeling on 1000 YouTube comments from a CNN-News18 video titled "Language War Escalates", which covered a widely publicized incident involving a female SBI manager refusing to speak in Kannada. The aim is to extract dominant themes in public discourse using natural language processing (NLP) techniques.


**Why YouTube?**

- Massive engagement: YouTube comment sections of popular media channels offer the user capacity for real-time public reactions and debates.

- Anonymity + virality: Enables spontaneous, emotionally charged responses, ideal for analyzing public sentiment.

- Cross-regional audience: Unlike regional forums, YouTube draws commenters from across India, allowing for a diverse range of views.


**Why This Video?**

We selected the CNN-News18 video because:

- It had the highest number of views and engagement at the point of this analysis among news videos on this topic (as of June 2025).

- CNN-News18 has national reach, making the comment section representative of a broader Indian audience.

- Over 1000 comments were available for analysis — enough for meaningful topic modeling.


**Methodology**

- Used Python, `spaCy`, and `scikit-learn` libraries.
- Preprocessed comments:
  - Cleaned and lowercased text
  - Removed stopwords (including custom ones)
  - Lemmatized using `spaCy`
- Vectorized using TF-IDF (unigrams + bigrams)
- Applied Latent Dirichlet Allocation (LDA) to extract 4 topics
- Extracted top 10 keywords per topic for interpretation


**Topics & Interpretations**

**Topic 1**  know, kannada, english, language, speak, hindi, customer, hai, idiot, service 

**Language as Communication Breakdown** : This represents emotionally charged frustration (**Idiot**) over **customer service** and the use (**Speak**) of **language** in formal set ups. A degree of emphasis of language barriers (**Know, kannada, English**) in service settings. 

**2**  language, bank, speak, sbi, people, manager, regional, state, hindi, local 

**Institutional Language Policy** : Discussion on of **SBI’s** **language** stance. Discussions about the **manager**, the bank, and broader expectations of language use in public institutions. An 

 **3**  language, learn, hindi, people, kannada, speak, local, karnataka, local language, state  
 
 **Language Learning & Identity Assertion** : Comments arguing whether people should learn and **speak** the regional language. Many assert Kannada's legitimacy (**local language**) in Karnataka and push back against the idea of **Hindi** imposition. 

  **4**  india, language, right, common, sense, hindi, state, north, common sense, problem 
  
  **Pan-Indian Tensions & Rights Discourse** : Another emotionally charged topic that talks about linguistic **right**s, **Hindi** imposition, and the **North-South divide**. A mix of emotional (**common sense**) and rational appeals around fairness (**right**), nationalism (**India**), and identity. |


**Project Structure**


Kannada_Youtube_Topic_Modeling/
│
├── comments.txt # YouTube comments data
├── topic_modeling.py # Main Python script
├── requirements.txt # Dependencies
└── README.md # Project description and results


**How to Run**

1. Ensure `comments.txt` is in the same folder as `topic_modeling.py`.
2. Run the script using:

```bash
python topic_modeling.py


**Why This Matters**

This project combines computational methods with sociopolitical analysis to understand language politics in India. It is part of a broader learning journey in:

Natural Language Processing (NLP)

Unsupervised Machine Learning

Cultural Analytics

Digital Sociology / Humanities