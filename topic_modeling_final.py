import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction import text

# Load spaCy English model
nlp = spacy.load('en_core_web_sm')

# Load comments file
with open('comments.txt', 'r', encoding='utf-8') as f:
    comments = [line.strip() for line in f if line.strip()]

# Preprocess + lemmatize function
def preprocess_lemma(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # keep only letters and spaces
    doc = nlp(text)
    # remove stopwords + tokens of length <= 2
    lemmas = [token.lemma_ for token in doc if not token.is_stop and len(token) > 2]
    return ' '.join(lemmas)

cleaned = [preprocess_lemma(comment) for comment in comments]

# Custom stopwords to remove common but unhelpful words in your data
custom_stopwords = set([
    'language', 'speak', 'people', 'state', 'know', 'local', 'like', 'dont', 'said', 'use', 'one', 'also'
])

# Combine sklearn default English stopwords with custom ones
all_stopwords = text.ENGLISH_STOP_WORDS.union(custom_stopwords)

# Vectorize with TF-IDF and bigrams
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.8,
    stop_words=all_stopwords
)
X = vectorizer.fit_transform(cleaned)

# LDA with 4 topics
lda = LatentDirichletAllocation(n_components=4, random_state=42)
lda.fit(X)

# Print top 10 words for each topic
feature_names = vectorizer.get_feature_names_out()
for idx, topic in enumerate(lda.components_):
    top_words = [feature_names[i] for i in topic.argsort()[-10:][::-1]]
    print(f"\n🧠 Topic #{idx + 1}:")
    print(" | ".join(top_words))
