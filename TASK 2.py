import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required resources (only run once)
nltk.download('punkt')
nltk.download('stopwords')

# Sample FAQ data
faq_data = {
    "What is your return policy?": "Our return policy allows returns within 30 days of purchase.",
    "How do I track my order?": "You can track your order using the tracking number sent to your email.",
    "What payment methods are accepted?": "We accept credit/debit cards, UPI, and PayPal.",
    "How do I contact customer service?": "You can reach us at support@example.com or call 1800-123-456.",
    "Do you ship internationally?": "Yes, we offer worldwide shipping."
}

# Preprocessing function
def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]  # Remove punctuation
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)

# Prepare data
questions = list(faq_data.keys())
preprocessed_questions = [preprocess(q) for q in questions]

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(preprocessed_questions)

# Matching function
def get_best_answer(user_question):
    user_input = preprocess(user_question)
    user_vec = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vec, question_vectors)
    best_match_index = similarities.argmax()
    return faq_data[questions[best_match_index]]

# Chat loop
print("FAQ Chatbot (type 'exit' to quit)\n")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Bot: Goodbye! 👋")
        break
    answer = get_best_answer(user_input)
    print("Bot:", answer)
