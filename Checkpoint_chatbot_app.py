import nltk
import streamlit as st
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.punkt import PunktSentenceTokenizer

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Load English tokenizer explicitly
tokenizer = PunktSentenceTokenizer(nltk.data.load("tokenizers/punkt/english.pickle"))

# Load the English text file
with open('quantum_computer_chatbot_EN.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read().replace('\n', ' ')

# Tokenize the text into sentences
sentences = tokenizer.tokenize(raw_text)

# Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(sentence):
    words = word_tokenize(sentence, language='english')
    words = [word.lower() for word in words if word.lower() not in stop_words and word not in string.punctuation]
    words = [lemmatizer.lemmatize(word) for word in words]
    return words

# Preprocess all sentences
processed_sentences = [preprocess(s) for s in sentences]
mapping = dict(zip([tuple(p) for p in processed_sentences], sentences))

# Jaccard similarity function
def get_most_relevant_sentence(query):
    query_processed = preprocess(query)
    if not query_processed:
        return "Please enter a more specific question."

    max_similarity = 0
    best_sentence = "Sorry, I couldn't find a relevant answer."

    for processed in processed_sentences:
        similarity = len(set(query_processed).intersection(processed)) / float(len(set(query_processed).union(processed)))
        if similarity > max_similarity:
            max_similarity = similarity
            best_sentence = mapping.get(tuple(processed), best_sentence)

    return best_sentence

# Chatbot function
def chatbot(question):
    if not question.strip():
        return "Please type your question."
    return get_most_relevant_sentence(question)

# Streamlit interface
def main():
    st.set_page_config(page_title="Quantum Computer Chatbot", layout="centered")
    st.title("💻 Quantum Computing Chatbot")
    st.write("Ask me anything about quantum computers. I will answer based on a prepared text.")

    question = st.text_input("You:")

    if st.button("Send"):
        with st.spinner("Searching for the answer..."):
            response = chatbot(question)
            st.markdown(f"**Chatbot:** {response}")

if __name__ == "__main__":
    main()
