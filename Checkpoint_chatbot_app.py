import nltk
import streamlit as st
import string
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Télécharger les ressources NLTK nécessaires (uniquement la première fois)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Charger le texte (met le bon chemin vers ton fichier ici si besoin)
with open('quantum_computer_chatbot_EN.txt', 'r', encoding='utf-8') as f:
    text = f.read().replace('\n', ' ')

# Tokenisation en phrases
sentences = sent_tokenize(text)

# Préparation du prétraitement
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(sentence):
    words = word_tokenize(sentence)
    words = [word.lower() for word in words if word.lower() not in stop_words and word not in string.punctuation]
    words = [lemmatizer.lemmatize(word) for word in words]
    return words

# Appliquer le prétraitement
processed_sentences = [preprocess(s) for s in sentences]
mapping = dict(zip([tuple(p) for p in processed_sentences], sentences))

# Fonction de similarité (Jaccard)
def get_most_relevant_sentence(query):
    query_processed = preprocess(query)
    if not query_processed:
        return "Veuillez entrer une question plus précise."

    max_similarity = 0
    best_sentence = "Désolé, je n'ai pas trouvé de réponse."

    for processed in processed_sentences:
        similarity = len(set(query_processed).intersection(processed)) / float(len(set(query_processed).union(processed)))
        if similarity > max_similarity:
            max_similarity = similarity
            best_sentence = mapping.get(tuple(processed), best_sentence)

    return best_sentence

# Fonction principale du chatbot
def chatbot(question):
    if not question.strip():
        return "Veuillez poser une question."
    return get_most_relevant_sentence(question)

# Interface Streamlit
def main():
    st.set_page_config(page_title="Chatbot Ordinateur Quantique", layout="centered")
    st.title("🧠 Chatbot - Ordinateur Quantique")
    st.write("Posez-moi une question sur les ordinateurs quantiques, je tenterai d'y répondre à partir du texte fourni.")

    question = st.text_input("Vous :")

    if st.button("Envoyer"):
        with st.spinner("Recherche de la réponse..."):
            response = chatbot(question)
            st.markdown(f"**Chatbot :** {response}")

if __name__ == "__main__":
    main()
