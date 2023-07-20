import streamlit as st

from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch
import re
from bs4 import BeautifulSoup

model = DistilBertForSequenceClassification.from_pretrained('output_model1_1')
tokenizer = DistilBertTokenizer.from_pretrained('output_model1_1')

file = open('stopwords.txt', 'r')
file_contents = file.read()
file.close()


def filtered_text(text):
    clean_text = remove_html_tags(text)
    filter_text = remove_stop_words(clean_text)
    return filter_text


def remove_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    clean_text = soup.get_text()
    return clean_text


def remove_stop_words(text):
    stop_words = file_contents
    tokens = re.findall(r'\w+', text)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(filtered_tokens)


def predict_class(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1)
    return predicted_class.item()


def main():
    st.title("Text Classification with DistilBERT")
    text = st.text_area("Enter a text:")
    if st.button("Classify"):
        if text.strip() != "":
            cleaned_text = filtered_text(text)
            predicted_class = predict_class(cleaned_text)
            if predicted_class == 0:
                st.success("Review : Negative")
            else:
                st.success("Review : Positive")
        else:
            st.warning("Please enter some text.")


if __name__ == "__main__":
    main()
