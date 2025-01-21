import streamlit as st
import pickle
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model(pickle_path):
    with open(pickle_path, 'rb') as f:
        config = pickle.load(f)
    return config

def analyze_review(review_text, rating, config, model_name="nlptown/bert-base-multilingual-uncased-sentiment"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)

    # Prepare input
    inputs = tokenizer(
        review_text,
        max_length=512,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    ).to(device)

    # Get prediction
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.softmax(outputs.logits, dim=1)
        sentiment_score = (torch.argmax(predictions, dim=1) + 1)[0] / 5.0
        confidence = torch.max(predictions, dim=1).values[0]

    # Normalize rating
    normalized_rating = rating / 5.0
    sentiment_rating_diff = abs(sentiment_score - normalized_rating)

    # Make decision using config parameters
    if confidence < config['sentiment_threshold']:
        decision = "REJECTED"
        reason = "Low confidence in sentiment analysis"
    elif sentiment_rating_diff > config['rating_sentiment_diff_threshold'] / 5.0:
        decision = "REJECTED"
        reason = f"Sentiment ({sentiment_score:.2f}) doesn't match rating ({normalized_rating:.2f})"
    else:
        decision = "APPROVED"
        reason = "Sentiment matches rating"

    return {
        "Decision": decision,
        "Reason": reason,
        "Confidence": f"{confidence:.2f}",
        "Sentiment Score": f"{sentiment_score:.2f}",
        "Rating": rating
    }

# Load the config once at startup
config = load_model('review_approval_model.pkl')

# Streamlit app
st.title("Review Approval System")
st.markdown("Enter your review text and rating to check if it would be approved or rejected.")

# User inputs
review_text = st.text_area("Review Text", placeholder="Enter your review here...")
rating = st.slider("Rating", min_value=1, max_value=5, step=1, value=5)

# Analyze button
if st.button("Analyze Review"):
    if review_text.strip():
        result = analyze_review(review_text, rating, config)
        st.markdown("### Analysis Result:")
        st.write(f"**Decision:** {result['Decision']}")
        st.write(f"**Reason:** {result['Reason']}")
        st.write(f"**Confidence:** {result['Confidence']}")
        st.write(f"**Sentiment Score:** {result['Sentiment Score']}")
        st.write(f"**Rating:** {result['Rating']}")
    else:
        st.error("Please enter a review text before analyzing.")
