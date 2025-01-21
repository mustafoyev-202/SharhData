import gradio as gr
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

    return f"""
    Decision: {decision}
    Reason: {reason}
    Confidence: {confidence:.2f}
    Sentiment Score: {sentiment_score:.2f}
    Rating: {rating}
    """

# Load the config once at startup
config = load_model('review_approval_model.pkl')

# Create Gradio interface
def create_gradio_app():
    with gr.Blocks(title="Review Approval System") as app:
        gr.Markdown("""
        # Review Approval System
        Enter your review text and rating to check if it would be approved or rejected.
        """)

        with gr.Row():
            with gr.Column():
                review_input = gr.Textbox(
                    label="Review Text",
                    placeholder="Enter your review here...",
                    lines=3
                )
                rating_input = gr.Slider(
                    minimum=1,
                    maximum=5,
                    step=1,
                    label="Rating",
                    value=5
                )
                analyze_button = gr.Button("Analyze Review")

            with gr.Column():
                output_text = gr.Textbox(
                    label="Analysis Result",
                    lines=6
                )

        analyze_button.click(
            fn=lambda review, rating: analyze_review(review, rating, config),
            inputs=[review_input, rating_input],
            outputs=output_text
        )
    return app

if __name__ == "__main__":
    app = create_gradio_app()
    app.launch()