
import streamlit as st
import joblib
import torch
from transformers import pipeline

# Page configuration
st.set_page_config(
    page_title="Multilingual Sentiment Analyzer",
    page_icon="🌐",
    layout="wide"
)

# Load Naive Bayes model
@st.cache_resource
def load_nb_model():
    model = joblib.load(
        "saved_models/naive_bayes_model.pkl"
    )
    vectorizer = joblib.load(
        "saved_models/tfidf_vectorizer.pkl"
    )
    return model, vectorizer

# Load multilingual Transformer
@st.cache_resource
def load_transformer():
    model_name = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"

    return pipeline(
        "sentiment-analysis",
        model=model_name,
        tokenizer=model_name,
        device=0 if torch.cuda.is_available() else -1
    )

# Convert Transformer output
def convert_sentiment(result):
    label = result["label"].lower()
    confidence = result["score"]

    if confidence < 0.60:
        return "Neutral"
    elif label == "positive":
        return "Positive"
    elif label == "negative":
        return "Negative"
    else:
        return "Neutral"

# Application header
st.title("🌐 Multilingual Sentiment Analysis")

st.markdown(
    "### Analyze customer reviews using Machine Learning and Transformers"
)

st.write(
    "Supported languages: English, Hindi and Odia"
)

st.divider()

# User input
st.subheader("Enter Your Review")

language = st.selectbox(
    "Select Language",
    ["English", "Hindi", "Odia"]
)

review = st.text_area(
    "Write your review:",
    placeholder="Enter your review here...",
    height=130
)

# Prediction button
if st.button("Analyze Sentiment", type="primary"):

    if not review.strip():
        st.warning("Please enter a review.")

    else:
        try:
            with st.spinner("Analyzing your review..."):

                nb_model, tfidf = load_nb_model()
                transformer = load_transformer()

                # Naive Bayes prediction
                vector = tfidf.transform([review])

                nb_prediction = nb_model.predict(vector)[0]

                nb_probabilities = nb_model.predict_proba(vector)[0]

                nb_confidence = max(nb_probabilities)

                # Transformer prediction
                raw_result = transformer(review)[0]

                transformer_prediction = convert_sentiment(
                    raw_result
                )

                transformer_confidence = raw_result["score"]

            # Display results
            st.divider()
            st.subheader("Analysis Results")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Model 1: Naive Bayes")
                st.success(nb_prediction)
                st.metric(
                    "Model Confidence",
                    f"{nb_confidence * 100:.2f}%"
                )

            with col2:
                st.markdown("### Model 2: Transformer")

                if transformer_prediction == "Positive":
                    st.success(transformer_prediction)
                elif transformer_prediction == "Negative":
                    st.error(transformer_prediction)
                else:
                    st.info(transformer_prediction)

                st.metric(
                    "Raw Model Confidence",
                    f"{transformer_confidence * 100:.2f}%"
                )

            st.divider()
            st.subheader("Model Comparison")

            if nb_prediction == transformer_prediction:
                st.success("Both models agree!")
            else:
                st.warning(
                    "The models produced different predictions."
                )

            st.caption(
                "Transformer neutral classification uses a "
                "60% confidence threshold. Confidence is not "
                "the same as prediction accuracy."
            )

        except Exception as e:
            st.error(f"An error occurred: {e}")

st.divider()

st.caption(
    "Academic Project | Multilingual Sentiment Analysis | "
    "English • Hindi • Odia"
)
