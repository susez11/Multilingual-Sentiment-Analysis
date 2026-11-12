# Multilingual Sentiment Analysis

## 1. Project Overview
A multilingual sentiment analysis application that classifies product reviews in **English, Hindi, and Odia** into three categories:
- Positive
- Negative
- Neutral

The project compares a traditional Machine Learning model (TF-IDF + Naive Bayes) with a pretrained Transformer model from Hugging Face.

## 2. Objectives
- Analyze reviews in three languages.
- Perform sentiment classification.
- Compare traditional ML and Transformer approaches.
- Evaluate performance using Accuracy, Precision, Recall and F1-score.
- Develop an interactive Streamlit application.

## 3. Technologies Used
- Python
- Google Colab
- Scikit-learn
- Hugging Face Transformers
- PyTorch
- Streamlit
- Pandas and Matplotlib

## 4. Methodology
1. Create and prepare a multilingual review dataset.
2. Preprocess the review text.
3. Train a TF-IDF + Naive Bayes classifier.
4. Perform sentiment prediction using a pretrained Transformer.
5. Evaluate both approaches using classification reports and confusion matrices.
6. Integrate the models into a Streamlit application.

## 5. Model Performance

| Model | Accuracy |
|---|---:|
| Naive Bayes | 85% |
| Transformer | 43% |

*Values represent the updated evaluation results obtained during the project.*

## 6. Repository Structure

```text
Multilingual-Sentiment-Analysis/
├── app.py
├── requirements.txt
├── saved_models/
│   ├── naive_bayes_model.pkl
│   └── tfidf_vectorizer.pkl
├── synthetic_multilingual_sentiment_uneven.csv
├── Multilingual_Sentiment_Analysis_Presentation.pptx
├── Demonstration.mp4
├── Report.pdf
└── multiling.ipynb - Colab.pdf
```

## 7. Installation and Execution

Clone the repository:

```bash
git clone https://github.com/susez11/Multilingual-Sentiment-Analysis.git
cd Multilingual-Sentiment-Analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## 8. Evaluation Metrics
- Accuracy
- Precision
- Recall
- Macro F1-score
- Confusion Matrix

## 9. Limitations
- The dataset is synthetically generated.
- Transformer performance varies across languages.
- The project may not generalize to all real-world reviews.

## 10. Individual Contributions

| Member | Roll Number | Contribution |
|---|---|---|
| Sushobhan Mandal | 2304128 | Model implementation, Transformer integration and Streamlit development |
| Rajarshi Kumar Paul | 2304108 | Dataset creation, preprocessing and Naive Bayes implementation |
| Rishin Tarafder | 2304110 | Performance evaluation, confusion matrices, visualizations and documentation |

## 11. Conclusion
The project demonstrates multilingual sentiment classification using traditional machine learning and pretrained Transformer models. Their performance is compared through quantitative evaluation and an interactive application.

**Institution:** KIIT University  
**Department:** Electronics and Telecommunication Engineering
