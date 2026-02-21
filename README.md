FYP: Stock Market Analysis Using Machine Learning

A web-based system that predicts stock price direction by combining:
- Numerical Model: Logistic Regression using historical stock indicators
- Sentiment Model: FinBERT analyzing financial news
- Web Dashboard: Flask + HTML/CSS/JS for interactive display

Main Objective
- Predict future stock movement (Short, Hold, Long)
- Analyze financial news entiment (Negative, Neutral, Positive)
- Present results in a simple web interface

General Flow:
1. User enters latest stock prices
2. Stock prices -> using Logistic Regression to predict movement, News headline -> using FinBERT to analyze sentiment
3. Combine results and display on website

Technologies:
- Python, pandas, numpy, scikit-learn, PyTorch
- Flask, HTML, CSS, JavaScript

Current Status:
- In progress
- Future improvements: UI/UX enhancement

Note:
- Saved model params are left empty due to git push file size limits, it is required to run all the model training first before actually using it
- You can use other dataset for model training, but the scoring might differ
