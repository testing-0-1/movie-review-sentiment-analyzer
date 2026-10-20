# Sentiment Analysis Dashboard

An interactive sentiment analysis application built with Python, Scikit-learn, and Streamlit to classify movie reviews as positive or negative.
## Live Demo

Try the application here: [**Movie Review Sentiment Analyzer**](https://movie-review-sentiment-analyzer-yj7e45pqvrkru3kx8vj89q.streamlit.app/)

Enter a movie review to predict whether its sentiment is positive or negative.
## Features

- **Review Analyzer:** Predict the sentiment of a review and display confidence.
- **Dataset Explorer:** Explore IMDb reviews through statistics and visualizations.
- **Model Dashboard:** View evaluation metrics and the confusion matrix.
- **Batch Analysis:** Analyze reviews from an uploaded CSV file.
- **Movie Explorer:** Explore movies and analyze critics' reviews from Rotten Tomatoes.

## Machine Learning

- **Dataset:** IMDb movie reviews
- **Text representation:** TF-IDF
- **Classification model:** Logistic Regression
- **Train/test split:** 80% / 20%
- **Test accuracy:** Approximately 89.26%

Predictions are not guaranteed to be correct for every review.

## Technologies

Python, Pandas, NumPy, Scikit-learn, Streamlit, Matplotlib, Seaborn, Jupyter Notebook.

## Run Locally

1. Clone this repository.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the required datasets and place them in the `data/` directory. See the project documentation for the required filenames.
4. ### Required Datasets

To use the **Movie Explorer**, download these two Rotten Tomatoes datasets:

1. `rotten_tomatoes_movies.csv` — contains movie information used to display and search the movie list.
2. `rotten_tomatoes_critic_reviews.csv` — contains critics' reviews that the application analyzes.

Place both files inside the `data/` folder, using **exactly these filenames**:

```text
data/
├── rotten_tomatoes_movies.csv
└── rotten_tomatoes_critic_reviews.csv
```

The IMDb dataset is also required for the Dataset Explorer and the application's other dataset-dependent features. Place it in the same folder as `IMDB Dataset.csv`.

These CSV files are not included in the GitHub repository because of their size. Download them separately from their original dataset source before running the application.
5. Launch the application:

   ```bash
   streamlit run app.py
   ```

## Purpose

This project demonstrates data exploration, text processing, machine learning, model evaluation, and interactive application development.
