
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.feature_extraction.text import CountVectorizer


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Sentiment AI",
    page_icon="🎬",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"


# ============================================================
# 2. LOAD MODEL AND DATASETS
# ============================================================

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_DIR / "sentiment_model.pkl")
    vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
    return model, vectorizer


@st.cache_data
def load_dataset():
    data = pd.read_csv(DATA_DIR / "IMDB Dataset.csv")

    data["word_count"] = (
        data["review"]
        .fillna("")
        .astype(str)
        .apply(lambda text: len(text.split()))
    )

    return data


@st.cache_data
def load_movies():
    columns = [
        "rotten_tomatoes_link",
        "movie_title",
        "movie_info",
        "content_rating",
        "genres",
        "original_release_date",
        "tomatometer_status",
        "tomatometer_rating",
        "audience_status",
        "audience_rating",
    ]

    return pd.read_csv(
        DATA_DIR / "rotten_tomatoes_movies.csv",
        usecols=columns,
        low_memory=False,
    )


@st.cache_data(show_spinner=False)
def load_movie_reviews(movie_link):
    """Read the large reviews CSV in chunks for one selected movie."""

    review_path = DATA_DIR / "rotten_tomatoes_critic_reviews.csv"

    columns = [
        "rotten_tomatoes_link",
        "critic_name",
        "publisher_name",
        "review_type",
        "review_date",
        "review_content",
    ]

    matching_reviews = []

    for chunk in pd.read_csv(
        review_path,
        usecols=columns,
        chunksize=100_000,
    ):
        matches = chunk[
            chunk["rotten_tomatoes_link"] == movie_link
        ]

        if not matches.empty:
            matching_reviews.append(matches)

    if not matching_reviews:
        return pd.DataFrame(columns=columns)

    return pd.concat(matching_reviews, ignore_index=True)


model, vectorizer = load_model()


# ============================================================
# 3. SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("🎬 Sentiment AI")
st.sidebar.caption("Movie Review Analysis")

page = st.sidebar.radio(
    "Navigation",
    [
        "Review Analyzer",
        "Dataset Explorer",
        "Model Dashboard",
        "Batch Analysis",
        "Movie Explorer",
    ],
)


# ============================================================
# PAGE 1: REVIEW ANALYZER
# ============================================================

if page == "Review Analyzer":

    st.title("🎬 Movie Review Sentiment Analyzer")
    st.write(
        "Enter a movie review to predict whether its sentiment "
        "is positive or negative."
    )

    review = st.text_area(
        "Enter your review",
        placeholder="This movie was amazing! The acting was excellent...",
        height=180,
    )

    if st.button("Analyze Sentiment", type="primary"):

        if not review.strip():
            st.warning("Please enter a review first.")

        else:
            features = vectorizer.transform([review])
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            confidence = max(probabilities) * 100

            st.divider()
            st.subheader("Analysis Result")

            if prediction == 1:
                st.success("😊 Positive sentiment")
            else:
                st.error("☹️ Negative sentiment")

            st.metric(
                "Estimated model confidence",
                f"{confidence:.1f}%",
            )

            st.progress(float(confidence / 100))

            st.caption(
                "This percentage is the model's estimated probability "
                "for its prediction, not a guarantee of correctness."
            )


# ============================================================
# PAGE 2: DATASET EXPLORER
# ============================================================

elif page == "Dataset Explorer":

    st.title("🔎 IMDb Dataset Explorer")
    st.write("Explore the dataset used to train the sentiment model.")

    try:
        data = load_dataset()

        total_reviews = len(data)
        positive_reviews = int(
            (data["sentiment"] == "positive").sum()
        )
        negative_reviews = int(
            (data["sentiment"] == "negative").sum()
        )
        average_length = data["word_count"].mean()

        st.subheader("Dataset Overview")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Reviews", f"{total_reviews:,}")
        col2.metric("Positive Reviews", f"{positive_reviews:,}")
        col3.metric("Negative Reviews", f"{negative_reviews:,}")
        col4.metric("Average Words", f"{average_length:.0f}")

        st.divider()
        st.subheader("Dataset Preview")

        sentiment_filter = st.selectbox(
            "Filter by sentiment",
            ["All", "Positive", "Negative"],
        )

        if sentiment_filter == "Positive":
            preview = data[data["sentiment"] == "positive"]
        elif sentiment_filter == "Negative":
            preview = data[data["sentiment"] == "negative"]
        else:
            preview = data

        st.dataframe(
            preview[["review", "sentiment", "word_count"]].head(10),
            use_container_width=True,
        )

        st.divider()
        st.subheader("Sentiment Distribution")

        counts = data["sentiment"].value_counts()

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(
            ["Positive", "Negative"],
            [
                int(counts.get("positive", 0)),
                int(counts.get("negative", 0)),
            ],
        )
        ax.set_ylabel("Number of Reviews")
        ax.set_title("Positive vs. Negative Reviews")
        st.pyplot(fig)
        plt.close(fig)

        st.divider()
        st.subheader("Review Length Analysis")

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(data["word_count"], bins=40)
            ax.set_xlabel("Number of Words")
            ax.set_ylabel("Number of Reviews")
            ax.set_title("Distribution of Review Lengths")
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            length_by_sentiment = (
                data.groupby("sentiment")["word_count"].mean()
            )

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(
                ["Negative", "Positive"],
                [
                    length_by_sentiment.get("negative", 0),
                    length_by_sentiment.get("positive", 0),
                ],
            )
            ax.set_ylabel("Average Number of Words")
            ax.set_title("Average Review Length by Sentiment")
            st.pyplot(fig)
            plt.close(fig)

        st.divider()
        st.subheader("Most Frequent Words")

        selected_sentiment = st.selectbox(
            "Choose a sentiment",
            ["Positive", "Negative"],
            key="word_sentiment",
        )

        reviews = data.loc[
            data["sentiment"] == selected_sentiment.lower(),
            "review",
        ].dropna().astype(str)

        word_vectorizer = CountVectorizer(
            stop_words="english",
            max_features=10,
            token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
        )

        word_matrix = word_vectorizer.fit_transform(reviews)
        frequencies = np.asarray(word_matrix.sum(axis=0)).flatten()
        words = word_vectorizer.get_feature_names_out()

        word_frequency = pd.DataFrame({
            "Word": words,
            "Frequency": frequencies,
        }).sort_values("Frequency", ascending=False)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(
            word_frequency["Word"][::-1],
            word_frequency["Frequency"][::-1],
        )
        ax.set_xlabel("Frequency")
        ax.set_title(f"Frequent Words in {selected_sentiment} Reviews")
        st.pyplot(fig)
        plt.close(fig)

    except FileNotFoundError:
        st.error("Could not find the IMDb dataset in the data folder.")

    except Exception as error:
        st.error(f"Dataset Explorer error: {error}")


# ============================================================
# PAGE 3: MODEL DASHBOARD
# ============================================================

elif page == "Model Dashboard":

    st.title("📊 Model Performance Dashboard")
    st.write("Evaluation results from the held-out test dataset.")

    accuracy = 0.8926086518
    precision = 0.893
    recall = 0.893
    f1 = 0.893

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{accuracy * 100:.2f}%")
    col2.metric("Macro Precision", f"{precision * 100:.1f}%")
    col3.metric("Macro Recall", f"{recall * 100:.1f}%")
    col4.metric("Macro F1-score", f"{f1 * 100:.1f}%")

    st.divider()
    st.subheader("Confusion Matrix")

    cm = np.array([
        [4325, 615],
        [450, 4527],
    ])

    fig, ax = plt.subplots(figsize=(7, 5))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Negative", "Positive"],
    )

    display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Logistic Regression")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Interpretation")

    st.write("- **True Negatives (4,325):** negative reviews correctly identified.")
    st.write("- **False Positives (615):** negative reviews classified as positive.")
    st.write("- **False Negatives (450):** positive reviews classified as negative.")
    st.write("- **True Positives (4,527):** positive reviews correctly identified.")

    st.info(
        "The model achieved approximately 89.26% accuracy on the test set. "
        "It can still struggle with sarcasm, context and mixed opinions."
    )

    st.caption(
        "These results were recorded during the earlier model evaluation."
    )


# ============================================================
# PAGE 4: BATCH ANALYSIS
# ============================================================

elif page == "Batch Analysis":

    st.title("📁 Batch Sentiment Analysis")
    st.write("Upload a CSV containing multiple reviews.")

    st.info(
        "Your CSV must contain a column named 'review', "
        "with one review per row."
    )

    uploaded_file = st.file_uploader(
        "Upload your reviews CSV",
        type=["csv"],
        key="batch_csv_uploader",
    )

    if uploaded_file is not None:

        try:
            data = pd.read_csv(uploaded_file)

            if "review" not in data.columns:
                st.error("The CSV must contain a 'review' column.")

            elif data.empty:
                st.warning("The uploaded CSV contains no rows.")

            else:
                st.subheader("Dataset Preview")
                st.dataframe(data.head(10), use_container_width=True)

                if st.button("Analyze All Reviews", type="primary"):

                    reviews = data["review"].fillna("").astype(str)
                    features = vectorizer.transform(reviews)

                    predictions = model.predict(features)
                    probabilities = model.predict_proba(features)

                    results = data.copy()

                    results["predicted_sentiment"] = [
                        "Positive" if prediction == 1 else "Negative"
                        for prediction in predictions
                    ]

                    results["confidence_percent"] = (
                        probabilities.max(axis=1) * 100
                    ).round(2)

                    positive_count = int(
                        (results["predicted_sentiment"] == "Positive").sum()
                    )
                    negative_count = int(
                        (results["predicted_sentiment"] == "Negative").sum()
                    )

                    st.divider()
                    st.subheader("Prediction Summary")

                    col1, col2, col3 = st.columns(3)

                    col1.metric("Total Reviews", len(results))
                    col2.metric("Positive Reviews", positive_count)
                    col3.metric("Negative Reviews", negative_count)

                    st.subheader("Predicted Results")
                    st.dataframe(results, use_container_width=True)

                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.bar(
                        ["Positive", "Negative"],
                        [positive_count, negative_count],
                    )
                    ax.set_ylabel("Number of Reviews")
                    ax.set_title("Predicted Sentiment Distribution")
                    st.pyplot(fig)
                    plt.close(fig)

                    csv_data = results.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        label="Download Results as CSV",
                        data=csv_data,
                        file_name="sentiment_analysis_results.csv",
                        mime="text/csv",
                    )

        except Exception as error:
            st.error(f"Could not process the CSV file: {error}")


# ============================================================
# PAGE 5: MOVIE EXPLORER
# ============================================================

elif page == "Movie Explorer":

    st.title("🎬 Movie Explorer")
    st.write(
        "Search for a movie, explore its Rotten Tomatoes information, "
        "and analyze its available critic reviews using your AI model."
    )

    try:
        movies = load_movies()

        movie_query = st.text_input(
            "Search for a movie",
            placeholder="For example: Percy Jackson",
        )

        if len(movie_query.strip()) < 2:
            st.info("Enter at least two characters to search for a movie.")

        else:
            matches = movies[
                movies["movie_title"].str.contains(
                    movie_query.strip(),
                    case=False,
                    na=False,
                    regex=False,
                )
            ].head(50)

            if matches.empty:
                st.warning("No matching movies found. Try another title.")

            else:
                st.write(f"Found {len(matches)} matching movie(s), "
                         "showing up to 50.")

                def format_movie_choice(index):
                    title = str(movies.loc[index, "movie_title"])
                    release_date = movies.loc[
                        index, "original_release_date"
                    ]

                    year = (
                        str(release_date)[:4]
                        if pd.notna(release_date)
                        else "Year unknown"
                    )

                    return f"{title} ({year})"

                selected_index = st.selectbox(
                    "Choose a movie",
                    matches.index.tolist(),
                    format_func=format_movie_choice,
                )

                movie = movies.loc[selected_index]

                st.divider()
                st.subheader(str(movie["movie_title"]))

                col1, col2, col3 = st.columns(3)

                tomatometer = movie["tomatometer_rating"]
                audience_rating = movie["audience_rating"]

                col1.metric(
                    "Tomatometer",
                    f"{tomatometer:.0f}%"
                    if pd.notna(tomatometer)
                    else "N/A",
                )

                col2.metric(
                    "Audience Rating",
                    f"{audience_rating:.0f}%"
                    if pd.notna(audience_rating)
                    else "N/A",
                )

                col3.metric(
                    "Content Rating",
                    str(movie["content_rating"])
                    if pd.notna(movie["content_rating"])
                    else "N/A",
                )

                if pd.notna(movie["genres"]):
                    st.write(f"**Genres:** {movie['genres']}")

                if pd.notna(movie["original_release_date"]):
                    st.write(
                        f"**Release date:** {movie['original_release_date']}"
                    )

                if pd.notna(movie["tomatometer_status"]):
                    st.write(
                        f"**Critics' status:** {movie['tomatometer_status']}"
                    )

                if pd.notna(movie["audience_status"]):
                    st.write(
                        f"**Audience status:** {movie['audience_status']}"
                    )

                if pd.notna(movie["movie_info"]):
                    st.write(str(movie["movie_info"]))

                st.divider()

                if st.button(
                    "Load Reviews and Analyze Sentiment",
                    type="primary",
                ):

                    movie_link = movie["rotten_tomatoes_link"]

                    with st.spinner(
                        "Finding critic reviews and analyzing sentiment..."
                    ):
                        reviews = load_movie_reviews(movie_link)

                    if reviews.empty:
                        st.warning(
                            "No critic reviews were found for this movie."
                        )

                    else:
                        reviews = reviews[
                            reviews["review_content"].notna()
                        ].copy()

                        reviews["review_content"] = (
                            reviews["review_content"].astype(str).str.strip()
                        )

                        reviews = reviews[
                            reviews["review_content"] != ""
                        ]

                        if reviews.empty:
                            st.warning(
                                "Reviews were found, but none contained "
                                "usable review text."
                            )

                        else:
                            # Most recent reviews first, up to 500
                            reviews = (
                                reviews.sort_values(
                                    "review_date",
                                    ascending=False,
                                    na_position="last",
                                )
                                .head(500)
                                .copy()
                            )

                            review_texts = reviews["review_content"]

                            review_features = vectorizer.transform(
                                review_texts
                            )

                            predictions = model.predict(review_features)
                            probabilities = model.predict_proba(
                                review_features
                            )

                            reviews["predicted_sentiment"] = [
                                "Positive" if prediction == 1 else "Negative"
                                for prediction in predictions
                            ]

                            reviews["confidence_percent"] = (
                                probabilities.max(axis=1) * 100
                            ).round(2)

                            positive_count = int(
                                (
                                    reviews["predicted_sentiment"] == "Positive"
                                ).sum()
                            )

                            negative_count = int(
                                (
                                    reviews["predicted_sentiment"] == "Negative"
                                ).sum()
                            )

                            total = len(reviews)

                            st.divider()
                            st.subheader("AI Sentiment Summary")

                            st.caption(
                                f"Analyzed {total} available critic reviews "
                                "(up to 500, most recent first)."
                            )

                            col1, col2, col3 = st.columns(3)

                            col1.metric("Reviews Analyzed", total)
                            col2.metric(
                                "Predicted Positive",
                                f"{positive_count} "
                                f"({positive_count / total * 100:.1f}%)",
                            )
                            col3.metric(
                                "Predicted Negative",
                                f"{negative_count} "
                                f"({negative_count / total * 100:.1f}%)",
                            )

                            fig, ax = plt.subplots(figsize=(7, 4))

                            ax.bar(
                                ["Positive", "Negative"],
                                [positive_count, negative_count],
                            )

                            ax.set_ylabel("Number of Reviews")
                            ax.set_title(
                                f"Predicted Sentiment: {movie['movie_title']}"
                            )

                            st.pyplot(fig)
                            plt.close(fig)

                            st.divider()
                            st.subheader("Individual Critic Reviews")

                            st.dataframe(
                                reviews[
                                    [
                                        "critic_name",
                                        "publisher_name",
                                        "review_date",
                                        "review_type",
                                        "predicted_sentiment",
                                        "confidence_percent",
                                        "review_content",
                                    ]
                                ].head(30),
                                use_container_width=True,
                            )

                            st.caption(
                                "Fresh/Rotten is the original Rotten Tomatoes "
                                "classification. Positive/Negative is your "
                                "model's prediction. They are not identical. "
                                "Confidence is an estimated probability, "
                                "not a guarantee of correctness."
                            )

                            csv_data = reviews.to_csv(
                                index=False
                            ).encode("utf-8")

                            st.download_button(
                                "Download Analyzed Reviews",
                                data=csv_data,
                                file_name="movie_sentiment_results.csv",
                                mime="text/csv",
                            )

    except FileNotFoundError:
        st.error(
            "Could not find the Rotten Tomatoes CSV files. Check that "
            "both files are in the project's data folder."
        )

    except Exception as error:
        st.error(f"Movie Explorer error: {error}")
