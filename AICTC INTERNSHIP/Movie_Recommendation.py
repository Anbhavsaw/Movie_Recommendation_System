import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load movies dataset
movies = pd.read_csv(r'D:\AICTC INTERNSHIP\movies.csv')

# Show dataset info
st.write("Movies loaded:", len(movies))
st.write(movies.head())        # Display the first few rows of the dataset

# Check if 'title' and 'genres' columns exist
if 'title' not in movies.columns or 'genres' not in movies.columns:
    st.error("CSV file must have 'title' and 'genres' columns.")
    st.stop()

# Clean the dataset i.e drop rows with missing values in 'title' or 'genres'
movies = movies[['title', 'genres']].dropna()

# Create a TF-IDF Vectorizer to convert the 'genres' text into numerical data
tfidf = TfidfVectorizer(stop_words='english')             #remove English stop words
tfidf_matrix = tfidf.fit_transform(movies['genres'])     # Fit and transform the 'genres' column into a TF-IDF matrix

# Calculate similarity scores between all movies using cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

 # Create a Series to map movie titles to their index number
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# Recommendation function
def get_recommendations(title, cosine_sim=cosine_sim):
    if title not in indices:
        return "Movie not found in dataset."
    idx = indices[title]                                                       # Get the index of the movie that matches the title
    sim_scores = list(enumerate(cosine_sim[idx]))                              # Get the pairwise similarity scores for all movies with that movie
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)          # Sort the movies based on the similarity scores
    sim_scores = sim_scores[1:6]                                               # Get the scores of the top 5 most similar movies (excluding itself)
    movie_indices = [i[0] for i in sim_scores]                                 # Get the indices of the top 5 most similar movies
    return movies['title'].iloc[movie_indices]                                 # Return the titles of the top 5 most similar movies

# Streamlit UI
st.title("🎬 Simple Movie Recommender")

movie_name = st.text_input("Enter a movie title (from dataset):")

if st.button("Recommend"):
    recommendations = get_recommendations(movie_name)
    if isinstance(recommendations, str):                            # Check if the recommendations is a string (error message)
        st.error(recommendations)                                   
    else:
        st.subheader("Top 5 Similar Movies:")                       
        for i, rec in enumerate(recommendations):
            st.write(f"{i+1}. {rec}")

# Show random titles for testing
st.write("🎞️ Some available movie titles:")
st.write(movies['title'].sample(10))

