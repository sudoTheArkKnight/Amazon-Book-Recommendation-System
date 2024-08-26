import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

# %%
# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# %%
# Load data
data = pd.read_csv("A:/00BaseProjects/Amazon Book Recommendation System/data/Books_df.csv")

# Drop unnecessary column and convert object columns to string
data.drop("Unnamed: 0", axis=1, inplace=True)
data = data.astype({col: 'string' for col in data.select_dtypes(['object']).columns})


# %%
# Define a function to tokenize and preprocess text
def preprocess_text(text):
    if pd.isnull(text):
        return []
    else:
        tokens = word_tokenize(text.lower())  # Convert to lowercase
        tokens = [t for t in tokens if t.isalpha()]  # Remove non-alphabetic tokens
        tokens = [t for t in tokens if t not in stopwords.words('english')]  # Remove stopwords
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(t) for t in tokens]  # Lemmatize tokens
        return tokens


# %%
# Apply preprocessing to relevant columns
data['text_tokenized'] = data['Title'].apply(preprocess_text)
data['Author_tokenized'] = data['Author'].apply(preprocess_text)
data['Main_Genre_tokenized'] = data['Main Genre'].apply(preprocess_text)
data['Sub_Genre_tokenized'] = data['Sub Genre'].apply(preprocess_text)

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

# Create a TF-IDF vectorizer object
vectorizer = TfidfVectorizer(max_features=5000)

# Fit the vectorizer to the tokenized text data and transform it into a matrix
tfidf_matrix = vectorizer.fit_transform([' '.join(tokens) for tokens in data['text_tokenized']])

# Convert the matrix to a dense array
tfidf_array = tfidf_matrix.toarray()

# Create a new dataframe with the TF-IDF features
tfidf_df = pd.DataFrame(tfidf_array, columns=vectorizer.get_feature_names_out())

# Merge the TF-IDF dataframe with the original data
data = pd.concat([data, tfidf_df], axis=1)

# Repeat the process for author, main genre, and sub genre columns
author_vectorizer = TfidfVectorizer(max_features=5000)
author_tfidf_matrix = author_vectorizer.fit_transform([' '.join(tokens) for tokens in data['Author_tokenized']])
author_tfidf_array = author_tfidf_matrix.toarray()
author_tfidf_df = pd.DataFrame(author_tfidf_array, columns=author_vectorizer.get_feature_names_out())
data = pd.concat([data, author_tfidf_df], axis=1)

main_genre_vectorizer = TfidfVectorizer(max_features=5000)
main_genre_tfidf_matrix = main_genre_vectorizer.fit_transform(
    [' '.join(tokens) for tokens in data['Main_Genre_tokenized']])
main_genre_tfidf_array = main_genre_tfidf_matrix.toarray()
main_genre_tfidf_df = pd.DataFrame(main_genre_tfidf_array, columns=main_genre_vectorizer.get_feature_names_out())
data = pd.concat([data, main_genre_tfidf_df], axis=1)

sub_genre_vectorizer = TfidfVectorizer(max_features=5000)
sub_genre_tfidf_matrix = sub_genre_vectorizer.fit_transform(
    [' '.join(tokens) for tokens in data['Sub_Genre_tokenized']])
sub_genre_tfidf_array = sub_genre_tfidf_matrix.toarray()
sub_genre_tfidf_df = pd.DataFrame(sub_genre_tfidf_array, columns=sub_genre_vectorizer.get_feature_names_out())
data = pd.concat([data, sub_genre_tfidf_df], axis=1)


# %%
# Function to find 5 closest titles
def find_closest_titles(book_title, data, tfidf_matrix):
    # Preprocess the input book title
    book_title_tokens = preprocess_text(book_title)

    # Convert the book title to a vector using the same vectorizer
    title_vector = vectorizer.transform([' '.join(book_title_tokens)])

    # Calculate the cosine similarity between the title vector and the entire TF-IDF matrix
    cosine_similarities = cosine_similarity(title_vector, tfidf_matrix).flatten()

    # Get the indices of the top 5 most similar books
    similar_indices = cosine_similarities.argsort()[-6:-1][::-1]

    # Get the titles of the most similar books
    similar_titles = data.iloc[similar_indices]['Title'].tolist()

    return similar_titles


# Example usage:
book_title = "Black Holes (L) : The Reith Lectures [Paperback] Hawking, Stephen"
closest_titles = find_closest_titles(book_title, data, tfidf_matrix)
print("The 5 closest titles to '{}':".format(book_title))
for i, title in enumerate(closest_titles, 1):
    print(f"{i}. {title}")
