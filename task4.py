import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split
import numpy as np

# Load MovieLens 100k dataset
def load_movie_data():
    movies = pd.read_csv('http://files.grouplens.org/datasets/movielens/ml-100k/u.item', 
                         sep='|', 
                         encoding='latin-1',
                         usecols=[0, 1, 2, 4],
                         names=['movie_id', 'title', 'release_date', 'genres'])
    # Replace NaN values with an empty string
    movies['genres'] = movies['genres'].fillna('')
    movies['genres'] = movies['genres'].apply(lambda x: x.replace('|', ' '))
    return movies

def load_ratings_data():
    ratings = pd.read_csv('http://files.grouplens.org/datasets/movielens/ml-100k/u.data', 
                          sep='\t', 
                          names=['user_id', 'movie_id', 'rating', 'timestamp'])
    return ratings

movies = load_movie_data()
ratings = load_ratings_data()

# Collaborative Filtering using Matrix Factorization
def collaborative_filtering(ratings):
    # Create user-item matrix
    user_item_matrix = ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)

    # Normalize the data by subtracting the mean rating for each user
    user_ratings_mean = np.mean(user_item_matrix.values, axis=1)
    ratings_demeaned = user_item_matrix.values - user_ratings_mean.reshape(-1, 1)

    # Perform Singular Value Decomposition (SVD)
    from scipy.sparse.linalg import svds
    U, sigma, Vt = svds(ratings_demeaned, k=50)

    # Convert sigma to diagonal matrix
    sigma = np.diag(sigma)

    # Predicted ratings
    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
    preds_df = pd.DataFrame(all_user_predicted_ratings, columns=user_item_matrix.columns)

    return preds_df

# Get top N recommendations for a user based on collaborative filtering
def get_top_n_recommendations_cf(preds_df, user_id, n=10):
    user_row_number = user_id - 1  # User ID starts at 1
    sorted_user_predictions = preds_df.iloc[user_row_number].sort_values(ascending=False)
    user_data = ratings[ratings.user_id == user_id]
    user_full = user_data.merge(movies, how='left', left_on='movie_id', right_on='movie_id').sort_values(['rating'], ascending=False)

    recommendations = (movies[~movies['movie_id'].isin(user_full['movie_id'])].
                       merge(pd.DataFrame(sorted_user_predictions).reset_index(), how='left',
                             left_on='movie_id',
                             right_on='movie_id').
                       rename(columns={user_row_number: 'Predictions'}).
                       sort_values('Predictions', ascending=False).
                       iloc[:n, :-1])

    return recommendations

# Content-Based Filtering
def content_based_filtering(movies):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres'])

    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim

# Get top N recommendations for a user based on content-based filtering
def get_top_n_recommendations_cb(user_id, n=10):
    user_ratings = ratings[ratings['user_id'] == user_id]
    user_movie_ids = user_ratings['movie_id']
    user_movie_indices = [movies[movies['movie_id'] == movie_id].index[0] for movie_id in user_movie_ids]

    cosine_sim = content_based_filtering(movies)

    sim_scores = cosine_sim[user_movie_indices].mean(axis=0)
    sim_scores = list(enumerate(sim_scores))
    sim_scores.sort(key=lambda x: x[1], reverse=True)

    top_n_indices = [i for i, score in sim_scores[:n]]
    top_n_movie_ids = movies.iloc[top_n_indices]['movie_id']
    return list(zip(top_n_movie_ids, [score for i, score in sim_scores[:n]]))

# Main function to demonstrate recommendations
def main():
    while True:
        try:
            user_id = int(input("Enter a user ID (1-943): "))
            if user_id < 1 or user_id > 943:
                print("User ID must be between 1 and 943. Please retry.")
                continue

            # Collaborative Filtering Recommendations
            preds_df = collaborative_filtering(ratings)
            top_n_cf = get_top_n_recommendations_cf(preds_df, user_id)
            print(f'\nTop 10 Collaborative Filtering recommendations for user {user_id}:')
            for index, row in top_n_cf.iterrows():
                print(f"Movie ID: {row['movie_id']}, Title of movie: {row['title']}")

            # Content-Based Filtering Recommendations
            top_n_cb = get_top_n_recommendations_cb(user_id)
            print(f'\nTop 10 Content-Based Filtering recommendations for user {user_id}:')
            for movie_id, score in top_n_cb:
                print(f'Movie ID: {movie_id}, Similarity Score of movie: {score}')

            another = input("\nWould you like to get recommendations for another user? (yes/no): ").strip().lower()
            if another != 'yes':
                break
        except ValueError:
            print("Please enter a valid user ID.")

if __name__ == "__main__":
    main()
