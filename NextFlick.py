from surprise import SVD, Dataset
from surprise import Reader
from surprise.model_selection import train_test_split
import pandas as pd
from thefuzz import process  

def find_book(user_input, title):
    try:     
        result = process.extractOne(user_input, title.tolist())
        if result:  
            match, score, *_ = result  
            if score > 65:  
                return match
        return None  
    except:
        return None

movies_pd = pd.read_csv('movies.csv')
rating_pd = pd.read_csv('ratings.csv', dtype={'userId': int, 'movieId': int, 'rating': float})

movies_pd["title"] = movies_pd["title"].str.strip().str.lower()

movie_stats = rating_pd.groupby("movieId").agg(avg_rating=("rating", "mean")).reset_index()
pd1 = pd.merge(movies_pd, movie_stats, on="movieId")

title = pd1["title"]
count = 0
movies = []

while count <= 3:
    count += 1
    user_input = input("what movie have you watched? ").strip().lower()
    match_book = find_book(user_input, title)
    if match_book is None:
        print("i can't find the movie")
        count -= 1
        continue
    user_rating = float(input(f"{match_book}:(1-5)"))
    movies.append({"title": match_book, "rating": user_rating})

user_ratings_df = pd.DataFrame(movies)
user_ratings_df = pd.merge(user_ratings_df, movies_pd, on="title")[["movieId", "rating"]]

new_user_id = rating_pd["userId"].max() + 1
user_ratings_df["userId"] = new_user_id

rating_pd = pd.concat([rating_pd, user_ratings_df], ignore_index=True)

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(rating_pd[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.25)

model = SVD()
model.fit(trainset)

all_movie_ids = set(movies_pd["movieId"])
watched_movie_ids = set(user_ratings_df["movieId"])
unwatched_movie_ids = all_movie_ids - watched_movie_ids

recommendations = []
for movie_id in unwatched_movie_ids:
    pred = model.predict(new_user_id, movie_id)
    recommendations.append({"movieId": movie_id, "predicted_rating": pred.est})

recommendations_df = pd.DataFrame(recommendations).sort_values(by="predicted_rating", ascending=False)
recommendations_df = pd.merge(recommendations_df, movies_pd, on="movieId")

print("Top 5 Recommended Movies:")
print(recommendations_df[["title", "predicted_rating"]].head(5))
