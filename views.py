import numpy as np
import os
import pandas as pd
from django.conf import settings
from sqlalchemy import create_engine
from django.http import JsonResponse
import ast
import nltk
import psycopg2

from django.shortcuts import render
from nltk.metrics.aline import similarity_matrix

def nav_home(request):
    return render(request, 'index.html',{'BOOKS':'http://127.0.0.1:8000/books/',
                                         'MOVIES':'http://127.0.0.1:8000/movies/', 
                                         'BLEND':'http://127.0.0.1:8000/blend/'})

def login_view(request):
    return render(request, 'login.html')

def signup_view(request):
    return render(request, 'signup.html')

def blend_view(request):
    return render(request, 'blend.html')

def sequill_view(request):
    return render(request, 'sequill.html')


#BOOK Recommendation Starts here
def recommend_books():
    books_path = os.path.join(settings.BASE_DIR, 'Syaahi','books.csv')
    users_path = os.path.join(settings.BASE_DIR, 'Syaahi','users.csv')
    ratings_path = os.path.join(settings.BASE_DIR,'Syaahi','ratings.csv') 

    books = pd.read_csv(books_path, dtype={'ISBN': str}, low_memory=False)
    users = pd.read_csv(users_path)
    ratings = pd.read_csv(ratings_path)
    
    'Popularity Based Recommender System'

    ratings_with_name = ratings.merge(books,on='ISBN')
    
    num_rating_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
    num_rating_df.rename(columns={'Book-Rating':'num_ratings'},inplace=True)
    avg_rating_df = ratings_with_name.groupby('Book-Title', as_index=False)['Book-Rating'].mean()
    avg_rating_df.rename(columns={'Book-Rating': 'avg_rating'}, inplace=True)
    popular_df = num_rating_df.merge(avg_rating_df,on='Book-Title')
    popular_df = popular_df[popular_df['num_ratings']>=250].sort_values('avg_rating',ascending=False).head(50)
    popular_df = popular_df.merge(books,on='Book-Title').drop_duplicates('Book-Title')[['Book-Title','Book-Author','Image-URL-M','num_ratings','avg_rating']]

    x = ratings_with_name.groupby('User-ID').count()['Book-Rating'] > 200
    eligible_users = x[x].index
    filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(eligible_users)]
    y = filtered_rating.groupby('Book-Title').count()['Book-Rating']>=50
    famous_books = y[y].index
    final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]
    pt = final_ratings.pivot_table(index='Book-Title',columns='User-ID',values='Book-Rating')
    pt.fillna(0,inplace=True)

    from sklearn.metrics.pairwise import cosine_similarity
    'calculating cosine_similarity of a single row with all the other rows i.e calculating similarity of a single book with all the other books'
    similarity_scores = cosine_similarity(pt)
    
    def recommend(book_name):
        index = np.where(pt.index==book_name)[0][0] #fetching index of the book
        similar_items = sorted(list(enumerate(similarity_scores[index])),key=lambda x:x[1],reverse=True)[1:5] 
    #enumerate is a loop over an iterable while keeping track of the index of the current item.
    #sorting is based on the basis of similarity scores in reverse
        recommendations = []    
            
        for i in similar_items:
            item = []
            temp_df = books[books['Book-Title'] == pt.index[i[0]]]
            title = temp_df.drop_duplicates('Book-Title')['Book-Title'].values[0]  # Get book title
            author = temp_df.drop_duplicates('Book-Author')['Book-Author'].values[0]  # Get book author
            recommendations.append(f"{title} by {author}")  # Append the formatted string to the list

        return recommendations

    thenotebook_recommendations = recommend('The Notebook')
    tokillamockingbird_recommendations = recommend('To Kill a Mockingbird')
    
    return {
        'thenotebook_recommendations': thenotebook_recommendations,
        'tokillamockingbird_recommendations': tokillamockingbird_recommendations,
    }

def book_recommendation_view(request):
    recommended = recommend_books()
    return render(request, 'books.html', {
        'thenotebook_recommendations': recommended['thenotebook_recommendations'],
        'tokillamockingbird_recommendations': recommended['tokillamockingbird_recommendations'],
    })

#MOVIE Recommendation Starts here
def recommend_movies():
    movies_path = os.path.join(settings.BASE_DIR, 'Syaahi','movies.csv')
    credits_path = os.path.join(settings.BASE_DIR, 'Syaahi','credits.csv')
    
    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)
    
    movies = movies.merge(credits,on='title')
    movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
    movies.dropna(inplace=True)
    
    def convert(obj): #here obj is in string'
        L = [] 
        for i in ast.literal_eval(obj): 
            L.append(i['name'])
        return L#here object is made in a list'
    
    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    
    def convert3(obj):
        L = []
        counter = 0
        for i in ast.literal_eval(obj): 
            if counter != 3:
                L.append(i['name'])
                counter+=1
            else:
                break
        return L
    movies['cast'] = movies['cast'].apply(convert3)
    
    def fetch_director(obj):
        L = []
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                L.append(i['name'])
                break
        return L
    movies['crew'] = movies['crew'].apply(fetch_director)
    
    movies['overview'] = movies['overview'].apply(lambda x:x.split()) #converts string in overview in a list
    movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
    movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
    movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
    movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])
    
    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    new_df = movies[['movie_id','title','tags']]
    new_df.loc[:, 'tags'] = new_df['tags'].apply(lambda x: " ".join(x))
    new_df.loc[:, 'tags'] = new_df['tags'].apply(lambda x: x.lower())
    
    from nltk.stem.porter import PorterStemmer
    ps = PorterStemmer()
    def stem(text):
        y = []
        for i in text.split():
            y.append(ps.stem(i))
        return " ".join(y)
    new_df.loc[:, 'tags'] = new_df['tags'].apply(stem)
    
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()
    
    from sklearn.metrics.pairwise import cosine_similarity
    similarity = cosine_similarity(vectors)
    sorted(list(enumerate(similarity[0])),reverse=True, key=lambda x:x[1])[1:6]
    
    def recommend(movie):
        movie_index = new_df[new_df['title'] == movie].index[0]
        distances = similarity[movie_index]
        movies_list = sorted(list(enumerate(distances)),reverse=True, key=lambda x:x[1])[1:6]
        recommended_movies = []
        for i in movies_list:
            recommended_movies.append(new_df.iloc[i[0]].title)
        return recommended_movies
    
    batman_recommendations = recommend('Batman Begins')
    avatar_recommendations = recommend('Avatar')
    
    return {
        'batman_recommendations': batman_recommendations,
        'avatar_recommendations': avatar_recommendations,
    }
    
def top_movies():
    top_movies_list = ["The Godfather", "The Shawshank Redemption", "Schindler's List", 
                       "Pulp Fiction", "The Lord of the Rings", "Forrest Gump", 
                       "Star Wars", "The Dark Knight", "Fight Club", "Goodfellas"]
    return top_movies_list

def movie_recommendation_view(request):
    recommended = recommend_movies()
    top_10 = top_movies()
    return render(request, 'movies.html', {
        'batman_recommendations': recommended['batman_recommendations'],
        'avatar_recommendations': recommended['avatar_recommendations'],
        'top_10': top_10
    })