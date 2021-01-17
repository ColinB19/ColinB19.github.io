---
title: A Simple Book Recommender with LightFM
date: 2021-01-16 12:00:00 +/-0800
categories: [Data Science, Book Recommender]
tags: [jupyter, machine learning, pandas, gradient descent]     # TAG names should always be lowercase
image: /assets/img/posts/basic-book-rec/header.jpg # from https://unsplash.com/photos/mcSDtbWXUZU
math: true
---

# Import and class definitions
Thanks to [Radu Marcusu](https://unsplash.com/photos/mbKApJz6RSU?utm_source=unsplash&utm_medium=referral&utm_content=creditShareLink) for the header photo!


```python
#standard stuff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

# model imports
import random
from scipy.sparse import coo_matrix as cm
import lightfm as lf

# table formatting
from IPython.display import display, HTML
```


```python
class Pipeline():

    def __init__(self):
        books = pd.DataFrame()
        ratings = pd.DataFrame()
        book_tags = pd.DataFrame()
        tags = pd.DataFrame()
        to_read = pd.DataFrame()

    def preprocess(self):
        '''
        This function is just a clean way to call all preprocessing steps. These steps
        include reading in the goodbooks-10k data, fixing some book id's and more.

        input:
            self - just class object

        output:
            null - no need to return anything
        '''
        self.read_data()
        self.fix_ids()

    def get_model(self, epochs=20, num_threads=2, loss='warp'):
        '''
        This is just a clean way to get the model. Creates interaction sparse matrices
        and then trains a model on them. Note: right now this is purely collaborative filtering,
        LightFM supports Hybrid filtering which we will incorporate in the next version.

        input:
            self - class object.
            epochs - number of epochs for gradient descent.
            num_threads - number of physical cores to parallelize over.
            loss - loss function type. See LightFM Documentation.

        output:
            model - our trained model.
        '''
        interactions = self.get_interactions()
        # user_meta, item_meta = self.get_meta()
        model = self.fit_model(interactions, epochs, num_threads, loss)

        return model


    def read_data(self):
        '''
        This function just reads in the goodbooks-10k data. Later should have the
        functionality to read in scraped data.

        input:
            self - class object

        output:
            null - no need to return anything
        '''
        self.books = pd.read_csv("goodbooks-10k/books.csv")
        self.ratings = pd.read_csv("goodbooks-10k/ratings.csv")

        # may not use these at first.
        self.book_tags = pd.read_csv("goodbooks-10k/book_tags.csv")
        self.tags = pd.read_csv("goodbooks-10k/tags.csv")
        self.to_read = pd.read_csv("goodbooks-10k/to_read.csv")

    def fix_ids(self):
        '''
        This function sets the bookand user id's to start at zero. It also changes the name of
        headers from user_id and book_id to uid and iid.

        input:
            self - class object

        output:
            null - no need to return anything
        '''
        # just changing to standard feature names
        self.ratings.rename(
            columns={'user_id': 'uid', 'book_id': 'iid'}, inplace=True)
        self.books.rename(columns={'book_id': 'iid'}, inplace=True)
        self.to_read.rename(
            columns={'user_id': 'uid', 'book_id': 'iid'}, inplace=True)

        # starting user and book indices from 0
        self.ratings.uid = self.ratings.uid - 1
        self.ratings.iid = self.ratings.iid - 1
        self.books.iid = self.books.iid - 1
        self.to_read.iid = self.to_read.iid - 1

        # this makes all the tags indexed by book_id instead of goodreads_book_id
        temp_books = self.books.set_index('goodreads_book_id')
        idMAP = temp_books['iid'].to_dict()
        self.book_tags['iid'] = self.book_tags.goodreads_book_id.map(idMAP)
        self.book_tags.drop('goodreads_book_id', inplace=True,axis = 1)

    def get_interactions(self):
        '''
        This function gets the interactions sparse matrix using a scipy sparse coo_matrix.

        inputs:
            self - class object

        outputs:
            ratSparse - sparse interactions matrix.
        '''
        numUsers = self.ratings.uid.max()+1
        numBooks = self.ratings.iid.max()+1

        ratSparse = cm((self.ratings.rating, (self.ratings.uid, self.ratings.iid)),
                       shape=(numUsers, numBooks))
        return ratSparse

#     def get_meta(self):
#         numUsers = self.ratings.uid.max()+1
#         numBooks = self.ratings.iid.max()+1
#         numUserFeatures = self.to_read.book_id.max+1
#         numItemFeatures = self.tags.book_id.max+1

    def fit_model(self, ratSparse, epochs, num_threads, loss):
        '''
        This gets a LightFM model and trains it on the sparse matrix.

        inputs:
            self - class object.
            ratSparse - sparse interactions matrix.
            epochs - number of epochs for gradient descent.
            num_threads - number of physical cores to parallelize over.
            loss - loss function type. See LightFM Documentation.

        outputs:
            model - our trained model.

        '''

        model = lf.LightFM(loss=loss)
        model.fit(ratSparse, epochs=epochs, num_threads=num_threads)

        return model

    def recommend_random(self, seed, model):
        '''
        This function will pick a random user out of the list of known users and print out their top
        10 recommendations! It will also print their top 10 rated items for comparison.

        inputs:
            self - class object.
            seed - random seed for reproducibility.

        outputs:
            null - no need to output anything right now.

        '''

        random.seed(seed)
        user = random.choices(self.ratings.uid.unique().tolist())[0]

        # now let's predict them on our trained model
        itemList = np.array(self.ratings.iid.unique().tolist())
        itemList.sort()

        knownRatings = pd.merge(self.ratings.query('uid == @user'),
                                self.books[['iid', 'title', 'authors']], on='iid', how='left')

        score = model.predict(user, itemList)
        suggested = self.books.loc[np.argsort(-score)][['title', 'authors']]

        print(color.BOLD + 'User {} known items: '.format(user) + color.END, end='\n')
        display(HTML(knownRatings[['title', 'authors', 'rating']]
                     .sort_values(by='rating', ascending=False).iloc[:10].to_html()))
        print(color.BOLD + 'Top 10 suggested items:' + color.END, end='\n')
        display(HTML(suggested[:10].to_html()))


class color:
    '''
    This is for printing purposes.
    '''
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


label_kwargs = {'fontfamily': 'sans-serif',
                'fontsize': 15}

title_kwargs = {'fontfamily': 'sans-serif',
                'fontsize': 25,
                'fontweight': 'bold'}

tick_kwargs = {'rotation': 'vertical'}
```

# Testing ground


```python
rec = Pipeline()
rec.preprocess()
model = rec.get_model(num_threads = 12)
```


```python
rec.recommend_random(seed = 1001, model = model)
```

**User 3898 known items:**


| title                           | authors                             | rating |
|:--------------------------------|:------------------------------------|-------:|
| The Green Mile                  | Stephen King                        | 5      |
| Room                            | Emma Donoghue                       | 5      |
| Eligible: A Modern Retelling... | Giovanni Rovelli                    | 5      |
| A Little Life                   | Hanya Yanagihara                    | 5      |
| Homegoing                       | Yaa Gyasi                           | 5      |
| All the Light We Cannot See     | Anthony Doerr                       | 5      |
| Case Histories (Jackson...      | Kate Atkinson                       | 5      |
| Rebecca                         | Daphne du Maurier, Sally Beauman    | 5      |
| Beautiful Ruins                 | Jess Walter                         | 5      |
| March: Book One (March, #1)     |John Lewis, Andrew Aydin, Nate Powell| 5      |


**Top 10 suggested items:**

| title                                 | authors                       |
|:--------------------------------------|------------------------------:|
| The Girl on the Train                 | Paula Hawkins                 |
| All the Light We Cannot See           | Anthony Doerr                 |
| The Goldfinch                         | Donna Tartt                   |
| Station Eleven                        | Emily St. John Mandel         |
| Go Set a Watchman                     | Harper Lee                    |
| Gone Girl                             | Gillian Flynn                 |
| Everything I Never Told You           | Celeste Ng                    |
| Fates and Furies                      | Lauren Groff                  |
| Life After Life                       | Kate Atkinson                 |
| My Brilliant Friend (The Neapolitan...| Elena Ferrante, Ann Goldstein |


## Some notes for moving forward.

1. now I need to do some gridSearch to find the best hyperparameters
    1. Since I need to preserve the entirety of a users reviews in either the test or train set and, when testing I need to keep some of the testers reviews to check the others, right? How do I do all this correctly?
2. find out what method/algo LightFM is using. Could another be better?
3. How do I read in a NEW user and give them recommendations?
    1. Maybe start with retraining the entire model.
    2. Then move on to batch updating or something like that
4. finally deployment, docker, AWS, etc.
