---
title: A Simple Book Recommender with LightFM
date: 2021-01-16 12:00:00 +/-0800
categories: [Data Science, Book Recommender]
tags: [machine learning, pandas, gradient descent]     # TAG names should always be lowercase
image: /assets/img/posts/basic-book-rec/header.jpg # from https://unsplash.com/photos/mcSDtbWXUZU
math: true
---
# Why a Recommender?
--------------------
Recommender systems are in use in many contemporary web service settings. The canonical example of a recommender system is employed by Netflix when trying to recommend what any particular user may want to watch. However, this technology can also be seen at Amazon, Youtube and LinkedIn. Companies leverage recommender systems to provide customers with the best experience possible (a pleasant side effect is the maximization of profits).

Since these systems are so prevalent in modern sales, I decided I would build one! Reading is one of my favorite hobbies. With this in mind, I figured a useful first from-scratch project would be to build a system that could recommend new books to me. I always feel a sense of emptiness after finishing a series (some of my favorites include *Harry Potter*, *Wheel of Time*, and *Earthsea*), the hope is that this system will help me dive right into a new series with little to no mourning period!

I will outline what a recommender does and the inner workings then get into a bit of code and some recommendations. Also check out [this](https://towardsdatascience.com/introduction-to-recommender-systems-6c66cf15ada) article if you're feeling extra ambitious!


# A Simple Recommender
----------------------
How can we begin to recommend items to users? First, we can look at the highest rated items and just recommend those to new users! Let's do this with the [goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k) data set and see what we get.

```python
import pandas as pd

books = pd.read_csv("goodbooks-10k/books.csv")
book_tags = pd.read_csv("goodbooks-10k/book_tags.csv")
ratings = pd.read_csv("goodbooks-10k/ratings.csv")
tags = pd.read_csv("goodbooks-10k/tags.csv")
to_read = pd.read_csv("goodbooks-10k/to_read.csv")

```

If we look at the **books** data frame, we can see that the average rating for each book is represented. Let's organize this data frame by the average rating and look at the top 10.

```python
books[['title','authors','average_rating']].sort_values(by='average_rating', ascending=False)[:10]
```

| title                                             | authors                                    | average_rating |
|:--------------------------------------------------|:-------------------------------------------|---------------:|
| The Complete Calvin and Hobbes                    | Bill Watterson 	                           | 4.82           |
| Harry Potter Boxed Set, Books 1-5 (Harry Potte... |	J.K. Rowling, Mary GrandPré 	             | 4.77           |
| Words of Radiance (The Stormlight Archive, #2) 	  | Brandon Sanderson 	                       | 4.77           |
| Mark of the Lion Trilogy 	                        | Francine Rivers 	                         | 4.76           |
| ESV Study Bible 	                                | Anonymous, Lane T. Dennis, Wayne A. Grudem | 4.76           |
| It's a Magical World: A Calvin and Hobbes Coll... | Bill Watterson 	                           | 4.75           |
| There's Treasure Everywhere: A Calvin and Hobb... |	Bill Watterson 	                           | 4.74           |
| Harry Potter Boxset (Harry Potter, #1-7) 	        | J.K. Rowling 	                             | 4.74           |
| Harry Potter Collection (Harry Potter, #1-6) 	    | J.K. Rowling 	                             | 4.73           |
| The Indispensable Calvin and Hobbes 	            | Bill Watterson 	                           | 4.73           |

Great! Now, whenever a new user wants to read some books, they can just get the top-10 list. There are, clearly, some issues with this. First of all: why are there so many repeat sets? It turns out that series can be entered into the database in many different ways, so different users can rate the same series under different names. Alright, easy enough to fix this, we can just drop repeats; however, you should notice that the repeats are actually different subsets of series. For Calvin and Hobbes we have the complete collection and then several individual books. Should we combine these? Maybe a user would like one book and not another? A second issue we run into is that what if our new user is looking for a specific genre? This list clearly is genre agnostic. We will have to create a more complex system if we want to personalize the recommendations. This is similar to the following issue: are we sure our new user will even like these books? Sure, we are better off suggesting these than the lowest rated books; however, we can be better than this!

If the *highest rated* books upsets you, we can also look at the most popular books.

```python
books[['title','authors','ratings_count']].sort_values(by='ratings_count',ascending=False)[:10]
```

| title                                             | authors                                    | ratings_count |
|:--------------------------------------------------|:-------------------------------------------|--------------:|
| The Hunger Games (The Hunger Games, #1) 	        | Suzanne Collins 	                         | 4780653       |
| Harry Potter and the Sorcerer's Stone (Harry P... | J.K. Rowling, Mary GrandPré 	             | 4602479       |
| Twilight (Twilight, #1) 	                        | Stephenie Meyer 	                         | 3866839       |
| To Kill a Mockingbird 	                          | Harper Lee 	                               | 3198671       |
| The Great Gatsby 	                                | F. Scott Fitzgerald 	                     | 2683664       |
| The Fault in Our Stars 	                          | John Green 	                               | 2346404       |
| The Hobbit 	                                      | J.R.R. Tolkien 	                           | 2071616       |
| The Catcher in the Rye 	                          | J.D. Salinger 	                           | 2044241       |
| Pride and Prejudice 	                            | Jane Austen 	                             | 2035490       |
| Angels & Demons (Robert Langdon, #1) 	            | Dan Brown 	                               | 2001311       |

Well, we got rid of the repeating entries issue. Hurrah! Alas, this does not really solve our other issues. How can we be better?

# Something much better
-----------------------

For now, I will just discuss the system that I am concerned with: the collaborative based filtering method. There are other methods (such as content based and hybrid methods); however, I will be using a collaborative system for the first iteration of this project so that is what I will discuss here.

<!-- In this section I will discuss the two (three if you count hybrid methods) paradigms of recommender systems: collaborative based filtering and content based filtering. I will discuss basic algorithms and ideas that fall under these umbrellas. Let's first look at content based filtering. -->

## Collaborative Filtering

Collaborative based recommender systems are systems that seek to recommend items to users based off of other users past interactions with the items in the system. A memory based collaborative filter assumes no underlying model of the data (which is stored in a user-item interaction matrix) and finds new items based off of similar users or similar items. This can be done with a [nearest neighbors]({% post_url 2020-10-15-KNN-from-scratch %}) algorithm. There are two types of memory based systems: user-user and item-item.

The user-user approach uses the rows (users) from the user-interaction matrix and treats them as vectors in the space of all possible items. A nearest neighbors search is done with the new user on each user vector, and recommendations are made based off of similar users.

The item-item approach is very similar to that of the user-user approach. This method uses the columns of the user-item interaction matrix as vectors in the space of all possible users. For the most positive items the new user previously rated, the item vectors can be compared with other item vectors in the matrix via a nearest neighbors search. The most similar items are then recommended. The downfall to the memory based approach is that it does not scale well with the age of the system. If you have millions of users and items, the nearest neighbors algorithm will slow significantly (on the order of $O(kmn)$ where $k$ is the number of neighbors considered and $m\times n$ is the user-item matrix dimensionality).

The system that we are concerned with is a model based collaborative filter called matrix factorization. In this method, it is assumed that the user-item interaction matrix can be decomposed into two matrices of lower dimension whose inner product is equal to the user-item interaction matrix. These lower dimensional matrices effectively represent the set of latent features on the users and items. The (non-mathematical) assumption is that there is a small (smaller than the original dimensionality of the matrix) set of latent features that appropriately describe each user's preferences and each item's characteristics. Formally: Let there be an $m \times n$ user-item interaction matrix $A$ such that
\$$ A = B\;C^{\text{T}}, \$$
where $B$ is an $m \times q$ matrix  of $q$ latent features describing all users and $C$ is a $q \times n$ matrix of $q$ latent features describing all items. Since every user will not have rated every single item, we will be searching for an approximate solution to this problem. What better way to do this than with gradient descent! Suppose each $B$ and $C$ are made up of vectors $b_i$ and $c_j$, each of length $q$. Then we must find
\$$ (B, C) = \underset{B, C}{argmin} \sum_{i = 0}^{m-1} \sum_{j = 0}^{m-1} \[(b_i)(c_j)^{\text{T}} - A_{ij} \]^2. \$$
Once this is done then the approximate user-item interaction matrix can be reconstructed with no missing values and each user can be provided with recommendations! Let's import a library that will do this for us on the goodbooks-10k data set.


# Building my first recommender using LightFM
---------------------------------------------
[LightFM](https://making.lyst.com/lightfm/docs/home.html) is a python implementation of a hybrid recommender system (you can input meta data on users and items as well, which I could implement in later iterations of this project).
> I had to use `pip install lightfm` rather than anaconda to install the package.

Let's import this package and see what we can do. We will also need to import a sparse matrix storage technique from `scipy`.
> A sparse matrix is a storage technique where you only store the positions and values of the non-zero elements of the matrix.



```python
from scipy.sparse import coo_matrix as cm
import lightfm as lf

# this is because I re-indexed all users and books to start from zero
numUsers = ratings.user_id.max()+1
numBooks = ratings.book_id.max()+1

ratSparse = cm((ratings.rating, (ratings.user_id, ratings.book_id)),
               shape=(numUsers, numBooks))

```

Now that we have our sparse matrix we can go ahead and perform our factorization. Note that LightFM handles most of the dirty work behind the scenes. All you have to do is specify the matrix you're decomposing, specify a loss function and some other optional inputs and you're good to go!
> I particularly like the built in parallelization. This is done using OMP in C!

```python
model = lf.LightFM(loss = 'warp')
# I have a Threadripper 2920X so I want to use quite a few threads.
model.fit(ratSparse, epochs = 30, num_threads = 12)
```

Finally we can predict what a random user might want to read!


```python
import numpy as np
import random


def recommend_random(seed, model):

    # let's just grab a random user
    random.seed(seed)
    user = random.choices(ratings.user_id.unique().tolist())[0]

    # now let's predict them on our trained model
    itemList = np.array(ratings.book_id.unique().tolist())
    itemList.sort()

    # let's see what they already like
    knownRatings = pd.merge(ratings.query('user_id == @user'),
                            books[['book_id', 'title', 'authors']], on='book_id', how='left')

    # here are the predictions                    
    score = model.predict(user, itemList)
    suggested = books.loc[np.argsort(-score)][['title', 'authors']]

    print(color.BOLD + 'User {} known items: '.format(user) + color.END, end='\n')
    display(HTML(knownRatings[['title', 'authors', 'rating']]
                 .sort_values(by='rating', ascending=False).iloc[:10].to_html()))
    print(color.BOLD + 'Top 10 suggested items:' + color.END, end='\n')
    display(HTML(suggested[:10].to_html()))

recommend_random(seed = 1001, model = model)
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

Great! This looks like it works in the way we would expect. Of course, I need to be a better data scientist and measure my success on a test set. That comes next along with hyperparameter tuning. I also need to figure out how to get this to work on NEW users. The collaborative method does suffer from a cold start problem (if the new user has no ratings then there is no way to predict the books they'll like), but we can alleviate this by asking a new user to rate books they've already read as well as rely on some possible meta-data for a content/hybrid based method.

## Topics for future exploration.

- I need to be a good data scientist and get a training/testing set and tune some hyperparameters (learning rate on gradient descent).
    - Here, testing means that I withhold some reviews and check how the model predicts them.
- LightFM has hybrid method functionality. Will this improve my results?
- How do I read in a NEW user and give them recommendations?
    - Maybe start with retraining the entire model. Very Slow.
    - Then move on to batch updating?
- Finally deployment, docker, AWS, etc.


_Thanks to [Radu Marcusu](https://unsplash.com/photos/mbKApJz6RSU?utm_source=unsplash&utm_medium=referral&utm_content=creditShareLink) for the header photo. Also, make sure to check out my [Github repo](https://github.com/ColinB19/BookRecommender) as I go through this project. As always please email me with any questions or comments you may have._
