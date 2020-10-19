---
title: EDA (Gooodbooks-10k)
date: 2020-10-09 16:00:00 +/-0800
categories: [Data Science, Book Recommender]
tags: [jupyter, eda, visualization, matplotlib, pandas]     # TAG names should always be lowercase
image: /assets/img/posts/book_rec_eda/header.jpg # from https://unsplash.com/photos/mcSDtbWXUZU
math: true
---

# Exploring the GoodBook-10K Dataset

I love books. They provide an escape from reality which, if we're honest, we all really need right now. Some of my favorite time spent with myself is when I'm pouring through *Lord of the Rings* or *The Wheel of Time*. Whenever I finish a series I am always looking toward the next great story! This is why I thought it would be fun to build a book recommender! For now I will be using the goodbook-10k data set available from [here](https://github.com/zygmuntz/goodbooks-10k) and [here](https://maciejkula.github.io/spotlight/datasets/goodbooks.html). This post goes through the exploratory data analysis (EDA) I performed on the data set before I did any modeling (which should always be the first step in any data science project!). [This person](https://www.kaggle.com/philippsp/book-recommender-collaborative-filtering-shiny) does some really great analysis on this data set (an older version of it). I will be reproducing some of their results here.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

label_kwargs = {'fontfamily': 'sans-serif',
                'fontsize': 15}

title_kwargs = {'fontfamily': 'sans-serif',
                'fontsize': 25,
                'fontweight': 'bold'}

tick_kwargs = {'rotation' : 'vertical'}
```


```python
# let's just read in all of the data that we'll need
books = pd.read_csv("goodbooks-10k/books.csv")
book_tags = pd.read_csv("goodbooks-10k/book_tags.csv")
ratings = pd.read_csv("goodbooks-10k/ratings.csv")
tags = pd.read_csv("goodbooks-10k/tags.csv")
to_read = pd.read_csv("goodbooks-10k/to_read.csv")
```


```python
# I always look at the beginning of the table to get a feel for the features and formatting
books.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>book_id</th>
      <th>goodreads_book_id</th>
      <th>best_book_id</th>
      <th>work_id</th>
      <th>books_count</th>
      <th>isbn</th>
      <th>isbn13</th>
      <th>authors</th>
      <th>original_publication_year</th>
      <th>original_title</th>
      <th>...</th>
      <th>ratings_count</th>
      <th>work_ratings_count</th>
      <th>work_text_reviews_count</th>
      <th>ratings_1</th>
      <th>ratings_2</th>
      <th>ratings_3</th>
      <th>ratings_4</th>
      <th>ratings_5</th>
      <th>image_url</th>
      <th>small_image_url</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2767052</td>
      <td>2767052</td>
      <td>2792775</td>
      <td>272</td>
      <td>439023483</td>
      <td>9.780439e+12</td>
      <td>Suzanne Collins</td>
      <td>2008.0</td>
      <td>The Hunger Games</td>
      <td>...</td>
      <td>4780653</td>
      <td>4942365</td>
      <td>155254</td>
      <td>66715</td>
      <td>127936</td>
      <td>560092</td>
      <td>1481305</td>
      <td>2706317</td>
      <td>https://images.gr-assets.com/books/1447303603m...</td>
      <td>https://images.gr-assets.com/books/1447303603s...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>3</td>
      <td>3</td>
      <td>4640799</td>
      <td>491</td>
      <td>439554934</td>
      <td>9.780440e+12</td>
      <td>J.K. Rowling, Mary GrandPré</td>
      <td>1997.0</td>
      <td>Harry Potter and the Philosopher's Stone</td>
      <td>...</td>
      <td>4602479</td>
      <td>4800065</td>
      <td>75867</td>
      <td>75504</td>
      <td>101676</td>
      <td>455024</td>
      <td>1156318</td>
      <td>3011543</td>
      <td>https://images.gr-assets.com/books/1474154022m...</td>
      <td>https://images.gr-assets.com/books/1474154022s...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>41865</td>
      <td>41865</td>
      <td>3212258</td>
      <td>226</td>
      <td>316015849</td>
      <td>9.780316e+12</td>
      <td>Stephenie Meyer</td>
      <td>2005.0</td>
      <td>Twilight</td>
      <td>...</td>
      <td>3866839</td>
      <td>3916824</td>
      <td>95009</td>
      <td>456191</td>
      <td>436802</td>
      <td>793319</td>
      <td>875073</td>
      <td>1355439</td>
      <td>https://images.gr-assets.com/books/1361039443m...</td>
      <td>https://images.gr-assets.com/books/1361039443s...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2657</td>
      <td>2657</td>
      <td>3275794</td>
      <td>487</td>
      <td>61120081</td>
      <td>9.780061e+12</td>
      <td>Harper Lee</td>
      <td>1960.0</td>
      <td>To Kill a Mockingbird</td>
      <td>...</td>
      <td>3198671</td>
      <td>3340896</td>
      <td>72586</td>
      <td>60427</td>
      <td>117415</td>
      <td>446835</td>
      <td>1001952</td>
      <td>1714267</td>
      <td>https://images.gr-assets.com/books/1361975680m...</td>
      <td>https://images.gr-assets.com/books/1361975680s...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>4671</td>
      <td>4671</td>
      <td>245494</td>
      <td>1356</td>
      <td>743273567</td>
      <td>9.780743e+12</td>
      <td>F. Scott Fitzgerald</td>
      <td>1925.0</td>
      <td>The Great Gatsby</td>
      <td>...</td>
      <td>2683664</td>
      <td>2773745</td>
      <td>51992</td>
      <td>86236</td>
      <td>197621</td>
      <td>606158</td>
      <td>936012</td>
      <td>947718</td>
      <td>https://images.gr-assets.com/books/1490528560m...</td>
      <td>https://images.gr-assets.com/books/1490528560s...</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>




```python
#this just tells me which id this data goes by (book_id, goodreads, or best_book)
ratings.sort_values('book_id').head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>book_id</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2174136</th>
      <td>29300</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>433265</th>
      <td>6590</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1907014</th>
      <td>7546</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3743260</th>
      <td>43484</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1266846</th>
      <td>18689</td>
      <td>1</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
# I want to organize every book and review according to it's goodreads id. First I'll make a dataframe that
# maps all the id's together. Then I make a dict that just maps between book_id and goodreads_book_id
id_key = books[['book_id','goodreads_book_id','best_book_id','work_id']]
book_to_goodreads_key = pd.Series(id_key.goodreads_book_id.values,index=id_key.book_id).to_dict()

# make a goodreads_book_id column in ratings and filling it!
ratings['goodreads_book_id'] = ratings['book_id'].map(book_to_goodreads_key)
ratings = ratings.drop(['book_id'],axis=1)

# Here I'm just dropping every id that isn't the goodreads id from the book list
books = books.drop(['book_id','best_book_id','work_id'],axis=1)
```


```python
books.sort_values('goodreads_book_id').head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>goodreads_book_id</th>
      <th>books_count</th>
      <th>isbn</th>
      <th>isbn13</th>
      <th>authors</th>
      <th>original_publication_year</th>
      <th>original_title</th>
      <th>title</th>
      <th>language_code</th>
      <th>average_rating</th>
      <th>ratings_count</th>
      <th>work_ratings_count</th>
      <th>work_text_reviews_count</th>
      <th>ratings_1</th>
      <th>ratings_2</th>
      <th>ratings_3</th>
      <th>ratings_4</th>
      <th>ratings_5</th>
      <th>image_url</th>
      <th>small_image_url</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>26</th>
      <td>1</td>
      <td>275</td>
      <td>439785960</td>
      <td>9.780440e+12</td>
      <td>J.K. Rowling, Mary GrandPré</td>
      <td>2005.0</td>
      <td>Harry Potter and the Half-Blood Prince</td>
      <td>Harry Potter and the Half-Blood Prince (Harry ...</td>
      <td>eng</td>
      <td>4.54</td>
      <td>1678823</td>
      <td>1785676</td>
      <td>27520</td>
      <td>7308</td>
      <td>21516</td>
      <td>136333</td>
      <td>459028</td>
      <td>1161491</td>
      <td>https://images.gr-assets.com/books/1361039191m...</td>
      <td>https://images.gr-assets.com/books/1361039191s...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2</td>
      <td>307</td>
      <td>439358078</td>
      <td>9.780439e+12</td>
      <td>J.K. Rowling, Mary GrandPré</td>
      <td>2003.0</td>
      <td>Harry Potter and the Order of the Phoenix</td>
      <td>Harry Potter and the Order of the Phoenix (Har...</td>
      <td>eng</td>
      <td>4.46</td>
      <td>1735368</td>
      <td>1840548</td>
      <td>28685</td>
      <td>9528</td>
      <td>31577</td>
      <td>180210</td>
      <td>494427</td>
      <td>1124806</td>
      <td>https://images.gr-assets.com/books/1387141547m...</td>
      <td>https://images.gr-assets.com/books/1387141547s...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>491</td>
      <td>439554934</td>
      <td>9.780440e+12</td>
      <td>J.K. Rowling, Mary GrandPré</td>
      <td>1997.0</td>
      <td>Harry Potter and the Philosopher's Stone</td>
      <td>Harry Potter and the Sorcerer's Stone (Harry P...</td>
      <td>eng</td>
      <td>4.44</td>
      <td>4602479</td>
      <td>4800065</td>
      <td>75867</td>
      <td>75504</td>
      <td>101676</td>
      <td>455024</td>
      <td>1156318</td>
      <td>3011543</td>
      <td>https://images.gr-assets.com/books/1474154022m...</td>
      <td>https://images.gr-assets.com/books/1474154022s...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>5</td>
      <td>376</td>
      <td>043965548X</td>
      <td>9.780440e+12</td>
      <td>J.K. Rowling, Mary GrandPré, Rufus Beck</td>
      <td>1999.0</td>
      <td>Harry Potter and the Prisoner of Azkaban</td>
      <td>Harry Potter and the Prisoner of Azkaban (Harr...</td>
      <td>eng</td>
      <td>4.53</td>
      <td>1832823</td>
      <td>1969375</td>
      <td>36099</td>
      <td>6716</td>
      <td>20413</td>
      <td>166129</td>
      <td>509447</td>
      <td>1266670</td>
      <td>https://images.gr-assets.com/books/1499277281m...</td>
      <td>https://images.gr-assets.com/books/1499277281s...</td>
    </tr>
    <tr>
      <th>23</th>
      <td>6</td>
      <td>332</td>
      <td>439139600</td>
      <td>9.780439e+12</td>
      <td>J.K. Rowling, Mary GrandPré</td>
      <td>2000.0</td>
      <td>Harry Potter and the Goblet of Fire</td>
      <td>Harry Potter and the Goblet of Fire (Harry Pot...</td>
      <td>eng</td>
      <td>4.53</td>
      <td>1753043</td>
      <td>1868642</td>
      <td>31084</td>
      <td>6676</td>
      <td>20210</td>
      <td>151785</td>
      <td>494926</td>
      <td>1195045</td>
      <td>https://images.gr-assets.com/books/1361482611m...</td>
      <td>https://images.gr-assets.com/books/1361482611s...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# let's look at the top books!
books[['title','authors','average_rating', 'ratings_count','image_url']].sort_values('average_rating',ascending=False).head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>authors</th>
      <th>average_rating</th>
      <th>ratings_count</th>
      <th>image_url</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3627</th>
      <td>The Complete Calvin and Hobbes</td>
      <td>Bill Watterson</td>
      <td>4.82</td>
      <td>28900</td>
      <td>https://images.gr-assets.com/books/1473064526m...</td>
    </tr>
    <tr>
      <th>3274</th>
      <td>Harry Potter Boxed Set, Books 1-5 (Harry Potte...</td>
      <td>J.K. Rowling, Mary GrandPré</td>
      <td>4.77</td>
      <td>33220</td>
      <td>https://s.gr-assets.com/assets/nophoto/book/11...</td>
    </tr>
    <tr>
      <th>861</th>
      <td>Words of Radiance (The Stormlight Archive, #2)</td>
      <td>Brandon Sanderson</td>
      <td>4.77</td>
      <td>73572</td>
      <td>https://images.gr-assets.com/books/1391535251m...</td>
    </tr>
    <tr>
      <th>8853</th>
      <td>Mark of the Lion Trilogy</td>
      <td>Francine Rivers</td>
      <td>4.76</td>
      <td>9081</td>
      <td>https://images.gr-assets.com/books/1349032180m...</td>
    </tr>
    <tr>
      <th>7946</th>
      <td>ESV Study Bible</td>
      <td>Anonymous, Lane T. Dennis, Wayne A. Grudem</td>
      <td>4.76</td>
      <td>8953</td>
      <td>https://images.gr-assets.com/books/1410151002m...</td>
    </tr>
    <tr>
      <th>4482</th>
      <td>It's a Magical World: A Calvin and Hobbes Coll...</td>
      <td>Bill Watterson</td>
      <td>4.75</td>
      <td>22351</td>
      <td>https://images.gr-assets.com/books/1437420710m...</td>
    </tr>
    <tr>
      <th>6360</th>
      <td>There's Treasure Everywhere: A Calvin and Hobb...</td>
      <td>Bill Watterson</td>
      <td>4.74</td>
      <td>16766</td>
      <td>https://s.gr-assets.com/assets/nophoto/book/11...</td>
    </tr>
    <tr>
      <th>421</th>
      <td>Harry Potter Boxset (Harry Potter, #1-7)</td>
      <td>J.K. Rowling</td>
      <td>4.74</td>
      <td>190050</td>
      <td>https://images.gr-assets.com/books/1392579059m...</td>
    </tr>
    <tr>
      <th>3752</th>
      <td>Harry Potter Collection (Harry Potter, #1-6)</td>
      <td>J.K. Rowling</td>
      <td>4.73</td>
      <td>24618</td>
      <td>https://images.gr-assets.com/books/1328867351m...</td>
    </tr>
    <tr>
      <th>6919</th>
      <td>The Indispensable Calvin and Hobbes</td>
      <td>Bill Watterson</td>
      <td>4.73</td>
      <td>14597</td>
      <td>https://s.gr-assets.com/assets/nophoto/book/11...</td>
    </tr>
  </tbody>
</table>
</div>




```python
books[['title','authors','average_rating', 'ratings_count','image_url']].sort_values('ratings_count',ascending=False).head(10)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>authors</th>
      <th>average_rating</th>
      <th>ratings_count</th>
      <th>image_url</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The Hunger Games (The Hunger Games, #1)</td>
      <td>Suzanne Collins</td>
      <td>4.34</td>
      <td>4780653</td>
      <td>https://images.gr-assets.com/books/1447303603m...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Harry Potter and the Sorcerer's Stone (Harry P...</td>
      <td>J.K. Rowling, Mary GrandPré</td>
      <td>4.44</td>
      <td>4602479</td>
      <td>https://images.gr-assets.com/books/1474154022m...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Twilight (Twilight, #1)</td>
      <td>Stephenie Meyer</td>
      <td>3.57</td>
      <td>3866839</td>
      <td>https://images.gr-assets.com/books/1361039443m...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>To Kill a Mockingbird</td>
      <td>Harper Lee</td>
      <td>4.25</td>
      <td>3198671</td>
      <td>https://images.gr-assets.com/books/1361975680m...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The Great Gatsby</td>
      <td>F. Scott Fitzgerald</td>
      <td>3.89</td>
      <td>2683664</td>
      <td>https://images.gr-assets.com/books/1490528560m...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>The Fault in Our Stars</td>
      <td>John Green</td>
      <td>4.26</td>
      <td>2346404</td>
      <td>https://images.gr-assets.com/books/1360206420m...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>The Hobbit</td>
      <td>J.R.R. Tolkien</td>
      <td>4.25</td>
      <td>2071616</td>
      <td>https://images.gr-assets.com/books/1372847500m...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>The Catcher in the Rye</td>
      <td>J.D. Salinger</td>
      <td>3.79</td>
      <td>2044241</td>
      <td>https://images.gr-assets.com/books/1398034300m...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Pride and Prejudice</td>
      <td>Jane Austen</td>
      <td>4.24</td>
      <td>2035490</td>
      <td>https://images.gr-assets.com/books/1320399351m...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Angels &amp; Demons  (Robert Langdon, #1)</td>
      <td>Dan Brown</td>
      <td>3.85</td>
      <td>2001311</td>
      <td>https://images.gr-assets.com/books/1303390735m...</td>
    </tr>
  </tbody>
</table>
</div>




```python
ratings.sort_values('goodreads_book_id').head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>rating</th>
      <th>goodreads_book_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>101595</th>
      <td>344</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>410222</th>
      <td>1634</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1852697</th>
      <td>25153</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>269829</th>
      <td>6307</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3066307</th>
      <td>35099</td>
      <td>4</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
books.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>goodreads_book_id</th>
      <th>books_count</th>
      <th>isbn13</th>
      <th>original_publication_year</th>
      <th>average_rating</th>
      <th>ratings_count</th>
      <th>work_ratings_count</th>
      <th>work_text_reviews_count</th>
      <th>ratings_1</th>
      <th>ratings_2</th>
      <th>ratings_3</th>
      <th>ratings_4</th>
      <th>ratings_5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.000000e+04</td>
      <td>10000.000000</td>
      <td>9.415000e+03</td>
      <td>9979.000000</td>
      <td>10000.000000</td>
      <td>1.000000e+04</td>
      <td>1.000000e+04</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>1.000000e+04</td>
      <td>1.000000e+04</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.264697e+06</td>
      <td>75.712700</td>
      <td>9.755044e+12</td>
      <td>1981.987674</td>
      <td>4.002191</td>
      <td>5.400124e+04</td>
      <td>5.968732e+04</td>
      <td>2919.955300</td>
      <td>1345.040600</td>
      <td>3110.885000</td>
      <td>11475.893800</td>
      <td>1.996570e+04</td>
      <td>2.378981e+04</td>
    </tr>
    <tr>
      <th>std</th>
      <td>7.575462e+06</td>
      <td>170.470728</td>
      <td>4.428619e+11</td>
      <td>152.576665</td>
      <td>0.254427</td>
      <td>1.573700e+05</td>
      <td>1.678038e+05</td>
      <td>6124.378132</td>
      <td>6635.626263</td>
      <td>9717.123578</td>
      <td>28546.449183</td>
      <td>5.144736e+04</td>
      <td>7.976889e+04</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000e+00</td>
      <td>1.000000</td>
      <td>1.951703e+08</td>
      <td>-1750.000000</td>
      <td>2.470000</td>
      <td>2.716000e+03</td>
      <td>5.510000e+03</td>
      <td>3.000000</td>
      <td>11.000000</td>
      <td>30.000000</td>
      <td>323.000000</td>
      <td>7.500000e+02</td>
      <td>7.540000e+02</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4.627575e+04</td>
      <td>23.000000</td>
      <td>9.780316e+12</td>
      <td>1990.000000</td>
      <td>3.850000</td>
      <td>1.356875e+04</td>
      <td>1.543875e+04</td>
      <td>694.000000</td>
      <td>196.000000</td>
      <td>656.000000</td>
      <td>3112.000000</td>
      <td>5.405750e+03</td>
      <td>5.334000e+03</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.949655e+05</td>
      <td>40.000000</td>
      <td>9.780452e+12</td>
      <td>2004.000000</td>
      <td>4.020000</td>
      <td>2.115550e+04</td>
      <td>2.383250e+04</td>
      <td>1402.000000</td>
      <td>391.000000</td>
      <td>1163.000000</td>
      <td>4894.000000</td>
      <td>8.269500e+03</td>
      <td>8.836000e+03</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>9.382225e+06</td>
      <td>67.000000</td>
      <td>9.780831e+12</td>
      <td>2011.000000</td>
      <td>4.180000</td>
      <td>4.105350e+04</td>
      <td>4.591500e+04</td>
      <td>2744.250000</td>
      <td>885.000000</td>
      <td>2353.250000</td>
      <td>9287.000000</td>
      <td>1.602350e+04</td>
      <td>1.730450e+04</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.328864e+07</td>
      <td>3455.000000</td>
      <td>9.790008e+12</td>
      <td>2017.000000</td>
      <td>4.820000</td>
      <td>4.780653e+06</td>
      <td>4.942365e+06</td>
      <td>155254.000000</td>
      <td>456191.000000</td>
      <td>436802.000000</td>
      <td>793319.000000</td>
      <td>1.481305e+06</td>
      <td>3.011543e+06</td>
    </tr>
  </tbody>
</table>
</div>




```python
# there's clearly something going on with pub year. Let's take a closer look
books.sort_values('original_publication_year',ascending = True).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>goodreads_book_id</th>
      <th>books_count</th>
      <th>isbn</th>
      <th>isbn13</th>
      <th>authors</th>
      <th>original_publication_year</th>
      <th>original_title</th>
      <th>title</th>
      <th>language_code</th>
      <th>average_rating</th>
      <th>ratings_count</th>
      <th>work_ratings_count</th>
      <th>work_text_reviews_count</th>
      <th>ratings_1</th>
      <th>ratings_2</th>
      <th>ratings_3</th>
      <th>ratings_4</th>
      <th>ratings_5</th>
      <th>image_url</th>
      <th>small_image_url</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2075</th>
      <td>19351</td>
      <td>266</td>
      <td>141026286</td>
      <td>9.780141e+12</td>
      <td>Anonymous, N.K. Sandars</td>
      <td>-1750.0</td>
      <td>Shūtur eli sharrī</td>
      <td>The Epic of Gilgamesh</td>
      <td>eng</td>
      <td>3.63</td>
      <td>44345</td>
      <td>55856</td>
      <td>2247</td>
      <td>1551</td>
      <td>5850</td>
      <td>17627</td>
      <td>17485</td>
      <td>13343</td>
      <td>https://s.gr-assets.com/assets/nophoto/book/11...</td>
      <td>https://s.gr-assets.com/assets/nophoto/book/50...</td>
    </tr>
    <tr>
      <th>2141</th>
      <td>1375</td>
      <td>255</td>
      <td>147712556</td>
      <td>9.780148e+12</td>
      <td>Homer, Robert Fagles, Bernard Knox</td>
      <td>-762.0</td>
      <td>Ἰλιάς ; Ὀδύσσεια</td>
      <td>The Iliad/The Odyssey</td>
      <td>eng</td>
      <td>4.03</td>
      <td>47825</td>
      <td>51098</td>
      <td>537</td>
      <td>916</td>
      <td>2608</td>
      <td>10439</td>
      <td>17404</td>
      <td>19731</td>
      <td>https://s.gr-assets.com/assets/nophoto/book/11...</td>
      <td>https://s.gr-assets.com/assets/nophoto/book/50...</td>
    </tr>
    <tr>
      <th>340</th>
      <td>1371</td>
      <td>1726</td>
      <td>140275363</td>
      <td>9.780140e+12</td>
      <td>Homer, Robert Fagles, Frédéric Mugler, Bernard...</td>
      <td>-750.0</td>
      <td>Ἰλιάς</td>
      <td>The Iliad</td>
      <td>eng</td>
      <td>3.83</td>
      <td>241088</td>
      <td>273565</td>
      <td>4763</td>
      <td>7701</td>
      <td>20845</td>
      <td>68844</td>
      <td>89384</td>
      <td>86791</td>
      <td>https://s.gr-assets.com/assets/nophoto/book/11...</td>
      <td>https://s.gr-assets.com/assets/nophoto/book/50...</td>
    </tr>
    <tr>
      <th>6165</th>
      <td>534289</td>
      <td>140</td>
      <td>069109750X</td>
      <td>9.780691e+12</td>
      <td>Anonymous, Richard Wilhelm, Cary F. Baynes, C....</td>
      <td>-750.0</td>
      <td>易 [Yì]</td>
      <td>The I Ching or Book of Changes</td>
      <td>eng</td>
      <td>4.18</td>
      <td>12781</td>
      <td>14700</td>
      <td>275</td>
      <td>178</td>
      <td>599</td>
      <td>2649</td>
      <td>4230</td>
      <td>7044</td>
      <td>https://images.gr-assets.com/books/1406503668m...</td>
      <td>https://images.gr-assets.com/books/1406503668s...</td>
    </tr>
    <tr>
      <th>78</th>
      <td>1381</td>
      <td>1703</td>
      <td>143039954</td>
      <td>9.780143e+12</td>
      <td>Homer, Robert Fagles, E.V. Rieu, Frédéric Mugl...</td>
      <td>-720.0</td>
      <td>Ὀδύσσεια</td>
      <td>The Odyssey</td>
      <td>eng</td>
      <td>3.73</td>
      <td>670326</td>
      <td>710757</td>
      <td>8101</td>
      <td>29703</td>
      <td>65629</td>
      <td>183082</td>
      <td>224120</td>
      <td>208223</td>
      <td>https://images.gr-assets.com/books/1390173285m...</td>
      <td>https://images.gr-assets.com/books/1390173285s...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# actually, this is fine. Gilgamesh was written ~ 1800 BC, let's look at the ratings
ratings['rating'].describe()
```




    count    5.976479e+06
    mean     3.919866e+00
    std      9.910868e-01
    min      1.000000e+00
    25%      3.000000e+00
    50%      4.000000e+00
    75%      5.000000e+00
    max      5.000000e+00
    Name: rating, dtype: float64




```python
# I want to look at english books for now
eng_books = books[books['language_code'] == 'eng']
non_eng_books = books[books['language_code'] != 'eng']
```


```python
eng_books.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>goodreads_book_id</th>
      <th>books_count</th>
      <th>isbn</th>
      <th>isbn13</th>
      <th>authors</th>
      <th>original_publication_year</th>
      <th>original_title</th>
      <th>title</th>
      <th>language_code</th>
      <th>average_rating</th>
      <th>ratings_count</th>
      <th>work_ratings_count</th>
      <th>work_text_reviews_count</th>
      <th>ratings_1</th>
      <th>ratings_2</th>
      <th>ratings_3</th>
      <th>ratings_4</th>
      <th>ratings_5</th>
      <th>image_url</th>
      <th>small_image_url</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2767052</td>
      <td>272</td>
      <td>439023483</td>
      <td>9.780439e+12</td>
      <td>Suzanne Collins</td>
      <td>2008.0</td>
      <td>The Hunger Games</td>
      <td>The Hunger Games (The Hunger Games, #1)</td>
      <td>eng</td>
      <td>4.34</td>
      <td>4780653</td>
      <td>4942365</td>
      <td>155254</td>
      <td>66715</td>
      <td>127936</td>
      <td>560092</td>
      <td>1481305</td>
      <td>2706317</td>
      <td>https://images.gr-assets.com/books/1447303603m...</td>
      <td>https://images.gr-assets.com/books/1447303603s...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>491</td>
      <td>439554934</td>
      <td>9.780440e+12</td>
      <td>J.K. Rowling, Mary GrandPré</td>
      <td>1997.0</td>
      <td>Harry Potter and the Philosopher's Stone</td>
      <td>Harry Potter and the Sorcerer's Stone (Harry P...</td>
      <td>eng</td>
      <td>4.44</td>
      <td>4602479</td>
      <td>4800065</td>
      <td>75867</td>
      <td>75504</td>
      <td>101676</td>
      <td>455024</td>
      <td>1156318</td>
      <td>3011543</td>
      <td>https://images.gr-assets.com/books/1474154022m...</td>
      <td>https://images.gr-assets.com/books/1474154022s...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2657</td>
      <td>487</td>
      <td>61120081</td>
      <td>9.780061e+12</td>
      <td>Harper Lee</td>
      <td>1960.0</td>
      <td>To Kill a Mockingbird</td>
      <td>To Kill a Mockingbird</td>
      <td>eng</td>
      <td>4.25</td>
      <td>3198671</td>
      <td>3340896</td>
      <td>72586</td>
      <td>60427</td>
      <td>117415</td>
      <td>446835</td>
      <td>1001952</td>
      <td>1714267</td>
      <td>https://images.gr-assets.com/books/1361975680m...</td>
      <td>https://images.gr-assets.com/books/1361975680s...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4671</td>
      <td>1356</td>
      <td>743273567</td>
      <td>9.780743e+12</td>
      <td>F. Scott Fitzgerald</td>
      <td>1925.0</td>
      <td>The Great Gatsby</td>
      <td>The Great Gatsby</td>
      <td>eng</td>
      <td>3.89</td>
      <td>2683664</td>
      <td>2773745</td>
      <td>51992</td>
      <td>86236</td>
      <td>197621</td>
      <td>606158</td>
      <td>936012</td>
      <td>947718</td>
      <td>https://images.gr-assets.com/books/1490528560m...</td>
      <td>https://images.gr-assets.com/books/1490528560s...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>11870085</td>
      <td>226</td>
      <td>525478817</td>
      <td>9.780525e+12</td>
      <td>John Green</td>
      <td>2012.0</td>
      <td>The Fault in Our Stars</td>
      <td>The Fault in Our Stars</td>
      <td>eng</td>
      <td>4.26</td>
      <td>2346404</td>
      <td>2478609</td>
      <td>140739</td>
      <td>47994</td>
      <td>92723</td>
      <td>327550</td>
      <td>698471</td>
      <td>1311871</td>
      <td>https://images.gr-assets.com/books/1360206420m...</td>
      <td>https://images.gr-assets.com/books/1360206420s...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>5107</td>
      <td>360</td>
      <td>316769177</td>
      <td>9.780317e+12</td>
      <td>J.D. Salinger</td>
      <td>1951.0</td>
      <td>The Catcher in the Rye</td>
      <td>The Catcher in the Rye</td>
      <td>eng</td>
      <td>3.79</td>
      <td>2044241</td>
      <td>2120637</td>
      <td>44920</td>
      <td>109383</td>
      <td>185520</td>
      <td>455042</td>
      <td>661516</td>
      <td>709176</td>
      <td>https://images.gr-assets.com/books/1398034300m...</td>
      <td>https://images.gr-assets.com/books/1398034300s...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1885</td>
      <td>3455</td>
      <td>679783261</td>
      <td>9.780680e+12</td>
      <td>Jane Austen</td>
      <td>1813.0</td>
      <td>Pride and Prejudice</td>
      <td>Pride and Prejudice</td>
      <td>eng</td>
      <td>4.24</td>
      <td>2035490</td>
      <td>2191465</td>
      <td>49152</td>
      <td>54700</td>
      <td>86485</td>
      <td>284852</td>
      <td>609755</td>
      <td>1155673</td>
      <td>https://images.gr-assets.com/books/1320399351m...</td>
      <td>https://images.gr-assets.com/books/1320399351s...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>77203</td>
      <td>283</td>
      <td>1594480001</td>
      <td>9.781594e+12</td>
      <td>Khaled Hosseini</td>
      <td>2003.0</td>
      <td>The Kite Runner</td>
      <td>The Kite Runner</td>
      <td>eng</td>
      <td>4.26</td>
      <td>1813044</td>
      <td>1878095</td>
      <td>59730</td>
      <td>34288</td>
      <td>59980</td>
      <td>226062</td>
      <td>628174</td>
      <td>929591</td>
      <td>https://images.gr-assets.com/books/1484565687m...</td>
      <td>https://images.gr-assets.com/books/1484565687s...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>13335037</td>
      <td>210</td>
      <td>62024035</td>
      <td>9.780062e+12</td>
      <td>Veronica Roth</td>
      <td>2011.0</td>
      <td>Divergent</td>
      <td>Divergent (Divergent, #1)</td>
      <td>eng</td>
      <td>4.24</td>
      <td>1903563</td>
      <td>2216814</td>
      <td>101023</td>
      <td>36315</td>
      <td>82870</td>
      <td>310297</td>
      <td>673028</td>
      <td>1114304</td>
      <td>https://images.gr-assets.com/books/1328559506m...</td>
      <td>https://images.gr-assets.com/books/1328559506s...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>5470</td>
      <td>995</td>
      <td>451524934</td>
      <td>9.780452e+12</td>
      <td>George Orwell, Erich Fromm, Celâl Üster</td>
      <td>1949.0</td>
      <td>Nineteen Eighty-Four</td>
      <td>1984</td>
      <td>eng</td>
      <td>4.14</td>
      <td>1956832</td>
      <td>2053394</td>
      <td>45518</td>
      <td>41845</td>
      <td>86425</td>
      <td>324874</td>
      <td>692021</td>
      <td>908229</td>
      <td>https://images.gr-assets.com/books/1348990566m...</td>
      <td>https://images.gr-assets.com/books/1348990566s...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Let's just get a feeling for where we have missing values
print(
    books.isna().sum(axis=0),
    ratings.isna().sum(axis=0),
    sep = '\n'*2
)
```

    goodreads_book_id               0
    books_count                     0
    isbn                          700
    isbn13                        585
    authors                         0
    original_publication_year      21
    original_title                585
    title                           0
    language_code                1084
    average_rating                  0
    ratings_count                   0
    work_ratings_count              0
    work_text_reviews_count         0
    ratings_1                       0
    ratings_2                       0
    ratings_3                       0
    ratings_4                       0
    ratings_5                       0
    image_url                       0
    small_image_url                 0
    dtype: int64

    user_id              0
    rating               0
    goodreads_book_id    0
    dtype: int64



```python
# We don't really need the publication year but we could look up 21 of them if we wanted.
```


```python
# Let's look at some of these other CSV's
print(book_tags.describe(),
      ratings.describe(),
      tags.describe(),
      to_read.describe(),sep = '\n'*4)
```

           goodreads_book_id         tag_id          count
    count       9.999120e+05  999912.000000  999912.000000
    mean        5.263442e+06   16324.527073     208.869633
    std         7.574057e+06    9647.846196    3501.265173
    min         1.000000e+00       0.000000      -1.000000
    25%         4.622700e+04    8067.000000       7.000000
    50%         3.948410e+05   15808.000000      15.000000
    75%         9.378297e+06   24997.000000      40.000000
    max         3.328864e+07   34251.000000  596234.000000



                user_id        rating  goodreads_book_id
    count  5.976479e+06  5.976479e+06       5.976479e+06
    mean   2.622446e+04  3.919866e+00       3.530194e+06
    std    1.541323e+04  9.910868e-01       6.342982e+06
    min    1.000000e+00  1.000000e+00       1.000000e+00
    25%    1.281300e+04  3.000000e+00       1.287300e+04
    50%    2.593800e+04  4.000000e+00       8.413500e+04
    75%    3.950900e+04  5.000000e+00       5.060378e+06
    max    5.342400e+04  5.000000e+00       3.328864e+07



                 tag_id
    count  34252.000000
    mean   17125.500000
    std     9887.845047
    min        0.000000
    25%     8562.750000
    50%    17125.500000
    75%    25688.250000
    max    34251.000000



                 user_id        book_id
    count  912705.000000  912705.000000
    mean    27668.980115    2454.739538
    std     14775.096388    2626.359921
    min         1.000000       1.000000
    25%     15507.000000     360.000000
    50%     27799.000000    1381.000000
    75%     40220.000000    3843.000000
    max     53424.000000   10000.000000



```python
# We can see there are some weird values in the count column of book_tags. Let's take a closer look.
book_tags.sort_values('count').head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>goodreads_book_id</th>
      <th>tag_id</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>922053</th>
      <td>18607805</td>
      <td>17246</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>922054</th>
      <td>18607805</td>
      <td>6552</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>922055</th>
      <td>18607805</td>
      <td>2272</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>959611</th>
      <td>22931009</td>
      <td>9221</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>922052</th>
      <td>18607805</td>
      <td>21619</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>922051</th>
      <td>18607805</td>
      <td>10197</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>29176</th>
      <td>3885</td>
      <td>10188</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29190</th>
      <td>3885</td>
      <td>25491</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29171</th>
      <td>3885</td>
      <td>5732</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29172</th>
      <td>3885</td>
      <td>21676</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29189</th>
      <td>3885</td>
      <td>18440</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29188</th>
      <td>3885</td>
      <td>22612</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29187</th>
      <td>3885</td>
      <td>29837</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29173</th>
      <td>3885</td>
      <td>29725</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29186</th>
      <td>3885</td>
      <td>30133</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29185</th>
      <td>3885</td>
      <td>24324</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29184</th>
      <td>3885</td>
      <td>33215</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29183</th>
      <td>3885</td>
      <td>13179</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29174</th>
      <td>3885</td>
      <td>20357</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29181</th>
      <td>3885</td>
      <td>8121</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# we should probably just set these to 1
book_tags.loc[book_tags['count'] < 0,'count'] = 1
book_tags.sort_values('count').head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>goodreads_book_id</th>
      <th>tag_id</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>572124</th>
      <td>901680</td>
      <td>21803</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29192</th>
      <td>3885</td>
      <td>12260</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29191</th>
      <td>3885</td>
      <td>17349</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29190</th>
      <td>3885</td>
      <td>25491</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29189</th>
      <td>3885</td>
      <td>18440</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29188</th>
      <td>3885</td>
      <td>22612</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29187</th>
      <td>3885</td>
      <td>29837</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29186</th>
      <td>3885</td>
      <td>30133</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29185</th>
      <td>3885</td>
      <td>24324</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29184</th>
      <td>3885</td>
      <td>33215</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29183</th>
      <td>3885</td>
      <td>13179</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29182</th>
      <td>3885</td>
      <td>24487</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29181</th>
      <td>3885</td>
      <td>8121</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29180</th>
      <td>3885</td>
      <td>5166</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29179</th>
      <td>3885</td>
      <td>8813</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29178</th>
      <td>3885</td>
      <td>14136</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29177</th>
      <td>3885</td>
      <td>5095</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29176</th>
      <td>3885</td>
      <td>10188</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29175</th>
      <td>3885</td>
      <td>11485</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29174</th>
      <td>3885</td>
      <td>20357</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Awesome. I'd like to know how many reviews each user has. Let's check it out
ratings.drop(['goodreads_book_id'],axis=1).groupby('user_id').count().describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>53424.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>111.868804</td>
    </tr>
    <tr>
      <th>std</th>
      <td>26.071224</td>
    </tr>
    <tr>
      <th>min</th>
      <td>19.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>96.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>111.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>128.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>200.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Wow, the minimum number of ratings is 19! that's pretty good. No need to mess with that. Let's look at
# duplicate values
ratings[ratings.duplicated(subset = ['user_id','goodreads_book_id'], keep = False)].head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>rating</th>
      <th>goodreads_book_id</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
# No duplicates in ratings! How about books?
books[books.duplicated(subset=['goodreads_book_id'], keep=False)].head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>goodreads_book_id</th>
      <th>books_count</th>
      <th>isbn</th>
      <th>isbn13</th>
      <th>authors</th>
      <th>original_publication_year</th>
      <th>original_title</th>
      <th>title</th>
      <th>language_code</th>
      <th>average_rating</th>
      <th>ratings_count</th>
      <th>work_ratings_count</th>
      <th>work_text_reviews_count</th>
      <th>ratings_1</th>
      <th>ratings_2</th>
      <th>ratings_3</th>
      <th>ratings_4</th>
      <th>ratings_5</th>
      <th>image_url</th>
      <th>small_image_url</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
# Great! No duplicates and the number of NaN's isn't that bad (especially in columns we care about).
# Ratings is already clean. Let's make an average ratings column!
averages = ratings.groupby('goodreads_book_id')['rating'].mean()
averages = averages.to_dict()
```


```python
# Let's get some plots going to assess the data.  Let's look at the distribution of ratings!
plt.figure(figsize=(15,10))

plt.hist(ratings['rating'],
         density = True,
         bins = 5,
         range = (ratings['rating'].min()-0.5,ratings['rating'].max()+0.5),
         edgecolor='Black',
         facecolor = 'Orange')
plt.title('Distribution of Ratings', title_kwargs)
plt.xlabel('Rating', label_kwargs)
plt.ylabel('Density', label_kwargs)
plt.ticklabel_format(axis = 'y', style = 'plain', scilimits = (0,0))
# plt.ticks(tick_kwargs)
# plt.ylabel(tick_kwargs)
plt.grid(True)
plt.show()
```


![png]({{"/assets/img/posts/book_rec_eda/output_24_0.png" | absolute_url}})



```python
# Let's see the number of ratings per book!
plt.figure(figsize=(15,10))

plt.hist(ratings.groupby('goodreads_book_id').count()['rating'],
         density = True,
         range = (ratings.groupby('goodreads_book_id').count()['rating'].min(),
                  5000),
                  #ratings.groupby('goodreads_book_id').count()['rating'].max()),
         bins = 200,
         edgecolor='Black',
         facecolor = 'Green')
plt.title('Distribution of Number of Ratings', title_kwargs)
plt.xlabel("Number of Ratings", label_kwargs)
plt.ylabel('Density', label_kwargs)
plt.ticklabel_format(axis = 'y', style = 'plain', scilimits = (0,0))
# plt.ticks(tick_kwargs)
# plt.ylabel(tick_kwargs)
plt.grid(True)
```

![png]({{"/assets/img/posts/book_rec_eda/output_25_0.png" | absolute_url}})



```python
# This looks Poissonian! Let's look at mean user rating. Hopefully we have user's that tend to like books.
plt.figure(figsize=(15,10))

plt.hist(ratings.groupby('user_id').mean()['rating'],
         density = True,
         range = (0,5.4),
         bins = 100,
         edgecolor='Black')
plt.title('Distribution of Av. Rating per User', title_kwargs)
plt.xlabel("Av. Rating", label_kwargs)
plt.ylabel('Density', label_kwargs)
plt.ticklabel_format(axis = 'y', style = 'plain', scilimits = (0,0))
# plt.ticks(tick_kwargs)
# plt.ylabel(tick_kwargs)
plt.grid(True)
```


![png]({{"/assets/img/posts/book_rec_eda/output_26_0.png" | absolute_url}})



```python
# Great! Most users tend to like the books they review. Let's get the average book rating per
# year (publication date)

id_to_pub_year = pd.Series(books.original_publication_year.values,index=id_key.goodreads_book_id).to_dict()
ratings['year'] = ratings['goodreads_book_id'].map(id_to_pub_year)
av_rating_on_year = ratings.groupby('year')['rating'].mean()
```


```python
plt.figure(figsize=(15,10))

plt.plot(av_rating_on_year)
plt.title('Average book ratings based on publication year', title_kwargs)
plt.xlabel('Publication year', label_kwargs)
plt.ylabel('Av. rating', label_kwargs)
# plt.ticks(tick_kwargs)
# plt.ylabel(tick_kwargs)
plt.grid(True)
plt.show()
```


![png]({{"/assets/img/posts/book_rec_eda/output_28_0.png" | absolute_url}})



```python
# This isn't super illuminating. Maybe we should bin together every decade? Let's get the goodreads id to
# decade map instead of year map!
id_to_decade_map = {}
for key,value in id_to_pub_year.items():
    if (value<1700):
        id_to_decade_map.update({key:1700})
    else:
        decade = np.floor(value/10)*10
        id_to_decade_map.update({key:decade})
```


```python
# Getting the average on the decade.
ratings['decade'] = ratings['goodreads_book_id'].map(id_to_decade_map)
av_rating_on_decade = ratings.groupby('decade')['rating'].mean()
stdev_rating_on_decade = ratings.groupby('decade')['rating'].std()
```


```python
plt.figure(figsize=(15,10))

plt.plot(av_rating_on_decade)
plt.title('Average book ratings based on publication decade', title_kwargs)
plt.xlabel('Publication decade', label_kwargs)
plt.ylabel('Av. rating', label_kwargs)
# plt.ticks(tick_kwargs)
# plt.ylabel(tick_kwargs)
plt.grid(True)
plt.show()
```


![png]({{"/assets/img/posts/book_rec_eda/output_31_0.png" | absolute_url}})



```python
# Great! Now we see that the average book rating has increased over time (roughly). So we should expect that our
# system will recommend more recent books. This will be a loose pattern as this pattern isn't very dramatic. Now
# I want to see how many reviews there are per year/decade
num_of_ratings_on_decade = ratings.groupby('decade')['decade'].count()
num_of_ratings_on_year = ratings.groupby('year')['year'].count()
```


```python
fig, ax = plt.subplots(2,figsize=(13,10))

ax[0].plot(num_of_ratings_on_year)
# ax[0].ticklabel_format(axis = 'y', style = 'plain', scilimits = (0,0))
ax[0].ticklabel_format(useOffset = False)
# ax[0].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
ax[0].set_title('Number of ratings per year', title_kwargs)
ax[0].set_xlabel('Publication year', label_kwargs)
ax[0].set_ylabel('Number of ratings', label_kwargs)
ax[0].grid(True)

ax[1].plot(num_of_ratings_on_decade)
ax[1].ticklabel_format(axis = 'y', style = 'plain', scilimits = (0,0))
ax[1].set_title('Number of ratings per decade', title_kwargs)
ax[1].set_xlabel('Publication decade', label_kwargs)
ax[1].set_ylabel('Number of ratings', label_kwargs)
ax[1].grid(True)


fig.tight_layout(pad=3.0)
```


![png]({{"/assets/img/posts/book_rec_eda/output_33_0.png" | absolute_url}})



```python
# What does the distribution of the ratings look like? What about based on tags?
```


```python
book_tags.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>goodreads_book_id</th>
      <th>tag_id</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>30574</td>
      <td>167697</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>11305</td>
      <td>37174</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>11557</td>
      <td>34173</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>8717</td>
      <td>12986</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>33114</td>
      <td>12716</td>
    </tr>
  </tbody>
</table>
</div>




```python
tags.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tag_id</th>
      <th>tag_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>34247</th>
      <td>34247</td>
      <td>Ｃhildrens</td>
    </tr>
    <tr>
      <th>34248</th>
      <td>34248</td>
      <td>Ｆａｖｏｒｉｔｅｓ</td>
    </tr>
    <tr>
      <th>34249</th>
      <td>34249</td>
      <td>Ｍａｎｇａ</td>
    </tr>
    <tr>
      <th>34250</th>
      <td>34250</td>
      <td>ＳＥＲＩＥＳ</td>
    </tr>
    <tr>
      <th>34251</th>
      <td>34251</td>
      <td>ｆａｖｏｕｒｉｔｅｓ</td>
    </tr>
  </tbody>
</table>
</div>




```python
# is there a relationship between how many tags a book has and a books average rating?
# What's the highest rated tag? First let's get a single df for tags.
book_tags = book_tags.merge(tags, on = 'tag_id', how = 'left')

```


```python
# let's get the number of tags per book
tags_per_book = book_tags.groupby('goodreads_book_id').count()
tags_per_book['tag_id'].unique()

```




    array([100,  94,  62,  56])




```python
tags_per_book.groupby('tag_id').count()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>tag_name</th>
    </tr>
    <tr>
      <th>tag_id</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>56</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>62</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>94</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>100</th>
      <td>9997</td>
      <td>9997</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Almost every book has 100 tags. This isn't helpful! How about the highest rated book tags?
book_tags.sort_values('count',ascending = False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>goodreads_book_id</th>
      <th>tag_id</th>
      <th>count</th>
      <th>tag_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8400</th>
      <td>865</td>
      <td>30574</td>
      <td>596234</td>
      <td>to-read</td>
    </tr>
    <tr>
      <th>614494</th>
      <td>2429135</td>
      <td>30574</td>
      <td>586235</td>
      <td>to-read</td>
    </tr>
    <tr>
      <th>911994</th>
      <td>18143977</td>
      <td>30574</td>
      <td>505884</td>
      <td>to-read</td>
    </tr>
    <tr>
      <th>200</th>
      <td>3</td>
      <td>30574</td>
      <td>496107</td>
      <td>to-read</td>
    </tr>
    <tr>
      <th>167200</th>
      <td>24280</td>
      <td>30574</td>
      <td>488469</td>
      <td>to-read</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>549167</th>
      <td>760168</td>
      <td>6985</td>
      <td>1</td>
      <td>childrens-picturebooks</td>
    </tr>
    <tr>
      <th>549166</th>
      <td>760168</td>
      <td>25034</td>
      <td>1</td>
      <td>read-as-a-kid</td>
    </tr>
    <tr>
      <th>549165</th>
      <td>760168</td>
      <td>7404</td>
      <td>1</td>
      <td>classic</td>
    </tr>
    <tr>
      <th>876683</th>
      <td>16170625</td>
      <td>580</td>
      <td>1</td>
      <td>20</td>
    </tr>
    <tr>
      <th>467265</th>
      <td>287655</td>
      <td>25153</td>
      <td>1</td>
      <td>read-in-2017</td>
    </tr>
  </tbody>
</table>
<p>999912 rows × 4 columns</p>
</div>




```python
av_book_rating = ratings.groupby('goodreads_book_id')['rating'].mean()
av_book_rating.head(10)
```




    goodreads_book_id
    1     4.443339
    2     4.358697
    3     4.351350
    5     4.418732
    6     4.430780
    8     4.736842
    10    4.699571
    11    4.109940
    13    4.305199
    21    4.134978
    Name: rating, dtype: float64




```python
tag_ratings = book_tags[['tag_id','goodreads_book_id']].merge(av_book_rating,
                                                              on = 'goodreads_book_id',
                                                              how = 'left')
```


```python
tag_ratings.sort_values('tag_id')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tag_id</th>
      <th>goodreads_book_id</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>623187</th>
      <td>0</td>
      <td>2983489</td>
      <td>3.443662</td>
    </tr>
    <tr>
      <th>709170</th>
      <td>0</td>
      <td>7494887</td>
      <td>3.837838</td>
    </tr>
    <tr>
      <th>694280</th>
      <td>0</td>
      <td>6952423</td>
      <td>4.048544</td>
    </tr>
    <tr>
      <th>397872</th>
      <td>0</td>
      <td>147074</td>
      <td>4.247826</td>
    </tr>
    <tr>
      <th>688062</th>
      <td>0</td>
      <td>6713071</td>
      <td>3.989899</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>278803</th>
      <td>34248</td>
      <td>60510</td>
      <td>4.011828</td>
    </tr>
    <tr>
      <th>749499</th>
      <td>34248</td>
      <td>9361589</td>
      <td>3.932480</td>
    </tr>
    <tr>
      <th>453801</th>
      <td>34249</td>
      <td>248871</td>
      <td>4.258065</td>
    </tr>
    <tr>
      <th>184150</th>
      <td>34250</td>
      <td>28866</td>
      <td>3.856688</td>
    </tr>
    <tr>
      <th>749526</th>
      <td>34251</td>
      <td>9361589</td>
      <td>3.932480</td>
    </tr>
  </tbody>
</table>
<p>999912 rows × 3 columns</p>
</div>




```python
av_tag_ratings = tag_ratings.groupby('tag_id').mean().merge(tags, on = 'tag_id').drop('goodreads_book_id',axis=1)
```


```python
av_tag_ratings.sort_values('rating',ascending = False).head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tag_id</th>
      <th>rating</th>
      <th>tag_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>277</th>
      <td>277</td>
      <td>4.818182</td>
      <td>101-bibles-tools</td>
    </tr>
    <tr>
      <th>30242</th>
      <td>30242</td>
      <td>4.818182</td>
      <td>theology-and-apologetics</td>
    </tr>
    <tr>
      <th>28859</th>
      <td>28859</td>
      <td>4.818182</td>
      <td>study-bibles</td>
    </tr>
    <tr>
      <th>4430</th>
      <td>4430</td>
      <td>4.818182</td>
      <td>bible-commentary</td>
    </tr>
    <tr>
      <th>28858</th>
      <td>28858</td>
      <td>4.818182</td>
      <td>study-bible</td>
    </tr>
    <tr>
      <th>4431</th>
      <td>4431</td>
      <td>4.818182</td>
      <td>bible-commentary-study</td>
    </tr>
    <tr>
      <th>26967</th>
      <td>26967</td>
      <td>4.818182</td>
      <td>scripture-commentaries</td>
    </tr>
    <tr>
      <th>7128</th>
      <td>7128</td>
      <td>4.818182</td>
      <td>christian-help</td>
    </tr>
    <tr>
      <th>14674</th>
      <td>14674</td>
      <td>4.793539</td>
      <td>hobbes</td>
    </tr>
    <tr>
      <th>28818</th>
      <td>28818</td>
      <td>4.768707</td>
      <td>strip</td>
    </tr>
    <tr>
      <th>28982</th>
      <td>28982</td>
      <td>4.766355</td>
      <td>sunday-comics</td>
    </tr>
    <tr>
      <th>32233</th>
      <td>32233</td>
      <td>4.763406</td>
      <td>watterson</td>
    </tr>
    <tr>
      <th>27539</th>
      <td>27539</td>
      <td>4.761364</td>
      <td>shelfari-humor</td>
    </tr>
    <tr>
      <th>6134</th>
      <td>6134</td>
      <td>4.754220</td>
      <td>calvin</td>
    </tr>
    <tr>
      <th>10910</th>
      <td>10910</td>
      <td>4.742424</td>
      <td>esv</td>
    </tr>
    <tr>
      <th>15053</th>
      <td>15053</td>
      <td>4.736986</td>
      <td>humor-comics</td>
    </tr>
    <tr>
      <th>32803</th>
      <td>32803</td>
      <td>4.736842</td>
      <td>world-shift</td>
    </tr>
    <tr>
      <th>6833</th>
      <td>6833</td>
      <td>4.736842</td>
      <td>childhood-classic</td>
    </tr>
    <tr>
      <th>5574</th>
      <td>5574</td>
      <td>4.736842</td>
      <td>boxset</td>
    </tr>
    <tr>
      <th>15306</th>
      <td>15306</td>
      <td>4.736842</td>
      <td>imaginative</td>
    </tr>
  </tbody>
</table>
</div>




```python
# it looks like the top tags all have the same rating. This could be because there are only a few books with
# these tags and are very highly rated. You will notice that these tags are all bible related. Let's exclude tags
# that only apply to a few books.

num_books_per_tag = book_tags.groupby('tag_id').count().drop(['goodreads_book_id','tag_name'],axis=1)
```


```python
num_books_per_tag.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
    <tr>
      <th>tag_id</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
tag_ratings = tag_ratings.merge(num_books_per_tag,on = 'tag_id')
```


```python
tag_ratings.sort_values('goodreads_book_id').head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tag_id</th>
      <th>goodreads_book_id</th>
      <th>rating</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>30574</td>
      <td>1</td>
      <td>4.443339</td>
      <td>9983</td>
    </tr>
    <tr>
      <th>258362</th>
      <td>33165</td>
      <td>1</td>
      <td>4.443339</td>
      <td>932</td>
    </tr>
    <tr>
      <th>250046</th>
      <td>17213</td>
      <td>1</td>
      <td>4.443339</td>
      <td>8316</td>
    </tr>
    <tr>
      <th>246919</th>
      <td>27535</td>
      <td>1</td>
      <td>4.443339</td>
      <td>3127</td>
    </tr>
    <tr>
      <th>245888</th>
      <td>16799</td>
      <td>1</td>
      <td>4.443339</td>
      <td>1031</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Now let's look at the averages again
av_tag_ratings = tag_ratings[tag_ratings['count'] > 10].groupby('tag_id').mean().merge(tags, on = 'tag_id').drop('goodreads_book_id',axis=1)
av_tag_ratings.sort_values('rating',ascending = False).head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tag_id</th>
      <th>rating</th>
      <th>count</th>
      <th>tag_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>639</th>
      <td>4537</td>
      <td>4.731122</td>
      <td>13.0</td>
      <td>bill-watterson</td>
    </tr>
    <tr>
      <th>895</th>
      <td>6136</td>
      <td>4.731122</td>
      <td>13.0</td>
      <td>calvin-hobbes</td>
    </tr>
    <tr>
      <th>894</th>
      <td>6135</td>
      <td>4.731122</td>
      <td>13.0</td>
      <td>calvin-and-hobbes</td>
    </tr>
    <tr>
      <th>1161</th>
      <td>7768</td>
      <td>4.669816</td>
      <td>16.0</td>
      <td>comic-strip</td>
    </tr>
    <tr>
      <th>1162</th>
      <td>7770</td>
      <td>4.643310</td>
      <td>17.0</td>
      <td>comic-strips</td>
    </tr>
    <tr>
      <th>917</th>
      <td>6346</td>
      <td>4.636292</td>
      <td>18.0</td>
      <td>cartoon</td>
    </tr>
    <tr>
      <th>4484</th>
      <td>28824</td>
      <td>4.499932</td>
      <td>14.0</td>
      <td>strips</td>
    </tr>
    <tr>
      <th>4096</th>
      <td>26552</td>
      <td>4.447420</td>
      <td>17.0</td>
      <td>sandman</td>
    </tr>
    <tr>
      <th>4631</th>
      <td>30084</td>
      <td>4.446729</td>
      <td>13.0</td>
      <td>the-sandman</td>
    </tr>
    <tr>
      <th>560</th>
      <td>3856</td>
      <td>4.426719</td>
      <td>17.0</td>
      <td>banda-desenhada</td>
    </tr>
    <tr>
      <th>4102</th>
      <td>26602</td>
      <td>4.408591</td>
      <td>11.0</td>
      <td>sarah-j-maas</td>
    </tr>
    <tr>
      <th>1929</th>
      <td>12679</td>
      <td>4.387189</td>
      <td>19.0</td>
      <td>funnies</td>
    </tr>
    <tr>
      <th>918</th>
      <td>6347</td>
      <td>4.380734</td>
      <td>30.0</td>
      <td>cartoons</td>
    </tr>
    <tr>
      <th>2061</th>
      <td>13532</td>
      <td>4.374822</td>
      <td>13.0</td>
      <td>graphic-comic</td>
    </tr>
    <tr>
      <th>805</th>
      <td>5557</td>
      <td>4.349220</td>
      <td>13.0</td>
      <td>box-sets</td>
    </tr>
    <tr>
      <th>1249</th>
      <td>8324</td>
      <td>4.333486</td>
      <td>14.0</td>
      <td>cosmere</td>
    </tr>
    <tr>
      <th>4183</th>
      <td>26966</td>
      <td>4.325058</td>
      <td>15.0</td>
      <td>scripture</td>
    </tr>
    <tr>
      <th>4620</th>
      <td>29849</td>
      <td>4.316637</td>
      <td>12.0</td>
      <td>the-hollows-series</td>
    </tr>
    <tr>
      <th>1015</th>
      <td>6985</td>
      <td>4.307168</td>
      <td>19.0</td>
      <td>childrens-picturebooks</td>
    </tr>
    <tr>
      <th>1948</th>
      <td>12811</td>
      <td>4.304813</td>
      <td>20.0</td>
      <td>game-of-thrones</td>
    </tr>
  </tbody>
</table>
</div>




```python
# good! We've removed the tags that have been tagged less than 10 times!
plt.figure(figsize=(20,5))

plt.bar(av_tag_ratings.sort_values('rating',ascending = False)['tag_name'].iloc[:50],
         av_tag_ratings.sort_values('rating',ascending = False)['rating'].iloc[:50])
plt.title('Average book ratings per book tag', title_kwargs)
plt.xlabel('Tag Name', label_kwargs)
plt.ylabel('Av. rating', label_kwargs)
plt.xticks(rotation=80)
# plt.ylabel(tick_kwargs)
plt.grid(True)
plt.show()
```


![png]({{"/assets/img/posts/book_rec_eda/output_51_0.png" | absolute_url}})



```python
# I can plot a scatter plot of Av rating and number of books to see if there is a coorelation!

plt.figure(figsize=(20,5))

plt.scatter(av_tag_ratings['count'],
            av_tag_ratings['rating'])
plt.title('Average book ratings as a function of number of tags', title_kwargs)
plt.xlabel('Number of tags', label_kwargs)
plt.ylabel('Av. rating', label_kwargs)
plt.xticks(rotation=80)
# plt.ylabel(tick_kwargs)
plt.grid(True)
plt.show()
```


![png]({{"/assets/img/posts/book_rec_eda/output_52_0.png" | absolute_url}})



```python
# This is really interesting!
av_tag_ratings.sort_values('count',ascending = False).head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tag_id</th>
      <th>rating</th>
      <th>count</th>
      <th>tag_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4698</th>
      <td>30574</td>
      <td>3.903186</td>
      <td>9983.0</td>
      <td>to-read</td>
    </tr>
    <tr>
      <th>1735</th>
      <td>11557</td>
      <td>3.903390</td>
      <td>9881.0</td>
      <td>favorites</td>
    </tr>
    <tr>
      <th>3428</th>
      <td>22743</td>
      <td>3.902365</td>
      <td>9858.0</td>
      <td>owned</td>
    </tr>
    <tr>
      <th>751</th>
      <td>5207</td>
      <td>3.902377</td>
      <td>9799.0</td>
      <td>books-i-own</td>
    </tr>
    <tr>
      <th>1310</th>
      <td>8717</td>
      <td>3.905203</td>
      <td>9776.0</td>
      <td>currently-reading</td>
    </tr>
    <tr>
      <th>2736</th>
      <td>18045</td>
      <td>3.898866</td>
      <td>9415.0</td>
      <td>library</td>
    </tr>
    <tr>
      <th>3431</th>
      <td>22753</td>
      <td>3.897385</td>
      <td>9221.0</td>
      <td>owned-books</td>
    </tr>
    <tr>
      <th>1772</th>
      <td>11743</td>
      <td>3.897280</td>
      <td>9097.0</td>
      <td>fiction</td>
    </tr>
    <tr>
      <th>4690</th>
      <td>30521</td>
      <td>3.900676</td>
      <td>8692.0</td>
      <td>to-buy</td>
    </tr>
    <tr>
      <th>2604</th>
      <td>17213</td>
      <td>3.883258</td>
      <td>8316.0</td>
      <td>kindle</td>
    </tr>
    <tr>
      <th>1378</th>
      <td>9221</td>
      <td>3.901959</td>
      <td>8239.0</td>
      <td>default</td>
    </tr>
    <tr>
      <th>1523</th>
      <td>10197</td>
      <td>3.886917</td>
      <td>8054.0</td>
      <td>ebook</td>
    </tr>
    <tr>
      <th>3118</th>
      <td>20774</td>
      <td>3.887336</td>
      <td>7561.0</td>
      <td>my-books</td>
    </tr>
    <tr>
      <th>498</th>
      <td>3389</td>
      <td>3.878758</td>
      <td>7242.0</td>
      <td>audiobook</td>
    </tr>
    <tr>
      <th>1524</th>
      <td>10210</td>
      <td>3.876774</td>
      <td>7203.0</td>
      <td>ebooks</td>
    </tr>
    <tr>
      <th>4998</th>
      <td>32586</td>
      <td>3.883867</td>
      <td>7192.0</td>
      <td>wish-list</td>
    </tr>
    <tr>
      <th>3134</th>
      <td>20849</td>
      <td>3.879223</td>
      <td>7000.0</td>
      <td>my-library</td>
    </tr>
    <tr>
      <th>499</th>
      <td>3392</td>
      <td>3.873962</td>
      <td>6862.0</td>
      <td>audiobooks</td>
    </tr>
    <tr>
      <th>2325</th>
      <td>15169</td>
      <td>3.885541</td>
      <td>6670.0</td>
      <td>i-own</td>
    </tr>
    <tr>
      <th>219</th>
      <td>1642</td>
      <td>3.877709</td>
      <td>6604.0</td>
      <td>adult</td>
    </tr>
  </tbody>
</table>
</div>




```python
# the highest counted tags don't really say much about the book. We can attribute this to user-entered tags aren't
# a good indicator of avg rating. Clearly the av. rating variance decreases with the number of tags which is really
# just a statement of the central limit theorem.
```


```python
# let's start pulling genres!
book_tags['tag_name'].str.lower()
```




    0                   to-read
    1                   fantasy
    2                 favorites
    3         currently-reading
    4               young-adult
                    ...        
    999907            neighbors
    999908      kindleunlimited
    999909         5-star-reads
    999910          fave-author
    999911             slowburn
    Name: tag_name, Length: 999912, dtype: object




```python
book_tags.tail(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>goodreads_book_id</th>
      <th>tag_id</th>
      <th>count</th>
      <th>tag_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>999892</th>
      <td>33288638</td>
      <td>2541</td>
      <td>9</td>
      <td>angsty</td>
    </tr>
    <tr>
      <th>999893</th>
      <td>33288638</td>
      <td>29401</td>
      <td>9</td>
      <td>tear-jerker</td>
    </tr>
    <tr>
      <th>999894</th>
      <td>33288638</td>
      <td>28420</td>
      <td>9</td>
      <td>sport</td>
    </tr>
    <tr>
      <th>999895</th>
      <td>33288638</td>
      <td>1060</td>
      <td>9</td>
      <td>4-5-stars</td>
    </tr>
    <tr>
      <th>999896</th>
      <td>33288638</td>
      <td>833</td>
      <td>9</td>
      <td>2016-releases</td>
    </tr>
    <tr>
      <th>999897</th>
      <td>33288638</td>
      <td>27821</td>
      <td>9</td>
      <td>single-mom</td>
    </tr>
    <tr>
      <th>999898</th>
      <td>33288638</td>
      <td>3392</td>
      <td>8</td>
      <td>audiobooks</td>
    </tr>
    <tr>
      <th>999899</th>
      <td>33288638</td>
      <td>25822</td>
      <td>8</td>
      <td>reviewed</td>
    </tr>
    <tr>
      <th>999900</th>
      <td>33288638</td>
      <td>855</td>
      <td>8</td>
      <td>2017-books</td>
    </tr>
    <tr>
      <th>999901</th>
      <td>33288638</td>
      <td>5207</td>
      <td>7</td>
      <td>books-i-own</td>
    </tr>
    <tr>
      <th>999902</th>
      <td>33288638</td>
      <td>28528</td>
      <td>7</td>
      <td>stand-alones</td>
    </tr>
    <tr>
      <th>999903</th>
      <td>33288638</td>
      <td>2132</td>
      <td>7</td>
      <td>alpha</td>
    </tr>
    <tr>
      <th>999904</th>
      <td>33288638</td>
      <td>17080</td>
      <td>7</td>
      <td>kick-ass-heroine</td>
    </tr>
    <tr>
      <th>999905</th>
      <td>33288638</td>
      <td>29299</td>
      <td>7</td>
      <td>tattoos</td>
    </tr>
    <tr>
      <th>999906</th>
      <td>33288638</td>
      <td>2101</td>
      <td>7</td>
      <td>all-time-favorite</td>
    </tr>
    <tr>
      <th>999907</th>
      <td>33288638</td>
      <td>21303</td>
      <td>7</td>
      <td>neighbors</td>
    </tr>
    <tr>
      <th>999908</th>
      <td>33288638</td>
      <td>17271</td>
      <td>7</td>
      <td>kindleunlimited</td>
    </tr>
    <tr>
      <th>999909</th>
      <td>33288638</td>
      <td>1126</td>
      <td>7</td>
      <td>5-star-reads</td>
    </tr>
    <tr>
      <th>999910</th>
      <td>33288638</td>
      <td>11478</td>
      <td>7</td>
      <td>fave-author</td>
    </tr>
    <tr>
      <th>999911</th>
      <td>33288638</td>
      <td>27939</td>
      <td>7</td>
      <td>slowburn</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python

```
