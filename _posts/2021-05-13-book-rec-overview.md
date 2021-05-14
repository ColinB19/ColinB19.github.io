---
title: Book Recommender Overview
date: 2021-05-13 12:00:00 +/-0800
categories: [Data Science, Book Recommender]
tags: [machine learning, flask, Postgres]
image: /assets/img/posts/book-rec/header1.jpg #https://unsplash.com/photos/o4-YyGi5JBc
math: true
---
# Where were you?
----------------------------
I'm back! After quite a long haiatus from posting I'm back with a vengeance. What's the deal? Well, what's the first thing to go when the stack of stuff to do gets too high to manage? Apparently, for me, it's writing. I have been working very hard on applying to new career opportunities, reading every book I can get my hands on, and writing code that I have put one of the key aspects of data science on the backburner: communication. Even my [Twitter](https://twitter.com/data_motivated) has suffered from this lull. 

Despite this temporary leave of absence I am still as committed as ever to talking about my journey into data science! I'll be posting more frequently from here on out.

# The Book Spot
-------------------

![jpg]({{"/assets/img/posts/book-rec/midpost1.jpg" | absolute_url}})
_Thanks to [Glen Noble](https://unsplash.com/photos/o4-YyGi5JBc) for this image._
<br>

I wanted to discuss my book recommender project as it has been finished(ish) and I learned A TON while doing it. This post will be the first of several where I dive into the different aspects of the project. Here I will just talk about the general overview of the project, a brief description of its parts, and some general concepts I learned or would do differently. The subsequent posts in this series will dive deeper into the aforementioned parts. 

## Heroku, thou art a cruel and angry god
Back in January, I posted a story on my first [basic book recommender]({% post_url 2021-01-17-A-basic-book-rec %}) outlining a matrix factorization algorithm I wrote using [LightFM](https://making.lyst.com/lightfm/docs/home.html). This package worked really well and I was very pleased. After completion of this model I wanted to deploy the model to an app for people to log into and get some periodic recommendations. I decided on a Heroku server for this job as it had a free tier and was fairly simple to get going. All I needed was:

1. An application ran through Flask (which I will discuss later).
2. A model training file to compute new recommendations.
3. A `PROCFILE` to tell Heroku how the app will run (in this case I used the package [gunicorn](https://gunicorn.org/). Don't worry, I'll cover this stuff in a future post).
4. A `requirements.txt` file to inform Heroku what packages it needed to install to run the application. 

This all seems very easy; however, I ran into a significant problem. The free version of Heroku only allows you a certain amount of storage space devoted to the slug (basically all the packages you're installing). As it turns out, LightFM required a fairly large slug. Thus, I needed a new recommendation engine. So I built one!

## Building my own recommender
I needed to build a recommender that didn't rely on too many external libraries. This resulted in me writing up my own *gradient descent* (GD) function. The format in its most basic elements is as follows:

```python
# gradient of mean square error
def grad_user(utility_matrix, user_features, item_features, lmbda):
    error = utility_matrix - user_features*item_features.T
    (-2/df.shape[0]) * (error*item_features) + 2*lmbda*user_features

def grad_item(utility_matrix, user_features, item_features, lmbda):
    error = utility_matrix - user_features*item_features.T
    (-2/df.shape[0])*((error.T) *user_features) + 2*lmbda*item_features

# here I'm just pretending the utility matrix, feature embeddings, epochs and regularization were estableshed prior. The actual code is more rigorous than this
def grad_descent(utility_matrix, user_features, item_features, lmbda, epochs, learning_rate)
    for i in range(epochs):
        # update the gradient based on new feature matrices
        gu = grad_user(utility_matrix, user_features, item_features, lmbda)
        gi = grad_item(utility_matrix, user_features, item_features, lmbda)

        user_features = user_features - learning_rate*grad_user
        item_features = item_features - learning_rate*grad_item
    return user_features, item_features
```

Essentially, the goal of matrix factorization is to take your user-item interaction matrix $A$ (a large $ m \times n$ matrix where $n$ is the number of users and $m$ is the number of books) and decompose it into two latent-feature matrices
\$$ A = B\;C^{\text{T}}, \$$
where $B$ is an $m \times q$ matrix  of $q$ latent features describing all users and $C$ is a $n \times q$ matrix of $q$ latent features describing all items. Matrices $B$ and $C$ are found iteratively through GD. If you want to learn more about this process you can check out my earlier post, [A Simple Book Recommender with LightFM]({% post_url 2021-01-17-A-basic-book-rec %}). Or you can hold out for a future post diving further into the code for this project!

Also be sure to check out this [post](https://towardsdatascience.com/recommender-systems-matrix-factorization-using-pytorch-bd52f46aa199) on medium for a deep dive into matrix factorization! It's good stuff.

## Creating a Flask App
The next challenge was creating an application for users to interact with. Since this is a very public-friendly project, I opted to create a full website instead of a simple API. In order to do this I leaned heavily on [Corey Schaffer's Youtube series on Flask](https://www.youtube.com/watch?v=MwZwr5Tvyxo&list=PL-osiE80TeTs4UjLw5MM6OjgkjFeUxCYH). I can't recommend these tutorials enough, Corey rocks!

This is the portion of the project that took me the longest and I learned the most. 

- First I learned basic Flask routes and some HTML to get an initial site running. Flask routes are essentially triggered by a user visiting certain pages of your site. So if a user navigates to `yoursite.com/home` then your flask app will run some code then possible render some html for you. An example:

```python
@app.route("/home")
def index():
    mult = 2*2
    return render_template('index.html')
```
This will load the variable mult with 4 then render the `index.html` template in your templates folder. This is the foundation of flask.

- I then put together various pages and routes. I am most proud of my search engine which queries a database I host on AWS (which I talk about in the next section). I'd like to get a search engine that can handle typos in the future, but let's just handle one step at a time. For this step I needed to learn SQLAlchemy, a package that allows python to interface with a SQL database. 

- I then had to learn about forms, models, registration and logging in of users with password encryption, environment variables and more! Most of this can be found in Corey's series (I seriously cannot recommend this enough). 

This part of my project was enourmous and will definitely make a good blog post where I dive into details. Be sure to check that out when it's up!

## Databasing and model training with AWS

I finally had my app, app host and model in a state I was comfortbale with; however, to finish deployment I needed a way to retrain my model to make new recommendations and a way to store/pull/push data to an online databse. For all of these tasks I chose AWS.

### AWS RDS
Amazon Web Services: Relational Database Server is a service that allows you to host a number of different types of databases (which can be free depending on how long you've been a member and how much data you're storing/moving around). I chose PostgreSQL since it's the database I see the most often on job postings. I spent some time learning PostgreSQL, SQL, and RDS; however, once you figure out the pricing scheme for AWS the process is pretty easy. 

### AWS EC2
I finally needed a place to retrain my model daily. Since I didn't want to leave my computer on my Linux partition 24/7, I decided cloud computing was the way to go. I messed with AWS Lambda for a while, but realized my code base was too large to host there. I decided to purchase a small EC2 instance and deploy my model there (I realized after the fact that this defeats the purpose of me creating my own algorithm to get a small Heroku slug size, oh well!). EC2 is slightly more complex to setup than RDS. I'll be going over both of these in detail in a future post.

# Moving Forward
----------------
In the future I would like to spend some more time on model training. The recommendations given are alright but not exactly what I expect. For example, I am a huge fan of *The Wheel of Time* by Robert Jordan and have rated all of those books on my account. I expect the algorithm to recommend some other fantasy authors like Brandon Sanderson and Steven Erikson; however, I tend to get more obscure recommendations. I suspect getting the MSE below 1 will likely help with this issue.

Be on the look out for more blog posts where I dive into the data, the model training, and the flask app and possible shorter blog posts on setting up Heroku and AWS! I'm also starting a new project on analyzing some data from the MLB. Be on the lookout for that! As always you can find me on [LinkedIn](https://www.linkedin.com/in/colin-bradley-data-motivated/) and [Twitter](https://twitter.com/home).

<br>
_Thanks to [Marten Newhall](https://unsplash.com/photos/uAFjFsMS3YY) for the header photo. Also, make sure to check out my [Github repo](https://github.com/ColinB19/BookRecommender) to look at my code for this and other projects. As always please email me with any questions or comments you may have._

