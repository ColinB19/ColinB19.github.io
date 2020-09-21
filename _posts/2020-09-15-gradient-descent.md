---
title: Gradient Descent
date: 2020-09-15 16:00:00 +/-0800
categories: [Data Science, From Scratch Algorithms]
tags: [jupyter, learning, machine learning]     # TAG names should always be lowercase
image: /assets/img/posts/gradient-descent/gd.jpg
math: true
---

# Gradient Descent

The goal of this notebook is to gain an intuitive understanding of gradient descent (GD). We will be roughly following along with the Gradient Descent chapter of *Data Science from Scratch* by Joel Grus; however, we will be tackling it with slightly more mathematical formality. I do not wish to discuss the meaning of a derivative.

GD is an algorithm to minimize a function by iteritavely taking small steps in the opposite direction of the gradient of the function. Remembering vector calculus, the gradient of a function is in the direction of maximum increase of the function. The gradient of a function $f(x_1, x_2, ..., x_n)$ is as follows: \$$ \vec{\nabla}f(x_1, x_2, ..., x_n) = \sum_{i = 1}^{n} \hat{x}_i \frac{\partial f}{\partial x_i}, \$$  where $\hat{x}_i$ denotes a unit vector in the direction of that particular coordinate. Let's start by writing an approximate algorithm to compute this.


```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import california_housing
```


```python
# writing a gradient algorithm
def der(f,x,k,h):
    '''
    Computes the derivative in the k-th direction.
    '''
    small_change = [val + (h if index==k else 0) for index,val in enumerate(x)]
    a = f(x)
    b = f(small_change)
    return (b - a)/h

def grad(f,x,h = 0.01):
    '''
    Computes the gradient via a for loop.
    '''
    return [der(f,x,i,h) for i in range(len(x))]
def f(x):
    '''
    Just a random function to start off with. Note we
    are using a scalar function... We could compute the gradient of a vector function
    as wel.. another time.
    '''
    return np.sum([np.sin(q**2) for q in x])
```


```python
grad(f,[1,2,3,1,0,8])
```




    [1.0689369475964217,
     -2.559569368171921,
     -5.547003376646842,
     1.0689369475964217,
     0.009999999983323349,
     5.070278199942724]



## An algorithm for GD

Now that we have a function that will compute the gradient, what do we do if we want to minimize a particular function $f(x_1, x_2, ..., x_n)$? We could, and will, compute the gradient of the function then take small steps in the negative direction of the gradient iteratively until we reach a local minimum! These step sizes are often called the *learning rate*. Picking a learning rate is not much of a science; too large a rate will cause your algorithm to diverge and skip the minimum entirely, too small a rate will take too long to reach the minimum. There is also something to be said for functions with more than one minimum (and possibly no global minimum). For now let's start off with an easy example: the magnitude of a vector.


```python
def GD_naiive(x,f,step):
    '''
    The actual GD step. The learning rate must be positive here since we
    take into acount the direction in the function.
    '''
    if (step <= 0):
        print("Pick a positive-definite step")
        return 0
    else:
        gradient = grad(f,x)
        x_n = [old - (step*gradient[index]) for index, old in enumerate(x)]
        return x_n
def mag(x):
    '''
    Just the magnitude of a vector.
    '''
    return np.sum([val**2 for val in x])
```


```python
# great, let's test it out. We start from a random position and iterate many times to get a "minimum".
vec = np.random.uniform(-1,1,5)
for i in range(10**4):
    vec = GD_naiive(vec,mag,0.01)
vec
```




    [-0.005000000000000022,
     -0.005000000000000022,
     -0.005000000000000022,
     -0.004999999999999978,
     -0.004999999999999978]



## Applying GD to Linear Regression
Check out this great [blog post](https://towardsdatascience.com/gradient-descent-from-scratch-e8b75fa986cc), I will be following it fairly closely.

We have seen how GD works in a first principles sense, but our goal is to become good Data Scientists. To do this we need to apply our knowledge to examples. Let's look at some basic linear regression. \$$\hat{y} = mx + b\$$ Linear regression is the act of fitting a line to data. An example of data with a linear relationship is the height of a person and pant sizes. The "fitting" part is the job of GD. The best fit line is one that minimizes the mean square error (MSE), or whatever error function you happen to choose. The square error for a particular data point is \$$SE_i = (y - y_i)^2,\$$where $y$ is the true value of the parameter and $y_i$ is a particular data point. Thus, the MSE is \$$MSE = \frac{1}{n}\sum_{i=1}^{n} (y - y_i)^2.\$$ We wish to minimize this error by optimizing the parameters that affect $y$! We could plug this into a gradient alogorithm; however, since we know the analytical expression for the error function the gradient is simple to calculate. Let
\$$f(m,b) \equiv \frac{1}{n}\sum_{i=1}^{n} (mx_i + b - y_i)^2.\$$ Then the gradient is
\$$\vec{\nabla}f(m,b) = \frac{1}{n}\sum_{i=1}^{n} (2x_i(mx_i + b - y_i)\hat{m}+ 2(mx_i + b - y_i)\hat{b}),\$$ where $\hat{m}$ and $\hat{b}$ denote unit vectors in the directions of the parameters. My claim is that $\hat{m}\cdot \hat{b} = 0$, this means that a change in the intercept of a line does not change its slope. Thus, we have a orthonormal basis set and can consider them as cartesian vectors (this also requires the space to be flat but I think we'll abstract ourselves from that for now).

Let's simulate some data and try to fit a line to it.


```python
# we're going to look at the same data as the blog post I linked to earlier!
housing_data = california_housing.fetch_california_housing()

```

<!--
```python
# pulling features and labels from dataset
Features = pd.DataFrame(housing_data.data, columns=housing_data.feature_names)
Target = pd.DataFrame(housing_data.target, columns=['Target'])
df = Features.join(Target)
df.head()
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
      <th>MedInc</th>
      <th>HouseAge</th>
      <th>AveRooms</th>
      <th>AveBedrms</th>
      <th>Population</th>
      <th>AveOccup</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8.3252</td>
      <td>41.0</td>
      <td>6.984127</td>
      <td>1.023810</td>
      <td>322.0</td>
      <td>2.555556</td>
      <td>37.88</td>
      <td>-122.23</td>
      <td>4.526</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8.3014</td>
      <td>21.0</td>
      <td>6.238137</td>
      <td>0.971880</td>
      <td>2401.0</td>
      <td>2.109842</td>
      <td>37.86</td>
      <td>-122.22</td>
      <td>3.585</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.2574</td>
      <td>52.0</td>
      <td>8.288136</td>
      <td>1.073446</td>
      <td>496.0</td>
      <td>2.802260</td>
      <td>37.85</td>
      <td>-122.24</td>
      <td>3.521</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5.6431</td>
      <td>52.0</td>
      <td>5.817352</td>
      <td>1.073059</td>
      <td>558.0</td>
      <td>2.547945</td>
      <td>37.85</td>
      <td>-122.25</td>
      <td>3.413</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.8462</td>
      <td>52.0</td>
      <td>6.281853</td>
      <td>1.081081</td>
      <td>565.0</td>
      <td>2.181467</td>
      <td>37.85</td>
      <td>-122.25</td>
      <td>3.422</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[['MedInc', 'Target']].describe()
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
      <th>MedInc</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>20640.000000</td>
      <td>20640.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.870671</td>
      <td>2.068558</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.899822</td>
      <td>1.153956</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.499900</td>
      <td>0.149990</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.563400</td>
      <td>1.196000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.534800</td>
      <td>1.797000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.743250</td>
      <td>2.647250</td>
    </tr>
    <tr>
      <th>max</th>
      <td>15.000100</td>
      <td>5.000010</td>
    </tr>
  </tbody>
</table>
</div>




```python
# I want to remove outliers. If you look at the descrepency between the 3rd quantile and the max on both columns,
# you can see the issue. We want to remove all rows where either MedInc > 5 or Target > 3 But we should look to
# see how many of these points there are
plt.figure(figsize=(15,5))

#Plotting data without connecting lines
plt.plot(df.MedInc,df.Target,'.')

#Labels and options
plt.title("Median Household Income vs Price of house")
plt.ylabel("Price of House")
plt.xlabel("Med Income")
plt.grid(True)
```


![png]({{"/assets/img/posts/gradient-descent/output_11_0.png" | absolute_url}})



```python
# let's ignore MedInc > 10 and Target > 5... we have that strange line at 5... I imagine this is a " > 5" data point.
df = df[df.MedInc < 10]
df = df[df.Target < 5]

```


```python
plt.figure(figsize=(15,5))

#Plotting data without connecting lines
plt.plot(df.MedInc,df.Target,'.')

#Labels and options
plt.title("Median Household Income vs Price of house")
plt.ylabel("Price of House")
plt.xlabel("Med Income")
plt.grid(True)
```


![png]({{"/assets/img/posts/gradient-descent/output_13_0.png" | absolute_url}})


```python
# now let's normalize these variables
def normalize(df,cols):
    for name in cols:
        minimum = df[name].min()
        maximum = df[name].max()
        df[name] = df[name].apply(lambda x: (x-minimum)/(maximum-minimum))
    return 0
```


```python
normalize(df,['Target','MedInc'])
```




    0




```python
df[['MedInc','Target']].describe()
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
      <th>MedInc</th>
      <th>Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>19599.000000</td>
      <td>19599.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.335776</td>
      <td>0.364730</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.162427</td>
      <td>0.199512</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.215180</td>
      <td>0.209256</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.312877</td>
      <td>0.327207</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.432987</td>
      <td>0.479964</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>
 -->



```python
plt.figure(figsize=(15,5))

#Plotting data without connecting lines
plt.plot(df.MedInc,df.Target,'.')

#Labels and options
plt.title("Median Household Income vs Price of house")
plt.ylabel("Normalized Price of House")
plt.xlabel("Normalized Med Income")
plt.grid(True)
```

![png]({{"/assets/img/posts/gradient-descent/output_17_0.png" | absolute_url}})



```python
# these functions will only work for this particular dataset
def MSE(data,m,b):
    '''
    Just computes the MSE.
    '''
    data['SE'] = (m*data['MedInc']+b - data['Target'])**2
    return (1/len(data))*data['SE'].sum()

def MSE_grad(data,m,b):
    '''
    Returns the gradient MSE.
    '''
    data['m_sum'] = 2*data['MedInc']*(m*data['MedInc']+b - data['Target'])
    data['b_sum'] = 2*(m*data['MedInc']+b - data['Target'])
    new_m = (1/len(data))*data['m_sum'].sum()
    new_b = (1/len(data))*data['b_sum'].sum()
    # print("m: ", new_m, "\nb: ", new_b)
    return {'m':new_m,'b':new_b}

def Lin_regression(data,lr = 0.05, epoch = 10,m = 1, b = 0):
    '''
    This performs regression with gradient descent
    '''
    log, mse = [],[]
    for i in range(epoch):
        grad_err = MSE_grad(data,m,b)
        m -= lr*grad_err['m']
        b -= lr*grad_err['b']
        mse.append(MSE(data,m,b))
        log.append([m,b])
    return m,b,log,mse

def line(data,m,b):
    '''
    This is just the eq. of a line
    '''
    ran = np.linspace(data['MedInc'].min(),data['MedInc'].max(),1000)
    return ran, m *ran + b

def plot_regression(data, pred_m, pred_b, log=None, title="Linear Regression With Gradient Descent"):
    '''
    This plots the results as well as the steps.
    '''
    ran, regr = line(data,pred_m,pred_b)

    plt.figure(figsize=(16,6))
    plt.rcParams['figure.dpi'] = 227
    plt.scatter(data['MedInc'], data['Target'], label='Data', c='#388fd8', s=6)
    if log != None:
        for i in range(len(log)):
            temp_ran, temp_regr = line(data,log[i][0],log[i][1])
            plt.plot(temp_ran, temp_regr, lw=1, c='#caa727', alpha=0.3)
    plt.plot(ran, regr, c='#ff7702', lw=2, label='Regression')
    plt.title(title, fontSize=14)
    plt.xlabel('Income', fontSize=11)
    plt.ylabel('Price', fontSize=11)
    plt.legend(frameon=True, loc=1, fontsize=10, borderpad=.6)
    plt.tick_params(direction='out', length=6, color='#a0a0a0', width=1, grid_alpha=.6)
    plt.grid(True)
    plt.show()
```


```python
final_m, final_b, mb_log, error_log = Lin_regression(data = df,lr = 0.075, epoch = 100,m = 0.5, b=0.1)
plot_regression(df,final_m,final_b,log = mb_log)
final_m, final_b, mb_log, error_log = Lin_regression(data = df,lr = 0.075, epoch = 100,m = 0.1, b=0.8)
plot_regression(df,final_m,final_b,log = mb_log)
final_m, final_b, mb_log, error_log = Lin_regression(data = df,lr = 0.075, epoch = 100,m = 0.85, b=-0.2)
plot_regression(df,final_m,final_b,log = mb_log)
```


![png]({{"/assets/img/posts/gradient-descent/output_20_0.png" | absolute_url}})



![png]({{"/assets/img/posts/gradient-descent/output_20_1.png" | absolute_url}})



![png]({{"/assets/img/posts/gradient-descent/output_20_2.png" | absolute_url}})
