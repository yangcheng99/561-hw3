# RegulizationFit Documentation
In this homework, we finetunes a blackbox regression model based on three methods: **Dropout**,**NoiseAddtion** and **Robustness**
# Installation
1. Install scikit-learn
```ruby
pip install scikit-learn
```
2. Install numpy
```ruby
pip install numpy
```
3. Install copy
```ruby
pip install copy
```
4. Install random
```ruby
pip install random
```
# Getting Started
You can see the validation part in the code for the basic usage of our model.

# Input Parameters
- f(object)
  - An existing model that takes $X\in \mathbb{R}^{n\times p}$ and $Y\in \mathbb{R}^{n}$ and returns a function that maps $X$ to $Y$
- X(numpy array)
  - The training data for the model f.
- Y(numpy array)
  - The training label for the model f.
- Training data $X\in \mathbb{R}^{n\times p}$ and $Y\in \mathbb{R}^{n}$
- method:(string) $\text{\{"Dropout", "NoiseAddition", "Robustness"\}}$
  - One regularization method from either $\{\text{Dropout, NoiseAddition, Robustness}\}$
- eval_criteria:(string) $\{\text{"MSE","MAD"}\}$
  - Way to evaluate the method. MSE refers to mean square error and MAD refers to mean absolute deviation.
  
- M(int)
  - The number of Monte Carlo repetitions
- K(int)
  - The number of folds in cross-validations 
- c(list)
  - a list of vector that is required in the robustness method
- para_list(list) $\text{\{"Default = None"\}}$
  - a list of parameters for the noise and dropout. 
  - For the dropout, it should be a list of dropout proportion between 0 and 1. The default setting is: $[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]$
  - For the noise addition, it should a list of varience greater than 0. The default setting is: $[0.5, 1, 2, 3, 4, 5, 6, 7, 8]$
# Output
- final_predictive_model(object)
  - The model trained by the best parameters.
  
# Basic Regulization Techniques 
**Noise Addition**

We add a noise varible which is gaussian with mean 0 and variance $\sigma^2$. The codes are shown below

```ruby
def noiseadd(X,var):
  """
  This function adds noise with a normal distribution with mean equals to 0 and the variance equals to input var
  X - the input data
  var - the square of sigma
  return - the X with added noise
  """
  #noise=[]
  noise = np.random.normal(0,var,X.shape)
  return X + noise 
```

**DropOut**

The idea of behind dropout is that we randomly drop serveral columns based on the *prop*, which is the proportion of dropout of the columns. The code are shown below
```ruby
def dropout(X,p):
  """
  This function drops some points in some features in the input X under Bernoulli distribution
  X - the input data
  p - the dropout proportion
  return - a output matrix with some points set to zero
  """
  # check the proportion is valid
  if (p < 0 or p >= 1): 
    raise ValueError("Invalid dropout Value!")
  else:
    dropout_matrix = np.random.binomial(1,p,X.shape)
    return X*dropout_matrix
```
```ruby
def dropout_v2(X,p, max_iters=None):
  """
  This function drops a subset of features in X across all samples followed by Bernoulli distribution
  X - the input data
  p - the dropout proportion
  max_iters - the number of candidate feature subsets to drop
  return - two lists: one of data matrices with zero-ed out features and the 
           other with the feature masks used for dropout
  """
  # check the proportion is valid
  if (p < 0 or p >= 1): 
    raise ValueError("Invalid dropout Value!")
  else:
    if max_iters is None:
        max_iters = int(X.shape[1]*p)**2
    dropout_matrix_options = []
    masks = []
    for _ in range(max_iters):
        features_to_drop = np.random.binomial(1,p,X.shape[1])
        masks.append(np.concatenate([features_to_drop for _ in range(X.shape[0])]).reshape(X.shape))
        dropout_matrix_options.append(features_to_drop)
    return [X*dropout_matrix for dropout_matrix in masks], dropout_matrix_options
  
```
# Parameter Finetuning
**DropOut/NoiseAddition**

For these two methods, we the Monte Carlo M times for each element in the paralist.We then find the correspoding MSE or MAD. After we have run all the experiment, we find the parameter with the smallest MSE or MAD and return the model associated with that parameter.

**Robust**

We do the sampling delta M times for each c in the c_list, where M is determined by size of X with a max limit. Then, we fit the model with X plus different delta and keep track of MSE. After that, we find the max delta by finding the max MSE and record MSE for each c. Next, by locating the min MSE of c, we can get the best c in the list of c. Finally, we return the model fitted by the best parameter.
