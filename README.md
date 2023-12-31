# machine_learning-andrew_ng

This is the [coursera course](https://www.coursera.org/learn/machine-learning-course) from Andrew Ng

# 1. Environment Setup

<br><br><br>

```bash
# 1. setup local environment
$ pyenv local 3.10.8
# $ pyenv local 3.8.15

# 2. setup virtual environment
$ python3.10 -m venv .venv
# $ python3.8 -m venv .venv

# 3. set intepreter
>> CMD + Shift + P ==> Interpreter ==> ./.venv/bin/python

# 4. check current python3 version
$ python3 --version
$ python3 -m pip --version
$ pip install --upgrade pip

# 5. activate virtual envrionment
$ source ./.venv/bin/activate

# 6. tensorflow installation
$ python3 -m pip install --default-timeout=10000 tensorflow-macos


$ pip install --upgrade pip
```

select current virutal environment

```bash
# downgrade python version
# 1. check all python versions
$ pyenv install --list

# 2. install 3.8.15
$ pyenv install 3.8.15

# 3. check all version
$ pyenv versions

# 4. remove .venv folder

# 5. repeat virual env setup steps

```

<br><br><br>

## 1.2 Tips

### 1.2.1 references

[numpy_docs](https://numpy.org/doc/stable/index.html)

<br><br><br>

# Week-1

## 1.1 Introduction

### 1.1.1 Intro

- Machine Learning
  - Grew out of work in AI
  - New capability for computers
- Exmaples:
  - Database mining
    - Large datasets from growth of automation/web
    - E.g. Web click data, medical records, biology, engineering
  - Applications can't program by hand
    - E.g. Autonomous helicopter, handwriting recognition, most of NLP, Computer Vision
  - Self-customizing programs
    - E.g. Amazon, Netflix product recommendations
  - Understanding human learning(brain, real AI)

<br><br><br>

### 1.1.2 What is Machine Learning?

- Machine Learning Definition

  - Arthur Samuel(1959). Machine Learning: Field of study that gives computers the ability to learn without being explicitly programmed
  - Tom Mitchell(1998) well-posed learning problem: A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improved with experience E.
    - Sample-1
      - `E` the exprience of playing many games of checkers
      - `T` the taks of playing checkers
      - `P` the prob that the program will win the next game
    - Sample-2
      - `E` Watching u label emails as spam or not spam
      - `T` Classifying emails as spam or not spam
      - `P` The number (or fraction) of emails correctly classified as spam/ not spam

- Machine learning algorithms:

  - Supervised learning
  - Unsupervised learning
  - others: Reinforcment learning, recommender systems

- Practical advice for applying learning algorithms

<br><br><br>

### 1.1.3 Supervised Learning

- We are given a data set and already know what our correct output should look like, having the idea that there is a relationship between the input and output
- `Regression`: we are trying to predict results within a continuous output, meaning that we are trying to map inut variables to some continous function

- `Classification`: we are trying to predict results in a discrete output. In other words, we are trying to map input variables into discrete categories

- Example 1:

  - Given data about size of houses on the real estate market, try to predict their price. Price as a function of size is a continuous output, this is a regression problem
  - By making our output as `whether the house sells for more or less than the asking price`, we turn it into a classification problem

- Example 2:

  - House price prediction

    - Supervised learning: "right answers" given
    - Regression: Predict continuous valued output (price)
    - <img src="./imgs/Xnip2023-02-09_14-55-58.jpg" alt="imgs" width="800" height="300"><br><br><br><br><br><br>

  - Breast cancer(malignant, benign)

    - Classification: discrete valued output(0 or 1)
    - <img src="./imgs/Xnip2023-02-09_14-59-40.jpg" alt="imgs" width="800" height="300"><br><br><br><br><br><br>

  - Breast cancer another approach
    - Deal with infinite features (SVM - supported vector machine)
    - Clump thickness, Uniformity of Cell Size, Uniformity of Cell Shape
    - <img src="./imgs/Xnip2023-02-09_15-01-53.jpg" alt="imgs" width="400" height="300"><br><br><br><br><br><br>

<br><br><br>

### 1.1.4 Unsupervised Learning

- Unsupervised Learning

  - We can derive structure from data where we don't neccessarily know the effect of the variables
  - we can derive this structure by clustering the data based on relationships among the variables in the data.
    - <img src="./imgs/Xnip2023-02-09_15-51-57.jpg" alt="imgs" width="350" height="300"><br><br><br><br><br><br>

- Example:
  - clustering: take a collection of 1,000,000 different genes, and find a way to automatically group these genes into groups that are somehow similar or related by different variables, such as lifespan, location, roles and so on
    - <img src="./imgs/Xnip2023-02-09_15-53-07.jpg" alt="imgs" width="800" height="200"><br><br><br><br><br><br>
    - <img src="./imgs/Xnip2023-02-09_15-54-07.jpg" alt="imgs" width="600" height="400"><br><br><br><br><br><br>
  - non-clustering: the "cocktail party algorithm", allows u to find structure in a chaotic environment (i.e. identityfing individual voices and music from a mesh of sounds at a cocktail party)
    - `[W,s,v] = svd((repmat(sum(x.*x, 1), size(x, 1), 1).*x)*x')`;
      - svd - singular value decomposition

<br><br><br>

## 1.2 Model and Cost Function

### 1.2.1 Model Representation

- x<sup>(i)</sup> denotes input
- y<sup>(i)</sup> denotes output
- (x<sup>(i)</sup>, y<sup>(i)</sup>); i = 1, ...,m - is called a training set
- to learn a function `h: X -> Y` so that h(x) i a "good" predictore for the corresponding value of y. this function called `hypothesis`

- Example
  - Housing Prices (Portland, OR)
    - supervised learning: given the "right answer" for each example in the data
    - Regression problem: predicted real-valued output
      - <img src="./imgs/Xnip2023-02-09_16-58-00.jpg" alt="imgs" width="600" height="250"><br><br><br><br><br><br>
      - <img src="./imgs/Xnip2023-02-09_16-59-09.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
      - <img src="./imgs/Xnip2023-02-09_16-59-56.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

### 1.2.2 Cost Function

- we can measure the accuracy of our hypothesis by using a `cost function`. this takes an average difference of all results of the hypothesis with inputs from x's and the actual ouput y's

  - <img src="./imgs/Xnip2023-02-13_15-00-33.jpg" alt="imgs" width="600" height="100"><br><br><br>
  - 1/2\*x where x is the mean of squares of h<sub>θ</sub>(x<sub>i</sub>) - y<sub>i</sub>, or the difference between `the predicted value and actual value` <br><br><br>

- this function is otherwise called the `Squared Error Function` or `Mean squared error`. The mean is halved (1/2) as a convenience for the computation of the gradient descent, ad the derivative term of the square function will cancel out the 1/2 term

  - <img src="./imgs/Xnip2023-02-13_14-53-25.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
  - <img src="./imgs/Xnip2023-02-13_14-54-27.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
  - <img src="./imgs/Xnip2023-02-13_14-54-49.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

### 1.2.3 Cost Function Intuition I

- Our training data set is scattered on the x-y plane. We are trying to make a straight line pass through all these scattered data points. The best possible line will be such so that the average squared vertical distances of the scattered points from the line will be the least. In such case, the value of J(θ<sub>0</sub>, θ<sub>1</sub>) will be 0

  - <img src="./imgs/Xnip2023-02-13_15-39-44.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

- when θ<sub>1</sub> = 1, we get a slope of 1 which goes through all single data in our model
- when θ<sub>1</sub> = 0.5, we see the vertical distance from out fit to the data points increase
  - <img src="./imgs/Xnip2023-02-13_15-39-15.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
- we should try to minimize the cost function, in this case, θ<sub>1</sub> = 1 is our global minimum
  - <img src="./imgs/Xnip2023-02-13_15-26-07.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

### 1.2.4 Cost Function Intuition II

- a contour line of two variable function has a constant value at all points of the same line
- The three green points below have the same value of J(θ<sub>0</sub>, θ<sub>1</sub>), the are found along the same line.

  - <img src="./imgs/Xnip2023-02-13_16-35-08.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

- when θ<sub>0</sub> = 360 and θ<sub>1</sub> = 0, the value of J(θ<sub>0</sub>, θ<sub>1</sub>) contour plot gets closer to the center thus reducing the cost function error
  - <img src="./imgs/Xnip2023-02-13_16-38-21.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
- the graph below minimizes the cost function as much as possible and consequently, the result of θ<sub>0</sub> and θ<sub>1</sub> tend to be around 0.12 and 250 respectively. Plotting those values on our graph seems to put our point in the center of the inner most `circle`

## 1.3 Parameter Learning

### 1.3.1 Gradient Descent

- we need to estimate the parameters in the hypothesis function. that's where `gradient descent` comes in. we put θ<sub>0</sub> on x axis and θ<sub>1</sub> on y axis, with the cost function on the vertical z axis.
- <img src="./imgs/Xnip2023-02-13_16-54-33.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

  - if choose a different start point
    - <img src="./imgs/Xnip2023-02-13_16-54-56.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

- the way we do this is by taking the derivative of our cost function. the slope of the tangent is the derivative at that point and it will give us a direction to move towards. - we make steps down the cost fucntion in the direction with the steepest descent - the size of each step is determined by the parameter `𝛂`, called learning rate

  - A smaller `𝛂` result in a smaller step
  - A larger `𝛂` result in a larger step
  - the direction in which the step is taken is determined by the partial derivative of J(θ<sub>0</sub>, θ<sub>1</sub>).
  - Depending on where on starts on the graph

- repeat until convergence:

  - j = 0,1 represents the feature index number
  - `:=` assignment operator
  - <img src="./imgs/Xnip2023-02-13_17-12-57.jpg" alt="imgs" width="300" height="70"><br><br><br><br><br><br>

- at each iteration j, one should simultaneously update the parameters θ<sub>1</sub>, θ<sub>2</sub>, ..., θ<sub>n</sub>. `Updating a specific parameter prior to calculating another one on the` j<sup>th</sup> `iteration would yield to a wrong implemention`

  - <img src="./imgs/Xnip2023-02-13_17-03-16.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

<br><br><br>

### 1.3.2 Gradient Descent Intuition

- <img src="./imgs/Xnip2023-02-13_17-35-38.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

- <img src="./imgs/Xnip2023-02-13_17-38-41.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

- if θ<sub>1</sub> stays in local minimal, then it stays unchanged as slope is 0

  - <img src="./imgs/Xnip2023-02-13_17-40-12.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

- Gradient descent can converge to local minimum, even with the learning rate `𝛂` fixed. As we approach a local minimum, gradient descent will automatically take smaller steps (slope to be 0 to the local minimum, so the slope is getting smaller). So no need to decrease `𝛂` over time
  - <img src="./imgs/Xnip2023-02-13_17-44-49.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

<br><br><br>

### 1.3.3 Gradient Descent for linear regression

- Substitute our actual cost function and our actual hypothesis and modify equation to

  - m - size of the traning set
  - θ<sub>0</sub> - the constant
  - θ<sub>1</sub> - the constant, changing simultaneously with θ<sub>1</sub>
  - x<sub>i</sub>, y<sub>i</sub> are values of the given training set(data)
  - <img src="./imgs/Xnip2023-03-23_16-51-50.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

- Derivative

  - <img src="./imgs/Xnip2023-03-23_16-55-36.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

- The point of all this is that if we start with a guess for our hypothesis and then repeatedly apply these gradient descent equations, our hypothesis will become more and more accurate.

- `Batch Gradient Descent` (scale better in large dataset)

  - Each step of gradient descent uses all the training example
  - bowl shape - convex function
    - no local minimal but one global minimal
  - <img src="./imgs/Xnip2023-03-23_16-56-50.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

- normal equation method

<br><br><br><br><br><br>

## 1.4 Linear Algrebra Review

<br><br><br>

### 1.4.1 Matrices and Vectors

- A<sub>ij</sub> = "i,j entry" in the i<sup>th</sup> row, j<sup>th</sup> column
- uppercase for matrix, lowercase for vector
- A vector with 'n' rows is referred to as an 'n'-dimensional vector.
- v<sub>j</sub> refers to the element in the ith row of the vector
- all our vectors and matrices will be 1-indexed. Note that for some programming languages, the arrays are 0-indexed.
- "Scalar" means that an object is a single value, not a vector or matrix.
- ℝ refers to the set of scalar real numbers.
- ℝ<sup>n</sup> refers to the set of n-dimensional vectors of real numbers

```py
% The ; denotes we are going back to a new row.
A = [1, 2, 3; 4, 5, 6; 7, 8, 9; 10, 11, 12]

% Initialize a vector
v = [1;2;3]

% Get the dimension of the matrix A where m = rows and n = columns
[m,n] = size(A)

% You could also store it this way
dim_A = size(A)

% Get the dimension of the vector v
dim_v = size(v)

% Now let's index into the 2nd row 3rd column of matrix A
A_23 = A(2,3)

```

<br><br><br>

### 1.4.2 Addition and Scalar mutiplication

- Addition and subtraction are element-wise
- In scalar multiplication, we simply multiply every element by the scalar value
- In scalar division, we simply divide every element by the scalar value

<br><br><br>

### 1.4.3 Matrix vector multiplication

- The result is a vector. The number of columns of the matrix must equal the number of rows of the vector.
- An m x n matrix multiplied by an n x 1 vector results in an m x 1 vector.

<br><br><br>

### 1.4.4 Matrix matrix multiplication

- <img src="./imgs/Xnip2023-03-23_19-06-04.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

- prediction of first h<sub>θ</sub>

  - <img src="./imgs/Xnip2023-03-23_19-13-39.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

- An m x n matrix multiplied by an n x o matrix results in an m x o matrix. In the above example, a 3 x 2 matrix times a 2 x 2 matrix resulted in a 3 x 2 matrix.

<br><br><br>

### 1.4.5 Matrix multiplication properties

- Commutative

  - reverse the order of matrices muliplication, it even result in different dimensions
  - <img src="./imgs/Xnip2023-03-23_19-21-25.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

- Associative

  - <img src="./imgs/Xnip2023-03-23_19-20-59.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

- `Identity Matrix`
  - <img src="./imgs/Xnip2023-03-23_19-25-39.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

<br><br><br>

### 1.4.6 inverse and transpose

- Matrix inverse

  - [calculate inverse of matrix manually](https://www.youtube.com/watch?v=Fg7_mv3izR0)
  - <img src="./imgs/Xnip2023-03-23_19-40-09.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

- Matrix transpose
  - <img src="./imgs/Xnip2023-03-23_19-43-46.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

<br><br><br><br><br><br>

# Week-2

<br><br><br>

## 2.1 Environment setup instructions

<br><br><br>

## 2.2 Multivariate Linear Regression

<br><br><br>

### 2.2.1 Multiple Features

- Notation:

  - n = the number of features
  - m = the number of training examples
  - x<sup>(i)</sup> = input (features) of i<sup>th</sup> training example
  - x<sup>(i)</sup><sub>j</sub> = value of feature j in i<sup>th</sup> training example

  - <img src="./imgs/Xnip2023-03-27_08-23-03.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

- hypothesis
  - The multivariable form of the hypothesis function accommodating these multiple features is as follows:
    - <img src="./imgs/Xnip2023-03-27_08-30-03.jpg" alt="imgs" width="700" height="80"><br><br><br>
  - 0 feature x<sup>(i)</sup><sub>0</sub> = 1
  - we can think about θ<sub>0</sub> as the basic price of a house, θ<sub>1</sub> as the price per square meter, θ<sub>2</sub> as the price per floor, etc. x<sub>1</sub> will be the number of square meters in the house, x<sub>2</sub> the number of floors, etc.
  - <img src="./imgs/Xnip2023-03-27_08-28-42.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

<br><br><br>

### 2.2.2 Gradient Descent for multiple variables

- Basic theory

  - <img src="./imgs/Xnip2023-03-27_08-36-08.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

- Gradient Descent

  - the gradient descent equation itself is generally the same form; we just have to repeat it for our 'n' features
  - <img src="./imgs/Xnip2023-03-27_08-40-44.jpg" alt="imgs" width="600" height="350"><br><br><br>

  - repeat until convergence:
  - <img src="./imgs/Xnip2023-03-27_08-41-41.jpg" alt="imgs" width="600" height="100"><br><br><br><br><br><br>

<br><br><br>

### 2.2.3 Gradient Descent in Practice I - Feature Scaling

- We can speed up gradient descent by having each of our input values in roughly the same range.

- `Feature Scaling`
  - Idea **make sure features are on a similar scale**.
    - or gradient descent will take a long time to converge
    - <img src="./imgs/Xnip2023-03-27_08-59-57.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
  - get every feature into approximately a -1 <= x<sub>i</sub> <= 1 range
    - <img src="./imgs/Xnip2023-03-27_09-02-24.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
- `Mean normalization`
  - Replace x<sub>i</sub> with x<sub>i</sub> - 𝜇<sub>i</sub> to make features have approximately zero mean (do not apply to x<sub>0</sub> = 1)
  - 𝜇<sub>i</sub> - `average` of all the values for feature (i)
  - s<sub>i</sub> - range of (max_value - min_value), the standard deviation
  - <img src="./imgs/Xnip2023-03-27_09-10-12.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

<br><br><br>

### 2.2.4 Gradient Descent in Practice II - Learning Rate

- Gradient descent

  - **Debugging gradient descent**: how to make sure gradient descent is working correctly
    J(θ) should decrease after every iteration

    - <img src="./imgs/Xnip2023-03-27_09-21-54.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

  - **Automatic convergence test**: eclare convergence if J(θ) decreases by less than E in one iteration, where E is some small value such as 10<sup>−3</sup>. However in practice it's difficult to choose this threshold value

  - how to choose learning rate `𝛼`
    - <img src="./imgs/Xnip2023-03-27_09-16-54.jpg" alt="imgs" width="500" height="100"><br><br><br><br><br><br>
    - if graph as below, the cost function is increasing, you probably need use a smaller learning rate `𝛼`
    - for sufficiently small `𝛼`, J(θ) should decrease on every iteration
    - but if `𝛼` is too small, gradient descent can be slow to converge
      - <img src="./imgs/Xnip2023-03-27_09-25-33.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

- Summary
  - if `𝛼` is too small: slow convergence
  - if `𝛼` is too large: J(θ) may not decrease on every iteration; may not converge
  - to chooes `𝛼`, try (3x than previous)
    - .... -.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, ...

<br><br><br>

### 2.2.5 Features and Polynomial Regression

- Housing prices prediction

  - h<sub></sub>(x) = θ<sub>0</sub> + θ<sub>1</sub> x _frontage_ + θ<sub>2</sub> x _depth_
    - <img src="./imgs/Xnip2023-03-27_09-56-04.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

- Choice of features

## 2.3 Computing Parameters Analytically

<br><br><br>

### 2.3.1 Normal Equation

- Intuition

  - how to minimize a quadratic function?
  - <img src="./imgs/Xnip2023-03-27_10-30-54.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

- Equation

  - <img src="./imgs/Xnip2023-03-27_10-31-38.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

- example

  - <img src="./imgs/Xnip2023-03-27_10-50-27.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

- 𝜃 = (X<sup>T</sup> X)<sup>-1</sup> X<sup>T</sup> y

  - <img src="./imgs/Xnip2023-03-30_08-41-57.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

- when `gradient descent`, when `normal equation`
  - <img src="./imgs/Xnip2023-03-30_08-45-49.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

<br><br><br>

### 2.3.2 Normal Equation Noninvertibility

- Normal equation

  - <img src="./imgs/Xnip2023-03-30_08-58-14.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

- what if is X<sup>T</sup>X non-invertible `Rarely`
  - Redundant features (linearly dependent) `Delete redundant feature`
    - e.g. x1 = size in feet<sup>2</sup>
    - x2 = size in m<sup>2</sup>
  - Too many features (e.g. m <= n)
    - Delete some features, or use regularization

<br><br><br>

## 2.4 Submitting Programming Assignments

### 2.4.1 Working on and submitting programming assignments (using python package - numpy)

<br><br><br><br><br><br>

## 2.5 Python/ Octave/ Matlab tutorial

### 2.5.7 Vectorization

- <img src="./imgs/Xnip2023-04-03_15-55-32.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
- <img src="./imgs/Xnip2023-04-03_15-59-39.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
- <img src="./imgs/Xnip2023-04-03_16-00-37.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
- <img src="./imgs/Xnip2023-04-03_16-11-10.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

<br><br><br><br><br><br>

# Week-3

## 3.1 Classification and Representation

### 3.1.1 Logistic regression - Classification

1. classification
   - we will focus on the **binary classification problem** in which y can take on only two values, 0 and 1.
   - <img src="./imgs/Xnip2023-04-03_16-24-29.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
   - <img src="./imgs/Xnip2023-04-03_16-36-13.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
   - <img src="./imgs/Xnip2023-04-03_16-39-13.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

### 3.1.2 Logistic regression - Hypothesis representation

1. Logistic Regression Model
   - <img src="./imgs/Xnip2023-04-04_07-39-58.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
2. interpretation of Hypothesis output
   - probability that y = 1, given x, parameterized by θ
   - <img src="./imgs/Xnip2023-04-04_07-44-54.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

### 3.1.3 Logistic regression - Decision Boundary

1. Logistic regression

   - <img src="./imgs/Xnip2023-04-04_07-53-12.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
   - <img src="./imgs/Xnip2023-04-04_08-09-32.jpg" alt="imgs" width="300" height="100"><br><br><br><br><br><br>

2. Decision Boundary

   - this line called decision boundary
   - <img src="./imgs/Xnip2023-04-04_07-58-50.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

3. Non-linear decision boundaries
   - <img src="./imgs/Xnip2023-04-04_08-06-54.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

<br><br><br><br><br><br>

## 3.2 Logistic Regression Model

### 3.2.1 Cost Function

1. Cost function

   - <img src="./imgs/Xnip2023-04-04_09-10-50.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
   - <img src="./imgs/Xnip2023-04-04_09-15-22.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

2. Logistic regression cost function
   - <img src="./imgs/Xnip2023-04-04_09-20-20.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
   - y = 1
     - If our correct answer 'y' is 0, then the cost function will be 0 if our hypothesis function also outputs 0. If our hypothesis approaches 1, then the cost function will approach infinity.
     - <img src="./imgs/Xnip2023-04-04_09-22-32.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
   - y = 0
     - If our correct answer 'y' is 1, then the cost function will be 0 if our hypothesis function outputs 1. If our hypothesis approaches 0, then the cost function will approach infinity.
     - <img src="./imgs/Xnip2023-04-04_09-28-43.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

### 3.2.2 Simplified cost function and gradient descent

1. logistic regression cost function

   - <img src="./imgs/Xnip2023-04-04_09-46-57.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
   - <img src="./imgs/Xnip2023-04-04_09-49-09.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
   - vectorized implementation
     - <img src="./imgs/Xnip2023-04-04_10-05-42.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

2. Gradient Descent
   - <img src="./imgs/Xnip2023-04-04_09-51-26.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
   - <img src="./imgs/Xnip2023-04-04_10-02-29.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

### 3.2.3 Advanced optimization

1. optimization

   - <img src="./imgs/Xnip2023-04-04_10-09-33.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
   - Conjugate gradient
   - [BFGS](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm)
   - [L-BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS)
     - <img src="./imgs/Xnip2023-04-04_10-27-44.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

2. example
   - <img src="./imgs/Xnip2023-04-04_10-58-17.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
   - <img src="./imgs/Xnip2023-04-04_16-35-55.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

<br><br><br><br><br><br>

## 3.3 Multiclass Classification

### 3.3.1 Multiclass classification: one-vs-all

1. Multiclass classification

   - <img src="./imgs/Xnip2023-04-04_16-41-15.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
   - <img src="./imgs/Xnip2023-04-04_16-42-30.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

2. one-vs-all (one-vs-rest)
   - <img src="./imgs/Xnip2023-04-04_16-45-26.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
   - <img src="./imgs/Xnip2023-04-04_16-54-41.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

<br><br><br><br><br><br>

## 3.4 Solving the problem of overfitting

### 3.4.1 The problem of overfitting

1. Linear regression

   - <img src="./imgs/Xnip2023-04-04_17-38-16.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

2. Logistic regression

   - <img src="./imgs/Xnip2023-04-04_17-40-59.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

3. Addressing overfitting

   - <img src="./imgs/Xnip2023-04-04_17-43-42.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

   - Options:
     1. Reduce number of features
        - Manually select which features to keep
        - model selection algorithm (later in course)
     2. Regularization
        - Keep all the features, but reduce magnitude/ values of parameters
        - Regularization works well when we have a lot of slightly useful features (Works well when we have a lot of features, each of which contributes a bit to predicting y.)

### 3.4.2 Cost function

1. Intuition
   - <img src="./imgs/Xnip2023-04-19_10-04-15.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
2. `Regularization`

   - <img src="./imgs/Xnip2023-04-19_09-55-27.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
   - <img src="./imgs/Xnip2023-04-19_09-59-15.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

3. if 𝞴 is too large (regularization parameter)
   - <img src="./imgs/Xnip2023-04-19_10-02-47.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

### 3.4.3 Regularized Linear Regression

1. Regularized linear regression
   - <img src="./imgs/Xnip2023-04-23_06-41-53.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
2. Gradient descent

   - <img src="./imgs/Xnip2023-04-19_10-25-40.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

3. Normal equation

   - <img src="./imgs/Xnip2023-04-19_10-28-32.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

4. Non-invertibility (optional/ advanced)

### 3.4.4 Regularized Logistic Regression

- Regularized Logistic Regression
  - <img src="./imgs/Xnip2023-04-23_06-51-02.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
- Gradient descent
  - <img src="./imgs/Xnip2023-04-23_06-50-36.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
- Advanced optimization
  - <img src="./imgs/Xnip2023-04-23_06-56-45.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

<br><br><br><br><br><br>

# Week-4

## 4.1 Motivations

### 4.1.1 Non-linear Hypotheses

1. Non-linear classification
   - <img src="./imgs/Xnip2023-04-23_07-30-26.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
2. Computer vision; car detection

   - <img src="./imgs/Xnip2023-04-23_07-31-58.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
   - <img src="./imgs/Xnip2023-04-23_07-32-19.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
   - <img src="./imgs/Xnip2023-04-23_07-37-38.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

   - features = (50 \* 50)<sup>2</sup> / 2

<br><br><br><br><br><br>

### 4.1.2 Neurons and the Brain

1. Neural Networks

   - Origins: Algorithms that try to mimic the brain
   - Was very widely used in 80s and early 90s; popularity diminished in late 90s
   - Recent resurgence: state-of-the-art technique for many applications

2. the 'one learning algorithm' hypothesis

   - Auditory cortex learns to see
   - <img src="./imgs/Xnip2023-04-23_07-48-53.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
   - somatosensory cortex learns to see
   - <img src="./imgs/Xnip2023-04-23_07-51-07.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
   - <img src="./imgs/Xnip2023-04-23_07-53-50.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

## 4.2 Neural Networks

## 4.2.1 Model Representation I

1. Neuron in the brain

   - <img src="./imgs/Xnip2023-04-23_09-58-29.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
   - <img src="./imgs/Xnip2023-04-23_10-01-04.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

2. Neuron model: logistic unit

   - <img src="./imgs/Xnip2023-04-23_10-41-16.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

3. Neural Network

   - <img src="./imgs/Xnip2023-04-23_10-49-41.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
   - <img src="./imgs/Xnip2023-04-23_10-54-10.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

   - **If network has s<sub>j</sub> units in layer j and s<sub>j+1</sub> units in layer j+1, then θ<sup>(j)</sup> will be of dimension s<sub>j+1</sub> X (s<sub>j</sub>+1)**
   - The +1 comes from the addition in θ<sup>(j)</sup> of the "bias nodes," x<sub>0</sub> and θ<sub>0</sub><sup>(j)</sup>. `In other words the output nodes will not include the bias nodes while the inputs will.`
   - Example: If layer 1 has 2 input nodes and layer 2 has 4 activation nodes. Dimension ofθ<sup>(1)</sup> is going to be 4x3 where s<sub>j</sub>=2 and s<sub>j+1</sub> =4, so s<sub>j+1</sub> x (s<sub>j</sub> + 1) = 4 x 3

## 4.2.2 Model Representation II

[link](https://www.coursera.org/learn/machine-learning-course/supplement/YlEVx/model-representation-ii)

1. Forward propagation: Vectorized implementation

   - <img src="./imgs/Xnip2023-04-24_09-15-53.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

2. Neural Network learning its own features

   - <img src="./imgs/Xnip2023-04-24_09-20-48.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

3. Other network architecture
   - <img src="./imgs/Xnip2023-04-24_09-26-38.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

## 4.3 Applications

### 4.3.1 Examples and Intuitions I

1. Non-linear claasification example: XOR/ XNOR

   - <img src="./imgs/Xnip2023-04-24_09-49-52.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

2. Simple example: AND

   - <img src="./imgs/Xnip2023-04-24_10-01-14.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

3. Example: OR function
   - <img src="./imgs/Xnip2023-04-24_10-01-14.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

### 4.3.2 Examples and Intuitions II

1. Negation

   - if and only if x1 = x2 = 0, h(x) = 1
   - <img src="./imgs/Xnip2023-04-24_10-12-43.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

2. Putting it together: x1 XNOR x2

   - <img src="./imgs/Xnip2023-04-24_10-20-12.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

3. Neural Network intuition

   - <img src="./imgs/Xnip2023-04-24_10-21-00.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

4. Handwritten digit classification
   - <img src="./imgs/Xnip2023-04-24_10-25-50.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

### 4.3.3 Multiclass Classification

1. multiple output units: one-vs-all

   - <img src="./imgs/Xnip2023-04-24_11-09-07.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
   - <img src="./imgs/Xnip2023-04-24_11-10-37.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

2. Question
   - `add one more bias unit`: `(5 + 1) x 10`
   - <img src="./imgs/Xnip2023-04-24_11-14-16.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

## 4.4 Review

<br><br><br><br><br><br>

# 5. Neural Network Learning

## 5.1 Cost Function and Backpropagation

### 5.1.1 Cost Function

1. Neural Network (classification)
   - <img src="./imgs/Xnip2023-04-25_08-58-11.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
2. Cost function
   - the doulbe sum simply adds up the logistic regression costs calculated for each cell in the ouput layer
   - the triple sum simply adds up the squares of all the individual Θs in the entire network
   - the i in the triple sum does not refer to training example i
   - <img src="./imgs/Xnip2023-04-25_09-16-42.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

### 5.1.2 Backpropagation Algorithm

- Backpropagation is neural-network terminology for minimizing our cost function.
- [doc_ref](https://www.coursera.org/learn/machine-learning-course/supplement/pjdBA/backpropagation-algorithm)

1. Gradient computation

   - <img src="./imgs/Xnip2023-04-25_09-33-57.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
   - <img src="./imgs/Xnip2023-04-25_09-33-33.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

2. Gradient computation: Backpropagation algorithm

   - inorder to compute derivative, use backpropagation
   - a<sup>(4)</sup><sub>j</sub> - activation of layer 4 node j unit
   - y<sub>j</sub> - j<sub>th</sub> element of vector y in our label training set
   - <img src="./imgs/Xnip2023-04-25_09-43-46.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

3. Backpropagation algorithm
   - <img src="./imgs/Xnip2023-04-25_09-47-28.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

### 5.1.3 Backpropagation Intuition

1. Forward propagation

   - <img src="./imgs/Xnip2023-06-20_11-24-23.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

2. what is backpropagation doing?

   - <img src="./imgs/Xnip2023-06-20_11-26-37.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

3. Forward propagation

   - difference between actual value y<sup>(i)</sup> and what was the value predicted a<sup>(4)</sup><sub>1</sub>
   - <img src="./imgs/Xnip2023-06-20_11-36-00.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

4. [Reading](https://www.coursera.org/learn/machine-learning-course/supplement/v5Bu8/backpropagation-intuition)
   - <img src="./imgs/Xnip2023-06-20_12-33-42.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>



<br><br><br>

## 5.2 Backpropagation in Practice 

<br><br><br>

### 5.2.1 Implementation note: unrollling parameters

0. Principle

   - In order to use optimizing functions such as "fminunc()", we will want to "unroll" all the elements and put them into one long vector:

1. Example

   - pullout 1 ~ 110, then pullout 111~220, then pollout 221~231 elements
   - <img src="./imgs/Xnip2023-06-20_13-26-01.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

   - 1~60, 61~71
   - <img src="./imgs/Xnip2023-06-20_13-30-00.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

2. Learning algorithm
   - <img src="./imgs/Xnip2023-06-20_14-07-11.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

<br><br><br>

### 5.2.2 Gradient Checking

1. Numerical estimation of gradients

   - <img src="./imgs/Xnip2023-06-21_09-03-35.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

2. Parameter vector θ

   - <img src="./imgs/Xnip2023-06-21_09-11-28.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

3. Calculation

   - <img src="./imgs/Xnip2023-06-21_09-14-53.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

4. Implementation Note:

   - Implement backprop to compute `DVec` (unrolled D<sup>(1)</sup>, D<sup>(2)</sup>, D<sup>(3)</sup>) .
   - Implement numerical gradient check to compute `gradApprox`
   - Make sure they give similar values
   - Turn off gradient checking. Using backprop code forlearing
   - numberical estimation of gradients is very `computational expensive`

5. Important
   - Be sure to disable your gradient checking code before training your classifier.If you run numerical gradient computation on every iteration of gradient descent (or in the inner loop of `costFuction(...)`) your code will be very slow

<br><br><br>

### 5.2.3 Random Initialization

1. Zero initialization

   - when backpropagate, all nodes will update to the same value repeatedly.
   - <img src="./imgs/Xnip2023-06-21_09-41-23.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

2. Random initialization: symmetry breaking

   - instead we can randomly initialize our weights for our 𝜣 matrices using the follwing method
   - <img src="./imgs/Xnip2023-06-21_09-44-45.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
   - hence we initialize each 𝜣<sup>(l)</sup><sub>ij</sub>, l - hidden layer l. to a randome value between [-𝜀,𝜀]. Using the above formula guarantees that we get the desired bound. the same procedure applies to all the 𝜣's

   ```python
    # If the dimensions of Theta1 is 10x11, Theta2 is 10x11 and Theta3 is 1x11.

    Theta1 = rand(10,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
    Theta2 = rand(10,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
    Theta3 = rand(1,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
   ```

3. quiz
   - <img src="./imgs/Xnip2023-06-21_09-48-00.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

### 5.2.4 Putting it Together

1. Training a neural network

   - Number of input units = dimension of features x<sup>(i)</sup>
   - Number of output units = number of classes
   - Number of hidden units per layer = usually more the better (must balance with cost of computation as it increases with more hidden units)
   - Defaults: 1 hidden layer. If you have more than 1 hidden layer, then it is recommended that you have the same number of units in every hidden layer.
   - <img src="./imgs/Xnip2023-06-21_10-14-23.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

2. Steps of training a neural network

   1. Randomly initialize weights. (normally relatively small near 0)
   2. Implement forward propagation to get h<sub>𝜣</sub>(x<sup>(i)</sup>) for any x<sup>(i)</sup>. (get estimated value of y)
   3. Implement code to compute cost function J(𝜣)
   4. Implement backprop to compute partial derivatives 𝜕/(𝜕𝜣<sup>(l)</sup><sub>jk</sub>)J(𝜣)
      - <img src="./imgs/Xnip2023-06-21_10-25-01.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
   5. use gradient checking to compare 𝜕/(𝜕𝜣<sup>(l)</sup><sub>jk</sub>)J(𝜣) computed using `backpropagation` vs. using `numerical estimate` of gradient of J(𝜣). then disable gradient checking code
   6. use gradient descent or advanced optimization method with backpropagation to try to minimize J(𝜣) as a function of parameters 𝜣. (non-convex, might be stuck on local optimal. But normally gradient descent can get a pretty good local minimal if it's not global minimal )
      - <img src="./imgs/Xnip2023-06-21_11-39-20.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

3. matplot
   1. backpropagation computes the direction of gradient.
   2. gradient descent seek a route down to the hill
      - <img src="./imgs/Xnip2023-06-27_14-40-50.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
4. quiz
   - <img src="./imgs/Xnip2023-06-27_14-43-06.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>


<br><br><br>

## 5.3 Application of Neural Networks

<br><br><br>

### 5.3.1 Autonomous Driving

1. using backpropagation to train

<br><br><br><br><br><br>

# 6. Advice for Applying Machine Learning

<br><br><br>

## 6.1 Evaluating a learning algorithm

### 6.1.1 Deciding what to try next

1. Debugging a learning algorithm

   - hypothesis makes unacceptably large errors in its predictions, what should u try next?
     - Get more training examples
     - Try smaller sets of features
     - Try getting additional features
     - Try adding polynomial features
     - Try decreasing 𝜆
     - Try increasing 𝜆
     - <img src="./imgs/Xnip2023-06-28_11-24-20.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

2. Machine learning diagonostic
   - diagostis: a test that you can to gain insight what is/isn't working with a learning algorithm, and gain guidance as to how best to improve its performance
   - diagostics can take time to implement, but doing so can be a very good use of your time.
3. quiz
   - <img src="./imgs/Xnip2023-06-28_11-28-38.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

<br><br><br>

### 6.1.2 Evaluating a hypothesis

1. Fails to generalize to new examples not in training set

   - <img src="./imgs/Xnip2023-06-28_11-35-13.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

2. Dataset

   - split 70/30 (make sure your data randomly shuffle before use it )
   - <img src="./imgs/Xnip2023-06-28_13-31-54.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
   - overfitting quiz
   - <img src="./imgs/Xnip2023-06-28_11-39-55.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

3. Training/ testing procedure for linear regression

   - learn parameter θ from training data (minimizing training error J(θ))
   - Compute test set error
   - <img src="./imgs/Xnip2023-06-28_13-34-40.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

4. Training/ testing procesdure for logistic regression

   - learn parameter θ from training data
   - compute test set error
   - misclassification error (0/1 misclassification error)
   - <img src="./imgs/Xnip2023-06-28_13-37-28.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

5. [ref](https://www.coursera.org/learn/machine-learning-course/supplement/aFpD3/evaluating-a-hypothesis)


<br><br><br>

### 6.1.3 Model selection and train/validation/test sets

1. Overfitting example

   - <img src="./imgs/Xnip2023-06-28_13-48-44.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

2. Model selection

   - <img src="./imgs/Xnip2023-06-28_14-23-29.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

3. Evaluting your hypothesis

   - Traning set 60%
   - cross validation set (cv) 20%
   - test set 20%
   - <img src="./imgs/Xnip2023-06-28_14-25-22.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

4. Train/ validation/ test error

   - <img src="./imgs/Xnip2023-06-28_14-26-11.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

5. test on cross validation set

   - pick the one with lowest cross validation error
   - <img src="./imgs/Xnip2023-06-28_14-28-31.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

6. Calculate three separate error values

   1. Optimize the parameters θ using the training set for each polynomial degree
   2. find the polynomial degress d with the least error using the cross validation set
   3. estimate the generalization error using the test set with J<sub>test</sub>(𝜣<sup>d</sup>), (d = theta from polynomial with lower error)

7. quiz

   - <img src="./imgs/Xnip2023-06-28_14-32-30.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

8. [ref](https://www.coursera.org/learn/machine-learning-course/supplement/XHQqO/model-selection-and-train-validation-test-sets)

<br><br><br>

## 6.2 Bias vs. Variance

### 6.2.1 Diagnosing Bias vs. Variance

1. Bias/ variance

   - <img src="./imgs/Xnip2023-06-28_15-12-35.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

   - Training error:
   - Cross validation error:
     - d = 1, underfitting, high bias
     - d = 4, overfitting, high variance
     - <img src="./imgs/Xnip2023-06-28_15-17-03.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

2. Diagonosing bias vs. variance

   - Suppose your learning algorithm is performing less well than you were hoping. (J<sub>cv</sub>(θ) or J<sub>test</sub>(θ) is high.) is it a bias problem or a variance problem?
   - <img src="./imgs/Xnip2023-06-28_15-25-23.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

3. quiz

   - <img src="./imgs/Xnip2023-06-28_15-24-56.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

4. summation

   - We need to distinguish whether bias or variance is the problem contributing to bad predictions.
   - High bias is underfitting and high variance is overfitting. Ideally, we need to find a golden mean between these two.
   - <img src="./imgs/Xnip2023-06-28_16-31-35.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

5. [ref](https://www.coursera.org/learn/machine-learning-course/supplement/81vp0/diagnosing-bias-vs-variance)

<br><br><br>

### 6.2.2 Regularization and Bias/Variance

1. Linear regression with regularization

   - <img src="./imgs/Xnip2023-06-28_16-40-01.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

2. choosing the regularization parameter 𝜆

   - <img src="./imgs/Xnip2023-06-28_16-41-24.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
   - <img src="./imgs/Xnip2023-06-28_16-53-26.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

3. Bias/variance as a function of the regularization paramer 𝜆
   - <img src="./imgs/Xnip2023-06-28_17-05-01.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
4. quiz

   - <img src="./imgs/Xnip2023-06-28_16-58-53.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

5. summation

   - in order to choose the model and the regularization term 𝜆, we need to:
     1. create a list of lambdas(i.e. 𝜆 ∈ {0,0.01,0.02,0.04,0.08,0.16,0.32,0.64,1.28,2.56,5.12,10.24})
     2. create a set of models with different degress or any other variants
     3. iterate throught the 𝜆s and for each 𝜆 go through all the models to learn some θ
     4. compute the cross validation error using the learned θ (computed with 𝜆) on the J<sub>CV</sub>(θ) `without` regularization or 𝜆 = 0
     5. select the best combo that produces the lowest error on the cross validation set
     6. using the best combo θ and 𝜆, apply it on J<sub>test</sub>(θ) to see if it has a good generalization of the problem

6. [ref](https://www.coursera.org/learn/machine-learning-course/supplement/JPJJj/regularization-and-bias-variance)

<br><br><br>

### 6.2.3 Learning curves

1. Learning curves

   - if training size is small, the training error will be small as well
   - the training error grows as training set # grows
   - As the training set gets larger, the error for a quadratic function increases.
   - The error value will plateau out after a certain m, or training set size.
   - <img src="./imgs/Xnip2023-06-29_09-04-49.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

2. High bias

   - increase training set size
   - `if a learning algoritm is suffering from high bias, getting more traning data will not (by itself) help much`
   - <img src="./imgs/Xnip2023-06-29_09-08-28.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

3. High variance
   - `if a learning algoritm is suffering from high variance, getting more training data is likely to help`
   - <img src="./imgs/Xnip2023-06-29_09-11-30.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
4. quiz

   - <img src="./imgs/Xnip2023-06-29_09-12-20.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

5. [ref](https://www.coursera.org/learn/machine-learning-course/supplement/79woL/learning-curves)

<br><br><br>

### 6.2.4 Deciding what to do next revisited

1. Debugging a learning algorithm

   - hypothesis makes unacceptably large errors in its predictions, what should u try next?
     - Get more training examples. `fix high variance`
     - Try smaller sets of features. `fix high variance`
     - Try getting additional features. `fix high bias`
     - Try adding polynomial features. `fix high bias`
     - Try decreasing 𝜆. `fix high bias`
     - Try increasing 𝜆. `fix high variance`

2. Neural networks and overfitting

   - A neural network with fewer parameters is **prone to underfitting**. It is also **computationally cheaper**.
   - A large neural network with more parameters is **prone to overfitting**. It is also **computationally expensive**. In this case you can use regularization (increase λ) to address the overfitting.
   - <img src="./imgs/Xnip2023-06-29_09-55-36.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

3. Model complexity effects

   - Lower-order polynomials (low model complexity) have high bias and low variance. In this case, the model fits poorly consistently.
   - Higher-order polynomials (high model complexity) fit the training data extremely well and the test data extremely poorly. These have low bias on the training data, but very high variance.
   - In reality, we would want to choose a model somewhere in between, that can generalize well but also fits the data reasonably well.

4. quiz

   - <img src="./imgs/Xnip2023-06-29_09-56-20.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

5. [ref](https://www.coursera.org/learn/machine-learning-course/supplement/llc5g/deciding-what-to-do-next-revisited)

## 6.3 Review

<br><br><br><br><br><br>

# 7. Machine Learning System Design

<br><br><br>

## 7.1 Building a Spam Classifier

<br><br><br>

### 7.1.1 Prioritizing what to work on

1. building a spam classifier

   - <img src="./imgs/Xnip2023-06-29_11-28-50.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
   - supervised learning. x = features of email. y = spam(1) or not spam(0). Features x: choose 100 words indicative of spam/ not spam
   - <img src="./imgs/Xnip2023-06-29_11-33-20.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
   - how to spend your time to make it have low error?
     - collect lots of data
       - e.g "honeypot" project
     - Develop sophisticated featurs based on email routing information (from email header)
     - Develop sophisticated features for message body, e.g. should "discount" and "discounts" be treated as the same word? how about "deal" and "Dealer"? Features about punctuation?
     - Develop sophisticated algorithm to detect misspellings (e.g. m0rtagae, med1cine, w4atches)

2. quiz
   - <img src="./imgs/Xnip2023-06-29_13-18-17.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

3. [ref](https://www.coursera.org/learn/machine-learning-course/supplement/0uu7a/prioritizing-what-to-work-on)


<br><br><br>

### 7.1.2 Error analysis

1. Recommended approach

   - start with a simple algorithm that you can implement quickly. implement it and test it on your cross-validation data
   - plot learning curves to decide if more data, more features, etc. are likely to help
   - error analysis: manually examine the examples (in cross validation set) that your algorithm made errors on. See if you spot any systematic trend in what type of examples it is making erros on

2. Error analysis

   - <img src="./imgs/Xnip2023-06-29_15-00-14.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

3. The importance of numerical evaluation
   - `It is very important to get error results as a single, numerical value. Otherwise it is difficult to assess your algorithm's performance.`
   - should discount/ discounts/ discounted/ discouting be treated as the same word?
     - can use 'stemming' software (e.g 'porter stemmer')
       - universe/ universty
     - error analysis may not be helpful for deciding if this is likely to improve performance. Only solution is to try it and see if it works
     - need numerical evaluation (e.g. cross validation error) of algorithm's performance with and without stemming
       - without stemming: `5% error`/ with stemming: `3% error`
       - distinguish upper vs. lower case (Mom/mon): `3.2%`
       - <img src="./imgs/Xnip2023-06-29_15-10-58.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
4. quiz
   - <img src="./imgs/Xnip2023-06-29_15-04-38.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
5. [ref](https://www.coursera.org/learn/machine-learning-course/supplement/Z11RP/error-analysis)

<br><br><br>

## 7.2 Handling skewed data

<br><br><br>

### 7.2.1 Error Metrics for Skewed Classes

1. Cancer classification example

   - <img src="./imgs/Xnip2023-06-29_15-47-51.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

2. Precision/ Recall

   - y = 1 in presence of rare class that we want to detect
   - <img src="./imgs/Xnip2023-06-29_16-01-40.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

3. quiz
   - <img src="./imgs/Xnip2023-06-29_15-58-25.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
   - <img src="./imgs/Xnip2023-06-29_15-59-53.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

<br><br><br>


### 7.2.2 Trading off precision and recall

1. trading off precision and recall

   - logistic regression: 0 <= h<sub>θ</sub>(x) <= 1
   - predict 1 if h<sub>θ</sub>(x) >= 0.5
   - predict 0 if h<sub>θ</sub>(x) >= 0.5
   - suppose we want to predict y= 1 (cancer) only if very confident
   - suppose we want to avoid missing too many cases of cancer(avoid false negatives)
   - more generally: predict 1 if h<sub>θ</sub>(x) >= threshold
   - <img src="./imgs/Xnip2023-06-29_16-17-14.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

2. F<sub>1</sub> Score (F score)

   - <img src="./imgs/Xnip2023-06-29_16-27-22.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

3. quiz
   - <img src="./imgs/Xnip2023-06-29_16-27-02.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

<br><br><br>

## 7.3 Using large data sets

<br><br><br>

### 7.3.1 Data for machine learning

1. designing a high accuracy learning system

   - <img src="./imgs/Xnip2023-06-29_16-37-17.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

2. large data rationale

   - <img src="./imgs/Xnip2023-06-29_16-48-10.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

3. quiz
   - <img src="./imgs/Xnip2023-06-29_16-50-49.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

<br><br><br>

## 7.4 review

- <img src="./imgs/Xnip2023-06-29_17-53-17.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

<br><br><br><br><br><br>

# 8. Support Vector Machines

<br><br><br>

## 8.1 Large margin classification

<br><br><br>

### 8.1.1 Optimization objective

1. Alternative view of logistic regression

   - <img src="./imgs/Xnip2023-06-29_18-20-02.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
   - <img src="./imgs/Xnip2023-06-29_18-19-31.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

2. support vector machine

   - <img src="./imgs/Xnip2023-06-29_18-32-40.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

3. SVM hypothesis

   - <img src="./imgs/Xnip2023-06-29_18-32-19.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

4. quiz
   - <img src="./imgs/Xnip2023-06-29_18-28-25.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

<br><br><br>

### 8.1.2 Large Margin Intuition

1. Support Vector Machine
   - <img src="./imgs/Xnip2023-06-29_19-20-11.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
2. SVM decision boundary

   - <img src="./imgs/Xnip2023-06-29_19-26-10.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>s

3. SVM decision bounday: linearly separable case
   - <img src="./imgs/Xnip2023-06-29_19-44-29.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>s
4. quiz

   - <img src="./imgs/Xnip2023-06-29_19-37-10.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>s

5. large margin classifier in presence of outliers
   - if C is very large use magenta
   - if C is not very large use black
   - <img src="./imgs/Xnip2023-06-29_19-44-16.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>s

<br><br><br>

### 8.1.3 Mathematics behind large margin classification

1. Vector inner product

   - ||u|| - length of vector u
   - p - `signed` length of projection of v onto u
   - <img src="./imgs/Xnip2023-06-29_19-59-08.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>s

2. SVM decision boundary
   - SVM minimize square norm
   - p<sup>(i)</sup> - projection of ith training example onto parameter vector θ
   - <img src="./imgs/Xnip2023-06-29_20-08-07.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
   - maximize distance between training example to the decision boundary
   - <img src="./imgs/Xnip2023-06-29_20-27-14.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

<br><br><br>

## 8.2 Kernels

<br><br><br>

### 8.2.1 Kernels I

1. non-linear decision boundary

   - <img src="./imgs/Xnip2023-06-30_08-50-05.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

2. kernel

   - <img src="./imgs/Xnip2023-06-30_08-53-46.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

3. kernel and similarity

   - 𝜎 - sigma lower case
   - l<sup>(1)</sup> - landmark 1
   - component-wise distance
   - <img src="./imgs/Xnip2023-06-30_09-01-01.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

4. example

   - <img src="./imgs/Xnip2023-06-30_09-14-30.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

5. quiz

   - <img src="./imgs/Xnip2023-06-30_09-13-58.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

6. predict 1 or 0
   - <img src="./imgs/Xnip2023-06-30_09-19-26.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

<br><br><br>

### 8.2.2 Kernel II

1. choosing the landmarks

   - where to get l<sup>(1)</sup>, l<sup>(2)</sup>, l<sup>(3)</sup>, ...?
   - choose landmark as exactly as training example
   - <img src="./imgs/Xnip2023-06-30_09-28-07.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

2. SVM with kernels

   - one of landmark will be 1 as x<sup>(i)</sup> = l<sup>(i)</sup>,
   - update and get new `f` feature set
   - <img src="./imgs/Xnip2023-06-30_09-34-56.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

3. SVM with kernels

   - update feature x to new feature f
   - for computational reason to write in this way **θ<sup>(T)</sup>Mθ**
   - <img src="./imgs/Xnip2023-06-30_09-45-38.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

4. SVM parameters

   - choose large 𝜎<sup>2</sup> if gaussian kernel falls smoothly you tend to get hypothesis very slowly as u change the input x
   - <img src="./imgs/Xnip2023-06-30_09-52-28.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

5. quiz
   - <img src="./imgs/Xnip2023-06-30_09-54-29.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

<br><br><br>

## 8.3 SVMs in Practice

### 8.3.1 Using an SVM

1. Use SVM

   - <img src="./imgs/Xnip2023-06-30_10-15-38.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

2. Kernel (`similarity`) functions

   - scaling before caclcualte new feature f
   - <img src="./imgs/Xnip2023-06-30_10-22-17.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

3. other choices of kernel

   - note: not all similarity functions `similarity(x, l)` make valid kernels. (need to satisfy techinical condition called "[Mercer's Theorem](https://en.wikipedia.org/wiki/Mercer%27s_theorem)") to make sure SVM packages' optimizations run correctly, and do not diverge).
   - many off-the-shelf kernels available

     - polynomial kernel
     - more esoteric: string kernel, chi-square kernel, histogram intersection kernel

   - <img src="./imgs/Xnip2023-06-30_10-50-12.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

4. quiz

   - <img src="./imgs/Xnip2023-06-30_10-48-31.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

5. multi-class classification

   - many SVM packages already have built-in multi-class classification functionality
   - <img src="./imgs/Xnip2023-06-30_10-53-38.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

6. logistic regression vs. SVMs
   - n = # of features, m = # of training examples
   1. if n is large (relative to m)
      - use logistic regression, or SVM without a kernel ("linear kernel")
   2. if n is small, m is intermediate:
      - use SVM with Gaussian kernel (non-linear classification)
   3. if n is small, m is large
      - create/ add more features, then use logistic regression or SVM without a kernel
   - neural network likely to work well for most of these settings, but may be slower to train
   - <img src="./imgs/Xnip2023-06-30_10-59-51.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

<br><br><br>

<br><br><br>

## 8.4 Review

<br><br><br><br><br><br>

# 9. Unsupervised Learning

<br><br><br>

## 9.1 Clustering

<br><br><br>

### 9.1.1 Unsupervised learning: introduction

1. Supervised learning
   - <img src="./imgs/Xnip2023-06-30_13-25-10.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
2. Unsupervised learning
   - without label
   - find some structure for us
   - <img src="./imgs/Xnip2023-06-30_13-26-25.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
3. Applications of clustering

   - Market segmentation
   - Social network analysis
   - Organize computing clusters
   - Astronomical data analysis
   - <img src="./imgs/Xnip2023-06-30_13-27-41.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

4. quiz
   - <img src="./imgs/Xnip2023-06-30_13-29-03.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

<br><br><br>

### 9.1.2 K-means algorithm

1. steps to cluster

   - cluster centriods - randomly initialize two points
   - K Means is an iterative algorithm

     - (1) cluster assigment step
     - (2) move centriod step

   - initialize
     - <img src="./imgs/Xnip2023-06-30_13-49-46.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
   - cluster
     - <img src="./imgs/Xnip2023-06-30_13-50-01.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
   - move centriod
     - <img src="./imgs/Xnip2023-06-30_13-50-16.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
   - cluster
     - <img src="./imgs/Xnip2023-06-30_13-50-35.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
   - move centriod
     - <img src="./imgs/Xnip2023-06-30_13-50-57.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
   - cluster
     - <img src="./imgs/Xnip2023-06-30_13-51-30.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
   - move centriod
     - <img src="./imgs/Xnip2023-06-30_13-51-44.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

2. K-means algorithm

   - Input:
     - K (# of cluster)
     - training set {x<sup>(1)</sup>, x<sup>(2)</sup>, ..., x<sup>(m)</sup>}
     - x<sup>(i)</sup> ∈ ℝ<sup>n</sup> (drop x<sub>(0)</sub> = 1 convention)
     - <img src="./imgs/Xnip2023-06-30_13-55-49.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

3. K-means algorithm

   - K upper case denote total # of centroid
   - k lower case denote kth centroid
   - c<sup>i</sup>, index (from 1 to K) of cluster centriod closest to x<sup>(i)</sup>
   - 𝜇<sup>k</sup>, average(mean) of points assigned to cluster k
     - if no points assign to 𝜇<sup>k</sup>, eleminate that centriod
   - <img src="./imgs/Xnip2023-06-30_14-15-54.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

4. quiz

   - <img src="./imgs/Xnip2023-06-30_14-08-05.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

5. K-means for non-separated clusters
   - <img src="./imgs/Xnip2023-06-30_14-18-15.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

<br><br><br>

### 9.1.3 Optimization Objective

1. K-means optimization objective

   - c<sup>i</sup> = index of cluster (1,2,..., K) of which example x<sup>(i)</sup> is currently assigned
   - 𝜇<sub>k</sub> = cluster centriod k ℝ (𝜇<sup>(k)</sup> ∈ ℝ<sup>n</sup>)
   - 𝜇<sub>c<sup>(i)</sup></sub> = cluster centriod of cluster to which example x<sup>(i)</sup> has been assigned
   - `distortion function`
   - <img src="./imgs/Xnip2023-06-30_15-06-57.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

2. k-means algorithm

   - <img src="./imgs/Xnip2023-06-30_15-06-22.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

3. quiz
   - <img src="./imgs/Xnip2023-06-30_15-05-31.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

<br><br><br>

### 9.1.4 Random Initialization

1. k-means algorithm

   - <img src="./imgs/Xnip2023-06-30_15-11-15.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

2. Random initialization

   - should have K < m
   - Randomly pick K training examples
   - set 𝜇<sub>1</sub>,...,𝜇<sub>K</sub> equal to these K example
   - <img src="./imgs/Xnip2023-06-30_15-17-24.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

3. local optima

   - <img src="./imgs/Xnip2023-06-30_15-20-52.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

4. Random initialization

   - <img src="./imgs/Xnip2023-06-30_15-24-24.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

5. quiz
   - <img src="./imgs/Xnip2023-06-30_15-23-36.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

<br><br><br>

### 9.1.5 Choosing the number of clusters

1. choosing the value of K

   - Elbow method
   - <img src="./imgs/Xnip2023-06-30_15-38-25.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

2. quiz

   - <img src="./imgs/Xnip2023-06-30_15-37-45.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

3. choosing the value of K
   - sometiems, you're running K-means to get clusters to use for some later/ downstream purpose. Evaluate K-means based on a metric for how well it performs for that later purpose.
   - <img src="./imgs/Xnip2023-06-30_15-43-23.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

<br><br><br>


## 9.2 Review

<br><br><br><br><br><br>

# 10. Dimensionality Reduction

<br><br><br>

## 10.1 Motivation

<br><br><br>

### 10.1.1 Data Compression

1. 2D -> 1D

   - <img src="./imgs/Xnip2023-07-26_15-42-07.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

2. 3D -> 2D

   - e.g. 10,000D -> 1,000D
   - project all data into a 2-D plain
   - <img src="./imgs/Xnip2023-07-26_15-47-55.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

3. quiz
   - <img src="./imgs/Xnip2023-07-26_15-49-24.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

<br><br><br>

### 10.1.2 Data Visulization

1. Data Visualization

   - <img src="./imgs/Xnip2023-07-26_18-39-36.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
   - reduce from 50-D to 2-D
   - <img src="./imgs/Xnip2023-07-26_18-40-51.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
   - <img src="./imgs/Xnip2023-07-26_18-44-35.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

2. quiz
   - <img src="./imgs/Xnip2023-07-26_18-44-06.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

<br><br><br>

## 10.2 Principal Component Analysis (PCA)

<br><br><br>

### 10.2.1 Principal Component Analysis Problem Formulation

1. PCA

   - find a good line to project data 2-D to 1-D
   - <img src="./imgs/Xnip2023-07-26_18-59-44.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

2. find k direction to project data

   - <img src="./imgs/Xnip2023-07-26_19-30-27.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

3. PCA is not linear regression

   - reduce vertical distance
   - reduce magnitude
   - <img src="./imgs/Xnip2023-07-26_19-33-27.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
   - <img src="./imgs/Xnip2023-07-26_19-34-04.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

4. quiz
   - <img src="./imgs/Xnip2023-07-26_19-36-27.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

<br><br><br>

### 10.2.2 Principal Component Analysis Algorithm

1. Data Preprocessing
   - Training set: x<sup>(1)</sup>, x<sup>(2)</sup>, ..., x<sup>(m)</sup>
   - Preprocessing (feature scaling/ mean normalization)
   - j stands for jth feature in ith traning data
   - s<sub>(j)</sub> is some beta value of feature j, it could be max - min value or standard deviation of feature j
   - <img src="./imgs/Xnip2023-07-27_09-31-44.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

2. Principal Component Analysis (PCA) algorithm
   - <img src="./imgs/Xnip2023-07-27_09-41-02.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

3. Principal Component Analysis (PCA) algorithm
   - Reduce data from n-dimensions to k-dimensions
   - Compute "covariance matrix"
   - Compute "eigenvectors" of matrix
   - Greek alphabet sigma
   - Summation symbols
   - svd - Singular value decomposition
   - (x<sup>(i)</sup>)(x<sup>(i)</sup>)<sup>T</sup> is n by n matrix
   - take first k vector
   - <img src="./imgs/Xnip2023-07-27_09-51-01.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

4. Principal Component Analysis (PCA) algorithm
   - From [U, S, V] = svd(Sigma), we get
   - Ureduce - n x k, 
   - x - example of training set, n x 1 
   - <img src="./imgs/Xnip2023-07-27_09-55-34.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>


5. Principal Component Analysis (PCA) algorithm
   - <img src="./imgs/Xnip2023-07-27_09-58-04.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

6. quiz
   - u<sup>(j)</sup>, n x 1  => (u<sup>(j)</sup>)<sup>T</sup>, 1 x n
   - x, n x 1 
   - <img src="./imgs/Xnip2023-07-27_10-04-33.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

<br><br><br>

## 10.3 Applying PCA

<br><br><br>

### 10.3.1 Reconstruction from compressed representation
1. Reconstruction from compressed representation
   - z = U<sup>T</sup><sub>reduce</sub> • x
   - X<sup>(1)</sup><sub>approx</sub> = U<sub>reduce</sub> • z<sub>(1)</sub>
   - <img src="./imgs/Xnip2023-07-27_10-22-34.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

2. quiz
   - <img src="./imgs/Xnip2023-07-27_10-18-03.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
   
   
3. summary
   - unlabel dataset
   - high-d to low-d
   - low-d back to high-d

<br><br><br>

### 10.3.2 Choosing the number of principal components
1. Choosing k (# of principle components)
   - Average squared project eorr:
   - total variation in the data:
   - typically, choose k to be smallest value so that 99% of variance is retained
   - <img src="./imgs/Xnip2023-07-27_11-50-04.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>


2. Choosing k (# of principle components)
   - S, diagonal matrix
   - <img src="./imgs/Xnip2023-07-27_11-55-31.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

3. Choosing k (# of principle components)
   - <img src="./imgs/Xnip2023-07-27_11-56-48.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

4. quiz
   - <img src="./imgs/Xnip2023-07-27_11-57-27.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

5. summary
   - you often find that PCA will be able to retain 99% of variance or some fraction of the variance even while compressing the data by a very large factor


<br><br><br>

### 10.3.3 Advice for applying PCA

1. Supervised learning speedup
   - PCA can speed up running time of a learning algo
   - run your PCA only on `training set`
   - <img src="./imgs/Xnip2023-07-27_13-32-42.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

2. application of PCA
   - Compression
      - Reduce memory/disk needed to store data
      - speed up learning algo
   - Visualization
   - <img src="./imgs/Xnip2023-07-27_13-34-44.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

3. Bad use of PCA: to prevent overfitting
   - <img src="./imgs/Xnip2023-07-27_13-37-35.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>
   

4. PCA is sometimes used where it shouldn't be
   - Design of ML system:
      - Getting training set ...........
      - Run PCA to reduce x in dimension to get z ...........
      - Train logistic regression on ...........
      - Test on test set: Map x to z run hypothesis on ...........
   - How about doing the whole thing without using PCA?
   - Before implementing PCA, first try running whatever you want to do with the original/raw data x only if that doesn't do what you want, then implement PCA and consider using z
   - <img src="./imgs/Xnip2023-07-27_13-44-12.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

5. quiz
   - <img src="./imgs/Xnip2023-07-27_13-45-06.jpg" alt="imgs" width="600" height="350"><br><br><br><br><br><br>

<br><br><br>

## 10.4 Review