# Machine Learning From Scratch

This repository contains from-scratch implementations of several machine-learning algorithms. 
It is intended for educational purposes, to make clear the ideas underlying these algorithms: 
How each algorithm represents a decision, what assumptions it makes about the data, and how 
those assumptions shape its behavior.

## Algorithms and inductive bias

An algorithm's **inductive bias** is the set of assumptions it uses to generalize beyond the examples it has seen. 
In this repository, we will compare the inductive biases of several algorithms.

### K-nearest neighbors

KNN predicts the label of a new example from the labels of nearby training examples. Its inductive bias is local smoothness: 
points that are close in feature space are expected to have similar labels. 

The choice of `k` controls the amount of smoothing. 
A small k is sensitive to noise, such as outliers or mislabeled examples. A large k smooths the boundary but may obscure real
local patterns. Here, bias means systematic error caused by an algorithm’s assumptions, which limit the patterns it can learn from the data.

Because KNN chooses neighbors by distance, features with larger numeric ranges can dominate the distance calculation and, in turn, the prediction. 
Standardizing numeric features puts them on comparable scales, so differences in units alone don’t determine which examples are nearest.

### Decision tree

A decision tree’s inductive bias is the assumption that class structure can be learned by repeatedly partitioning the feature space into regions,
using one feature at a time. This creates rectangular regions of the feature space, with a predicted class assigned to instances in each region. 
The tree can model nonlinear patterns without scaling features, but it tends to favor patterns that can be described with axis-aligned splits. 
Example: splitting the feature spaace into two regions "age <= 30 or > 30", which is aligned with the age axis.

Limiting a tree’s depth limits how many times the feature space can be partitioned. A shallow tree makes only a few partitions, so its decision 
regions stay simple; this can miss real patterns in the data. A deeper tree can add more partitions to capture more detailed patterns, but may also 
learn quirks or noise specific to the training data. Choosing a depth balances these effects: too little depth can underfit, while too much can overfit.


### Random forest

A random forest’s inductive bias builds on the decision tree’s: each tree repeatedly partitions the feature space using one feature at a time, creating 
rectangular regions with a predicted class assigned to each. The forest combines the trees’ predictions by majority vote. 

Because each tree is trained on a different random sample of the data and considers random subsets of features when making splits, the trees create 
different partitions of the feature space. The forest combines their predictions through majority voting, allowing it to form more flexible class 
regions—for example, several separate regions or a step-like boundary that approximates a curve. These regions are still built from axis-aligned splits, 
but they need not have the simple shape produced by a single tree.

The ntree parameter sets how many trees vote in the forest. With few trees, random differences between trees can sway the majority vote and lead to 
incorrect predictions. Adding trees usually makes the vote more reliable and can improve prediction accuracy, but the benefit tends to level off. More 
trees do not guarantee correct predictions, especially if the trees make similar errors.

### Multinomial naive Bayes

Multinomial Naive Bayes is designed for count-based features, such as word frequencies in a document. Its inductive bias is that word counts provide 
evidence for a class and are conditionally independent given that class. For example, seeing “urgent” can count as evidence that an email is work-related. 
Independence means the model treats this evidence separately from evidence from other words: it approximates the probability of seeing both “urgent” and
"meeting” in a work email as

P(urgent ∩ meeting | work) ≈ P(urgent | work) × P(meeting | work)

The words may actually be related, but this simplifying assumption makes the model fast and easy to interpret, and it can work well for text classification. 
It is less suitable for continuous-valued measurements, which are not naturally represented as counts.

The alpha parameter controls additive smoothing. It adds a small pseudo-count to each word in each class, so a word absent from the training data for a class still gets a 
nonzero probability. A larger alpha smooths the probabilities more, making word frequencies less different across classes; if it is too large, useful class-specific
 word patterns may be weakened.

### Neural network

A feed-forward neural network passes information through a sequence of layers. At a neuron receiving three inputs, it first forms a **linear combination**:

`z = w1*x1 + w2*x2 + w3*x3 + b`

Here, `x1`, `x2`, and `x3` are the inputs; `w1`, `w2`, and `w3` are learned weights that control each input’s contribution; and `b` is a learned bias. 
The neuron then applies the sigmoid activation:

`a = 1 / (1 + e^(-z))`

This transforms `z` into an output `a` between 0 and 1, which is passed to the next layer. The sigmoid makes the transformation nonlinear. Without a nonlinear 
activation, stacking layers would still amount to a single linear transformation.

The network’s inductive bias is flexible: it assumes useful patterns can be learned by combining information from many features across layers, rather than relying on 
explicit threshold rules or local neighborhoods. 

More hidden layers allow more successive transformations and can help represent complex patterns, but can also make 
training harder and increase the risk of overfitting. The network’s behavior also depends on its width, initialization, optimization, and regularization. Comparing 
training and test performance over time helps show whether the learned patterns generalize or the network is beginning to overfit.

## Demonstration

The repository small Parkinson’s dataset for a quick demo. Run all cells in parkinsons_demo.ipynb from the repository root to compare KNN, 
decision tree, random forest, and neural network on the same train/test split. The notebook saves parameter-performance plots and a three-panel
neural-network training-history figure to results/parkinsons/ and figures/parkinsons/. The data is in data/parkinsons_train.csv and data/parkinsons_test.csv.

For a text classification demo, open naive_bayes_demo.ipynb⁠￼. It shows how reviews are tokenized and represented as bag-of-words vectors, how Multinomial 
Naive Bayes uses class priors and class-conditional word probabilities, and how additive smoothing works. It also compares performance across alpha values, 
which control the strength of smoothing.

## Project layout

```text
algorithms/              from-scratch algorithm implementations
parkinsons_demo.ipynb    KNN, decision tree, random forest, neural network demonstration
naive_bayes_demo.ipynb   naive bayes demonstration
data/                    datasets and fixed train/test splits
figures/                 generated plots from demonstration
results/                 generated metrics and training histories from demonstration
docs/guides/             setup, coding best practices, testing, tooling, and CI guides
tests/                   automated tests
```

## Setup

From the repository root, install the project dependencies, including development and testing tools:

```bash
python -m pip install -r requirements.txt
```

For guidance on development, testing, tools, and contributions, see [`docs/guides/`](docs/guides/).

## Continuous integration

On every push and pull request, GitHub Actions checks formatting and linting, runs type checks, 
validates the notebook, and runs the automated tests.