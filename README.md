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

### Multinomial naive Bayes

Multinomial Naive Bayes is designed for count-based features, such as word frequencies in a document. Its inductive bias is that word counts provide 
evidence for a class and are conditionally independent given that class. For example, seeing “urgent” can count as evidence that an email is work-related. 
Independence means the model treats this evidence separately from evidence from other words: it approximates the probability of seeing both “urgent” and
"meeting” in a work email as

P(urgent ∩ meeting | work) ≈ P(urgent | work) × P(meeting | work)

The words may actually be related, but this simplifying assumption makes the model fast and easy to interpret, and it can work well for text classification. 
It is less suitable for continuous-valued measurements, which are not naturally represented as counts.

### Neural network

A feed-forward neural network makes predictions by passing inputs through layers of learned transformations and nonlinear activation functions. 
Its inductive bias is relatively flexible: layers can combine information from many features to learn complex patterns and interactions, without requiring 
explicit threshold rules or local relationships. This flexibility means the network’s behavior depends strongly on its architecture, initialization, 
optimization, and regularization. Comparing training and test performance over time helps show whether it is learning patterns that generalize or beginning to overfit.

## Demonstration

The repository includes a small, deterministic Parkinson’s dataset for quick demos. Run all cells in parkinsons_demo.ipynb from the repository root to compare KNN, 
decision tree, random forest, and neural network on the same train/test split. The notebook saves parameter-performance plots and a three-panel neural-network training-history 
figure to results/parkinsons/ and figures/parkinsons/. The data is in data/parkinsons_train.csv and data/parkinsons_test.csv.

For a text-classification demo, open naive_bayes_demo.ipynb⁠￼. It uses a compact set of labeled music reviews to demonstrate Multinomial Naive Bayes, including tokenization,
bag-of-words probabilities, class priors, additive smoothing, and an alpha performance sweep.

## Project layout

```text
algorithms/              algorithm implementations
parkinsons_demo.ipynb     numeric tabular-model demonstration
naive_bayes_demo.ipynb    text-classification demonstration
data/                    demonstration datasets and fixed splits
figures/                 generated plots
results/                 generated metrics and training histories
docs/guides/             setup, coding, testing, tooling, and CI guides
tests/                   automated tests
```

## Setup

Install the project dependencies, including development and testing tools, with:

```bash
python -m pip install -r requirements.txt
```

See [`docs/guides/`](docs/guides/) for development, testing, tooling, and contribution guidance.

## Continuous integration

GitHub Actions runs the project checks on pushes and pull requests. The workflow checks formatting and linting, runs type checks, validates the notebook file, and runs the automated tests.
