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



### Multinomial naive Bayes

Multinomial naive Bayes is designed for count-based features such as word or token frequencies. Its inductive bias is that feature counts provide class evidence and are conditionally independent given the class. This strong assumption makes the model fast and interpretable for text classification, even when the features are not truly independent. It is not used in the Parkinson's demonstration because that dataset contains continuous biomedical measurements rather than token counts.

### Neural network

A feed-forward neural network represents a prediction as a composition of learned linear transformations and nonlinear activation functions. Its inductive bias is flexible: the target is assumed to be expressible through distributed combinations of features rather than only local neighborhoods or explicit threshold rules. This flexibility allows the network to model complex interactions, but optimization, initialization, architecture, and regularization have a larger effect on its behavior. Training and test histories help reveal whether learning is generalizing or overfitting.

## Demonstration

The repository includes a small, deterministic Parkinson's dataset split so the examples run quickly. The notebook demonstrates KNN, the decision tree, the random forest, and the neural network on the same train/test data:

```text
parkinsons_demo.ipynb
data/parkinsons_train.csv
data/parkinsons_test.csv
results/parkinsons/
figures/parkinsons/
```

The notebook produces parameter-performance plots for KNN, the decision tree, and the random forest, together with a three-panel neural-network training-history figure. Open [`parkinsons_demo.ipynb`](parkinsons_demo.ipynb) from the repository root and run all cells.

For the text-specific model, open [`naive_bayes_demo.ipynb`](naive_bayes_demo.ipynb). It uses a compact subset of real labeled music reviews to show tokenization, bag-of-words probabilities, class priors, additive smoothing, and an `alpha` performance sweep.

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
