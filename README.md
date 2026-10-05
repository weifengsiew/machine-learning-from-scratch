# Machine Learning From Scratch

This repository is a compact, educational implementation of several machine-learning algorithms. The code is intended to make the main modeling ideas visible: how each algorithm represents a decision, what assumptions it makes about data, and how those assumptions affect its behavior.

## Algorithms and inductive bias

An algorithm's **inductive bias** is the set of assumptions it uses to generalize beyond the examples it has seen. The algorithms in this repository make different assumptions, so comparing them is useful even when they are trained on the same data.

### K-nearest neighbors

KNN predicts a new example from the labels of nearby training examples. Its inductive bias is local smoothness: points that are close in feature space are expected to have similar labels. The choice of `k` controls the amount of smoothing. A small `k` can preserve local structure but is sensitive to noise; a large `k` produces a smoother, more biased decision boundary. Because distance is central to the method, numeric features should usually be standardized first.

### Decision tree

A decision tree recursively divides the feature space with threshold-based questions. Its inductive bias is that useful class structure can be represented by a sequence of axis-aligned partitions. Trees can capture nonlinear relationships and do not require feature scaling, but unrestricted depth can make them fit noise. Limiting the maximum depth makes the bias–variance trade-off explicit: shallow trees are simpler, while deeper trees represent more detailed decision regions.

### Random forest

A random forest combines many decision trees trained with data and feature randomness. It retains the tree's partition-based bias while reducing the variance of any one tree through aggregation and majority voting. More trees generally make predictions more stable, and limiting tree depth provides another way to control model complexity. The model is often effective when the data contains nonlinearities and feature interactions.

### Multinomial naive Bayes

Multinomial naive Bayes is designed for count-based features such as word or token frequencies. Its inductive bias is that feature counts provide class evidence and are conditionally independent given the class. This strong assumption makes the model fast and interpretable for text classification, even when the features are not truly independent. It is not used in the Parkinson's demonstration because that dataset contains continuous biomedical measurements rather than token counts.

### Neural network

A feed-forward neural network represents a prediction as a composition of learned linear transformations and nonlinear activation functions. Its inductive bias is flexible: the target is assumed to be expressible through distributed combinations of features rather than only local neighborhoods or explicit threshold rules. This flexibility allows the network to model complex interactions, but optimization, initialization, architecture, and regularization have a larger effect on its behavior. Training and test histories help reveal whether learning is generalizing or overfitting.

## Demonstration

The repository includes a small, deterministic Parkinson's dataset split so the examples run quickly. The notebook demonstrates KNN, the decision tree, the random forest, and the neural network on the same train/test data:

```text
demonstrations/parkinsons_demo.ipynb
data/parkinsons_train.csv
data/parkinsons_test.csv
results/parkinsons/
figures/parkinsons/
```

The notebook produces parameter-performance plots for KNN, the decision tree, and the random forest, together with a three-panel neural-network training-history figure. Open [`demonstrations/parkinsons_demo.ipynb`](demonstrations/parkinsons_demo.ipynb) from the repository root and run all cells.

## Project layout

```text
algorithms/              algorithm implementations
demonstrations/          runnable Jupyter demonstrations
data/                    demonstration datasets and fixed splits
figures/                 generated plots
results/                 generated metrics and training histories
docs/guides/             setup, coding, testing, tooling, and CI guides
tests/                   automated tests
```

## Setup

Install the runtime dependencies with:

```bash
python -m pip install -r requirements.txt
```

For development and testing, install:

```bash
python -m pip install -r requirements-dev.txt
```

See [`docs/guides/`](docs/guides/) for development, testing, tooling, and contribution guidance.

## Continuous integration

GitHub Actions runs the project checks on pushes and pull requests. The workflow checks formatting and linting, runs type checks, validates the notebook file, and runs the automated tests.
