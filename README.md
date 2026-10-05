# Machine Learning From Scratch

Machine-learning algorithms developed across HW1–HW4, with the improved Decision Tree, Random Forest, and Neural Network versions from `ML_Algorithm_Compare` selected as the current implementations.

## Included algorithms

- **KNN:** original HW1 implementation
- **Decision Tree:** improved implementation with configurable maximum depth
- **Multinomial Naive Bayes:** original HW2 implementation
- **Random Forest:** improved implementation with configurable maximum depth
- **Neural Network:** improved implementation with multiclass labels and metrics

Decision Tree, Random Forest, and Neural Network are copied from `ML_Algorithm_Compare/src/shared/algorithms/`. KNN and Multinomial Naive Bayes are copied from their original homework implementations.

## Project layout

```text
algorithms/              one current implementation per algorithm
docs/guides/             setup, coding, testing, tooling, and CI guides
```

## Setup

The algorithm files retain their original coursework interfaces. Consult the original homework requirements and `ML_Algorithm_Compare/requirements.txt` when running experiments.

See [`docs/guides/`](docs/guides/) for the project’s development and contribution guides.

## Parkinson's demonstration

The demonstration uses a small, deterministic Parkinson’s train/test split so that the notebook remains quick to run while still producing the final-project-style figures:

```text
demonstrations/parkinsons_demo.ipynb
data/parkinsons_train.csv
data/parkinsons_test.csv
results/parkinsons/
figures/parkinsons/
```

Open `demonstrations/parkinsons_demo.ipynb` from the repository root and run all cells. It produces parameter-performance plots for KNN, Decision Tree, and Random Forest, plus the three-panel Neural Network training-history figure.

## Inductive bias discussion

The final-project comparison emphasizes that model performance depends on the assumptions each algorithm makes about the data. KNN is a non-parametric, distance-based baseline: it assumes that observations with the same label are close in feature space. This is why the notebook standardizes the numeric attributes before distance computation. A small `k` preserves local structure but is sensitive to noise, while a larger `k` smooths the decision boundary and may underfit.

The Decision Tree assumes that labels can be predicted by recursively partitioning the feature space using feature thresholds. This creates piecewise, box-like decision regions and naturally captures nonlinear interactions without feature scaling. The improved `maximum_depth` parameter makes the bias–variance trade-off explicit: shallow trees have more bias and less variance, while deeper trees can fit more complex structure but may overfit.

Random Forest keeps the tree-based partitioning bias but reduces the variance of an individual tree through bootstrap samples, random feature selection, and majority voting. Increasing the number of trees usually stabilizes the estimate, although it increases training time. The improved forest also passes a maximum depth to each tree, providing an additional regularization control.

The Neural Network is a flexible parametric model that assumes the target can be represented by compositions of smooth nonlinear transformations. It can learn distributed feature interactions that are difficult to express as local neighborhoods or a short sequence of thresholds, but it has higher variance and depends more strongly on initialization, optimization, architecture, and regularization. The learning-history plot helps show whether improvements on the training set also transfer to the test set.

Multinomial Naive Bayes is intentionally not used in the Parkinson’s notebook. Its inductive bias is that token counts provide class evidence and are conditionally independent given the class, which is appropriate for text but not for continuous biomedical measurements.

## Origin

The code was developed as part of machine-learning coursework covering supervised learning, text classification, ensemble methods, and neural networks.

See [`docs/guides/`](docs/guides/) for the project’s development and contribution guides.
