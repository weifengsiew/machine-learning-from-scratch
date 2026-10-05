import numpy as np

from ml_from_scratch import KNNClassifier


X = np.array([[0.0], [1.0], [10.0]])
y = np.array([0, 0, 1])
model = KNNClassifier(k=3).fit(X, y)
print(model.predict(np.array([[0.5], [9.0]])))
