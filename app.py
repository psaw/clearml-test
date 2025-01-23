from clearml import Task
from sklearn.datasets import make_classification
import pandas as pd


task = Task.init(project_name="Learn ClearML", task_name="task-1")

X, y = make_classification(10000)
X = pd.DataFrame(X)

X.to_csv("data.csv", index=False)

