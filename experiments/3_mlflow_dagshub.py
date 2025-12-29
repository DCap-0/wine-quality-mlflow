from typing import cast
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.utils import Bunch
import seaborn as sns
import mlflow
from mlflow import sklearn as mlflow_sklearn

# Dagshub setup
import dagshub
dagshub.init(repo_owner='Dhruv-Kapri',
             repo_name='wine-quality-mlflow', mlflow=True)


def conf_mat(y_true, y_pred, target_names):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    path = "3_Confusion-matrix.png"
    plt.savefig(path)
    plt.close()

    return path


# Load dataset (need target_names)
wine = cast(Bunch, load_wine())
X = wine.data
y = wine.target
target_names = wine.target_names

# params
seed = 42
test_size = 0.2
n_estimators = 10
max_depth = 40

# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=seed)

# Exp name (Runs register under default without it)
mlflow.set_experiment('wine-quality-mlflow')

# train and test model
with mlflow.start_run():
    rf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=seed)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))

    mlflow.log_param('test_size', test_size)
    mlflow.log_param('n_estimators', n_estimators)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_metric('accuracy', accuracy)

    # log artifacts using mlflow
    cm_path = conf_mat(y_test, y_pred, target_names)
    mlflow.log_artifact(cm_path)

    # Log the model
    mlflow_sklearn.log_model(
        rf,
        name="Random-Forest-Model"
    )
    mlflow.log_artifact(__file__)

    mlflow.set_tags(
        {"Author": 'Dhruv', "Project": "Wine Classification", "File": "3"})

    print(accuracy)
