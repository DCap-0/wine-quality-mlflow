import pandas as pd
from typing import cast
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import Bunch
import mlflow
from mlflow import sklearn as mlflow_sklearn
from mlflow import data as mlflow_data

# Dagshub setup
import dagshub
dagshub.init(repo_owner='Dhruv-Kapri',
             repo_name='wine-quality-mlflow', mlflow=True)


# Load dataset (need target_names)
wine = cast(Bunch, load_wine())
X = wine.data
y = wine.target
target_names = wine.target_names

# params
seed = 42
test_size = 0.2

# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=seed)

# Defining the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20, 30]
}

rf = RandomForestClassifier(random_state=seed)

# Applying GridSearchCV
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring="accuracy",
    cv=5,
    n_jobs=-1,
    verbose=2
)

# Exp name (Runs register under default without it)
mlflow.set_experiment('wine-quality-mlflow')

# train and test model
with mlflow.start_run():
    grid_search.fit(X_train, y_train)

    # log all the child runs
    for i in range(len(grid_search.cv_results_['params'])):
        with mlflow.start_run(nested=True) as child:
            mlflow.log_params(grid_search.cv_results_["params"][i])
            mlflow.log_metric("accuracy", grid_search.cv_results_[
                              "mean_test_score"][i])

    # Displaying the best parameters and the best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    mlflow.log_params(best_params)
    mlflow.log_metric('cv_accuracy', best_score)

    # log CV table
    cv_results_df = pd.DataFrame(grid_search.cv_results_)
    cv_results_df.to_csv("gridsearch_results.csv", index=False)
    mlflow.log_artifact("gridsearch_results.csv")

    # log datasets
    train_df = pd.DataFrame(X_train, columns=wine.feature_names)
    train_df["target"] = y_train
    mlflow.log_input(mlflow_data.from_pandas(train_df), "training")

    test_df = pd.DataFrame(X_test, columns=wine.feature_names)
    test_df["target"] = y_test
    mlflow.log_input(mlflow_data.from_pandas(test_df), "testing")

    # Log source code
    mlflow.log_artifact(__file__)

    # Log the best model
    mlflow_sklearn.log_model(
        grid_search.best_estimator_,
        name="Random-Forest-Model"
    )

    mlflow.set_tags(
        {"Author": 'Dhruv', "Project": "Wine Classification", "File": "5"})

    print(best_params)
    print(best_score)
