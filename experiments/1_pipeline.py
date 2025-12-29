from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Load dataset
X, y = load_wine(return_X_y=True)

# params
seed = 42
test_size = 0.2
n_estimators = 6
max_depth = 10

# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=seed)


# train and test model
rf = RandomForestClassifier(
    n_estimators=n_estimators, max_depth=max_depth, random_state=seed)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(accuracy)
