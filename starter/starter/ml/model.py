from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from starter.ml.data import process_data
from sklearn.metrics import accuracy_score


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    # Define the model pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier())
    ])

    # Define hyperparameters for tuning
    param_grid = {
        'model__n_estimators': [100, 200],
        'model__max_depth': [3, 5, 7],
        'model__min_samples_split': [2, 5, 10]
    }

    # Perform GridSearchCV
    grid_search = GridSearchCV(pipeline,
                               param_grid,
                               cv=5,
                               scoring='accuracy',
                               n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Return the best model
    return grid_search.best_estimator_


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision,
    recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)


def evaluate_model_on_slices(model,
                             data,
                             categorical_features,
                             label,
                             encoder,
                             lb):
    """Evaluate model performance on slices of categorical features."""
    results = {}
    for feature in categorical_features:
        feature_values = data[feature].unique()
        for value in feature_values:
            subset = data[data[feature] == value]
            if not subset.empty:
                X_subset, y_subset, _, _ = process_data(
                    subset,
                    categorical_features=categorical_features,
                    label=label,
                    training=False,
                    encoder=encoder,
                    lb=lb
                )
                preds = inference(model, X_subset)
                accuracy = accuracy_score(y_subset, preds)
                results[f"{feature}_{value}"] = accuracy
    return results
