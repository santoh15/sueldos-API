from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import numpy as np
import xgboost as xgb

def split_data(df_entrenar):
    """
    Splits the input DataFrame into training, validation, and test sets.
    The target variable '_sal' is separated from the features, and log transformation is applied to the target.

    Args:
        df_entrenar (pd.DataFrame): The input DataFrame containing features and target variable.

    Returns:
        tuple: A tuple containing the training features, validation features, test features,
               training target (log-transformed), validation target (log-transformed), and test target (log-transformed).
    """

    df_train_val,df_test=train_test_split(df_entrenar,test_size=0.2,random_state=1)
    df_train,df_val=train_test_split(df_train_val, test_size=0.25,random_state=1)
    df_train=df_train.reset_index(drop=True)
    df_val=df_val.reset_index(drop=True)
    df_test=df_test.reset_index(drop=True)
    y_train=df_train['_sal']
    y_val=df_val['_sal']
    y_test=df_test['_sal']
    del df_train['_sal']
    del df_val['_sal']
    del df_test['_sal']
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)
    y_test_log = np.log1p(y_test)
    return df_train, df_val, df_test, y_train_log, y_val_log, y_test_log


def number_estimators_model(df_train, df_val, y_train_log, y_val_log):
    """
    Trains an XGBoost regression model on the provided DataFrame and target variable.
    

    Args:
        df (pd.DataFrame): The input DataFrame containing features.
        _sal (pd.Series): The target variable representing salaries.

    """
    model_xgb= xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, early_stopping_rounds=50, random_state=42)
    model_xgb.fit(df_train, y_train_log, eval_set=[(df_val, y_val_log)], verbose=30 )
    print(f"The optimal number of estimators was: {model_xgb.best_iteration}")
    return model_xgb.best_iteration


def merge_data_for_full_train(df_train, df_val, y_train_log, y_val_log):
    """
    Merges the training and validation DataFrames and their corresponding target variables for full training.

    Args:
        df_train (pd.DataFrame): The training features DataFrame.
        df_val (pd.DataFrame): The validation features DataFrame.
        y_train_log (pd.Series): The log-transformed target variable for training.
        y_val_log (pd.Series): The log-transformed target variable for validation.

    Returns:
        tuple: A tuple containing the merged features DataFrame and the merged target variable Series.
    """
    df_full_train = pd.concat([df_train, df_val], axis=0)
    y_full_log = pd.concat([y_train_log, y_val_log], axis=0)
    return df_full_train, y_full_log




def tune_xgboost_hyperparameters(df_train, y_train_log, number_estimators, n_iterations=40,):
    """
    Performs hyperparameter tuning for an XGBoost regression model using RandomizedSearchCV.
    Args:
        df_train (pd.DataFrame): The training features DataFrame.
        y_train_log (pd.Series): The log-transformed target variable for training.
        n_iterations (int): The number of random combinations of hyperparameters to try. Default is 20.
    """

    xgb_base = xgb.XGBRegressor(random_state=42)

    param_dist = {
        'n_estimators': [number_estimators],
        'learning_rate': [0.01,0.03, 0.05,0.07, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9, 12],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5, 7]
    }
    random_search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_dist,
        n_iter=n_iterations,           
        scoring='neg_mean_absolute_error',
        cv=10,                          
        verbose=2,                     
        random_state=42,
        n_jobs=-1                      
    )
    print(f"Initializing hyperparameter search ({n_iterations} iterations)...")
    random_search.fit(df_train, y_train_log)
    print("-" * 30)
    print("¡Search completed!")
    print("Best hyperparameters found:")
    print(random_search.best_params_)
    return random_search.best_estimator_


def feature_importance(model_xgb, df_full):
    """
    
    This function takes the trained model and the full dataframe, and prints the top 10 most important variables according to the model.
    Args:
        model_xgb: The trained XGBoost model.
        df_full: The full dataframe used for training the model.
    
    """
    importances = pd.DataFrame({
        'Variable': df_full.columns,
        'Importance': model_xgb.feature_importances_
    })
    top_variables = importances.sort_values(by='Importance', ascending=False).head(10)
    print("Top 10 Variables por importance:")
    print(top_variables)


def bootstrap(y_true, y_pred, metric_fn, n_bootstrap=1000, ci=95):
    """
    Computes a confidence interval for a metric using bootstrap resampling.

    Args:
        y_true (np.array): True target values.
        y_pred (np.array): Predicted values.
        metric_fn (function): Metric function to apply (e.g., mean_absolute_error).
        n_bootstrap (int): Number of bootstrap samples.
        ci (int): Confidence interval percentage (e.g., 95).

    Returns:
        tuple: Lower and upper bounds of the confidence interval.
    """

    n = len(y_true)
    scores = []

    for _ in range(n_bootstrap):
        idx = np.random.choice(range(n), size=n, replace=True)
        score = metric_fn(y_true[idx], y_pred[idx])
        scores.append(score)

    alpha = (100 - ci) / 2
    lower = np.percentile(scores, alpha)
    upper = np.percentile(scores, 100 - alpha)

    return lower, upper

def calculate_confidence_intervals(y_true_array, y_pred_array):
    """
    Calculates confidence intervals for MAE and R2 Score using bootstrap resampling.

    Args:
        y_true_array (np.array): True target values.
        y_pred_array (np.array): Predicted values.
    """
    lower_mae, upper_mae = bootstrap(
    y_true=y_true_array, 
    y_pred=y_pred_array, 
    metric_fn=mean_absolute_error, 
    n_bootstrap=1000, 
    ci=95
    )
    print(f"Intervalo de confianza del 95% para MAE: ${lower_mae:,.2f} a ${upper_mae:,.2f}")

    lower_r2, upper_r2 = bootstrap(
    y_true=y_true_array, 
    y_pred=y_pred_array, 
    metric_fn=r2_score, 
    n_bootstrap=1000, 
    ci=95
    )

    print(f"Intervalo de confianza del 95% para R2: {lower_r2:.2f} a {upper_r2:.2f}")