import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score


def prediction(model_xgb, df_test):
    """ 
    
    This function takes the trained model and the test dataframe, and returns the predicted salaries in real values.
    Args:
        model_xgb: The trained XGBoost model.
        df_test: The test dataframe.
    Returns:
        y_pred_real: The predicted salaries in real values.
    
    """
    y_pred_log = model_xgb.predict(df_test)
    y_pred_real = np.expm1(y_pred_log)
    return y_pred_real

def metrics(y_test, y_pred_real):
    """
    
    This function takes the true salaries and the predicted salaries, and prints the Mean Absolute Error and R2 Score.
    Args:
        y_test: The true salaries.
        y_pred_real: The predicted salaries in real values.

    """
    mae = mean_absolute_error(y_test, y_pred_real)
    r2 = r2_score(y_test, y_pred_real)
    print("-" * 30)
    print(f"Mean Absolute Error: ${mae:,.2f} pesos.")
    print(f"R2 Score: {r2:.2f}")
