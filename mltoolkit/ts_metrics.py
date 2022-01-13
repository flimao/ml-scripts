import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error as smape, \
                            mean_squared_error as smse, \
                            mean_absolute_error as smae, \
                            r2_score as sr2

from pmdarima.arima.arima import ARIMA

def mostrar_metricas(
        y_true: pd.Series, y_pred: pd.Series, 
        n: int = None, dof: int = None, 
        calc_r2: bool = False,
        *args, **kwargs
    ) -> dict:

    metrics = calc_metricas(
        y_true = y_true, y_pred = y_pred,
        n = n, dof = dof,
        *args, **kwargs
    )

    espacos = max([ len(k) for k in metrics.keys() ]) + 4

    print('Métricas:')

    for n, v in metrics.items():
        if v is None:
            continue
        
        fmt = '.3e'
        if n in ['MAPE', 'R²', 'R² adj.']:
            fmt = '.3%'
        
        print(f'{n:>{espacos}s}: {v:{fmt}}')
        
    return metrics

def calc_metricas(
        y_true: pd.Series, y_pred: pd.Series, 
        n: int = None, dof: int = None, 
        calc_r2: bool = False,
        *args, **kwargs
) -> dict:
    
    metrics = {}

    metrics['MAPE'] = smape(*args, y_true = y_true, y_pred = y_pred, **kwargs)
    metrics['RMSE'] = smse(*args, y_true = y_true, y_pred = y_pred, squared = False, **kwargs)
    metrics['MAE'] = smae(*args, y_true = y_true, y_pred = y_pred, **kwargs)

    if calc_r2:
        metrics['R²'] = sr2(*args, y_true = y_true, y_pred = y_pred, **kwargs)
    else:
        metrics['R²'] = None

    if metrics['R²'] is not None and \
       n is not None and \
       dof is not None and \
       n - dof > 1:
        
        metrics['R² adj.'] = 1 - (1 - metrics['R²'])*(n - 1) / (n - dof - 1)
    
    else:
        metrics['R² adj.'] = None
    
    return metrics
