import numpy as np
import pandas as pd

from typing import Tuple
from sklearn.metrics import mean_absolute_percentage_error as fmape, \
                            mean_squared_error as fmse, \
                            mean_absolute_error as fmae, \
                            r2_score as fr2, \
                            f1_score as ff1, \
                            precision_score as fprecision, \
                            recall_score as frecall, \
                            accuracy_score as faccuracy, \
                            roc_auc_score as fauroc, \
                            confusion_matrix, cohen_kappa_score

def tunar_hiperparams(
    tipo_modelo, 
    selec_hiperparams: Tuple[pd.Index],
    X_train: np.array or pd.DataFrame or pd.Series,
    X_test: np.array or pd.DataFrame or pd.Series,
    y_train: np.array or pd.Series,
    y_test: np.array or pd.Series,
    fixed_hiperparams: dict or None = None,
    tipo_dado: str = 'quantitativo',
    metricas: dict or list or None = None,
    scaler = None,
    func_metricas_kwargs: dict = dict()
) -> pd.DataFrame:

    if metricas is None and tipo_dado not in [ 'quantitativo', 'qualitativo']:
        raise NameError("'tipo_dado' deve ser 'qualitativo' ou 'quantitativo'.")

    if scaler is not None:
        sc = scaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

    if fixed_hiperparams is None:
        fixed_hiperparams = dict()
    # create test cases

    comb_idx = pd.MultiIndex.from_product(selec_hiperparams)

    dict_metricas = {
        'mape': fmape,
        'rmse': lambda *args, **kwargs: fmse(squared = False, *args, **kwargs),
        'mae': fmae,
        'r2': fr2,
        'f1': ff1,
        'recall': frecall,
        'precision': fprecision,
        'auroc': fauroc,
    }

    # dicionário de métricas
    if metricas is None:
        if tipo_dado == 'quantitativo':
            metricas = {
                'mape': fmape,
                'rmse': lambda *args, **kwargs: fmse(squared = False, *args, **kwargs),
                'mae': fmae,
                'r2': fr2
            }
            
        else:
            metricas = {
                'auroc': fauroc,
                'f1': ff1,
                'recall': frecall,
                'precision': fprecision
            }
    
    elif isinstance(metricas, list):
        metricas_ = { nome_metrica: dict_metricas[nome_metrica] for nome_metrica in metricas }
        metricas = metricas_

    # create dataframe
    score_cols = ['modelo'] + list(metricas.keys())

    # create apply function
    def calc_scores(s, nomes_hiperparams, fixed_hiperparams = fixed_hiperparams, metricas = metricas):
        modelo_hiperparams = { nome: valor for nome, valor in zip(nomes_hiperparams, s.name)}
        modelo_hiperparams.update(fixed_hiperparams)
        
        modelo = tipo_modelo(**modelo_hiperparams)

        modelo.fit(X_train, y_train)
        s['modelo'] = modelo
        
        y_pred = modelo.predict(X_test)

        metricas_args = ( y_test, y_pred )

        for col, fcalc in metricas.items():
            s[col] = fcalc(*metricas_args, **func_metricas_kwargs)

        return s

    scores = pd.DataFrame([], columns = score_cols, index = comb_idx)
    comb_nomes = scores.index.names
    scores = scores.apply(calc_scores, nomes_hiperparams = comb_nomes, fixed_hiperparams = fixed_hiperparams, metricas = metricas, axis = 1)
    scores = scores.sort_values(by = list(scores.columns[1:]), ascending = False)
    return scores

def calc_confusion_matrix(threshold, y_true, y_prob, beta = 1):
    y_pred = (y_prob > threshold.values[0]) * 1
    [[tn, fp], [fn, tp]] = confusion_matrix(y_true = y_true, y_pred = y_pred)

    s = pd.Series([tn, fp, fn, tp], index = ['tn', 'fp', 'fn', 'tp'])

    return s

def calc_cohen_kappa_score(y_true, y_prob, threshold):
    y_pred = y_prob > threshold
    ck = cohen_kappa_score(y1 = y_true, y2 = y_pred)
    return ck

def calc_threshold_optimization(y_true, y_prob, beta = 1, calcs = dict()):

    # generate threshold dataframe
    thresholds = pd.DataFrame(np.linspace(0, 1, 101), columns = ['threshold'])

    # calc tn, fp, fn, tp for each threshold
    threshold_conf_mat = thresholds.apply(
            calc_confusion_matrix, 
            y_true = y_true, y_prob = y_prob, 
            axis = 1
        )

    threshold_opt = pd.concat([
        thresholds,
        threshold_conf_mat
    ], axis = 1)

    # calc a few other metrics
    threshold_opt['precision'] = (
        (threshold_opt.tp / (threshold_opt.tp + threshold_opt.fp))
            .where(threshold_opt.tp + threshold_opt.fp > 0, np.nan)
    )

    threshold_opt['recall'] = (
        (threshold_opt.tp / (threshold_opt.tp + threshold_opt.fn))
            .where(threshold_opt.tp + threshold_opt.fn > 0, np.nan)
    )

    threshold_opt['accuracy'] = (
        (threshold_opt.tp + threshold_opt.tn) /
        (threshold_opt.tn + threshold_opt.fp + threshold_opt.fn + threshold_opt.tp)
    )

    threshold_opt[f'f{beta:n}_score'] = (
        (
            (1+beta**2) * threshold_opt.precision * threshold_opt.recall /
            (threshold_opt.precision * beta**2 + threshold_opt.recall)
        )
            .where(
                threshold_opt.precision * beta**2 + threshold_opt.recall > 0,
                np.nan
            )
    )

    threshold_opt['cohen_kappa_score'] = np.vectorize(
        lambda threshold: calc_cohen_kappa_score(
            y_true = y_true,
            y_prob = y_prob,
            threshold = threshold)
    )(threshold_opt['threshold'])

    for name, fun_config in calcs.items():
        fcalc = fun_config['fcalc']
        kwds = fun_config['kwds']

        fapply = lambda row: fcalc(row, **kwds)

        threshold_opt[name] = threshold_opt.apply(fapply, axis = 1)

        try:
            if fun_config['normalize']:
                colvals = threshold_opt[name]
                threshold_opt[name] = (colvals - colvals.min()) / (colvals.max() - colvals.min())
        except KeyError:
            pass

    # return the dataframe (index is threshold level)
    return threshold_opt.set_index('threshold')

# cost function calculated over the confusim matrix entries
def calc_confusion_matrix_cost(row, cost_matrix):
    tn = row['tn']
    fp = row['fp']
    fn = row['fn']
    tp = row['tp']
    
    each_cost = np.asarray([[tn, fp], [fn, tp]]) * np.asarray(cost_matrix)

    return each_cost.sum()