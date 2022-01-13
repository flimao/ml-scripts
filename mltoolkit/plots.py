from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as spst

from sklearn.metrics import mean_absolute_percentage_error as fmape, \
                            mean_squared_error as fmse, \
                            mean_absolute_error as fmae, \
                            r2_score as fr2, \
                            f1_score as ff1, \
                            precision_score as fprecision, \
                            recall_score as frecall, \
                            accuracy_score as faccuracy, \
                            roc_auc_score as fauroc, \
                            roc_curve, \
                            brier_score_loss as fbrier, \
                            log_loss as flogloss

from sklearn.utils.validation import check_is_fitted

from sklearn.exceptions import NotFittedError

from .metrics import calc_threshold_optimization

def plot_true_pred(
    modelos, 
    X_test: np.array or pd.Series or pd.DataFrame,
    y_test: np.array or pd.Series or pd.DataFrame,
) -> plt.Figure:
    
    # Vamos criar um gráfico para comparar os Valores Reais com os Preditos
    fig = plt.figure()

    for i, (nome_modelo, modelo) in enumerate(modelos.items()):
        y_pred = modelo.predict(X_test)    

        # metricas
        metricas_kwargs = dict(
            y_true = y_test,
            y_pred = y_pred
        )

        r2 = fr2(**metricas_kwargs)
        mape = fmape(**metricas_kwargs)
        rmse = fmse(squared = False, **metricas_kwargs)
        mae = fmae(**metricas_kwargs)

        label = nome_modelo
        label += f'\n(R² = {r2:.3%}, MAPE = {mape:.3%}, RMSE = {rmse:.2f}, MAE = {mae:.2f})'

        cor = f'C{i}'
        plt.plot(y_pred, y_test, 'bo', markersize = 10, markerfacecolor = cor, markeredgecolor = cor, alpha = 0.3, label = label)

    plt.ylabel("True Value", fontsize=12)
    plt.xlabel("Predict Value", fontsize=12)

    plt.title('Comparação Valor Predito x Valor Real', fontsize=12)

    # mostra os valores preditos e originais
    xl = np.arange(min(y_test), 1.2*max(y_test),(max(y_test)-min(y_test))/10)
    yl = xl
    plt.plot(xl, yl, 'r--')

    plt.legend()
    plt.show()
    return fig

def plot_roc_auc_curve(models, X_test, y_test, X_train = None, y_train = None):
    
    models = models.copy()

    fig, ax = plt.subplots()

    model_idx = pd.Index([ model_name for model_name in models.keys() ], name = 'model_name')
    model_df = pd.DataFrame([], index = model_idx, columns = ['model', 'auroc', 'brier', 'logloss', 'f1', 'precision', 'recall', 'accuracy'])
    
    for model_name, model in models.items():
        
        model_df.at[model_name, 'model'] = model

        try:
            check_is_fitted(model)
        except NotFittedError:
            model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = fauroc(y_test, y_prob)

        model_df.at[model_name, 'auroc'] = auc
        model_df.at[model_name, 'accuracy'] = faccuracy(y_test, y_pred)
        model_df.at[model_name, 'f1'] = ff1(y_test, y_pred)
        model_df.at[model_name, 'precision'] = fprecision(y_test, y_pred)
        model_df.at[model_name, 'recall'] = frecall(y_test, y_pred)
        model_df.at[model_name, 'brier'] = fbrier(y_test, y_prob)
        model_df.at[model_name, 'logloss'] = flogloss(y_test, y_prob)

    model_df = model_df.sort_values(by = list(model_df.columns[1:]), ascending = False)
    
    for name, (model, auc) in model_df[['model', 'auroc']].iterrows():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        ax.plot(fpr, tpr, label=f'{name} ROC (AUC = {auc:.2%})')

    ax.plot([0, 1], [0, 1], linestyle = '--', color = 'gray', alpha = 0.8)
    ax.set_xlim([-0.02, 1.0])
    ax.set_ylim([0.0, 1.02])

    pctformatter = lambda x, pos: f'{x:.0%}'

    ax.set_xlabel('False Positive Rate\nFall-out\n1 - Recall of negative class')
    ax.xaxis.set_major_formatter(pctformatter)
    
    ax.set_ylabel('True Positive Rate\nRecall of positive class')
    ax.yaxis.set_major_formatter(pctformatter)
    
    ax.set_title('Receiver Operating Characteristic Curve')
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    return model_df.sort_values(by = list(model_df.columns[1:]), ascending = False)

def plot_precision_recall_curve(
    y_true: np.array or pd.Series, 
    y_prob: np.array or pd.Series, 
    threshold_opt: pd.DataFrame or None = None
):

    if threshold_opt is None:
        threshold_opt = calc_threshold_optimization(
            y_true = y_true, 
            y_prob = y_prob
        )

    # plotar o gráfico precision recall curve
    fig, ax = plt.subplots()

    # ligar o grid e configurá-lo
    plt.minorticks_on()
    ax.grid(alpha = 0.2, which = 'major', linestyle = '--')
    ax.grid(alpha = 0.2, which = 'minor', linestyle = '--')

    ax.scatter(threshold_opt.index, threshold_opt.precision, label = 'Precision')
    ax.scatter(threshold_opt.index, threshold_opt.recall, label = 'Recall')

    fmtr = lambda x, pos: f'{x:.0%}'
    ax.xaxis.set_major_formatter(fmtr)
    ax.yaxis.set_major_formatter(fmtr)

    ax.set_xlabel('Classification threshold')
    ax.set_title('Precision / Recall - Threshold Curve')

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])

    plt.legend()
    plt.show()

    return fig

def plot_threshold_curve(
    y_true: np.array or pd.Series, 
    y_prob: np.array or pd.Series, 
    metrics: str or list or dict,
    best_threshold: bool = True,
    minimize: bool = False,
    threshold_opt: pd.DataFrame or None = None):

    if threshold_opt is None:
        threshold_opt = calc_threshold_optimization(
            y_true = y_true, 
            y_prob = y_prob
        )
    

    # plotar o gráfico com threshold no eixo X e a metrica/estimador no eixo Y
    fig, ax = plt.subplots()

    # ligar o grid e configurá-lo
    plt.minorticks_on()
    ax.grid(alpha = 0.2, which = 'major', linestyle = '--')
    ax.grid(alpha = 0.2, which = 'minor', linestyle = '--')

    if isinstance(metrics, str): # nome de uma coluna só
        ax.scatter(threshold_opt.index, threshold_opt[metrics], label = metrics)
        metric_label = metrics

        axtitle = f"Threshold optimization curve - '{metric_label}'"

        ax.set_ylabel(metric_label)

        # plotar uma linha vertical com a otimização do threshold (maximização ou minimização)
        if best_threshold:
            if minimize:
                best_threshold_x = threshold_opt.index[threshold_opt[metrics].argmin()]
            else:
                best_threshold_x = threshold_opt.index[threshold_opt[metrics].argmax()]

            # plotar a linha vertical no melhor threshold em vermelho
            ax.axvline(
                x = best_threshold_x,
                color = 'red',
                linestyle = '--',
                label = fr'Best $\theta$ = {best_threshold_x:.1%}'
            )
    
    elif isinstance(metrics, list) or isinstance(metrics, dict):
        # lista com metricas ou dsicionário com chave = nome da coluna e valor = nome da metrica
        
        axtitle = f"Threshold optimization curve\n"

        # para cada metrica (funciona para lista e para as chaves de um dicionário)
        for i, metric in enumerate(metrics):

            try:
                # se for um dicionário, o valor é o nome da métrica
                name_metric = metrics[metric]
            except TypeError:
                # se não, o nome da métrica é o próprio nome do campo
                name_metric = metric
            
            # plotar os pontos
            color = f'C{i}'
            ax.scatter(threshold_opt.index, threshold_opt[metric], 
                label = name_metric, color = color
            )
            
            axtitle += f"'{name_metric}', "

            # plotar uma linha vertical com a otimização do threshold (maximização ou minimização)
            if best_threshold:
                if minimize:
                    best_threshold_x = threshold_opt.index[threshold_opt[metric].argmin()]
                else:
                    best_threshold_x = threshold_opt.index[threshold_opt[metric].argmax()]

                # plotar a linha vertical no melhor threshold com a mesma cor da metrica correspondente
                ax.axvline(
                    x = best_threshold_x,
                    color = color,
                    linestyle = '--',
                    label = fr"Best $\theta$, '{name_metric}' = {best_threshold_x:.1%}"
                )
        
        # remover a vírgula extra
        axtitle = axtitle[:-2]
        ax.set_ylabel('Metric')

    # boniteza do gráfico
    fmtr = lambda x, pos: f'{x:.0%}'
    ax.xaxis.set_major_formatter(fmtr)

    ax.set_xlabel(r'Classification threshold ($\theta$)')
    ax.set_xlim([-0.01, 1.01])
    
    ax.set_title(axtitle)

    plt.legend()
    plt.show()

    return fig

def plot_heatmap_chisq(
    data: pd.DataFrame,
    cols_labels: List[str] or None = None,
    fmt: str = '.1%',
    fig_kwds: dict = dict(),
    chisq: bool = False,
    filtro: str or None = None,
    cmap_lims: list = [0, 1]
) -> plt.Figure:

    data = data.copy()

    cols_campos = data.columns

    col1 = data[cols_campos[0]]
    col2 = data[cols_campos[1]]

    if cols_labels is None:
        cols_labels = cols_campos

    # tabela de contingencia
    contingencia = pd.crosstab(col1, col2)  
    # normalizar pelas linhas: o % plotado é a célula dividido pelo total da margem vertical
    contingencia_pct = contingencia.apply(lambda col: col / contingencia.sum(axis = 1))

    # figura
    fig, ax = plt.subplots(**fig_kwds)
      
    # plotar heatmap
    sns.heatmap(
        contingencia_pct, 
        annot = True, fmt = fmt, 
        cmap = 'viridis', vmax = cmap_lims[1], vmin = cmap_lims[0],
        ax = ax
    )

    # colorbar
    cbar = ax.collections[-1].colorbar
    cbar.ax.yaxis.set_major_formatter(lambda y, pos: f'{y:{fmt}}')

    # eixo x: resultado
    ax.set_xlabel(cols_labels[1])

    # eixo y
    ax.set_ylabel(cols_labels[0])

    # titulo de cada subgrafico
    axtitle = f"'{cols_labels[0]}' vs '{cols_labels[1]}'"
    if chisq:
        pvalor = spst.chi2_contingency(contingencia)[1]
        axtitle += f"\n(p-valor teste $\chi ^2$ = {pvalor:.2%})"

    ax.set_title(axtitle)

    # titulo do gráfico
    suptitle = 'Tabelas de contingência: % do total de cols, margem vertical'
    if filtro is not None:
        suptitle += f'\n(filtro: {filtro})'
    plt.suptitle(suptitle)

    plt.tight_layout()
    plt.show()

    return fig