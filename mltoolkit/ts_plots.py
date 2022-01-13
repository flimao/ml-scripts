
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_percentage_error as smape, mean_squared_error as smse

def dados_faltantes(tswide: pd.DataFrame, produtos: List = None) -> None:
    """Function to greet

    Parameters
    ----------
    tswide : pandas.DataFrame
        timeseries in wide format, per product

    """

    if produtos is None:
        produtos = tswide.columns
        n_produtos = produtos.shape[0]
    
    else:
        n_produtos = len(produtos)

    fig, axs = plt.subplots(nrows = n_produtos, ncols = 1, sharex = True, figsize = (10, 10))

    for i, produto in enumerate(produtos):
        data_full = tswide[produto]
        data_missings = data_full[data_full.isna()]
        
        ax = axs[i]
        sns.scatterplot(data = data_full, ax = ax, color = 'gray', alpha = 0.5)
        if data_missings.shape[0] > 0:
            for x in data_missings.index:
                ax.axvline(x = x, color = 'red')

        ax.set_title(f"Produto: '{produto}'")
        ax.set_ylabel('Faturamento')

    fig.suptitle('Dados faltantes', color = 'red', fontweight = 1000)
    plt.tight_layout()
    plt.show()

def decomp_fourier(serie_fat: pd.Series, produto: str, c: str) -> object:
    decomp = seasonal_decompose(serie_fat)

    fig, axs = plt.subplots(nrows = 4, figsize = (10, 8), sharex = True)

    ts_filtro = serie_fat

    sns.lineplot(data = ts_filtro, ax = axs[0], color = c)
    axs[0].set_title('Serie')
    axs[0].set_ylabel('Faturamento')

    sns.lineplot(data = decomp.trend, ax = axs[1], color = c)
    axs[1].set_ylabel('R$ 100')

    sns.lineplot(data = decomp.seasonal, ax = axs[2], color = c)
    axs[2].set_ylabel('R$')

    resid_standard = (decomp.resid - decomp.resid.mean()) / decomp.resid.std()
    sns.scatterplot(data = resid_standard, ax = axs[3], color = c)
    axs[3].set_ylabel('Resíduos padrão')

    fig.suptitle(f"Decomposicao temporal: produto '{produto}'")
    plt.show()

    return decomp

def ajuste_grafico(
    modelo: object, produto: str, 
    serie_teste: pd.Series, serie_treino: pd.Series or None = None, 
    ci: bool = False, in_sample: bool = True, preds_metrics: bool = True,
) -> pd.Series:

    n_test_periods = serie_teste.shape[0]

    arr_preds = modelo.predict(n_periods = n_test_periods, return_conf_int = ci)
    idx = pd.date_range(freq = 'MS', start = serie_teste.index[0], periods = n_test_periods)
    
    if ci:
        preds = pd.Series(arr_preds[0], index = idx)
        preds_bounds = pd.DataFrame(arr_preds[1], columns = ['lb', 'ub'], index = idx)
    else:
        preds = pd.Series(arr_preds, index = idx)
    
    if in_sample and serie_treino is not None:
        arr_preds_in_sample = modelo.predict_in_sample()
        idx_in_sample = serie_treino.index
        arr_preds_in_sample = arr_preds_in_sample[-len(idx_in_sample):]
        preds_in_sample = pd.Series(arr_preds_in_sample, index = idx_in_sample)
    
    preds.name = 'yearly_preds'

    palette = sns.color_palette(None, 4)
    
    label_preds = 'Previsão'
    if preds_metrics:
        kwargs_metrics = dict(
            y_true = serie_teste.dropna(), 
            y_pred = preds[serie_teste.dropna().index]
        )
        mape = smape(**kwargs_metrics)
        rmse = smse(**kwargs_metrics, squared = False)
        label_preds += f'\n(MAPE = {mape:.3%}, RMSE = {rmse:.2e})'

    preds.plot(label = label_preds, color = palette[2])
    if ci:
        plt.fill_between(idx, preds_bounds['lb'], preds_bounds['ub'], alpha = 0.2, color = palette[2])
    if in_sample:
        preds_in_sample.plot(color = palette[2])
    
    serie_teste.plot(label = 'Conjunto de teste', color = palette[1])
    
    if serie_treino is not None:
        serie_treino.plot(label = 'Conjunto de treino', color = palette[0])

    plt.legend()
    plt.ylabel('Faturamento total')
    plt.title(f"Predição contra conjunto de teste (produto '{produto}')")
    plt.show()

    return preds