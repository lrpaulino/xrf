import warnings
from itertools import chain
from dataclasses import dataclass

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from joblib import Parallel, delayed

from sklearn.cross_decomposition import PLSRegression
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import make_scorer, root_mean_squared_error


@dataclass
class OptimizedPLS(BaseEstimator, RegressorMixin):
    """
    Um regressor PLS com seleção automática de variáveis latentes (LV)
    usando validação cruzada com blocos contíguos e mínimo (ou regra de
    1 erro-padrão) do RMSECV.

    Parâmetros
    ----------
    max_lv : int, padrão=10
        Número máximo de variáveis latentes a serem testadas. É
        automaticamente limitado ao número de variáveis (colunas) de X.
    n_splits : int, padrão=10
        Número de divisões para a validação cruzada com blocos contíguos
    scale : bool, padrão=True
        Se True, escalona X internamente em cada ajuste do PLS
    n_jobs : int, padrão=-1
        Número de processos paralelos usados para testar os diferentes
        números de LV
    verbose : bool, padrão=False
        Se True, exibe uma barra de progresso durante o ajuste
    rule : {'min', '1se'}, padrão='min'
        Critério de seleção do número de LVs:
        - 'min': escolhe o nº de LVs que minimiza o RMSECV médio
        - '1se': escolhe o MENOR nº de LVs cujo RMSECV médio esteja
          dentro de 1 erro-padrão do mínimo (regra "one-standard-error",
          favorece modelos mais parcimoniosos)

    Atributos (após fit())
    -----------------------
    best_n_lv_ : int
        Número ótimo de variáveis latentes selecionado
    best_rmscv_ : float
        RMSECV médio associado a best_n_lv_
    rmscv_values_ : list[float]
        RMSECV médio para cada nº de LV testado, ordenado por nLV
        (rmscv_values_[i] corresponde a nLV = i + 1)
    rmscv_scores_ : dict[int, np.ndarray]
        RMSE de cada fold, por nº de LV testado
    final_model_ : PLSRegression
        Modelo final ajustado com best_n_lv_ componentes
    """
    max_lv: int = 10
    n_splits: int = 10
    scale: bool = True
    n_jobs: int = -1
    verbose: bool = False
    rule: str = 'min'

    def fit(self, X, y):
        """
        Ajusta o modelo PLS com o número ótimo de LVs usando validação cruzada.

        Parâmetros
        ----------
        X : array-like de forma (n_amostras, n_características)
            Dados de entrada
        y : array-like de forma (n_amostras,)
            Valores alvo

        Retorna
        -------
        self : objeto
            Retorna a instância ajustada
        """
        X, y = check_X_y(X, y, y_numeric=True)
        self.n_features_in_ = X.shape[1]

        if self.rule not in ('min', '1se'):
            raise ValueError("rule deve ser 'min' ou '1se'")

        # O PLS não aceita n_components > n_features; limita automaticamente.
        effective_max_lv = min(self.max_lv, self.n_features_in_)
        if effective_max_lv < self.max_lv:
            warnings.warn(
                f"max_lv={self.max_lv} excede o número de variáveis "
                f"({self.n_features_in_}); testando até {effective_max_lv} LVs."
            )
        lv_values = range(1, effective_max_lv + 1)

        # Validação cruzada com blocos contíguos
        kfcv = KFold(n_splits=self.n_splits, shuffle=False)
        metrics = {'rmse': make_scorer(root_mean_squared_error)}

        # Testar diferentes números de variáveis latentes em paralelo.
        # X e y são passados explicitamente (em vez de capturados por
        # closure) para permitir que o joblib faça memory-mapping
        # automático de arrays grandes entre os processos.
        def process_n_lv(n_lv, X, y):
            pls_model = PLSRegression(n_components=n_lv, scale=self.scale)
            scores = cross_validate(
                pls_model, X, y,
                cv=kfcv, scoring=metrics, n_jobs=1,
            )
            return n_lv, scores['test_rmse']

        iterator = Parallel(
            return_as='generator_unordered',
            n_jobs=self.n_jobs,
        )(
            delayed(process_n_lv)(n_lv, X, y)
            for n_lv in lv_values
        )
        if self.verbose:
            iterator = tqdm(iterator, total=len(lv_values), desc='Testando nº de LVs')

        rmscv_scores = {}
        for n_lv, scores in iterator:
            rmscv_scores[n_lv] = scores

        # IMPORTANTE: com 'generator_unordered' os resultados chegam fora
        # de ordem (na ordem em que cada job termina, não na ordem em que
        # foi submetido). Por isso reordenamos explicitamente por nLV em
        # vez de assumir que a posição na lista corresponde ao nLV testado.
        ordered_lvs = sorted(rmscv_scores)
        rmscv_values = [rmscv_scores[lv].mean() for lv in ordered_lvs]

        best_idx = int(np.argmin(rmscv_values))
        if self.rule == 'min':
            self.best_n_lv_ = ordered_lvs[best_idx]
            self.best_rmscv_ = rmscv_values[best_idx]
        else:  # '1se'
            best_scores = rmscv_scores[ordered_lvs[best_idx]]
            se_at_best = best_scores.std(ddof=1) / np.sqrt(len(best_scores))
            threshold = rmscv_values[best_idx] + se_at_best
            # menor nLV cujo RMSECV médio ainda está dentro de 1 SE do mínimo
            chosen = next(
                lv for lv, m in zip(ordered_lvs, rmscv_values) if m <= threshold
            )
            self.best_n_lv_ = chosen
            self.best_rmscv_ = rmscv_scores[chosen].mean()

        self.rmscv_values_ = rmscv_values
        self.rmscv_scores_ = rmscv_scores

        # Ajustar modelo final com número ótimo de LVs
        self.final_model_ = PLSRegression(n_components=self.best_n_lv_, scale=self.scale)
        self.final_model_.fit(X, y)

        return self

    def predict(self, X):
        """
        Prever valores usando o modelo ajustado.

        Parâmetros
        ----------
        X : array-like de forma (n_amostras, n_características)
            Dados de entrada para predição

        Retorna
        -------
        y_pred : ndarray, forma (n_amostras,)
            Valores previstos
        """
        check_is_fitted(self, 'final_model_')
        X = check_array(X)
        return self.final_model_.predict(X).ravel()

    def plot(self, ax=None, show=True):
        """
        Plot cross-validation RMSECV scores across different LV counts.

        Displays a line plot with error bars showing mean RMSECV and standard
        deviation for each number of latent variables tested. The selected
        LV count (self.best_n_lv_, conforme self.rule) is marked with a red 'x'.

        Parâmetros
        ----------
        ax : matplotlib.axes.Axes, opcional
            Eixo existente para desenhar o gráfico. Se None, cria uma nova figura.
        show : bool, padrão=True
            Se True, chama plt.show(). Defina como False se quiser apenas
            obter (fig, ax) para salvar ou compor com outros gráficos.

        Retorna
        -------
        fig, ax : a figura e o eixo do matplotlib
        """
        check_is_fitted(self, 'rmscv_scores_')

        data = pd.DataFrame(
            chain.from_iterable(
                [(k, v_i) for v_i in v]
                for k, v in self.rmscv_scores_.items()
            ), columns=['nLV', 'RMSECV'],
        )

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.figure

        sns.lineplot(
            data=data,
            x='nLV', y='RMSECV',
            errorbar=('sd', 0.5),
            color='cornflowerblue', alpha=0.8,
            ax=ax,
        )
        # Usa o LV efetivamente selecionado (respeita self.rule), em vez de
        # recalcular um argmin independente que poderia divergir dele.
        ax.plot(
            self.best_n_lv_,
            self.best_rmscv_,
            marker='x', color='r', markersize=10,
        )
        ax.set_title(f'Best nLV: {self.best_n_lv_}')
        fig.tight_layout()

        if show:
            plt.show()

        return fig, ax
