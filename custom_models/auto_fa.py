import numpy as np

from sklearn.covariance import oas
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis


class AutoFactorAnalysis(FactorAnalysis):
    """FactorAnalysis com seleção automática do número de fatores."""

    _VALID_METHODS = ('kc', 'ekc', 'pa', 'map')

    def __init__(
        self,
        method: str = 'kc',
        *,
        tol=1e-2,
        copy=True,
        max_iter=1000,
        noise_variance_init=None,
        svd_method='randomized',
        iterated_power=3,
        rotation=None,
        random_state: int = 0,
    ):
        super().__init__(
            n_components=None,
            tol=tol,
            copy=copy,
            max_iter=max_iter,
            noise_variance_init=noise_variance_init,
            svd_method=svd_method,
            iterated_power=iterated_power,
            rotation=rotation,
            random_state=random_state,
        )
        self.method = method

    @staticmethod
    def _correlation_eigenvalues(X):
        """Padroniza X e retorna os autovalores da matriz de correlação
        (com encolhimento Oracle Approximating Shrinkage), em ordem decrescente."""
        X_std = StandardScaler().fit_transform(X)
        corr_matrix, _ = oas(X_std)
        return np.linalg.eigvalsh(corr_matrix)[::-1]

    @classmethod
    def _kaiser_criterion(cls, X) -> int:
        eigenvalues = cls._correlation_eigenvalues(X)
        n_components = int(np.sum(eigenvalues >= 1))
        return max(n_components, 1)

    @classmethod
    def _empirical_kaiser_criterion(cls, X) -> int:
        eigenvalues = cls._correlation_eigenvalues(X)
        n, p = X.shape  # n amostras, p variáveis
        ref = np.zeros(p)
        cumsum_ref = 0.0
        for j in range(p):
            v_j = (p - cumsum_ref) / (p - j) if j > 0 else 1.0
            ref[j] = max((1 + np.sqrt(p / n)) ** 2 * v_j, 1.0)
            cumsum_ref += ref[j]
        n_components = 0
        for j in range(p):
            if eigenvalues[j] > ref[j]:
                n_components += 1
            else:
                break
        return max(n_components, 1)

    @classmethod
    def _parallel_analysis(cls, X, n_iter: int = 100, percentile: float = 95, random_state=None) -> int:
        rng = np.random.default_rng(random_state)
        n, p = X.shape
        eigenvalues = cls._correlation_eigenvalues(X)

        random_eigenvalues = np.empty((n_iter, p))
        for i in range(n_iter):
            X_random = rng.standard_normal(size=(n, p))
            random_eigenvalues[i] = cls._correlation_eigenvalues(X_random)

        ref = np.percentile(random_eigenvalues, percentile, axis=0)

        n_components = int(np.sum(eigenvalues > ref))
        return max(n_components, 1)

    @classmethod
    def _velicers_map(cls, X) -> int:
        """Critério MAP (Minimum Average Partial) de Velicer (1976)."""
        X_std = StandardScaler().fit_transform(X)
        corr_matrix, _ = oas(X_std)
        n, p = X.shape

        eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        max_m = min(p - 1, n - 1)  # não faz sentido extrair mais que isso
        map_values = np.empty(max_m + 1)

        off_diag = corr_matrix - np.diag(np.diag(corr_matrix))
        map_values[0] = np.sum(off_diag ** 2) / (p * (p - 1))

        for m in range(1, max_m + 1):
            loadings = eigenvectors[:, :m] * np.sqrt(np.maximum(eigenvalues[:m], 0))
            residual = corr_matrix - loadings @ loadings.T

            d = np.sqrt(np.abs(np.diag(residual)))
            d[d == 0] = np.finfo(float).eps  # evita divisão por zero
            partial_corr = residual / np.outer(d, d)

            off_diag = partial_corr - np.diag(np.diag(partial_corr))
            map_values[m] = np.sum(off_diag ** 2) / (p * (p - 1))

        n_components = int(np.argmin(map_values))
        return max(n_components, 1)

    def _select_n_components(self, X) -> int:
        if self.method == 'kc':
            return self._kaiser_criterion(X)
        elif self.method == 'ekc':
            return self._empirical_kaiser_criterion(X)
        elif self.method == 'pa':
            return self._parallel_analysis(X, random_state=self.random_state)
        elif self.method == 'map':
            return self._velicers_map(X)
        raise ValueError(
            f"method deve ser um de {self._VALID_METHODS} "
            f"recebido '{self.method}'"
        )

    def fit(self, X, y=None):
        self.n_components = self._select_n_components(X)
        return super().fit(X)

    def transform(self, X, y=None):
        return super().transform(X)
