from sklearn.pipeline import FeatureUnion
from itertools import izip
from sklearn.externals.joblib import Parallel, delayed
from sklearn.pipeline import _fit_transform_one, _transform_one


class FeatureDictionary(FeatureUnion):
    def __init__(self, transformer_list, n_jobs=1, transformer_weights=None):
        super(FeatureDictionary, self).__init__(self, transformer_list, n_jobs, transformer_weights)

    def fit_transform(self, X, y=None, **fit_params):
        """Fit all transformers using X, transform the data and concatenate
        results.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Input data to be transformed.

        Returns
        -------
        named_transformed_data : list<tuple<string, matrix>>
                where string is a name of transformer and matrix is a result of transformation
        """
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(trans, name, X, y,
                                        self.transformer_weights, **fit_params)
            for name, trans in self.transformer_list)

        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)
        names, _ = self.tranformer_list

        named_transformed_data = [(name, transformed_data) for name, transformed_data in izip(names, Xs)]
        return named_transformed_data

    def transform(self, X):
        """Transform X separately by each transformer, concatenate results.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Input data to be transformed.

        Returns
        -------
        named_transformed_data : list<tuple<string, matrix>>
                where string is a name of transformer and matrix is a result of transformation
        """
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, name, X, self.transformer_weights)
            for name, trans in self.transformer_list)
        names, _ = self.tranformer_list
        named_transformed_data = [(name, transformed_data) for name, transformed_data in izip(names, Xs)]
        return named_transformed_data
