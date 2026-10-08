"""
=============================
GridSearchCV using a pipeline
=============================

Finds the optimal development and CapeCod parameters
using the unweighted CDR score, then evaluates the best model on two
unseen valuation periods using the IBNR RMSE.

Since selecting development factors is a transformation,
it can be pipelined with an estimator
"""
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from tryangle import CapeCod, Development
from tryangle.metrics import neg_cdr_scorer, neg_ibnr_scorer
from tryangle.model_selection import TriangleSplit
from tryangle.utils.datasets import load_sample

X = load_sample("swiss")
latest_valuation = X.triangle.latest_diagonal.valuation[0]
valuation_dates = X.triangle.valuation.drop_duplicates().sort_values().to_numpy()
valuation_dates = valuation_dates[valuation_dates <= latest_valuation]
X_train = X[X.triangle.valuation <= valuation_dates[-3]]
tscv = TriangleSplit(n_splits=5)

param_grid = {
    "dev__n_periods": range(15, 20),
    "dev__drop_high": [True, False],
    "dev__drop_low": [True, False],
    "cc__decay": [0.25, 0.5, 0.75, 0.95],
}

pipe = Pipeline([("dev", Development()), ("cc", CapeCod())])

model = GridSearchCV(
    pipe, param_grid=param_grid, scoring=neg_cdr_scorer, cv=tscv, verbose=1, n_jobs=-1
)
model.fit(X_train, X_train)
print(model.best_params_)
print("Two-period holdout RMSE:", -neg_ibnr_scorer(model.best_estimator_, X_train, X))

# TODO add plotting
