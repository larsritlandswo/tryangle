# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import chainladder as cl
import pytest
from sklearn.pipeline import Pipeline

from tryangle import CapeCod, Development
from tryangle.core.base import TryangleData
from tryangle.core.methods import Chainladder
from tryangle.metrics.score import (
    neg_ave_scorer,
    neg_cdr_scorer,
    neg_ibnr_scorer,
    neg_weighted_ave_scorer,
    neg_weighted_cdr_scorer,
)
from tryangle.utils.datasets import load_sample


@pytest.mark.parametrize(
    "scorer, true_score",
    [
        (neg_ave_scorer, -2222.357324),
        (neg_weighted_ave_scorer, -2372.370821),
        (neg_cdr_scorer, -5064.408237),
        (neg_weighted_cdr_scorer, -5891.385703),
    ],
)
def test_scorers(scorer, true_score):
    X = TryangleData(cl.load_sample("raa"))
    score = scorer(Chainladder(), X, X)
    assert (score - true_score) < 0.001


def test_ibnr_scorer_uses_two_future_valuation_periods():
    X = load_sample("swiss")
    valuation_dates = X.triangle.valuation.drop_duplicates().sort_values().to_numpy()
    valuation_dates = valuation_dates[
        valuation_dates <= X.triangle.latest_diagonal.valuation[0]
    ]
    X_train = X[X.triangle.valuation <= valuation_dates[-3]]
    estimator = Pipeline([("dev", Development()), ("cc", CapeCod())]).fit(
        X_train, X_train
    )

    score = neg_ibnr_scorer(estimator, X_train, X)

    assert score < 0


def test_ibnr_scorer_requires_two_future_valuation_periods():
    X = load_sample("swiss")
    estimator = Pipeline([("dev", Development()), ("cc", CapeCod())]).fit(X, X)

    with pytest.raises(ValueError, match="2 future valuation periods"):
        neg_ibnr_scorer(estimator, X, X)
