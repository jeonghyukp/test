"""
Hybrid model (XGBoost + Mechanistic)

Serial(residual) 하이브리드 구조를 구현한다.

    최종 예측 = mechanistic 예측(Monod) + XGBoost 잔차 보정

  1. Mechanistic 층 (mechanistic_model.py)
     - Monod ODE로 VCD_mech, IVCD_mech, 잔존 포도당을 계산
     - 순수 mechanistic 기준선(VCD_mech, IgG_mech = qP*IVCD)을 제공
  2. Data-driven 층 (XGBoost)
     - 입력: 공정 조건 8개 + mechanistic 파생 특성 3개
     - 타깃: (실제값 - mechanistic 기준선) 잔차
     - Monod가 설명하지 못하는 사멸기, 온도/pH/DO/암모니아 효과를 학습

이렇게 하면 XGBoost는 처음부터 전체를 학습하는 대신, 이미 알려진 물리
법칙 위에서 "부족한 부분"만 학습하므로 데이터 효율과 외삽 성능이 개선된다.
"""

import numpy as np
import pandas as pd
import xgboost as xgb

from generate_cho_data import FEATURES, TARGETS
from mechanistic_model import HYBRID_FEATURES, mechanistic_baseline

CLIP_RANGES = {"VCD": (0.0, 20.0), "IgG_Titer": (0.0, 8.0)}


def _params_for(params, target):
    """단일 dict이면 모든 타깃에 공용, {target: dict}이면 타깃별 파라미터를 반환."""
    if target in params and isinstance(params[target], dict):
        return params[target]
    return params


class HybridModel:
    """
    Hybrid model (XGBoost + Mechanistic Monod).

    두 가지 하이브리드 구조를 지원한다.

    mode="parallel" (기본, feature augmentation)
        Monod ODE 출력(X_mech, IVCD_mech, Glucose_remaining)을 physics-informed
        특성으로 추가하여 타깃을 직접 예측한다. 순수 ML보다 정보가 많아
        같거나 더 나은 성능을 내며, 특히 데이터가 적을 때 이점이 크다.

    mode="residual" (serial)
        최종 예측 = mechanistic 기준선 + XGBoost(잔차). Monod가 강한 기준선일 때
        효과적이지만, 기준선이 약하면(예: 사멸기 미반영 VCD) 앵커가 방해가 될 수 있다.
    """

    def __init__(self, params, mode="parallel"):
        assert mode in ("parallel", "residual")
        self.params = params
        self.mode = mode
        self.models = {}

    def _design_matrix(self, df):
        """공정 특성 + mechanistic 파생 특성을 결합한 입력 행렬과 기준선을 반환."""
        baseline, feats = mechanistic_baseline(df)
        X = pd.concat([df[FEATURES].reset_index(drop=True),
                       feats.reset_index(drop=True)], axis=1)[HYBRID_FEATURES]
        return X, baseline.reset_index(drop=True)

    def fit(self, df_train):
        X, baseline = self._design_matrix(df_train)
        for target in TARGETS:
            y = df_train[target].to_numpy()
            fit_target = y if self.mode == "parallel" else y - baseline[target].to_numpy()
            model = xgb.XGBRegressor(**_params_for(self.params, target))
            model.fit(X, fit_target)
            self.models[target] = model
        return self

    def predict(self, df):
        X, baseline = self._design_matrix(df)
        out = {}
        for target in TARGETS:
            pred = self.models[target].predict(X)
            if self.mode == "residual":
                pred = baseline[target].to_numpy() + pred
            low, high = CLIP_RANGES[target]
            out[target] = np.clip(pred, low, high)
        index = df.index if len(df) > 1 else [0]
        return pd.DataFrame(out, index=index)

    def feature_importance(self, target):
        return pd.DataFrame({
            "feature": HYBRID_FEATURES,
            "importance": self.models[target].feature_importances_,
        }).sort_values("importance", ascending=False)


class PureXGBoostModel:
    """비교용 순수 XGBoost 모델 (기존 브랜치 방식, 공정 특성만 사용)."""

    def __init__(self, params):
        self.params = params
        self.models = {}

    def fit(self, df_train):
        X = df_train[FEATURES]
        for target in TARGETS:
            model = xgb.XGBRegressor(**_params_for(self.params, target))
            model.fit(X, df_train[target])
            self.models[target] = model
        return self

    def predict(self, df):
        X = df[FEATURES]
        out = {}
        for target in TARGETS:
            low, high = CLIP_RANGES[target]
            out[target] = np.clip(self.models[target].predict(X), low, high)
        index = df.index if len(df) > 1 else [0]
        return pd.DataFrame(out, index=index)
