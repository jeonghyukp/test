"""
Hybrid model (XGBoost + Mechanistic Monod) - 예측 유틸리티

학습된 하이브리드 모델(hybrid_model.pkl)로 배양 조건에서 VCD와 IgG를 예측한다.
mechanistic 기준선(Monod)과 hybrid 최종 예측을 함께 확인할 수 있다.
"""

import joblib
import numpy as np
import pandas as pd

from generate_cho_data import FEATURES
from mechanistic_model import mechanistic_baseline

HYBRID_MODEL_PATH = "hybrid_model.pkl"


def load_model(path=HYBRID_MODEL_PATH):
    return joblib.load(path)


def _to_frame(conditions):
    if isinstance(conditions, pd.DataFrame):
        return conditions[FEATURES].copy()
    return pd.DataFrame([conditions])[FEATURES]


def predict_all(conditions, model=None, with_mechanistic=False):
    """
    배양 조건으로부터 VCD와 IgG를 예측한다.

    with_mechanistic=True 이면 Monod 기준선(VCD_mech, IgG_mech)도 함께 반환하여
    mechanistic 기여와 hybrid 최종 예측을 비교할 수 있다.
    """
    if model is None:
        model = load_model()
    X = _to_frame(conditions)
    pred = model.predict(X)

    if with_mechanistic:
        mech, _ = mechanistic_baseline(X)
        pred = pred.copy()
        pred["VCD_mech"] = np.clip(mech["VCD"].to_numpy(), 0, None)
        pred["IgG_mech"] = np.clip(mech["IgG_Titer"].to_numpy(), 0, None)

    if len(X) == 1:
        return {k: float(v) for k, v in pred.iloc[0].items()}
    return pred


def predict_trajectory(base_conditions, days=range(0, 15), model=None):
    """고정 조건에서 배양일별 VCD/IgG 궤적(mechanistic + hybrid)을 예측한다."""
    if model is None:
        model = load_model()
    frame = pd.DataFrame([{**base_conditions, "Culture_Day": d} for d in days])
    out = predict_all(frame, model=model, with_mechanistic=True)
    out.insert(0, "Culture_Day", list(days))
    return out


def main():
    print("=" * 60)
    print("Hybrid Model (XGBoost + Monod) - 예측 예제")
    print("=" * 60)
    model = load_model()

    print("\n[예제 1] 단일 조건 예측 (mechanistic + hybrid)")
    conditions = {
        "Temperature": 34.0, "pH": 7.1, "Dissolved_Oxygen": 55.0,
        "Glucose": 5.0, "Glutamine": 4.0, "Ammonia": 2.0,
        "Seeding_Density": 0.5, "Culture_Day": 10,
    }
    pred = predict_all(conditions, model=model, with_mechanistic=True)
    print(f"  VCD  : mechanistic {pred['VCD_mech']:.2f}  ->  hybrid {pred['VCD']:.2f} million cells/mL")
    print(f"  IgG  : mechanistic {pred['IgG_mech']:.2f}  ->  hybrid {pred['IgG_Titer']:.2f} g/L")

    print("\n[예제 2] 기준 조건 배양 궤적 (Day 10-14)")
    base = {
        "Temperature": 34.0, "pH": 7.1, "Dissolved_Oxygen": 55.0,
        "Glucose": 5.0, "Glutamine": 4.0, "Ammonia": 2.0, "Seeding_Density": 0.5,
    }
    traj = predict_trajectory(base, model=model)
    print(traj[traj.Culture_Day >= 10].round(3).to_string(index=False))
    d14 = float(traj[traj.Culture_Day == 14]["IgG_Titer"].iloc[0])
    print(f"\n  Day 14 hybrid IgG titer = {d14:.3f} g/L  (데이터 캘리브레이션 목표 3.0)")


if __name__ == "__main__":
    main()
