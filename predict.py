"""
CHO cell 배양 공정 - VCD & IgG 예측 유틸리티

학습된 두 XGBoost 모델을 이용해 배양 조건으로부터
VCD(Viable Cell Density)와 IgG Titer를 예측한다.
"""

import json

import joblib
import numpy as np
import pandas as pd

from generate_cho_data import FEATURES

MODEL_PATHS = {
    "VCD": "xgboost_vcd_model.pkl",
    "IgG_Titer": "xgboost_igg_model.pkl",
}
CLIP_RANGES = {"VCD": (0.0, 20.0), "IgG_Titer": (0.0, 8.0)}


def load_models():
    """저장된 VCD/IgG 모델을 로드한다."""
    return {target: joblib.load(path) for target, path in MODEL_PATHS.items()}


def _to_frame(conditions):
    """dict 또는 DataFrame 입력을 특성 순서에 맞는 DataFrame으로 변환한다."""
    if isinstance(conditions, pd.DataFrame):
        return conditions[FEATURES].copy()
    return pd.DataFrame([conditions])[FEATURES]


def predict_all(conditions, models=None):
    """
    배양 조건으로부터 VCD와 IgG를 함께 예측한다.

    Parameters
    ----------
    conditions : dict | pd.DataFrame
        Temperature, pH, Dissolved_Oxygen, Glucose, Glutamine,
        Ammonia, Seeding_Density, Culture_Day 를 포함해야 한다.
    models : dict, optional
        미리 로드한 모델. None이면 파일에서 로드한다.

    Returns
    -------
    dict  (단일 입력)  또는  pd.DataFrame (배치 입력)
        VCD, IgG_Titer 예측값.
    """
    if models is None:
        models = load_models()
    X = _to_frame(conditions)

    result = {}
    for target, model in models.items():
        low, high = CLIP_RANGES[target]
        result[target] = np.clip(model.predict(X), low, high)

    if len(X) == 1:
        return {k: float(v[0]) for k, v in result.items()}
    return pd.DataFrame(result, index=X.index)


def predict_trajectory(base_conditions, days=range(0, 15), models=None):
    """고정 조건에서 배양일별 VCD/IgG 궤적을 예측한다."""
    if models is None:
        models = load_models()
    rows = [{**base_conditions, "Culture_Day": d} for d in days]
    frame = pd.DataFrame(rows)
    preds = predict_all(frame, models=models)
    preds.insert(0, "Culture_Day", list(days))
    return preds


def optimize_conditions(target="IgG_Titer", models=None, n_grid=8):
    """
    지정한 타깃을 최대화하는 배양 조건을 그리드 탐색으로 찾는다.
    최적 조건에서의 두 타깃 예측값을 함께 반환한다.
    """
    if models is None:
        models = load_models()

    grids = {
        "Temperature": np.linspace(30, 37.5, n_grid),
        "pH": np.linspace(6.7, 7.4, n_grid),
        "Dissolved_Oxygen": np.linspace(20, 100, n_grid),
        "Glucose": np.linspace(1, 10, n_grid),
        "Glutamine": np.linspace(0.5, 6, n_grid),
        "Ammonia": np.linspace(0, 12, n_grid),
        "Seeding_Density": np.array([0.5, 1.0]),
        "Culture_Day": np.array([7, 10, 12, 14]),
    }
    # 전수 그리드는 비싸므로 랜덤 서치로 대규모 후보를 평가한다.
    rng = np.random.default_rng(0)
    n_candidates = 200_000
    candidates = pd.DataFrame({
        feat: rng.choice(values, n_candidates) for feat, values in grids.items()
    })[FEATURES]

    preds = predict_all(candidates, models=models)
    best_idx = preds[target].idxmax()

    best = candidates.loc[best_idx].to_dict()
    best["Culture_Day"] = int(best["Culture_Day"])
    best["Predicted_VCD"] = round(float(preds.loc[best_idx, "VCD"]), 4)
    best["Predicted_IgG_Titer"] = round(float(preds.loc[best_idx, "IgG_Titer"]), 4)
    return best


def print_model_info(models=None):
    """모델 성능 및 특성 중요도를 출력한다."""
    if models is None:
        models = load_models()
    try:
        with open("metrics.json") as f:
            metrics = json.load(f)
    except FileNotFoundError:
        metrics = {}

    print("\n" + "=" * 60)
    print("XGBoost VCD & IgG 예측 모델 정보")
    print("=" * 60)
    for target, model in models.items():
        print(f"\n[{target}]")
        if target in metrics:
            m = metrics[target]
            print(f"  Test R2={m['test_r2']:.4f}  RMSE={m['test_rmse']:.4f}  MAE={m['test_mae']:.4f}")
        imp = sorted(zip(FEATURES, model.feature_importances_),
                     key=lambda x: x[1], reverse=True)
        print("  특성 중요도:")
        for feat, val in imp:
            print(f"    {feat:<18} {val:.4f}")


def main():
    print("=" * 60)
    print("CHO Cell 배양 공정 - VCD & IgG 예측 예제")
    print("=" * 60)
    models = load_models()

    # 예제 1: 단일 예측
    print("\n[예제 1] 단일 조건 예측")
    conditions = {
        "Temperature": 34.0, "pH": 7.1, "Dissolved_Oxygen": 55.0,
        "Glucose": 5.0, "Glutamine": 4.0, "Ammonia": 2.0,
        "Seeding_Density": 0.5, "Culture_Day": 10,
    }
    for k, v in conditions.items():
        print(f"    {k}: {v}")
    pred = predict_all(conditions, models=models)
    print(f"  -> VCD       : {pred['VCD']:.3f} million cells/mL")
    print(f"  -> IgG Titer : {pred['IgG_Titer']:.3f} g/L")

    # 예제 2: 배치 예측
    print("\n[예제 2] 배치 예측")
    batch = pd.DataFrame({
        "Temperature": [37.0, 34.0, 31.0],
        "pH": [7.0, 7.1, 7.2],
        "Dissolved_Oxygen": [40, 55, 70],
        "Glucose": [3.0, 5.0, 7.0],
        "Glutamine": [2.0, 4.0, 5.0],
        "Ammonia": [3.0, 2.0, 1.0],
        "Seeding_Density": [0.4, 0.5, 0.8],
        "Culture_Day": [7, 10, 12],
    })
    print(pd.concat([batch, predict_all(batch, models=models).round(3)], axis=1).to_string(index=False))

    # 예제 3: 배양 궤적
    print("\n[예제 3] 배양 궤적 예측 (T=34C, pH=7.1)")
    base = {
        "Temperature": 34.0, "pH": 7.1, "Dissolved_Oxygen": 55.0,
        "Glucose": 5.0, "Glutamine": 4.0, "Ammonia": 2.0, "Seeding_Density": 0.5,
    }
    traj = predict_trajectory(base, models=models)
    print(traj.round(3).to_string(index=False))

    # 예제 4: 모델 정보
    print_model_info(models)

    # 예제 5: IgG 최대화 조건 탐색
    print("\n[예제 5] IgG 최대화 배양 조건 탐색")
    best = optimize_conditions(target="IgG_Titer", models=models)
    for k, v in best.items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
