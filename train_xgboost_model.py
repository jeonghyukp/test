"""
CHO cell 배양 공정 - VCD & IgG 예측 XGBoost 모델 학습

두 개의 XGBoost 회귀 모델을 학습한다.
  - VCD (Viable Cell Density, million cells/mL)
  - IgG_Titer (g/L)

각 타깃별로 성능을 평가하고, 실제-예측 산점도 / 잔차 / 특성 중요도 그래프와
대표 조건에서의 배양 궤적(VCD, IgG vs 배양일) 그래프를 저장한다.
"""

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

from generate_cho_data import FEATURES, TARGETS

plt.style.use("seaborn-v0_8-darkgrid")

MODEL_PATHS = {
    "VCD": "xgboost_vcd_model.pkl",
    "IgG_Titer": "xgboost_igg_model.pkl",
}
TARGET_UNITS = {"VCD": "million cells/mL", "IgG_Titer": "g/L"}
XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_lambda=1.0,
    random_state=42,
    objective="reg:squarederror",
)


def load_and_prepare_data(data_path="cho_culture_data.csv"):
    df = pd.read_csv(data_path)
    X = df[FEATURES]
    y = df[TARGETS]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_one_model(target, X_train, y_train, X_test, y_test):
    """단일 타깃에 대한 XGBoost 모델을 학습한다."""
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    return model


def evaluate(model, target, X_train, y_train, X_test, y_test):
    """타깃별 성능 지표를 계산한다."""
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    cv = cross_val_score(
        xgb.XGBRegressor(**XGB_PARAMS),
        X_train, y_train, cv=5, scoring="r2",
    )
    metrics = {
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, pred_train))),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred_test))),
        "train_mae": float(mean_absolute_error(y_train, pred_train)),
        "test_mae": float(mean_absolute_error(y_test, pred_test)),
        "train_r2": float(r2_score(y_train, pred_train)),
        "test_r2": float(r2_score(y_test, pred_test)),
        "cv_r2_mean": float(cv.mean()),
        "cv_r2_std": float(cv.std()),
    }
    print(f"\n[{target}]  ({TARGET_UNITS[target]})")
    print(f"  Train : RMSE={metrics['train_rmse']:.4f}  MAE={metrics['train_mae']:.4f}  R2={metrics['train_r2']:.4f}")
    print(f"  Test  : RMSE={metrics['test_rmse']:.4f}  MAE={metrics['test_mae']:.4f}  R2={metrics['test_r2']:.4f}")
    print(f"  5-fold CV R2 = {metrics['cv_r2_mean']:.4f} +/- {metrics['cv_r2_std']:.4f}")
    return metrics, pred_test


def plot_evaluation(models, data, predictions, metrics):
    """타깃(행) x [실제-예측, 잔차, 특성 중요도](열) 그래프."""
    X_train, X_test, y_train, y_test = data
    fig, axes = plt.subplots(len(TARGETS), 3, figsize=(17, 10))
    fig.suptitle("CHO Cell Culture - VCD & IgG Prediction (XGBoost)",
                 fontsize=15, fontweight="bold")

    for row, target in enumerate(TARGETS):
        unit = TARGET_UNITS[target]
        y_true = y_test[target].values
        y_pred = predictions[target]

        # 1) 실제 vs 예측
        ax = axes[row, 0]
        ax.scatter(y_true, y_pred, alpha=0.5, s=25, color="#2c7fb8")
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, "k--", lw=2)
        ax.set_xlabel(f"Actual {target} ({unit})")
        ax.set_ylabel(f"Predicted {target} ({unit})")
        ax.set_title(f"{target}: Actual vs Predicted (R2={metrics[target]['test_r2']:.3f})",
                     fontweight="bold")

        # 2) 잔차 분포
        ax = axes[row, 1]
        residuals = y_true - y_pred
        ax.hist(residuals, bins=30, color="#7fcdbb", edgecolor="white")
        ax.axvline(0, color="k", linestyle="--", lw=2)
        ax.set_xlabel(f"Residual ({unit})")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{target}: Residual Distribution", fontweight="bold")

        # 3) 특성 중요도
        ax = axes[row, 2]
        imp = pd.DataFrame({
            "feature": FEATURES,
            "importance": models[target].feature_importances_,
        }).sort_values("importance")
        ax.barh(imp["feature"], imp["importance"], color="#41b6c4")
        ax.set_xlabel("Importance")
        ax.set_title(f"{target}: Feature Importance", fontweight="bold")

    plt.tight_layout()
    plt.savefig("model_evaluation.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("\n[OK] 그래프 저장: model_evaluation.png")


def plot_culture_trajectory(models):
    """대표 배양 조건에서 배양일에 따른 VCD/IgG 예측 궤적."""
    days = np.arange(0, 15)
    base = {
        "Temperature": 34.0, "pH": 7.1, "Dissolved_Oxygen": 55.0,
        "Glucose": 5.0, "Glutamine": 4.0, "Ammonia": 2.0, "Seeding_Density": 0.5,
    }
    grid = pd.DataFrame([{**base, "Culture_Day": d} for d in days])[FEATURES]
    vcd_pred = np.clip(models["VCD"].predict(grid), 0, None)
    igg_pred = np.clip(models["IgG_Titer"].predict(grid), 0, None)

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.plot(days, vcd_pred, "o-", color="#2c7fb8", lw=2, label="VCD")
    ax1.set_xlabel("Culture Day")
    ax1.set_ylabel("VCD (million cells/mL)", color="#2c7fb8")
    ax1.tick_params(axis="y", labelcolor="#2c7fb8")

    ax2 = ax1.twinx()
    ax2.plot(days, igg_pred, "s--", color="#d95f0e", lw=2, label="IgG Titer")
    ax2.set_ylabel("IgG Titer (g/L)", color="#d95f0e")
    ax2.tick_params(axis="y", labelcolor="#d95f0e")
    ax2.grid(False)

    plt.title("Predicted Culture Trajectory (T=34C, pH=7.1, Glc=5 g/L)",
              fontweight="bold")
    fig.tight_layout()
    plt.savefig("culture_trajectory.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("[OK] 그래프 저장: culture_trajectory.png")


def main():
    print("=" * 60)
    print("CHO Cell 배양 공정 - VCD & IgG 예측 모델 학습")
    print("=" * 60)

    X_train, X_test, y_train, y_test = load_and_prepare_data()
    print(f"\n훈련 샘플: {len(X_train)} | 테스트 샘플: {len(X_test)} | 특성: {len(FEATURES)}")

    models, all_metrics, predictions = {}, {}, {}
    for target in TARGETS:
        model = train_one_model(target, X_train, y_train[target], X_test, y_test[target])
        metrics, pred_test = evaluate(model, target, X_train, y_train[target],
                                      X_test, y_test[target])
        joblib.dump(model, MODEL_PATHS[target])
        models[target] = model
        all_metrics[target] = metrics
        predictions[target] = pred_test

    plot_evaluation(models, (X_train, X_test, y_train, y_test), predictions, all_metrics)
    plot_culture_trajectory(models)

    with open("metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("학습 완료! 생성된 파일:")
    print("  - xgboost_vcd_model.pkl / xgboost_igg_model.pkl : 학습된 모델")
    print("  - model_evaluation.png    : 타깃별 평가 그래프")
    print("  - culture_trajectory.png  : 배양 궤적 예측 그래프")
    print("  - metrics.json            : 성능 지표")
    print("=" * 60)


if __name__ == "__main__":
    main()
