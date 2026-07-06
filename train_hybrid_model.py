"""
Hybrid model 학습 및 순수 XGBoost 대비 비교

동일한 데이터(cho_culture_data.csv)와 동일한 train/test split에서 세 가지를 비교한다.
  1. Mechanistic-only : Monod ODE 기준선 (ML 없음)
  2. Pure XGBoost     : 공정 특성만 사용 (기존 브랜치 방식)
  3. Hybrid           : Monod mechanistic + XGBoost 잔차 보정

추가로, 학습 데이터 크기에 따른 성능(learning curve)을 비교하여
mechanistic 지식이 데이터 효율에 주는 이점을 확인한다.
"""

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from generate_cho_data import FEATURES, TARGETS
from hybrid_model import HybridModel, PureXGBoostModel
from mechanistic_model import mechanistic_baseline
from train_xgboost_model import load_and_prepare_data, resolve_params

plt.style.use("seaborn-v0_8-darkgrid")

TARGET_UNITS = {"VCD": "million cells/mL", "IgG_Titer": "g/L"}
HYBRID_MODEL_PATH = "hybrid_model.pkl"


def _metrics(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def split_frames():
    """train_xgboost_model과 동일한 split을 DataFrame 형태로 재구성한다."""
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
    return train, test


MODEL_ORDER = ["Mechanistic", "PureXGBoost", "Hybrid-residual", "Hybrid-parallel"]


def evaluate_all(train, test, params):
    """모든 모델을 학습/평가하고 타깃별 지표를 반환한다."""
    # 1) mechanistic-only (Monod, ML 없음)
    mech_test, _ = mechanistic_baseline(test)

    # 2) pure XGBoost (기존 브랜치 방식)
    pure = PureXGBoostModel(params).fit(train)
    pure_pred = pure.predict(test)

    # 3) hybrid - residual(serial) 및 parallel(feature augmentation)
    hybrid_res = HybridModel(params, mode="residual").fit(train)
    hybrid_par = HybridModel(params, mode="parallel").fit(train)
    res_pred = hybrid_res.predict(test)
    par_pred = hybrid_par.predict(test)

    results = {}
    for target in TARGETS:
        y_true = test[target].to_numpy()
        results[target] = {
            "Mechanistic": _metrics(y_true, np.clip(mech_test[target].to_numpy(), 0, None)),
            "PureXGBoost": _metrics(y_true, pure_pred[target].to_numpy()),
            "Hybrid-residual": _metrics(y_true, res_pred[target].to_numpy()),
            "Hybrid-parallel": _metrics(y_true, par_pred[target].to_numpy()),
        }
    # parallel 하이브리드를 배포 모델로 사용
    return results, pure, hybrid_par, pure_pred, par_pred


def print_comparison(results):
    print("\n" + "=" * 72)
    print("모델 비교 (테스트셋)")
    print("=" * 72)
    for target in TARGETS:
        print(f"\n[{target}]  ({TARGET_UNITS[target]})")
        print(f"  {'Model':<18}{'R2':>10}{'RMSE':>10}{'MAE':>10}")
        print("  " + "-" * 48)
        for name in MODEL_ORDER:
            m = results[target][name]
            print(f"  {name:<18}{m['r2']:>10.4f}{m['rmse']:>10.4f}{m['mae']:>10.4f}")
        gain = results[target]["Hybrid-parallel"]["r2"] - results[target]["PureXGBoost"]["r2"]
        print(f"  -> Hybrid(parallel) R2가 Pure XGBoost 대비 {gain:+.4f}")


def learning_curve(train, test, params, fractions=(0.05, 0.1, 0.2, 0.4, 0.7, 1.0), seed=42):
    """학습 데이터 비율에 따른 테스트 R^2를 pure vs hybrid로 비교한다."""
    rng = np.random.default_rng(seed)
    n = len(train)
    curve = {t: {"frac": [], "PureXGBoost": [], "Hybrid": []} for t in TARGETS}

    for frac in fractions:
        k = max(30, int(round(frac * n)))
        idx = rng.choice(n, size=min(k, n), replace=False)
        sub = train.iloc[idx]

        pure_pred = PureXGBoostModel(params).fit(sub).predict(test)
        hybrid_pred = HybridModel(params).fit(sub).predict(test)
        for target in TARGETS:
            y_true = test[target].to_numpy()
            curve[target]["frac"].append(len(sub))
            curve[target]["PureXGBoost"].append(r2_score(y_true, pure_pred[target]))
            curve[target]["Hybrid"].append(r2_score(y_true, hybrid_pred[target]))
    return curve


def plot_comparison(results, curve, hybrid, test, hybrid_pred):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Hybrid (XGBoost + Monod) vs Pure XGBoost", fontsize=15, fontweight="bold")
    models = ["Mechanistic", "PureXGBoost", "Hybrid-residual", "Hybrid-parallel"]
    colors = {"Mechanistic": "#b0b0b0", "PureXGBoost": "#2c7fb8",
              "Hybrid-residual": "#f0a35e", "Hybrid-parallel": "#d95f0e"}
    hybrid_color = colors["Hybrid-parallel"]

    for row, target in enumerate(TARGETS):
        unit = TARGET_UNITS[target]

        # (1) test R2 막대 비교
        ax = axes[row, 0]
        r2s = [results[target][m]["r2"] for m in models]
        ax.bar(range(len(models)), r2s, color=[colors[m] for m in models])
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(["Mech", "Pure", "Hyb-res", "Hyb-par"], fontsize=9)
        ax.set_ylim(min(0, min(r2s)) - 0.05, 1.0)
        ax.set_ylabel("Test R2")
        ax.set_title(f"{target}: Test R2 by model", fontweight="bold")
        for i, v in enumerate(r2s):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

        # (2) 학습곡선 (parallel hybrid vs pure)
        ax = axes[row, 1]
        c = curve[target]
        ax.plot(c["frac"], c["PureXGBoost"], "o-", color=colors["PureXGBoost"], label="Pure XGBoost")
        ax.plot(c["frac"], c["Hybrid"], "s--", color=hybrid_color, label="Hybrid (parallel)")
        ax.set_xlabel("Training samples")
        ax.set_ylabel("Test R2")
        ax.set_title(f"{target}: Learning curve", fontweight="bold")
        ax.legend()

        # (3) hybrid 실제 vs 예측
        ax = axes[row, 2]
        y_true = test[target].to_numpy()
        ax.scatter(y_true, hybrid_pred[target], alpha=0.5, s=22, color=hybrid_color)
        lims = [min(y_true.min(), hybrid_pred[target].min()),
                max(y_true.max(), hybrid_pred[target].max())]
        ax.plot(lims, lims, "k--", lw=2)
        ax.set_xlabel(f"Actual {target} ({unit})")
        ax.set_ylabel(f"Hybrid predicted ({unit})")
        ax.set_title(f"{target}: Hybrid fit (R2={results[target]['Hybrid-parallel']['r2']:.3f})",
                     fontweight="bold")

    plt.tight_layout()
    plt.savefig("hybrid_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("\n[OK] 그래프 저장: hybrid_comparison.png")


def plot_decomposition(hybrid):
    """대표 조건에서 mechanistic core + XGBoost 보정의 분해를 시각화한다."""
    from mechanistic_model import mechanistic_baseline

    days = np.arange(0, 15)
    base = {
        "Temperature": 34.0, "pH": 7.1, "Dissolved_Oxygen": 55.0,
        "Glucose": 5.0, "Glutamine": 4.0, "Ammonia": 2.0, "Seeding_Density": 0.5,
    }
    grid = pd.DataFrame([{**base, "Culture_Day": d} for d in days])
    mech, _ = mechanistic_baseline(grid)
    hyb = hybrid.predict(grid)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle("Hybrid decomposition: Monod core + XGBoost correction (T=34C)",
                 fontweight="bold")
    for ax, target, unit in zip(axes, TARGETS, ["million cells/mL", "g/L"]):
        ax.plot(days, mech[target], "--", color="#7f7f7f", lw=2, label="Mechanistic (Monod)")
        ax.plot(days, hyb[target], "o-", color="#d95f0e", lw=2, label="Hybrid (final)")
        ax.fill_between(days, mech[target], hyb[target], color="#fdd0a2", alpha=0.5,
                        label="XGBoost correction")
        ax.set_xlabel("Culture Day")
        ax.set_ylabel(f"{target} ({unit})")
        ax.set_title(target, fontweight="bold")
        ax.legend()
    plt.tight_layout()
    plt.savefig("hybrid_decomposition.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("[OK] 그래프 저장: hybrid_decomposition.png")


def main():
    print("=" * 68)
    print("Hybrid Model (XGBoost + Mechanistic Monod) 학습 및 비교")
    print("=" * 68)

    train, test = split_frames()
    print(f"\n훈련 샘플: {len(train)} | 테스트 샘플: {len(test)}")

    # 기존 브랜치의 튜닝 결과를 baseline/hybrid에 동일 적용 (공정 비교)
    params = {t: resolve_params(t)[0] for t in TARGETS}
    print(f"하이퍼파라미터 소스: {resolve_params('VCD')[1]}")

    results, pure, hybrid, pure_pred, hybrid_pred = evaluate_all(train, test, params)
    print_comparison(results)

    print("\n학습곡선(learning curve) 계산 중...")
    curve = learning_curve(train, test, params)

    plot_comparison(results, curve, hybrid, test, hybrid_pred)
    plot_decomposition(hybrid)

    joblib.dump(hybrid, HYBRID_MODEL_PATH)
    with open("hybrid_comparison.json", "w") as f:
        json.dump({"test_metrics": results,
                   "learning_curve": curve}, f, indent=2)

    print(f"\n[OK] 하이브리드 모델 저장: {HYBRID_MODEL_PATH}")
    print("[OK] 비교 결과 저장: hybrid_comparison.json")

    print("\nHybrid 잔차 모델 특성 중요도 (상위 5):")
    for target in TARGETS:
        top = hybrid.feature_importance(target).head(5)
        pairs = ", ".join(f"{r.feature} {r.importance:.2f}" for r in top.itertuples())
        print(f"  [{target}] {pairs}")


if __name__ == "__main__":
    main()
