"""
CHO cell VCD & IgG 모델 - Bayesian Optimization 하이퍼파라미터 튜닝

Optuna의 TPE(Tree-structured Parzen Estimator) sampler를 사용한다. TPE는
관측된 (하이퍼파라미터, 성능) 쌍으로부터 확률 모델을 갱신하며 다음 후보를
제안하는 Bayesian optimization 계열 알고리즘이다.

각 타깃(VCD, IgG_Titer)에 대해 5-fold 교차검증 R^2를 최대화하는
하이퍼파라미터를 탐색하고, 결과를 best_params.json에 저장한다.
이후 train_xgboost_model.py가 이 파일을 자동으로 읽어 최적 파라미터로 학습한다.

사용법:
    python hyperparameter_tuning.py                # 두 타깃, 각 60 trial
    python hyperparameter_tuning.py --trials 100   # trial 수 지정
    python hyperparameter_tuning.py --target VCD   # 특정 타깃만
"""

import argparse
import json

import numpy as np
import optuna
import xgboost as xgb
from sklearn.model_selection import cross_val_score

from generate_cho_data import TARGETS
from train_xgboost_model import (
    BEST_PARAMS_PATH,
    FIXED_PARAMS,
    TARGET_UNITS,
    load_and_prepare_data,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _suggest_params(trial):
    """탐색 공간에서 XGBoost 하이퍼파라미터 후보를 제안한다."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 700, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-3, 5.0, log=True),
    }


def tune_target(target, X_train, y_train, n_trials=60, seed=42):
    """단일 타깃에 대해 Bayesian optimization으로 하이퍼파라미터를 탐색한다."""
    y = y_train[target]

    def objective(trial):
        params = {**FIXED_PARAMS, **_suggest_params(trial)}
        model = xgb.XGBRegressor(**params)
        scores = cross_val_score(model, X_train, y, cv=5, scoring="r2", n_jobs=-1)
        return scores.mean()

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    print(f"\n[{target}] ({TARGET_UNITS[target]}) Bayesian optimization - {n_trials} trials")
    baseline = _default_cv_score(X_train, y)
    print(f"  기본 파라미터 CV R2 = {baseline:.4f}")

    def _report(study, trial):
        if (trial.number + 1) % 10 == 0:
            print(f"  trial {trial.number + 1:>3}/{n_trials}  "
                  f"best CV R2 = {study.best_value:.4f}")

    study.optimize(objective, n_trials=n_trials, callbacks=[_report])

    print(f"  완료: best CV R2 = {study.best_value:.4f}  "
          f"(기본 대비 {study.best_value - baseline:+.4f})")
    return {
        "params": study.best_params,
        "cv_r2": float(study.best_value),
        "baseline_cv_r2": float(baseline),
        "n_trials": n_trials,
    }


def _default_cv_score(X_train, y):
    """비교 기준선: 기본 하이퍼파라미터의 5-fold CV R^2."""
    from train_xgboost_model import DEFAULT_TUNABLE

    model = xgb.XGBRegressor(**{**FIXED_PARAMS, **DEFAULT_TUNABLE})
    return float(cross_val_score(model, X_train, y, cv=5, scoring="r2", n_jobs=-1).mean())


def main():
    parser = argparse.ArgumentParser(description="Bayesian optimization 하이퍼파라미터 튜닝")
    parser.add_argument("--trials", type=int, default=60, help="타깃별 trial 수")
    parser.add_argument("--target", choices=TARGETS, help="특정 타깃만 튜닝 (기본: 전체)")
    args = parser.parse_args()

    print("=" * 60)
    print("CHO Cell VCD & IgG - Bayesian Optimization 하이퍼파라미터 튜닝")
    print("=" * 60)

    X_train, _, y_train, _ = load_and_prepare_data()
    print(f"튜닝 데이터: {len(X_train)} 샘플 (train split)")

    targets = [args.target] if args.target else TARGETS

    # 기존 결과가 있으면 유지하고, 튜닝한 타깃만 갱신한다.
    try:
        with open(BEST_PARAMS_PATH) as f:
            results = json.load(f)
    except FileNotFoundError:
        results = {}

    for target in targets:
        results[target] = tune_target(target, X_train, y_train, n_trials=args.trials)

    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print(f"[OK] 최적 하이퍼파라미터 저장: {BEST_PARAMS_PATH}")
    for target in targets:
        r = results[target]
        print(f"\n[{target}]  CV R2 {r['baseline_cv_r2']:.4f} -> {r['cv_r2']:.4f}")
        for k, v in r["params"].items():
            print(f"    {k:<18} {v}")
    print("\n다음 단계: python train_xgboost_model.py  (튜닝된 파라미터로 재학습)")
    print("=" * 60)


if __name__ == "__main__":
    main()
