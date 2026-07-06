import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

# 스타일 설정
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_and_prepare_data(data_path='cho_culture_data.csv'):
    """데이터 로드 및 전처리"""
    df = pd.read_csv(data_path)

    # 특성과 타겟 분리
    X = df.drop('IgG_Titer', axis=1)
    y = df['IgG_Titer']

    # 훈련/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X, y, X_train, X_test, y_train, y_test

def train_xgboost_model(X_train, X_test, y_train, y_test):
    """XGBoost 모델 학습"""
    print("\n" + "=" * 60)
    print("XGBoost 모델 학습")
    print("=" * 60)

    # XGBoost 회귀 모델 생성
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='reg:squarederror',
        verbosity=1
    )

    # 모델 학습
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    return model

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """모델 평가"""
    print("\n" + "=" * 60)
    print("모델 평가")
    print("=" * 60)

    # 예측
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # 평가 지표
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print(f"\n훈련 데이터:")
    print(f"  RMSE: {train_rmse:.4f}")
    print(f"  MAE:  {train_mae:.4f}")
    print(f"  R²:   {train_r2:.4f}")

    print(f"\n테스트 데이터:")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAE:  {test_mae:.4f}")
    print(f"  R²:   {test_r2:.4f}")

    metrics = {
        'train_rmse': float(train_rmse),
        'test_rmse': float(test_rmse),
        'train_mae': float(train_mae),
        'test_mae': float(test_mae),
        'train_r2': float(train_r2),
        'test_r2': float(test_r2),
    }

    return metrics, y_train_pred, y_test_pred

def plot_results(model, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, metrics):
    """결과 시각화"""
    print("\n" + "=" * 60)
    print("결과 시각화")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('CHO Cell 배양 공정 IgG 예측 모델 평가', fontsize=16, fontweight='bold')

    # 1. 훈련/테스트 실제값 vs 예측값
    ax = axes[0, 0]
    ax.scatter(y_train, y_train_pred, alpha=0.5, label='훈련 데이터', s=30)
    ax.scatter(y_test, y_test_pred, alpha=0.5, label='테스트 데이터', s=30)
    ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2)
    ax.set_xlabel('실제 IgG Titer (g/L)', fontsize=11)
    ax.set_ylabel('예측 IgG Titer (g/L)', fontsize=11)
    ax.set_title('실제값 vs 예측값', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 잔차 분포
    ax = axes[0, 1]
    train_residuals = y_train - y_train_pred
    test_residuals = y_test - y_test_pred
    ax.hist(train_residuals, bins=30, alpha=0.6, label='훈련 잔차')
    ax.hist(test_residuals, bins=30, alpha=0.6, label='테스트 잔차')
    ax.axvline(0, color='k', linestyle='--', lw=2)
    ax.set_xlabel('잔차 (실제값 - 예측값)', fontsize=11)
    ax.set_ylabel('빈도', fontsize=11)
    ax.set_title('잔차 분포', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 특성 중요도
    ax = axes[1, 0]
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    ax.barh(feature_importance['feature'], feature_importance['importance'])
    ax.set_xlabel('중요도', fontsize=11)
    ax.set_title('특성 중요도', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # 4. 성능 지표
    ax = axes[1, 1]
    ax.axis('off')

    metrics_text = f"""
    성능 지표

    훈련 데이터:
      • RMSE: {metrics['train_rmse']:.4f} g/L
      • MAE:  {metrics['train_mae']:.4f} g/L
      • R²:   {metrics['train_r2']:.4f}

    테스트 데이터:
      • RMSE: {metrics['test_rmse']:.4f} g/L
      • MAE:  {metrics['test_mae']:.4f} g/L
      • R²:   {metrics['test_r2']:.4f}

    모델 구성:
      • 트리 개수: 200
      • 최대 깊이: 6
      • 학습률: 0.1
    """

    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('igg_prediction_results.png', dpi=300, bbox_inches='tight')
    print("✓ 그래프 저장: igg_prediction_results.png")

def plot_feature_analysis(X_train, model):
    """특성 분석 시각화"""
    print("\n특성 분석 시각화 생성 중...")

    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(feature_importance)))
    bars = ax.barh(feature_importance['feature'], feature_importance['importance'], color=colors)

    ax.set_xlabel('중요도 점수', fontsize=12, fontweight='bold')
    ax.set_title('XGBoost 모델 - 특성 중요도', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # 값 표시
    for i, (idx, row) in enumerate(feature_importance.iterrows()):
        ax.text(row['importance'], i, f" {row['importance']:.4f}", va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ 그래프 저장: feature_importance.png")

def save_model(model, model_path='xgboost_igg_model.pkl'):
    """모델 저장"""
    joblib.dump(model, model_path)
    print(f"\n✓ 모델 저장: {model_path}")

def main():
    print("\n" + "=" * 60)
    print("CHO Cell 배양 공정 IgG 예측 모델")
    print("=" * 60)

    # 데이터 로드
    print("\n데이터 로드 중...")
    X, y, X_train, X_test, y_train, y_test = load_and_prepare_data()
    print(f"✓ 데이터 로드 완료")
    print(f"  - 훈련 샘플: {len(X_train)}")
    print(f"  - 테스트 샘플: {len(X_test)}")
    print(f"  - 특성 개수: {X_train.shape[1]}")

    # 모델 학습
    model = train_xgboost_model(X_train, X_test, y_train, y_test)
    print("\n✓ 모델 학습 완료")

    # 모델 평가
    metrics, y_train_pred, y_test_pred = evaluate_model(model, X_train, X_test, y_train, y_test)

    # 결과 시각화
    plot_results(model, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, metrics)
    plot_feature_analysis(X_train, model)

    # 모델 저장
    save_model(model)

    # 메트릭 저장
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("✓ 메트릭 저장: metrics.json")

    print("\n" + "=" * 60)
    print("모델 학습 완료!")
    print("=" * 60)
    print("\n생성된 파일:")
    print("  • xgboost_igg_model.pkl - 학습된 모델")
    print("  • igg_prediction_results.png - 평가 그래프")
    print("  • feature_importance.png - 특성 중요도 그래프")
    print("  • metrics.json - 성능 지표")

if __name__ == "__main__":
    main()
