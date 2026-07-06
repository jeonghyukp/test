# CHO Cell 배양 공정 VCD & IgG 예측 모델 (XGBoost)

XGBoost를 이용한 CHO(Chinese Hamster Ovary) cell 배양 공정의 **VCD(Viable Cell Density, 생존 세포 밀도)** 와 **IgG(면역글로불린 G) titer** 예측 모델입니다.

## 프로젝트 개요

CHO cell은 바이오제약 산업에서 재조합 항체 생산에 가장 널리 쓰이는 세포주입니다. 이 프로젝트는 배양 공정 조건(온도, pH, 용존산소, 영양분, 접종 밀도, 배양일)으로부터 **두 가지 핵심 공정 변수**를 동시에 예측합니다.

- **VCD** — 배양 시점의 생존 세포 밀도 (million cells/mL)
- **IgG Titer** — 누적 항체 생산량 (g/L)

각 타깃에 대해 독립적인 XGBoost 회귀 모델을 학습하여, 타깃별로 성능과 특성 중요도를 따로 해석할 수 있습니다.

## 모델 성능

| 타깃 | 단위 | Test R² | Test RMSE | Test MAE | 5-fold CV R² |
|------|------|---------|-----------|----------|--------------|
| **VCD** | million cells/mL | 0.971 | 0.628 | 0.483 | 0.967 ± 0.003 |
| **IgG_Titer** | g/L | 0.954 | 0.203 | 0.152 | 0.944 ± 0.003 |

5-fold 교차검증의 낮은 표준편차(±0.003)는 모델이 안정적으로 일반화됨을 보여줍니다.

## 데이터 설계 (생물학적 근거)

학습 데이터(1,200 샘플)는 실제 CHO 유가배양(fed-batch)에서 관찰되는 현상을 반영하도록 생성했습니다.

1. **VCD 종형 곡선** — 세포 밀도는 배양일에 따라 성장기 → 정체기 → 사멸기의 종형(bell-shaped) 궤적을 그립니다.
2. **성장 반응 계수** — 온도/pH/DO의 최적점(가우시안), 포도당·글루타민의 Monod 포화, 암모니아 독성 억제를 6개 계수의 **기하평균**으로 결합합니다. (단순 곱은 한 인자만 낮아도 성장이 0으로 붕괴하므로 비현실적)
3. **IgG 누적 (IVCD)** — 항체는 생존 세포의 시간 적분(Integrated Viable Cell Density)에 비례하여 누적됩니다. 가우시안 VCD 곡선을 해석적으로 적분(오차함수)하여 계산합니다.
4. **저온 생산성 상승** — 비생산성(qP)은 경미한 저온 배양(mild hypothermia)에서 상승합니다. 이는 CHO 배양에서 잘 알려진 현상으로, **낮은 온도에서 VCD는 감소하지만 IgG titer는 오히려 증가**하는 트레이드오프를 만듭니다.

이 설계 덕분에 두 타깃은 단순 비례가 아닌 **조건 의존적 상관관계**(Pearson r ≈ 0.26)를 가집니다.

## 특성 (Features)

| 특성 | 단위 | 범위 | 설명 |
|------|------|------|------|
| **Temperature** | °C | 30 – 37.5 | 배양 온도 |
| **pH** | - | 6.7 – 7.4 | pH 값 |
| **Dissolved_Oxygen** | % | 20 – 100 | 용존산소 포화도 |
| **Glucose** | g/L | 0 – 10 | 포도당 농도 |
| **Glutamine** | mM | 0 – 6 | 글루타민 농도 |
| **Ammonia** | mM | 0 – 12 | 암모니아 농도 (대사 부산물, 독성) |
| **Seeding_Density** | million cells/mL | 0.2 – 1.0 | 접종(inoculation) 밀도 |
| **Culture_Day** | day | 0 – 14 | 배양 일수 |

> **참고**: 이전 버전에서 입력이던 `Cell_Density`는 이제 예측 대상 `VCD`로 이동했습니다. 세포 밀도를 입력으로 넣고 세포 밀도를 예측하는 순환 참조를 피하기 위해, 실제 공정 입력인 `Seeding_Density`(접종 밀도)를 특성으로 추가했습니다.

## 프로젝트 구조

```
.
├── generate_cho_data.py       # CHO 배양 데이터 생성 (VCD + IgG 타깃)
├── train_xgboost_model.py     # VCD/IgG 두 XGBoost 모델 학습 및 평가
├── predict.py                 # 예측 유틸리티 (단일/배치/궤적/최적화)
├── requirements.txt           # Python 의존성
├── cho_culture_data.csv       # 생성된 학습 데이터 (1200 샘플)
├── xgboost_vcd_model.pkl      # 학습된 VCD 모델
├── xgboost_igg_model.pkl      # 학습된 IgG 모델
├── metrics.json               # 타깃별 성능 지표
├── model_evaluation.png       # 타깃별 평가 그래프 (실제-예측/잔차/중요도)
└── culture_trajectory.png     # 배양일에 따른 VCD/IgG 궤적 예측
```

## 설치 및 실행

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 데이터 생성
python generate_cho_data.py

# 3. 모델 학습 및 평가
python train_xgboost_model.py

# 4. 예측 예제 실행
python predict.py
```

## 사용 예제

### VCD와 IgG 동시 예측

```python
from predict import predict_all

conditions = {
    "Temperature": 34.0, "pH": 7.1, "Dissolved_Oxygen": 55.0,
    "Glucose": 5.0, "Glutamine": 4.0, "Ammonia": 2.0,
    "Seeding_Density": 0.5, "Culture_Day": 10,
}

pred = predict_all(conditions)
print(f"VCD       : {pred['VCD']:.3f} million cells/mL")   # -> 9.240
print(f"IgG Titer : {pred['IgG_Titer']:.3f} g/L")          # -> 2.494
```

### 배치 예측

```python
import pandas as pd
from predict import predict_all

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

print(predict_all(batch))
# 37C: VCD 높고 IgG 낮음  /  31C: VCD 낮지만 IgG 높음 (저온 생산성 효과)
```

### 배양 궤적 예측

```python
from predict import predict_trajectory

base = {
    "Temperature": 34.0, "pH": 7.1, "Dissolved_Oxygen": 55.0,
    "Glucose": 5.0, "Glutamine": 4.0, "Ammonia": 2.0, "Seeding_Density": 0.5,
}
trajectory = predict_trajectory(base, days=range(0, 15))
print(trajectory)   # 배양일별 VCD(종형)와 IgG(누적) 예측
```

### IgG 최대화 조건 탐색

```python
from predict import optimize_conditions

best = optimize_conditions(target="IgG_Titer")
print(best)   # 저온·고영양·저암모니아·장기배양 조건에서 IgG 최대
```

## 모델 구성

두 타깃 모두 동일한 XGBoost 하이퍼파라미터를 사용합니다.

```python
XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_lambda=1.0,
    objective="reg:squarederror",
)
```

## 특성 중요도

| 순위 | VCD | IgG_Titer |
|------|-----|-----------|
| 1 | Culture_Day (0.70) | Culture_Day (0.63) |
| 2 | Seeding_Density (0.13) | Ammonia (0.18) |
| 3 | Temperature (0.07) | Seeding_Density (0.07) |
| 4 | Glucose (0.03) | Glucose (0.04) |
| 5 | Ammonia (0.02) | Temperature (0.04) |

- **VCD**: 배양일과 접종 밀도가 세포 성장 궤적을 지배
- **IgG**: 배양일 다음으로 **암모니아 독성**이 중요 — 대사 부산물이 항체 생산을 제한하는 실제 현상 반영

## 시각화

- **model_evaluation.png**: 2행(VCD, IgG) × 3열(실제 vs 예측 산점도 · 잔차 분포 · 특성 중요도)
- **culture_trajectory.png**: 고정 조건에서 배양일에 따른 VCD(종형)와 IgG(누적) 예측 궤적 (이중 y축)

## 활용 사례

1. **공정 최적화** — 목표 IgG를 달성하는 배양 조건 탐색
2. **실시간 모니터링** — 현재 조건에서의 VCD/IgG 예상값 추정
3. **의사결정 지원** — 저온 전환 시점 등 공정 전략 시뮬레이션
4. **수확 시점 결정** — VCD 피크 및 IgG 누적 곡선 기반 harvest timing

## 주의사항

- 모델은 물리·생물학적 관계를 반영한 **시뮬레이션 데이터**로 학습되었습니다. 실제 배양 데이터로 재학습하면 실무에 활용할 수 있습니다.
- 예측은 학습 데이터 범위 내에서 가장 신뢰할 수 있습니다.

## 개선 방향

1. 실제 CHO 배양 공정 데이터로 재학습
2. Bayesian Optimization 기반 하이퍼파라미터 튜닝
3. VCD·IgG를 함께 예측하는 multi-output / 시계열 모델 (LSTM 등)
4. Quantile regression으로 예측 불확실성 정량화

---

**버전**: 2.0 (VCD + IgG 멀티 타깃)
