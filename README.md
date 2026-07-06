# CHO Cell 배양 공정 IgG 예측 모델 (XGBoost)

XGBoost를 이용한 CHO(Chinese Hamster Ovary) cell 배양 공정의 IgG(면역글로불린 G) 예측 모델입니다.

## 프로젝트 개요

CHO cell은 바이오제약 산업에서 재조합 항체 및 단백질 생산에 사용되는 가장 일반적인 세포주입니다. 이 프로젝트는 배양 공정의 여러 조건(온도, pH, 포도당 농도 등)을 기반으로 IgG 생산량을 예측하는 머신러닝 모델을 제공합니다.

## 모델 성능

| 지표 | 훈련 데이터 | 테스트 데이터 |
|------|-----------|-------------|
| **RMSE** | 0.0213 g/L | 0.7035 g/L |
| **MAE** | 0.0152 g/L | 0.4334 g/L |
| **R²** | 0.9999 | 0.9192 |

- **테스트 R² = 0.9192**: 모델이 테스트 데이터의 약 92%의 분산을 설명
- **테스트 MAE = 0.4334 g/L**: 평균 예측 오차가 약 0.43 g/L

## 프로젝트 구조

```
.
├── generate_cho_data.py          # CHO cell 데이터 생성 스크립트
├── train_xgboost_model.py        # XGBoost 모델 학습 및 평가
├── predict_igg.py                # IgG 예측 함수 및 유틸리티
├── requirements.txt              # Python 의존성
├── cho_culture_data.csv          # 생성된 학습 데이터 (1000 샘플)
├── xgboost_igg_model.pkl         # 학습된 XGBoost 모델
├── metrics.json                  # 모델 성능 지표
├── igg_prediction_results.png    # 모델 평가 그래프
└── feature_importance.png        # 특성 중요도 시각화
```

## 특성 (Features)

모델이 사용하는 8가지 배양 공정 특성:

| 특성 | 단위 | 범위 | 설명 |
|------|------|------|------|
| **Temperature** | °C | 32-37 | 배양 온도 |
| **pH** | - | 6.8-7.4 | pH 값 |
| **Dissolved_Oxygen** | % | 20-100 | 용존산소 포화도 |
| **Glucose** | g/L | 0-10 | 포도당 농도 |
| **Glutamine** | mM | 0-2 | 글루타민 농도 |
| **Ammonia** | mM | 0-10 | 암모니아 농도 |
| **Cell_Density** | million cells/mL | 0.1-10 | 세포 밀도 |
| **Culture_Day** | days | 1-14 | 배양 일수 |

## 설치 및 사용

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 데이터 생성

```bash
python generate_cho_data.py
```

- 1000개의 샘플을 포함한 CHO cell 배양 데이터를 생성합니다.
- 파일: `cho_culture_data.csv`

### 3. 모델 학습

```bash
python train_xgboost_model.py
```

- XGBoost 모델을 학습하고 평가합니다.
- 생성 파일:
  - `xgboost_igg_model.pkl` - 학습된 모델
  - `igg_prediction_results.png` - 성능 평가 그래프
  - `feature_importance.png` - 특성 중요도 시각화
  - `metrics.json` - 성능 지표

### 4. 예측 수행

```bash
python predict_igg.py
```

## 프로그래밍 사용 예제

### 단일 IgG 예측

```python
from predict_igg import predict_igg

# 배양 조건 설정
temperature = 36.5      # °C
ph = 7.2               # pH
dissolved_oxygen = 60  # %
glucose = 5.0         # g/L
glutamine = 1.5       # mM
ammonia = 2.0         # mM
cell_density = 5.0    # million cells/mL
culture_day = 10      # days

# IgG Titer 예측
predicted_igg = predict_igg(
    temperature, ph, dissolved_oxygen, glucose, 
    glutamine, ammonia, cell_density, culture_day
)

print(f"Predicted IgG Titer: {predicted_igg:.4f} g/L")
```

### 배치 예측

```python
import pandas as pd
from predict_igg import predict_batch

# 데이터 준비
data = pd.DataFrame({
    'Temperature': [36.0, 36.5, 37.0],
    'pH': [7.0, 7.2, 7.4],
    'Dissolved_Oxygen': [40, 60, 80],
    'Glucose': [3.0, 5.0, 7.0],
    'Glutamine': [1.0, 1.5, 2.0],
    'Ammonia': [1.0, 2.0, 3.0],
    'Cell_Density': [3.0, 5.0, 7.0],
    'Culture_Day': [7, 10, 14]
})

# 배치 예측
predictions = predict_batch(data)
print(predictions)
```

### 최적 배양 조건 찾기

```python
from predict_igg import optimize_conditions

# IgG 생산을 최대화하는 최적 조건 찾기
optimal = optimize_conditions()
print("Optimal Culture Conditions:")
for key, value in optimal.items():
    print(f"  {key}: {value}")
```

## 모델 아키텍처

**XGBoost Regressor 파라미터:**
- `n_estimators`: 200 (부스팅 라운드)
- `max_depth`: 6 (트리 최대 깊이)
- `learning_rate`: 0.1 (학습률)
- `subsample`: 0.8 (샘플링 비율)
- `colsample_bytree`: 0.8 (특성 샘플링 비율)
- `objective`: reg:squarederror (회귀 목표함수)

## 특성 중요도

모델의 특성 중요도 순서 (상위 5개):
1. **Culture_Day**: 배양 일수가 가장 중요한 인자
2. **Cell_Density**: 세포 밀도
3. **Glucose**: 포도당 농도
4. **Dissolved_Oxygen**: 용존산소
5. **pH**: pH 값

더 자세한 정보는 `feature_importance.png`를 참조하세요.

## 데이터 특성

### 데이터 생성 방식

모델 학습을 위해 현실적인 시뮬레이션 데이터를 생성했습니다:

- **샘플 수**: 1000개
- **테스트 분할**: 80% 훈련, 20% 테스트
- **특성 간 상호작용**: 온도×포도당, 세포밀도×배양일수 상호작용 포함
- **노이즈**: N(0, 0.3) 가우시안 노이즈 추가

### 데이터 통계

```
Temperature:        평균 34.45°C,    표준편차 1.46
pH:                평균 7.10,       표준편차 0.18
Dissolved_Oxygen:  평균 60%,        표준편차 23.09
Glucose:          평균 5.0 g/L,    표준편차 2.89
Glutamine:        평균 1.0 mM,     표준편차 0.58
Ammonia:          평균 5.0 mM,     표준편차 2.89
Cell_Density:     평균 5.05 M cells/mL, 표준편차 2.89
Culture_Day:      평균 7.43일,     표준편차 4.10
IgG_Titer:        평균 1.19 g/L,   표준편차 2.54
```

## 모델 평가 그래프

### igg_prediction_results.png
4개의 서브플롯을 포함:
1. **실제값 vs 예측값**: 모델 예측의 정확도 시각화
2. **잔차 분포**: 예측 오차의 분포
3. **특성 중요도**: 각 특성이 모델에 미치는 영향
4. **성능 지표**: RMSE, MAE, R² 값 요약

### feature_importance.png
각 특성의 XGBoost 중요도 점수를 막대 그래프로 표시합니다.

## 활용 사례

이 모델은 다음과 같은 용도로 활용될 수 있습니다:

1. **배양 공정 최적화**: 최적의 IgG 생산 조건 탐색
2. **공정 제어**: 실시간 배양 조건 모니터링 및 조정
3. **의사결정 지원**: 새로운 배양 시나리오의 예측
4. **품질 관리**: 예상 생산량 기반 품질 확보
5. **자동화**: 공정 자동화 시스템의 입력으로 활용

## 주의사항

- 모델은 시뮬레이션 데이터로 학습되었습니다. 실제 배양 공정 데이터로 재학습하면 성능이 개선됩니다.
- 예측은 학습 데이터의 범위 내에서 가장 신뢰할 수 있습니다.
- 극단적인 입력값에 대해서는 예측의 신뢰도가 낮을 수 있습니다.

## 파일 설명

### 소스 코드

- **generate_cho_data.py**: 
  - CHO cell 배양 공정 데이터 생성
  - 비선형 관계와 상호작용 효과를 포함한 현실적인 데이터 시뮬레이션

- **train_xgboost_model.py**:
  - XGBoost 회귀 모델 학습
  - 교차 검증 및 모델 평가
  - 결과 시각화 및 저장

- **predict_igg.py**:
  - 학습된 모델을 이용한 IgG 예측 함수
  - 단일 예측, 배치 예측, 최적 조건 탐색 기능

### 생성된 아티팩트

- **cho_culture_data.csv**: 1000개 샘플의 학습 데이터
- **xgboost_igg_model.pkl**: joblib으로 저장된 학습된 모델
- **metrics.json**: 모델 평가 지표 (JSON 형식)
- **igg_prediction_results.png**: 4개 서브플롯이 포함된 평가 그래프
- **feature_importance.png**: 특성 중요도 시각화

## 라이선스

이 프로젝트는 교육 및 연구 목적으로 제공됩니다.

## 참고자료

- XGBoost 문서: https://xgboost.readthedocs.io/
- Scikit-learn: https://scikit-learn.org/
- CHO cell 배양 관련 논문 및 자료 참조

## 개선 방향

1. **실제 데이터 통합**: 실제 CHO cell 배양 공정 데이터로 모델 재학습
2. **하이퍼파라미터 최적화**: Bayesian Optimization 활용
3. **앙상블 모델**: 여러 모델을 결합한 고성능 예측기
4. **시계열 분석**: 배양 시간에 따른 역학 고려
5. **불확실성 정량화**: Bayesian 방식의 불확실성 추정

---

**버전**: 1.0  
**생성 날짜**: 2026-07-06  
**연락처**: jeonghyukp@gmail.com
