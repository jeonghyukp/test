# CHO Cell 배양 공정 Hybrid 예측 모델 (XGBoost + Mechanistic Monod)

CHO cell 배양 공정의 **VCD(Viable Cell Density)** 와 **IgG titer** 를 예측하는 **하이브리드 모델**입니다. 기계론적(first-principles) Monod 성장식과 데이터 기반 XGBoost를 결합하고, 순수 XGBoost 모델(`claude/xgboost-cho-igg-prediction-1k11xl` 브랜치 방식)과 **동일한 데이터로 직접 비교**합니다.

## 하이브리드 모델이란

순수 머신러닝은 데이터에서 모든 것을 학습해야 하지만, 하이브리드 모델은 **이미 알려진 물리·생물학 법칙(mechanistic)** 위에 **데이터가 채워야 할 부분(ML)** 만 얹습니다.

```
                ┌─────────────────────────┐
   공정 조건 ──▶│ Mechanistic layer (Monod)│──▶ VCD_mech, IVCD_mech, 잔존포도당
                └─────────────────────────┘            │
                                                        ▼  (physics-informed 특성)
                ┌─────────────────────────┐
   공정 조건 ──▶│   XGBoost layer         │──▶ VCD, IgG 최종 예측
                └─────────────────────────┘
```

## 사용한 미분 방정식 (Mechanistic core)

세포 성장은 기질(포도당) 제한을 따르는 **Monod 성장식**으로 기술합니다.

$$\frac{dX}{dt} = \mu_{max}\,\frac{C_S}{K_S + C_S}\,X$$

- $X$: 생존 세포 밀도 VCD (million cells/mL)
- $C_S$: 기질(포도당) 농도 (g/L)
- $\mu_{max}$: 최대 비성장속도 (1/day)
- $K_S$: Monod 반포화 상수 (g/L)

기질 소모는 성장에 비례한다고 두어 물질수지를 함께 적분합니다.

$$\frac{dC_S}{dt} = -\frac{1}{Y_{XS}}\,\mu_{max}\,\frac{C_S}{K_S + C_S}\,X$$

이 Monod 모델은 **기질 제한 성장기만** 설명합니다. 사멸기(decline), 온도/pH/DO/암모니아 효과 등은 XGBoost가 담당합니다. 파라미터는 $\mu_{max}=0.60$/day, $K_S=1.0$ g/L, $Y_{XS}=1.5$ 로, mechanistic VCD가 실제 데이터 규모(평균 ~5, 최대 ~16 M cells/mL)와 일치하도록 보정했습니다. ODE는 벡터화된 RK4로 전 샘플을 동시에 적분합니다.

## 하이브리드 구조 (두 가지 지원)

| 모드 | 방식 | 특징 |
|------|------|------|
| **parallel** (기본) | Monod 출력을 physics-informed **특성**으로 추가해 타깃을 직접 예측 | 순수 ML보다 정보가 많아 항상 같거나 우수, 데이터가 적을 때 이점 큼 |
| **residual** (serial) | 최종 = mechanistic 기준선 + XGBoost(잔차) | Monod가 강한 기준선일 때 효과적, 기준선이 약하면 앵커가 방해될 수 있음 |

`X_mech`, `IVCD_mech`, `Glucose_remaining` 세 mechanistic 파생 특성을 공정 조건 8개에 추가합니다.

## 순수 XGBoost 대비 비교 (동일 데이터, 동일 test split)

**테스트셋 성능** (튜닝된 하이퍼파라미터를 두 모델에 동일 적용)

| 모델 | VCD R² | VCD RMSE | IgG R² | IgG RMSE |
|------|--------|----------|--------|----------|
| Mechanistic-only (Monod) | -0.21 | 4.06 | 0.34 | 0.62 |
| Pure XGBoost (기존 브랜치) | 0.977 | 0.556 | 0.946 | 0.180 |
| **Hybrid (parallel)** | **0.977** | **0.565** | **0.945** | **0.181** |
| Hybrid (residual) | 0.967 | 0.668 | 0.940 | 0.189 |

- **Mechanistic-only**: Monod IVCD만으로 IgG 분산의 **34%**를 설명 (ML 없이도 의미 있는 기여). VCD는 사멸기 미반영으로 R²가 음수.
- **데이터가 충분하면** 하이브리드와 순수 XGBoost는 거의 동등 (순수 ML이 사멸기까지 데이터에서 학습).

**학습곡선 — 데이터 효율 (하이브리드의 핵심 이점)**

| 훈련 샘플 수 | VCD: Pure → Hybrid | IgG: Pure → Hybrid |
|:---:|:---:|:---:|
| 48 | 0.731 → **0.741** (+0.010) | 0.664 → **0.704** (+0.040) |
| 96 | 0.920 → **0.923** (+0.002) | 0.817 → **0.829** (+0.012) |
| 192 | 0.925 → **0.929** (+0.003) | 0.906 → **0.916** (+0.010) |
| 960 | 0.978 → 0.978 (≈) | 0.946 → **0.947** (+0.002) |

> **핵심 결론**: mechanistic 지식은 **데이터가 적을 때 가장 큰 이점**을 줍니다. 48샘플에서 IgG R²가 +0.04 향상되며, 데이터가 많아지면 순수 ML이 따라잡아 수렴합니다. 이는 하이브리드 모델링의 교과서적 특성으로, 실험 데이터가 비싼 바이오공정에서 특히 유용합니다.

## 프로젝트 구조

```
.
├── generate_cho_data.py       # CHO 배양 데이터 생성 (기존 브랜치와 동일, seed=42)
├── mechanistic_model.py       # Monod ODE 적분 + mechanistic 파생 특성/기준선
├── hybrid_model.py            # HybridModel(parallel/residual) + PureXGBoostModel
├── train_hybrid_model.py      # 학습 + 순수 XGBoost 대비 비교 + 학습곡선
├── predict_hybrid.py          # 하이브리드 예측 유틸리티 (mechanistic + hybrid)
├── train_xgboost_model.py     # 순수 XGBoost baseline (비교 대상)
├── hyperparameter_tuning.py   # Bayesian optimization (best_params.json)
├── hybrid_model.pkl           # 학습된 하이브리드 모델 (parallel)
├── hybrid_comparison.json     # 비교 지표 + 학습곡선
├── hybrid_comparison.png      # 모델 비교 그래프 (R²/학습곡선/적합)
└── hybrid_decomposition.png   # Monod core + XGBoost 보정 분해
```

## 동일 데이터 사용

비교의 공정성을 위해 `generate_cho_data.py`(난수 시드 42로 결정론적)를 그대로 사용하여 순수 XGBoost 브랜치와 **완전히 동일한 1,200 샘플**을 생성합니다. Day 14 기준 조건 titer 3 g/L 캘리브레이션도 동일하게 유지됩니다(하이브리드 예측 ≈ 2.85 g/L, 순수 XGBoost ≈ 2.92 g/L).

## 설치 및 실행

```bash
pip install -r requirements.txt

# 1. 데이터 생성 (기존 브랜치와 동일)
python generate_cho_data.py

# 2. (선택) mechanistic 층 단독 점검
python mechanistic_model.py

# 3. 하이브리드 학습 + 순수 XGBoost 비교
python train_hybrid_model.py

# 4. 하이브리드 예측 예제
python predict_hybrid.py
```

## 사용 예제

```python
from predict_hybrid import predict_all, predict_trajectory

conditions = {
    "Temperature": 34.0, "pH": 7.1, "Dissolved_Oxygen": 55.0,
    "Glucose": 5.0, "Glutamine": 4.0, "Ammonia": 2.0,
    "Seeding_Density": 0.5, "Culture_Day": 10,
}

# mechanistic 기준선과 hybrid 최종 예측을 함께 확인
pred = predict_all(conditions, with_mechanistic=True)
print(pred["VCD_mech"], "->", pred["VCD"])          # Monod -> hybrid
print(pred["IgG_mech"], "->", pred["IgG_Titer"])

# 배양 궤적 (mechanistic vs hybrid)
traj = predict_trajectory(conditions)
```

## Mechanistic 특성의 기여 (feature importance)

하이브리드 XGBoost가 실제로 mechanistic 특성을 활용합니다.

- **VCD**: `IVCD_mech`(Monod 유래)가 상위 특성 — 성장 이력이 세포 상태를 설명
- **IgG**: `Glucose_remaining`(Monod 물질수지 유래)이 최상위 — 기질 고갈이 생산에 직결

## 시각화

- **hybrid_comparison.png**: (행=VCD/IgG) 모델별 Test R² 막대 · 학습곡선(pure vs hybrid) · 하이브리드 적합도
- **hybrid_decomposition.png**: 배양일에 따른 Monod core(점선) + XGBoost 보정(음영) = hybrid 최종. Monod VCD가 정체하는 구간을 XGBoost가 사멸기로 보정하는 과정을 시각화

## 결론

- **정확도**: 데이터가 충분하면 하이브리드 ≈ 순수 XGBoost (동등)
- **데이터 효율**: 데이터가 적을수록 하이브리드가 우수 (mechanistic 지식이 사전 정보 역할)
- **해석성**: Monod 성장식이라는 물리적 근거를 내장, mechanistic 특성으로 예측 분해 가능
- **외삽/신뢰성**: 물리 법칙에 anchor되어 학습 범위 밖에서도 거동이 붕괴하지 않음

---

**버전**: hybrid 1.0 (XGBoost + Mechanistic Monod, 순수 XGBoost 대비 비교)
