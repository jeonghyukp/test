import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# 랜덤 시드 설정
np.random.seed(42)

def generate_cho_culture_data(n_samples=1000):
    """
    CHO cell 배양 공정 데이터 생성

    Features:
    - Temperature (°C): 32-37
    - pH: 6.8-7.4
    - Dissolved_Oxygen (%): 20-100
    - Glucose (g/L): 0-10
    - Glutamine (mM): 0-2
    - Ammonia (mM): 0-10
    - Cell_Density (million cells/mL): 0.1-10
    - Culture_Day: 1-14

    Target:
    - IgG_Titer (g/L): 0-10
    """

    data = {
        'Temperature': np.random.uniform(32, 37, n_samples),
        'pH': np.random.uniform(6.8, 7.4, n_samples),
        'Dissolved_Oxygen': np.random.uniform(20, 100, n_samples),
        'Glucose': np.random.uniform(0, 10, n_samples),
        'Glutamine': np.random.uniform(0, 2, n_samples),
        'Ammonia': np.random.uniform(0, 10, n_samples),
        'Cell_Density': np.random.uniform(0.1, 10, n_samples),
        'Culture_Day': np.random.randint(1, 15, n_samples),
    }

    df = pd.DataFrame(data)

    # IgG Titer 생성 (복잡한 비선형 관계)
    # 기본값
    igg_titer = np.ones(n_samples) * 0.5

    # Temperature의 영향 (37°C에 가까울수록 증가, 하지만 너무 높으면 감소)
    temp_effect = -2 * (df['Temperature'] - 36.5) ** 2 + 1.5
    igg_titer += temp_effect

    # pH의 영향 (최적 pH ~7.2)
    ph_effect = -5 * (df['pH'] - 7.2) ** 2 + 0.8
    igg_titer += ph_effect

    # Dissolved Oxygen의 영향 (40-80% 범위에서 최적)
    do_effect = -0.015 * (df['Dissolved_Oxygen'] - 60) ** 2 + 0.6
    igg_titer += do_effect

    # Glucose의 영향 (높을수록 좋지만, 너무 높으면 독성)
    glucose_effect = 0.4 * df['Glucose'] - 0.02 * df['Glucose'] ** 2
    igg_titer += glucose_effect

    # Glutamine의 영향
    glutamine_effect = 0.5 * df['Glutamine'] - 0.15 * df['Glutamine'] ** 2
    igg_titer += glutamine_effect

    # Ammonia의 영향 (높을수록 나쁨)
    ammonia_effect = -0.08 * df['Ammonia']
    igg_titer += ammonia_effect

    # Cell Density의 영향 (높을수록 IgG 생산 증가)
    cell_effect = 0.3 * np.log(df['Cell_Density'] + 0.1)
    igg_titer += cell_effect

    # Culture Day의 영향 (초기에 빠르게 증가, 나중에 느려짐)
    day_effect = 0.15 * np.log(df['Culture_Day'])
    igg_titer += day_effect

    # 상호작용 효과: Temperature와 Glucose
    igg_titer += 0.02 * df['Temperature'] * df['Glucose']

    # 상호작용 효과: Cell_Density와 Culture_Day
    igg_titer += 0.05 * np.log(df['Cell_Density'] + 0.1) * np.log(df['Culture_Day'])

    # 노이즈 추가
    igg_titer += np.random.normal(0, 0.3, n_samples)

    # IgG Titer를 0-10 범위로 클리핑
    igg_titer = np.clip(igg_titer, 0, 10)

    df['IgG_Titer'] = igg_titer

    return df

if __name__ == "__main__":
    print("=" * 60)
    print("CHO Cell 배양 공정 데이터 생성")
    print("=" * 60)

    # 데이터 생성
    df = generate_cho_culture_data(n_samples=1000)

    # 데이터 저장
    df.to_csv('cho_culture_data.csv', index=False)
    print(f"\n✓ 데이터 생성 완료: {len(df)} 샘플")
    print(f"✓ 파일 저장: cho_culture_data.csv\n")

    # 데이터 정보 출력
    print("데이터 통계:")
    print(df.describe())

    print("\n특성 정보:")
    print(df.info())
