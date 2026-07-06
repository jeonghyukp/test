import pandas as pd
import numpy as np
import joblib
import json

def load_model(model_path='xgboost_igg_model.pkl'):
    """저장된 모델 로드"""
    model = joblib.load(model_path)
    return model

def predict_igg(temperature, ph, dissolved_oxygen, glucose, glutamine, ammonia, cell_density, culture_day, model=None):
    """
    CHO cell 배양 공정에서 IgG Titer 예측

    Parameters:
    -----------
    temperature : float
        배양 온도 (°C), 범위: 32-37
    ph : float
        pH 값, 범위: 6.8-7.4
    dissolved_oxygen : float
        용존산소 (%), 범위: 20-100
    glucose : float
        포도당 농도 (g/L), 범위: 0-10
    glutamine : float
        글루타민 농도 (mM), 범위: 0-2
    ammonia : float
        암모니아 농도 (mM), 범위: 0-10
    cell_density : float
        세포 밀도 (million cells/mL), 범위: 0.1-10
    culture_day : int
        배양 일수, 범위: 1-14
    model : xgboost.XGBRegressor
        학습된 모델. None이면 저장된 모델 로드

    Returns:
    --------
    float
        예측된 IgG Titer (g/L)
    """
    if model is None:
        model = load_model()

    # 입력 데이터를 DataFrame으로 변환
    input_data = pd.DataFrame({
        'Temperature': [temperature],
        'pH': [ph],
        'Dissolved_Oxygen': [dissolved_oxygen],
        'Glucose': [glucose],
        'Glutamine': [glutamine],
        'Ammonia': [ammonia],
        'Cell_Density': [cell_density],
        'Culture_Day': [culture_day]
    })

    # 예측
    prediction = model.predict(input_data)[0]

    # 예측값을 0-10 범위로 클리핑
    prediction = np.clip(prediction, 0, 10)

    return prediction

def predict_batch(data_df, model=None):
    """
    배치 예측

    Parameters:
    -----------
    data_df : pd.DataFrame
        예측할 데이터 (특성 컬럼 포함)
    model : xgboost.XGBRegressor
        학습된 모델. None이면 저장된 모델 로드

    Returns:
    --------
    np.array
        예측된 IgG Titer
    """
    if model is None:
        model = load_model()

    predictions = model.predict(data_df)
    predictions = np.clip(predictions, 0, 10)

    return predictions

def get_model_info(model=None):
    """모델 정보 출력"""
    if model is None:
        model = load_model()

    print("\n" + "=" * 60)
    print("XGBoost IgG 예측 모델 정보")
    print("=" * 60)

    # 메트릭 로드
    try:
        with open('metrics.json', 'r') as f:
            metrics = json.load(f)

        print("\n모델 성능:")
        print(f"  테스트 RMSE: {metrics['test_rmse']:.4f} g/L")
        print(f"  테스트 MAE:  {metrics['test_mae']:.4f} g/L")
        print(f"  테스트 R²:   {metrics['test_r2']:.4f}")

    except FileNotFoundError:
        print("\n메트릭 파일을 찾을 수 없습니다.")

    print("\n특성 중요도:")
    for i, importance in enumerate(model.feature_importances_):
        feature_name = model.get_booster().feature_names[i]
        print(f"  {feature_name}: {importance:.4f}")

    print("\n모델 파라미터:")
    print(f"  트리 개수: {model.n_estimators}")
    print(f"  최대 깊이: {model.max_depth}")
    print(f"  학습률: {model.learning_rate}")

def optimize_conditions(model=None):
    """
    IgG 예측을 최대화하는 최적 배양 조건 제시

    Returns:
    --------
    dict
        최적 조건과 예측 IgG Titer
    """
    if model is None:
        model = load_model()

    # 그리드 서치를 통한 최적 조건 찾기
    temperatures = np.linspace(32, 37, 10)
    phs = np.linspace(6.8, 7.4, 10)
    dos = np.linspace(20, 100, 10)
    glucoses = np.linspace(0, 10, 10)
    glutamines = np.linspace(0, 2, 10)
    ammonia_values = np.linspace(0, 10, 10)
    cell_densities = np.linspace(0.1, 10, 10)
    culture_days = np.array([7, 10, 12, 14])  # 주요 배양 일수

    best_prediction = 0
    best_conditions = None

    print("\n최적 조건 탐색 중...")

    # 효율성을 위해 샘플링
    for temp in temperatures:
        for ph in phs:
            for do in dos:
                for glucose in glucoses:
                    for glutamine in glutamines:
                        for ammonia in ammonia_values:
                            for cell_density in cell_densities:
                                for day in culture_days:
                                    prediction = predict_igg(
                                        temp, ph, do, glucose, glutamine, ammonia, cell_density, day, model
                                    )

                                    if prediction > best_prediction:
                                        best_prediction = prediction
                                        best_conditions = {
                                            'Temperature': round(temp, 2),
                                            'pH': round(ph, 2),
                                            'Dissolved_Oxygen': round(do, 2),
                                            'Glucose': round(glucose, 2),
                                            'Glutamine': round(glutamine, 2),
                                            'Ammonia': round(ammonia, 2),
                                            'Cell_Density': round(cell_density, 2),
                                            'Culture_Day': int(day),
                                            'Predicted_IgG_Titer': round(best_prediction, 4)
                                        }

    return best_conditions

def main():
    """예제 실행"""
    print("\n" + "=" * 60)
    print("CHO Cell IgG Titer 예측")
    print("=" * 60)

    # 모델 로드
    model = load_model()

    # 예제 1: 단일 예측
    print("\n예제 1: 단일 예측")
    print("-" * 40)

    example_conditions = {
        'temperature': 36.5,
        'ph': 7.2,
        'dissolved_oxygen': 60,
        'glucose': 5.0,
        'glutamine': 1.5,
        'ammonia': 2.0,
        'cell_density': 5.0,
        'culture_day': 10
    }

    predicted_igg = predict_igg(**example_conditions, model=model)

    print("배양 조건:")
    for key, value in example_conditions.items():
        print(f"  {key}: {value}")

    print(f"\n예측 IgG Titer: {predicted_igg:.4f} g/L")

    # 예제 2: 배치 예측
    print("\n예제 2: 배치 예측")
    print("-" * 40)

    # 샘플 데이터 생성
    sample_data = pd.DataFrame({
        'Temperature': [36.0, 36.5, 37.0],
        'pH': [7.0, 7.2, 7.4],
        'Dissolved_Oxygen': [40, 60, 80],
        'Glucose': [3.0, 5.0, 7.0],
        'Glutamine': [1.0, 1.5, 2.0],
        'Ammonia': [1.0, 2.0, 3.0],
        'Cell_Density': [3.0, 5.0, 7.0],
        'Culture_Day': [7, 10, 14]
    })

    batch_predictions = predict_batch(sample_data, model=model)

    print("입력 데이터:")
    print(sample_data)

    print("\n예측 결과:")
    for i, pred in enumerate(batch_predictions):
        print(f"  샘플 {i+1}: {pred:.4f} g/L")

    # 예제 3: 모델 정보
    get_model_info(model)

    # 예제 4: 최적 조건 탐색
    print("\n최적 배양 조건:")
    print("-" * 40)
    optimal_conditions = optimize_conditions(model)

    for key, value in optimal_conditions.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
