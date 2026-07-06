"""
Mechanistic layer - Monod 성장 동역학

Hybrid 모델의 기계론적(first-principles) 부분이다. 다음 Monod 성장식을 사용한다.

    dX/dt = mu_max * (C_S / (K_S + C_S)) * X

여기서
    X    : 생존 세포 밀도 VCD (million cells/mL)
    C_S  : 기질(포도당) 농도 (g/L)
    mu_max : 최대 비성장속도 (1/day)
    K_S  : Monod 반포화 상수 (g/L)

기질 소모는 세포 성장에 비례한다고 두어 물질수지를 함께 적분한다.

    dC_S/dt = -(1/Y_XS) * mu_max * (C_S / (K_S + C_S)) * X

이 mechanistic 모델은 기질 제한 성장기(growth phase)만 설명한다.
사멸기(decline), 온도/pH/DO/암모니아 효과 등 Monod가 담지 못하는 동역학은
hybrid 모델의 XGBoost 잔차 보정이 담당한다.
"""

import numpy as np
import pandas as pd

from generate_cho_data import FEATURES

# Monod 동역학 파라미터 (mechanistic 기준선이 실제 VCD 스케일과 일치하도록 보정)
# mu_max/Yxs는 Monod 성장이 데이터의 VCD 규모(평균~5, 최대~16 M cells/mL)에
# 맞도록, qP_nom은 IgG_mech 평균이 실제 IgG 평균에 맞도록 선정했다.
MU_MAX = 0.60    # 최대 비성장속도 (1/day)
KS = 1.0         # Monod 반포화 상수 (g/L glucose)
YXS = 1.5        # 수율 (million cells/mL 당 g/L glucose)
QP_NOMINAL = 0.024  # 공칭 비생산성 (IgG_mech 기준선 스케일, g/L per IVCD)

# mechanistic 층이 생성하는 파생 특성
MECH_FEATURES = ["X_mech", "IVCD_mech", "Glucose_remaining"]
HYBRID_FEATURES = FEATURES + MECH_FEATURES

T_MAX = 14       # 최대 배양일
DT = 0.05        # 적분 스텝 (day)


def _derivatives(X, Cs, mu_max, Ks, Yxs):
    """Monod 성장 및 기질 소모 미분값."""
    mu = mu_max * Cs / (Ks + Cs)
    dX = mu * X
    dCs = -(1.0 / Yxs) * mu * X
    return dX, dCs


def integrate_population(X0, Cs0, mu_max=MU_MAX, Ks=KS, Yxs=YXS,
                         t_max=T_MAX, dt=DT):
    """
    여러 배양(샘플)의 Monod ODE를 벡터화된 RK4로 동시에 적분한다.

    Parameters
    ----------
    X0, Cs0 : array-like
        각 샘플의 초기 VCD(접종 밀도)와 초기 기질(포도당) 농도.

    Returns
    -------
    X_arr, Cs_arr, IVCD_arr : np.ndarray, shape (t_max+1, n_samples)
        정수 배양일(0..t_max)에서의 VCD, 잔존 기질, 누적 생존세포 적분값(IVCD).
    """
    X = np.asarray(X0, dtype=float).copy()
    Cs = np.asarray(Cs0, dtype=float).copy()
    n = X.shape[0]
    ivcd = np.zeros(n)

    X_arr = np.zeros((t_max + 1, n))
    Cs_arr = np.zeros((t_max + 1, n))
    IVCD_arr = np.zeros((t_max + 1, n))
    X_arr[0], Cs_arr[0], IVCD_arr[0] = X, Cs, ivcd

    steps = int(round(t_max / dt))
    for step in range(1, steps + 1):
        k1X, k1C = _derivatives(X, Cs, mu_max, Ks, Yxs)
        k2X, k2C = _derivatives(X + 0.5 * dt * k1X, Cs + 0.5 * dt * k1C, mu_max, Ks, Yxs)
        k3X, k3C = _derivatives(X + 0.5 * dt * k2X, Cs + 0.5 * dt * k2C, mu_max, Ks, Yxs)
        k4X, k4C = _derivatives(X + dt * k3X, Cs + dt * k3C, mu_max, Ks, Yxs)

        X_new = X + dt / 6.0 * (k1X + 2 * k2X + 2 * k3X + k4X)
        Cs_new = Cs + dt / 6.0 * (k1C + 2 * k2C + 2 * k3C + k4C)

        ivcd = ivcd + 0.5 * (X + np.clip(X_new, 0, None)) * dt  # 사다리꼴 적분
        X = np.clip(X_new, 0.0, None)
        Cs = np.clip(Cs_new, 0.0, None)

        t = step * dt
        nearest = round(t)
        if abs(t - nearest) < dt / 2 and 0 <= nearest <= t_max:
            X_arr[nearest] = X
            Cs_arr[nearest] = Cs
            IVCD_arr[nearest] = ivcd
    return X_arr, Cs_arr, IVCD_arr


def compute_mechanistic_features(df):
    """
    공정 조건 DataFrame으로부터 mechanistic 파생 특성을 계산한다.

    - X_mech           : Monod ODE로 예측한 배양일 t에서의 VCD
    - IVCD_mech        : 0~t 생존 세포 적분값 (IgG 생산의 기계론적 구동력)
    - Glucose_remaining: Monod 물질수지로 예측한 잔존 포도당

    접종 밀도(Seeding_Density)를 초기 VCD, 측정 포도당(Glucose)을 초기 기질로
    두고 배양일(Culture_Day)까지 적분한다.
    """
    X0 = df["Seeding_Density"].to_numpy(dtype=float)
    Cs0 = df["Glucose"].to_numpy(dtype=float)
    days = df["Culture_Day"].to_numpy(dtype=int)

    X_arr, Cs_arr, IVCD_arr = integrate_population(X0, Cs0)
    cols = np.arange(len(df))

    feats = pd.DataFrame(index=df.index)
    feats["X_mech"] = X_arr[days, cols]
    feats["IVCD_mech"] = IVCD_arr[days, cols]
    feats["Glucose_remaining"] = Cs_arr[days, cols]
    return feats


def mechanistic_baseline(df):
    """
    순수 mechanistic 예측(기준선)을 반환한다. XGBoost 보정 없이 Monod만 사용.

    - VCD_mech : Monod ODE VCD
    - IgG_mech : 공칭 qP * IVCD (Luedeking-Piret의 성장연계 항 단순화)
    """
    feats = compute_mechanistic_features(df)
    baseline = pd.DataFrame(index=df.index)
    baseline["VCD"] = feats["X_mech"]
    baseline["IgG_Titer"] = QP_NOMINAL * feats["IVCD_mech"]
    return baseline, feats


if __name__ == "__main__":
    # mechanistic 층 단독 점검
    df = pd.read_csv("cho_culture_data.csv")
    feats = compute_mechanistic_features(df)
    baseline, _ = mechanistic_baseline(df)

    print("=" * 60)
    print("Mechanistic (Monod) 층 점검")
    print("=" * 60)
    print(f"\n파라미터: mu_max={MU_MAX} /day, Ks={KS} g/L, Yxs={YXS}, qP_nom={QP_NOMINAL}")
    print("\nmechanistic 파생 특성 통계:")
    print(feats.describe().round(3).to_string())

    print("\nmechanistic 기준선 vs 실제 (상관계수):")
    for target in ["VCD", "IgG_Titer"]:
        r = np.corrcoef(baseline[target], df[target])[0, 1]
        print(f"  {target}: Pearson r = {r:.3f}")
