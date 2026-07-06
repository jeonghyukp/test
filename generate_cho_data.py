"""
CHO cell 배양 공정 데이터 생성

VCD(Viable Cell Density)와 IgG Titer를 함께 예측하기 위한 학습 데이터를 생성한다.
데이터는 실제 CHO 유가배양(fed-batch)에서 관찰되는 다음 현상을 반영하도록 모사한다.

  1. VCD는 배양일에 따라 종형(bell-shaped) 곡선을 그린다 (성장기 -> 정체기 -> 사멸기).
  2. 세포 성장은 온도/pH/DO의 최적점, 영양분(포도당/글루타민)의 Monod 포화,
     암모니아 독성에 의해 결정된다.
  3. IgG는 생존 세포의 시간 적분(IVCD)에 비례하여 누적된다.
  4. 비생산성(qP)은 경미한 저온 배양(mild hypothermia)에서 상승한다 (실제 CHO 현상).
"""

import numpy as np
import pandas as pd
from scipy.special import erf

np.random.seed(42)

# 특성 범위 (공정 입력 변수)
FEATURE_RANGES = {
    "Temperature": (30.0, 37.5),       # 배양 온도 (C) - 저온 productivity shift 포함
    "pH": (6.7, 7.4),                  # pH
    "Dissolved_Oxygen": (20.0, 100.0), # 용존산소 (%)
    "Glucose": (0.0, 10.0),            # 포도당 (g/L)
    "Glutamine": (0.0, 6.0),           # 글루타민 (mM)
    "Ammonia": (0.0, 12.0),            # 암모니아 (mM)
    "Seeding_Density": (0.2, 1.0),     # 접종 밀도 (million cells/mL)
    "Culture_Day": (0, 14),            # 배양 일수 (day)
}

FEATURES = list(FEATURE_RANGES.keys())
TARGETS = ["VCD", "IgG_Titer"]


def _optimum_factor(x, opt, width):
    """최적점 opt에서 1.0, 멀어질수록 0으로 감소하는 가우시안 형태의 반응 계수."""
    return np.exp(-((x - opt) ** 2) / (2.0 * width ** 2))


def _monod(substrate, ks):
    """Monod 포화 계수: 기질이 충분하면 1, 부족하면 0에 가까워진다."""
    return substrate / (ks + substrate)


def _growth_factor(df):
    """공정 조건으로부터 세포 성장 잠재력 (0~1)을 계산한다.

    6개 반응 계수의 기하평균을 사용한다. 단순 곱을 쓰면 한 인자만 낮아도
    growth factor가 0으로 붕괴하지만, 실제 배양에서는 한 조건이 다소
    최적에서 벗어나도 세포가 완전히 죽지 않으므로 기하평균이 더 현실적이다.
    """
    temp_f = _optimum_factor(df["Temperature"], opt=36.5, width=3.0)
    ph_f = _optimum_factor(df["pH"], opt=7.10, width=0.30)
    do_f = _optimum_factor(df["Dissolved_Oxygen"], opt=55.0, width=32.0)
    glc_f = _monod(df["Glucose"], ks=0.8)
    gln_f = _monod(df["Glutamine"], ks=0.4)
    amm_f = 1.0 / (1.0 + (df["Ammonia"] / 9.0) ** 2)  # 암모니아 독성 억제
    factors = np.vstack([temp_f, ph_f, do_f, glc_f, gln_f, amm_f])
    eps = 1e-6
    return np.exp(np.mean(np.log(factors + eps), axis=0))


def _viable_cell_density(df, growth):
    """배양일에 따른 종형 VCD 곡선."""
    t = df["Culture_Day"]
    vmax = 18.0 * growth * (0.4 + 0.6 * (df["Seeding_Density"] / 1.0))
    t_peak = 6.0 + 3.0 * growth      # 조건이 좋을수록 피크가 늦고 높다
    width = 2.5 + 1.0 * growth
    vcd = vmax * np.exp(-((t - t_peak) ** 2) / (2.0 * width ** 2))
    return vcd, vmax, t_peak, width


def _integrated_vcd(df, vmax, t_peak, width):
    """0일부터 t일까지 생존 세포 적분값 IVCD (가우시안의 해석적 적분)."""
    t = df["Culture_Day"]
    norm = vmax * width * np.sqrt(2.0 * np.pi) * 0.5
    ivcd = norm * (
        erf((t - t_peak) / (width * np.sqrt(2.0)))
        - erf((0.0 - t_peak) / (width * np.sqrt(2.0)))
    )
    return np.clip(ivcd, 0.0, None)


def _igg_titer(df, growth, ivcd):
    """IVCD와 비생산성(qP)으로부터 누적 IgG titer(g/L)를 계산한다."""
    # 저온에서 qP 상승 (mild hypothermia productivity boost)
    hypothermia_boost = 1.0 + 0.18 * (37.0 - df["Temperature"])
    nutrient_f = _monod(df["Glucose"], 1.0) * _monod(df["Glutamine"], 0.5)
    amm_f = 1.0 / (1.0 + (df["Ammonia"] / 8.0) ** 2)
    qp = 25.0 * np.clip(hypothermia_boost, 0.6, 2.2) * (0.5 + 0.5 * nutrient_f) * amm_f
    # 단위 변환 상수 (pg/cell/day * Mcells/mL*day -> g/L 스케일 조정)
    igg = qp * ivcd * 1.3e-3
    return igg


def generate_cho_culture_data(n_samples=1200):
    """CHO cell 배양 공정 데이터 (VCD, IgG_Titer 타깃 포함)를 생성한다."""
    data = {}
    for feat, (low, high) in FEATURE_RANGES.items():
        if feat == "Culture_Day":
            data[feat] = np.random.randint(low, high + 1, n_samples)
        else:
            data[feat] = np.random.uniform(low, high, n_samples)
    df = pd.DataFrame(data)

    growth = _growth_factor(df)
    vcd, vmax, t_peak, width = _viable_cell_density(df, growth)
    ivcd = _integrated_vcd(df, vmax, t_peak, width)
    igg = _igg_titer(df, growth, ivcd)

    # 측정 노이즈 추가
    vcd = vcd + np.random.normal(0.0, 0.35, n_samples)
    igg = igg + np.random.normal(0.0, 0.15, n_samples)

    df["VCD"] = np.clip(vcd, 0.0, 20.0)
    df["IgG_Titer"] = np.clip(igg, 0.0, 8.0)
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("CHO Cell 배양 공정 데이터 생성 (VCD + IgG)")
    print("=" * 60)

    df = generate_cho_culture_data(n_samples=1200)
    df.to_csv("cho_culture_data.csv", index=False)

    print(f"\n[OK] 데이터 생성 완료: {len(df)} 샘플")
    print(f"[OK] 특성 {len(FEATURES)}개, 타깃 {len(TARGETS)}개")
    print("[OK] 파일 저장: cho_culture_data.csv\n")

    print("데이터 통계:")
    print(df.describe().round(3).to_string())

    print("\n타깃 상관관계 (VCD vs IgG_Titer):")
    print(f"  Pearson r = {df['VCD'].corr(df['IgG_Titer']):.3f}")
