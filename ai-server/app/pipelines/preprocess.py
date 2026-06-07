import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple


# ─────────────────────────────────────────────────────────
# S 레이블 초기 계산용 피처 목록
# 실제 가중치(w1~w4)는 GRU 모델 내부의 nn.Parameter로 학습됩니다.
# 여기서는 학습 초기 레이블 생성을 위해 균등 평균(1/4)을 사용합니다.
# ─────────────────────────────────────────────────────────
S_FEATURE_COLS = ["accuracy", "inverse_hit_rate", "attack_item_efficiency", "hp_retention_rate"]

# ─────────────────────────────────────────────────────────
# C 레이블 판정 임계값
# ─────────────────────────────────────────────────────────
C_TILT_DROP_THRESHOLD     = 0.15   # 틸트: 평균 대비 15% 이상 폭락 (기존 0.30)
C_CHAIN_HIT_HP_THRESHOLD  = 0.25   # 연쇄 피격: 5초 내 체력 25% 이상 감소 (기존 0.30)
C_PANIC_ACC_THRESHOLD     = 0.70   # 패닉 난사: 마지막 10초 명중률 평균 대비 70% 이하 (기존 0.50)
C_CHAIN_HIT_WINDOW        = 5      # 연쇄 피격 감지 윈도우 (초) - 슬라이딩 윈도우 5초와 정합성 일치
C_PANIC_WINDOW            = 10     # 패닉 난사 감지 윈도우 (초)


def clean_and_feature_engineering(raw_data: Dict[str, Any]) -> pd.DataFrame:
    """
    클라이언트에서 이미 계산된 5개 피처 값을 수신하여 결측치를 처리합니다.
    서버에서 피처를 재계산하지 않습니다. (책임 분리)

    클라이언트(유니티)에서 전송해야 할 필드 (time_series_frames 내부):
      - apm                   : 행동 수 / 클리어 시간 (초당 행동 횟수)
      - inverse_hit_rate      : 1 - (피격 횟수 / 적 발사 탄 수)
      - hp_retention_rate     : 현재 체력 / 최대 체력
      - accuracy              : 명중 탄 수 / 전체 발사 탄 수
      - attack_item_efficiency: 1 - (기본 피해 * 명중 수) / 실제 총 피해

    Parameters
    ----------
    raw_data : 웨이브 단위 원시 JSON 데이터 (백엔드에서 수신)

    Returns
    -------
    (feature_df, raw_df) : 피처 DataFrame과 원본 DataFrame 튜플
    """
    if isinstance(raw_data, list):
        frames = raw_data
    else:
        frames = raw_data.get("time_series_frames", [])
        
    if not frames:
        return pd.DataFrame(), pd.DataFrame()

    df = pd.DataFrame(frames)

    # ── 1. 결측치 전방 대체 (Forward Fill) ────────────────────────────────
    # 네트워크 렉 등으로 인한 NaN 값을 직전 초(t-1) 데이터로 채움
    df = df.ffill()

    # ── 2. 필수 피처 컬럼 존재 여부 검증 ─────────────────────────────────
    feature_cols = [
        "apm",
        "inverse_hit_rate",
        "hp_retention_rate",
        "accuracy",
        "attack_item_efficiency",
    ]
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"클라이언트 전송 데이터에 필수 피처가 누락되었습니다: {missing}\n"
            f"유니티에서 time_series_frames 내에 해당 필드를 포함하여 전송해야 합니다."
        )

    # ── 3. 범위 보정 및 스케일링 (수동 정규화) ─────────────────────────────────
    # 기존에 MinMaxScaler를 썼으나 추론 시 스케일러가 유실되는 문제를 막기 위해 수동으로 0~1로 맞춥니다.
    # APM은 최대 500으로 가정하고 나누어 0.0 ~ 1.0으로 정규화
    df[["apm"]] = df[["apm"]].clip(lower=0.0, upper=500.0) / 500.0
    
    # 나머지 4개 비율(Rate) 피처는 본래 0.0 ~ 1.0 범위이므로 클리핑만 수행
    rate_cols = ["inverse_hit_rate", "hp_retention_rate", "accuracy", "attack_item_efficiency"]
    df[rate_cols] = df[rate_cols].clip(lower=0.0, upper=1.0)

    return df[feature_cols], df  # 원본 df도 C 레이블 계산에 재활용


# ─────────────────────────────────────────────────────────────────────────────
# S (숙련도) 레이블 추론
# ─────────────────────────────────────────────────────────────────────────────

def compute_skill_label(feature_df: pd.DataFrame) -> float:
    """
    S 레이블 초기값을 계산합니다.

    실제 가중치(w1~w4)는 GRU 모델의 nn.Parameter(skill_weights)로 학습되며,
    이 함수는 학습 초기에 사용될 합리적인 초기 레이블(정답)을 제공하기 위해
    4개 피처의 균등 평균(각 0.25)으로 계산합니다.

    모델이 학습됨에 따라 자체적으로 최적 가중치를 찾아나갑니다.

    Returns
    -------
    S : float, 0.0 ~ 1.0
    """
    means = feature_df[S_FEATURE_COLS].mean()
    # 균등 평균: 각 피처에 동일한 비중 (1/4)
    s = float(means.mean())
    
    # ── [수정 1] 명중률 폭락 시 숙련도 패널티 적용 (탄 흩뿌리기 방어) ──
    # 웨이브 평균 명중률이 30% 미만인 경우, 명중률 비율만큼 S 점수 감쇄
    avg_accuracy = means["accuracy"]
    if avg_accuracy < 0.30:
        penalty = avg_accuracy / 0.30
        s = s * penalty
        
    # ── [수정 2] 시그모이드를 이용한 대비 늘리기 (Contrast Stretching) ──
    # 애매한 0.4~0.6 대역의 점수들을 양극단(0과 1)으로 확장하여 알파값 변별력 확보
    k = 8.0
    s = 1.0 / (1.0 + np.exp(-k * (s - 0.5)))
    
    return float(np.clip(s, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# C (이탈 위험도) 레이블 추론
# ─────────────────────────────────────────────────────────────────────────────

def _is_tilt(feature_df: pd.DataFrame, wave_index: int, session_history: list) -> bool:
    """
    [C 기준 1] 틸트(Tilt): 세션 평균 대비 마지막 웨이브 지표 15% 이상 폭락.
    session_history: 이전 웨이브들의 feature_df 리스트 (시간순)
    마지막 웨이브에서만 판정. 이전 데이터가 없으면 현재 웨이브 내 전후반 전조를 감지합니다.
    """
    # ── [개선] 학습 파이프라인에서 session_history가 주어지지 않는 경우, 현재 웨이브 내 자체 틸트 감지 ──
    if not session_history or wave_index == 0:
        n = len(feature_df)
        if n < 10:
            return False
        # 전반부와 후반부로 나누어 급격한 명중률/회피율 하락 여부 검출
        half = n // 2
        first_half = feature_df.iloc[:half]
        second_half = feature_df.iloc[half:]
        
        avg_acc_1 = first_half["accuracy"].mean()
        avg_acc_2 = second_half["accuracy"].mean()
        avg_inv_1 = first_half["inverse_hit_rate"].mean()
        avg_inv_2 = second_half["inverse_hit_rate"].mean()
        
        acc_drop = (avg_acc_1 - avg_acc_2) / max(avg_acc_1, 1e-6)
        inv_drop = (avg_inv_1 - avg_inv_2) / max(avg_inv_1, 1e-6)
        return acc_drop >= C_TILT_DROP_THRESHOLD and inv_drop >= C_TILT_DROP_THRESHOLD

    # 이전 웨이브 전체 평균 (명중률, 회피율)
    prev_accuracies = [df["accuracy"].mean() for df in session_history]
    prev_inv_hits   = [df["inverse_hit_rate"].mean() for df in session_history]
    session_avg_acc = np.mean(prev_accuracies)
    session_avg_inv = np.mean(prev_inv_hits)

    # 현재 웨이브 평균
    cur_acc = feature_df["accuracy"].mean()
    cur_inv = feature_df["inverse_hit_rate"].mean()

    # 두 지표 모두 15% 이상 폭락했을 경우 틸트 판정
    acc_drop = (session_avg_acc - cur_acc) / max(session_avg_acc, 1e-6)
    inv_drop = (session_avg_inv - cur_inv) / max(session_avg_inv, 1e-6)
    return acc_drop >= C_TILT_DROP_THRESHOLD and inv_drop >= C_TILT_DROP_THRESHOLD


def _is_instant_quit_on_death(raw_data: Dict[str, Any]) -> bool:
    """
    [C 기준 2] 칼종료: HP가 0이 되었거나 비정상 게임 종료(fail_safe).
    """
    if isinstance(raw_data, list):
        frames = raw_data
        fail_safe = False
    else:
        frames = raw_data.get("time_series_frames", [])
        fail_safe = raw_data.get("wave_meta", {}).get("fail_safe", False)
        
    if len(frames) < 2:
        return False

    # 마지막 프레임 HP 확인
    last_frame = frames[-1]
    max_hp  = last_frame.get("max_hp", 1)
    hp_lost = last_frame.get("hp_lost", 0)
    is_dead = hp_lost >= max_hp

    # [개선] fail_safe가 False(정상종료)인 로그일지라도, 사망에 도달한 세션이라면 이탈 후보(C=1)로 완화 판정합니다.
    return is_dead or fail_safe


def _is_chain_hit_collapse(raw_data: Dict[str, Any]) -> bool:
    """
    [C 기준 3] 연쇄 피격 붕괴: 5초 내 최대 체력 25% 이상 급감.
    """
    if isinstance(raw_data, list):
        frames = raw_data
        fail_safe = False
    else:
        frames = raw_data.get("time_series_frames", [])
        fail_safe = raw_data.get("wave_meta", {}).get("fail_safe", False)
        
    # [개선] fail_safe 여부와 상관없이 플레이어의 피격 피해량이 급감하여 붕괴 상태에 이른 전조를 포착합니다.
    if len(frames) < C_CHAIN_HIT_WINDOW:
        return False

    # 마지막 N초 윈도우에서 HP 감소율 계산
    tail_frames = frames[-C_CHAIN_HIT_WINDOW:]
    max_hp = tail_frames[0].get("max_hp", 1)
    hp_lost_sum = sum(f.get("hp_lost", 0) for f in tail_frames)

    hp_drop_rate = hp_lost_sum / max(max_hp, 1)
    return hp_drop_rate >= C_CHAIN_HIT_HP_THRESHOLD or fail_safe


def _is_panic_spray(feature_df: pd.DataFrame, raw_data: Dict[str, Any]) -> bool:
    """
    [C 기준 4] 패닉 난사: 마지막 10초 명중률이 웨이브 평균 대비 70% 이하로 급감 (APM 조건은 실시간 렉 및 조작 감쇄 고려해 제외).
    """
    if isinstance(raw_data, list):
        frames = raw_data
        fail_safe = False
    else:
        frames = raw_data.get("time_series_frames", [])
        fail_safe = raw_data.get("wave_meta", {}).get("fail_safe", False)
        
    n = len(feature_df)
    if n < C_PANIC_WINDOW or len(frames) == 0:
        return False

    # [수정] 유저가 낙담하여 조작을 주춤하거나(APM 저하) 손을 떼는 상황도 이탈 위험으로 판단하기 위해,
    # APM 상승 강제 조건을 제외하고 명중률의 극단적 폭락(70% 이하) 자체만으로 패닉 위기(C=1)로 탐지합니다.
    wave_avg_acc  = feature_df["accuracy"].mean()
    tail_avg_acc  = feature_df["accuracy"].iloc[-C_PANIC_WINDOW:].mean()

    # 명중률 급감 검출
    acc_collapse  = tail_avg_acc <= wave_avg_acc * C_PANIC_ACC_THRESHOLD
    return acc_collapse


def compute_churn_label(
    feature_df: pd.DataFrame,
    raw_data: Dict[str, Any],
    wave_index: int = 0,
    session_history: list = None,
) -> int:
    """
    C (이탈 위험도) 레이블 추론.
    아래 4가지 기준 중 하나라도 해당하면 C=1 (이탈 위험).

    기준:
      1. 틸트(Tilt): 세션 평균 대비 명중률/회피율 30% 이상 폭락
      2. 칼종료(Instant Quit on Death): HP 0 직후 즉시 세션 종료
      3. 연쇄 피격 붕괴(Chain-hit Collapse): 5초 내 체력 25% 이상 급감 후 종료
      4. 패닉 난사(Panic Spray): 마지막 10초 명중률 평균 대비 50% 이하 급감

    Returns
    -------
    C : int, 0 또는 1
    """
    if session_history is None:
        session_history = []

    if _is_tilt(feature_df, wave_index, session_history):
        return 1
    if _is_instant_quit_on_death(raw_data):
        return 1
    if _is_chain_hit_collapse(raw_data):
        return 1
    if _is_panic_spray(feature_df, raw_data):
        return 1
    return 0