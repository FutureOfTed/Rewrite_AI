import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple


# ─────────────────────────────────────────────────────────
# S 레이블 계산에 사용되는 가중치 (합계 = 1.0)
# ─────────────────────────────────────────────────────────
S_WEIGHTS = {
    "accuracy":              0.30,
    "inverse_hit_rate":      0.30,
    "attack_item_efficiency":0.20,
    "hp_retention_rate":     0.20,
}

# ─────────────────────────────────────────────────────────
# C 레이블 판정 임계값
# ─────────────────────────────────────────────────────────
C_TILT_DROP_THRESHOLD     = 0.30   # 틸트: 평균 대비 30% 이상 폭락
C_CHAIN_HIT_HP_THRESHOLD  = 0.50   # 연쇄 피격: 5~7초 내 체력 50% 이상 감소
C_PANIC_ACC_THRESHOLD     = 0.50   # 패닉 난사: 마지막 10초 명중률 평균 대비 50% 이하
C_CHAIN_HIT_WINDOW        = 7      # 연쇄 피격 감지 윈도우 (초)
C_PANIC_WINDOW            = 10     # 패닉 난사 감지 윈도우 (초)


def clean_and_feature_engineering(raw_data: Dict[str, Any]) -> pd.DataFrame:
    """
    원천 데이터를 받아 5개 핵심 피처를 계산하고 결측치를 처리합니다.

    Parameters
    ----------
    raw_data : 웨이브 단위 원시 JSON 데이터 (백엔드에서 수신)

    Returns
    -------
    feature_cols('apm', 'inverse_hit_rate', 'hp_retention_rate',
                 'accuracy', 'attack_item_efficiency') 만 담긴 DataFrame
    """
    frames = raw_data.get("time_series_frames", [])
    if not frames:
        return pd.DataFrame()

    df = pd.DataFrame(frames)

    # ── 1. 결측치 전방 대체 (Forward Fill) ────────────────────────────────
    # 네트워크 렉 등으로 인한 NaN 값을 직전 초(t-1) 데이터로 채움
    df = df.ffill()

    # ── 2. APM 재계산: 행동 횟수 / 한 웨이브 클리어 시간 ─────────────────
    # clear_time_sec: wave_meta에 담겨있는 웨이브 클리어 시간 (초)
    clear_time_sec = raw_data.get("wave_meta", {}).get("clear_time_sec", None)
    total_actions  = df["atk_clicks_total"].sum()   # 웨이브 전체 행동 횟수 합계
    if clear_time_sec and clear_time_sec > 0:
        # 웨이브 전체 APM을 초당 단위로 환산 후 각 프레임에 동일하게 할당
        wave_apm = total_actions / clear_time_sec
    else:
        # clear_time_sec 미제공 시 실제 프레임 개수로 대체
        wave_apm = total_actions / max(len(df), 1)
    df["apm"] = wave_apm  # 모든 프레임에 웨이브 기준 APM을 기입

    # ── 3. 5대 핵심 피처 계산 ─────────────────────────────────────────────

    # 1) APM: 위에서 이미 계산 완료

    # 2) Inverse Hit Rate: 1 - (피격 횟수 / 적이 발사한 총 탄 수)
    df["inverse_hit_rate"] = 1.0 - (
        df["hitbox_collisions"] / df["enemy_atk_spawned"].replace(0, 1)
    )

    # 3) HP Retention Rate: 1 - (잃은 체력 / 최대 체력)
    df["hp_retention_rate"] = 1.0 - (
        df["hp_lost"] / df["max_hp"].replace(0, 1)
    )

    # 4) Accuracy: 명중 탄 수 / 전체 발사 탄 수
    df["accuracy"] = df["atk_clicks_hit"] / df["atk_clicks_total"].replace(0, 1)

    # 5) Attack Item Efficiency: 1 - (기본 피해량 * 명중 탄 수) / 가한 총 피해량
    df["attack_item_efficiency"] = 1.0 - (
        (df["base_dmg_expected"] * df["atk_clicks_hit"])
        / df["actual_dmg_dealt"].replace(0, 1)
    )

    # ── 4. 범위 보정 (0.0 이하 클리핑) ───────────────────────────────────
    feature_cols = [
        "apm",
        "inverse_hit_rate",
        "hp_retention_rate",
        "accuracy",
        "attack_item_efficiency",
    ]
    df[feature_cols] = df[feature_cols].clip(lower=0.0)

    return df[feature_cols], df  # 원본 df도 레이블 계산에 재활용


# ─────────────────────────────────────────────────────────────────────────────
# S (숙련도) 레이블 추론
# ─────────────────────────────────────────────────────────────────────────────

def compute_skill_label(feature_df: pd.DataFrame) -> float:
    """
    S = w1*Accuracy + w2*Inv_Hit_Rate + w3*Item_Efficiency + w4*HP_Retention
    (웨이브 전체 프레임의 평균값으로 계산)

    Returns
    -------
    S : float, 0.0 ~ 1.0
    """
    means = feature_df.mean()
    s = (
        S_WEIGHTS["accuracy"]               * means.get("accuracy", 0.0)
        + S_WEIGHTS["inverse_hit_rate"]      * means.get("inverse_hit_rate", 0.0)
        + S_WEIGHTS["attack_item_efficiency"]* means.get("attack_item_efficiency", 0.0)
        + S_WEIGHTS["hp_retention_rate"]     * means.get("hp_retention_rate", 0.0)
    )
    # 부동소수점 오차 보정
    return float(np.clip(s, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# C (이탈 위험도) 레이블 추론
# ─────────────────────────────────────────────────────────────────────────────

def _is_tilt(feature_df: pd.DataFrame, wave_index: int, session_history: list) -> bool:
    """
    [C 기준 1] 틸트(Tilt): 세션 평균 대비 마지막 웨이브 지표 30% 이상 폭락.
    session_history: 이전 웨이브들의 feature_df 리스트 (시간순)
    마지막 웨이브에서만 판정. 이전 데이터가 없으면 False.
    """
    if not session_history or wave_index == 0:
        return False

    # 이전 웨이브 전체 평균 (명중률, 회피율)
    prev_accuracies = [df["accuracy"].mean() for df in session_history]
    prev_inv_hits   = [df["inverse_hit_rate"].mean() for df in session_history]
    session_avg_acc = np.mean(prev_accuracies)
    session_avg_inv = np.mean(prev_inv_hits)

    # 현재 웨이브 평균
    cur_acc = feature_df["accuracy"].mean()
    cur_inv = feature_df["inverse_hit_rate"].mean()

    # 두 지표 모두 30% 이상 폭락했을 경우 틸트 판정
    acc_drop = (session_avg_acc - cur_acc) / max(session_avg_acc, 1e-6)
    inv_drop = (session_avg_inv - cur_inv) / max(session_avg_inv, 1e-6)
    return acc_drop >= C_TILT_DROP_THRESHOLD and inv_drop >= C_TILT_DROP_THRESHOLD


def _is_instant_quit_on_death(raw_data: Dict[str, Any]) -> bool:
    """
    [C 기준 2] 칼종료: HP가 0이 된 후 1~2초 이내 게임 종료.
    raw_data의 time_series_frames에서 hp_lost 누적이 max_hp에 도달하는 시점을 감지.
    실제 구현 시 백엔드의 게임 종료 타임스탬프와 사망 타임스탬프 차이가 필요하나,
    현재 데이터 구조에서는 마지막 프레임의 HP가 0인지 여부로 근사.
    """
    frames = raw_data.get("time_series_frames", [])
    if len(frames) < 2:
        return False

    # 마지막 프레임 HP 확인
    last_frame = frames[-1]
    max_hp  = last_frame.get("max_hp", 1)
    hp_lost = last_frame.get("hp_lost", 0)
    # HP가 0이 된 것(사망)을 마지막 프레임 hp_lost == max_hp로 판정
    is_dead = hp_lost >= max_hp

    # 세션 종료 플래그 (백엔드에서 별도로 전달 시 활용)
    # 현재 구조에서는 fail_safe 필드가 비정상 종료를 의미하는 것으로 해석
    fail_safe = raw_data.get("wave_meta", {}).get("fail_safe", False)

    return is_dead and fail_safe


def _is_chain_hit_collapse(raw_data: Dict[str, Any]) -> bool:
    """
    [C 기준 3] 연쇄 피격 붕괴: 사망 직전 5~7초 내 최대 체력 50% 이상 급감 후 종료.
    """
    frames = raw_data.get("time_series_frames", [])
    fail_safe = raw_data.get("wave_meta", {}).get("fail_safe", False)
    if not fail_safe or len(frames) < C_CHAIN_HIT_WINDOW:
        return False

    # 마지막 N초 윈도우에서 HP 감소율 계산
    tail_frames = frames[-C_CHAIN_HIT_WINDOW:]
    max_hp = tail_frames[0].get("max_hp", 1)
    hp_lost_sum = sum(f.get("hp_lost", 0) for f in tail_frames)

    hp_drop_rate = hp_lost_sum / max(max_hp, 1)
    return hp_drop_rate >= C_CHAIN_HIT_HP_THRESHOLD


def _is_panic_spray(feature_df: pd.DataFrame, raw_data: Dict[str, Any]) -> bool:
    """
    [C 기준 4] 패닉 난사: 마지막 10초 명중률이 웨이브 평균 대비 50% 이하이면서
    APM은 오히려 올라간 상태.
    """
    fail_safe = raw_data.get("wave_meta", {}).get("fail_safe", False)
    if not fail_safe:
        return False

    n = len(feature_df)
    if n < C_PANIC_WINDOW:
        return False

    wave_avg_acc  = feature_df["accuracy"].mean()
    tail_avg_acc  = feature_df["accuracy"].iloc[-C_PANIC_WINDOW:].mean()
    tail_avg_apm  = feature_df["apm"].iloc[-C_PANIC_WINDOW:].mean()
    total_avg_apm = feature_df["apm"].mean()

    # 명중률 50% 이하 급감 + APM은 평균 이상 (패닉 난사 특징)
    acc_collapse  = tail_avg_acc <= wave_avg_acc * C_PANIC_ACC_THRESHOLD
    apm_elevated  = tail_avg_apm >= total_avg_apm
    return acc_collapse and apm_elevated


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
      3. 연쇄 피격 붕괴(Chain-hit Collapse): 5~7초 내 체력 50% 이상 급감 후 종료
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
