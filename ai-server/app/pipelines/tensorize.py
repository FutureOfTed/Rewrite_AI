import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Dict, Any, Tuple, List


def create_sliding_windows(
    df: pd.DataFrame,
    window_size: int = 30,
    stride: int = 1,
) -> np.ndarray:
    """
    피처 데이터프레임을 받아 30초 단위 슬라이딩 윈도우 텐서를 생성합니다.
    이상치 필터링(AFK/자리비움, 매크로 의심)이 포함됩니다.

    Parameters
    ----------
    df          : clean_and_feature_engineering() 의 반환값 (feature_df)
    window_size : 윈도우 크기 (기본값 30초)
    stride      : 슬라이딩 이동 간격 (기본값 1초)

    Returns
    -------
    ndarray : shape [Batch, Timesteps(30), Features(5)]
    """
    if len(df) < window_size:
        return np.empty((0, window_size, df.shape[1]))

    windows: List[np.ndarray] = []
    data_values = df.values

    for i in range(0, len(data_values) - window_size + 1, stride):
        window = data_values[i : i + window_size]

        # ── 이상치 처리 (Outlier Handling) ─────────────────────────────────
        # APM은 첫 번째 컬럼 ('apm') 기준
        apms = window[:, 0]

        # 1. 30초 내내 APM이 0 → AFK(자리비움) 또는 이동 거리 없는 구간 제외
        if np.all(apms == 0):
            continue

        # 2. APM 500 초과 → 매크로(불법 자동화) 의심 데이터 제외
        if np.any(apms > 500):
            continue

        windows.append(window)

    if not windows:
        return np.empty((0, window_size, df.shape[1]))

    # 최종 텐서 형태: [Batch, Timesteps(30), Features(5)]
    return np.array(windows, dtype=np.float32)


def normalize_tensor(tensor: np.ndarray) -> np.ndarray:
    """
    Min-Max Scaler를 적용하여 모든 피처 값을 0.0 ~ 1.0 사이로 정규화합니다.
    기울기 폭발 방지 및 피처 간 스케일 차이 극복.

    Notes
    -----
    - 정규화는 배치 전체 분포를 기준으로 수행됩니다.
    - 학습 데이터의 scaler를 저장하여 추론 시에도 동일하게 적용해야 합니다.

    Returns
    -------
    ndarray : shape [Batch, Timesteps(30), Features(5)], 값 범위 0.0 ~ 1.0
    """
    if tensor.size == 0:
        return tensor

    batch_size, timesteps, features = tensor.shape

    # 2D로 펼쳐서 스케일링
    flattened = tensor.reshape(-1, features)

    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    normalized_flattened = scaler.fit_transform(flattened)

    # 다시 3D 텐서 형태로 복원
    return normalized_flattened.reshape(batch_size, timesteps, features).astype(np.float32)


def build_dataset(
    sample_list: List[Dict],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    여러 웨이브 샘플을 받아 학습용 텐서(X)와 레이블 벡터(y_s, y_c)를 조립합니다.

    Parameters
    ----------
    sample_list : [
        {
            "tensor":  np.ndarray  # shape [Batch, 30, 5] (슬라이딩 윈도우 결과)
            "s_label": float       # 숙련도 (0.0 ~ 1.0)
            "c_label": int         # 이탈 위험도 (0 또는 1)
        },
        ...
    ]

    Returns
    -------
    X   : np.ndarray [TotalBatch, 30, 5]
    y_s : np.ndarray [TotalBatch]  숙련도 레이블
    y_c : np.ndarray [TotalBatch]  이탈 위험도 레이블
    """
    X_parts:   List[np.ndarray] = []
    y_s_parts: List[np.ndarray] = []
    y_c_parts: List[np.ndarray] = []

    for sample in sample_list:
        tensor  = sample["tensor"]   # [Batch, 30, 5]
        s_label = sample["s_label"]  # float
        c_label = sample["c_label"]  # int

        n = tensor.shape[0]
        if n == 0:
            continue

        X_parts.append(tensor)
        y_s_parts.append(np.full(n, s_label, dtype=np.float32))
        y_c_parts.append(np.full(n, c_label, dtype=np.int32))

    if not X_parts:
        return np.empty((0,)), np.empty((0,)), np.empty((0,))

    X   = np.concatenate(X_parts,   axis=0)
    y_s = np.concatenate(y_s_parts, axis=0)
    y_c = np.concatenate(y_c_parts, axis=0)

    return X, y_s, y_c


def split_dataset(
    X: np.ndarray,
    y_s: np.ndarray,
    y_c: np.ndarray,
    train_ratio: float = 0.8,
    val_ratio:   float = 0.1,
    test_ratio:  float = 0.1,
    random_seed: int   = 42,
) -> Tuple:
    """
    전체 데이터를 Train(80%) / Validation(10%) / Test(10%)로 무작위 분할합니다.

    Returns
    -------
    (X_train, X_val, X_test,
     y_s_train, y_s_val, y_s_test,
     y_c_train, y_c_val, y_c_test)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "분할 비율의 합이 1.0이 되어야 합니다."

    # 1차 분할: Train vs (Val + Test)
    val_test_ratio = val_ratio + test_ratio
    X_train, X_temp, y_s_train, y_s_temp, y_c_train, y_c_temp = train_test_split(
        X, y_s, y_c,
        test_size=val_test_ratio,
        random_state=random_seed,
        shuffle=True,
    )

    # 2차 분할: Validation vs Test (val_test 중 val_ratio 비율만큼)
    val_ratio_of_temp = val_ratio / val_test_ratio
    X_val, X_test, y_s_val, y_s_test, y_c_val, y_c_test = train_test_split(
        X_temp, y_s_temp, y_c_temp,
        test_size=(1 - val_ratio_of_temp),
        random_state=random_seed,
        shuffle=True,
    )

    return (
        X_train, X_val, X_test,
        y_s_train, y_s_val, y_s_test,
        y_c_train, y_c_val, y_c_test,
    )
