import FinanceDataReader as fdr
import pandas as pd
import numpy as np

# ============================================================
# 1. 삼성전자(005930) 일봉 수집 → 주봉 변환
#    앞 14주 NaN 방지를 위해 start_date 보다 ~4개월 앞서 수집
# ============================================================
symbol = '005930'
start_date = '2024-05-27'
end_date   = '2026-05-27'

# 지표 길이(14주) + 평활(3주) 여유분 → 17주 ≈ 120일 먼저 수집
WARMUP_WEEKS = 17
warmup_start = (pd.Timestamp(start_date) - pd.DateOffset(weeks=WARMUP_WEEKS)).strftime('%Y-%m-%d')

df_raw = fdr.DataReader(symbol, warmup_start, end_date)
print("원본 컬럼:", df_raw.columns.tolist())
print(f"일봉 데이터 수 (워밍업 포함): {len(df_raw)}일")

df_weekly_full = df_raw.resample('W-MON').agg({
    'Open':   'first',
    'High':   'max',
    'Low':    'min',
    'Close':  'last',
    'Volume': 'sum'
}).dropna()

print(f"주봉 데이터 수 (워밍업 포함): {len(df_weekly_full)}주\n")

# ============================================================
# 2-A. MFI (Money Flow Index, 14주)
# ============================================================
length_mfi = 14

tp     = (df_weekly_full['High'] + df_weekly_full['Low'] + df_weekly_full['Close']) / 3
raw_mf = tp * df_weekly_full['Volume']   # 버그 수정: t * f_weekly → tp * df_weekly_full

pos_mf = pd.Series(np.where(tp > tp.shift(1), raw_mf, 0), index=df_weekly_full.index)
neg_mf = pd.Series(np.where(tp < tp.shift(1), raw_mf, 0), index=df_weekly_full.index)

pos_sum = pos_mf.rolling(window=length_mfi, min_periods=length_mfi).sum()
neg_sum = neg_mf.rolling(window=length_mfi, min_periods=length_mfi).sum()

mfr = pos_sum / neg_sum.replace(0, np.nan)
df_weekly_full['MFI_14'] = 100 - (100 / (1 + mfr))

# ============================================================
# 2-B. ATR (Average True Range, 14주)
# ============================================================
length_atr = 20

h_l  = df_weekly_full['High'] - df_weekly_full['Low']
h_pc = (df_weekly_full['High'] - df_weekly_full['Close'].shift(1)).abs()
l_pc = (df_weekly_full['Low']  - df_weekly_full['Close'].shift(1)).abs()

tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)

df_weekly_full['ATR_14'] = tr.ewm(alpha=1/length_atr, min_periods=length_atr,
                                  adjust=False).mean()

# ============================================================
# 2-C. Stochastic Oscillator (14, 3, 3)
#       버그 수정: 10 * ... → 100 * ...
#       버그 수정: 괄호 누락 → (100 * (close - lowest) / ...)
# ============================================================
length_k = 14
smooth_k = 3
smooth_d = 3

lowest  = df_weekly_full['Low'].rolling(window=length_k, min_periods=length_k).min()
highest = df_weekly_full['High'].rolling(window=length_k, min_periods=length_k).max()

fast_k = (100 * (df_weekly_full['Close'] - lowest)) / (highest - lowest)  # 버그 수정

df_weekly_full['STOCHk_14_3_3'] = fast_k.rolling(window=smooth_k, min_periods=smooth_k).mean()
df_weekly_full['STOCHd_14_3_3'] = df_weekly_full['STOCHk_14_3_3'].rolling(
    window=smooth_d, min_periods=smooth_d).mean()

# ============================================================
# 3. 워밍업 구간 제거 → 원하는 시작일부터만 사용
# ============================================================
df_weekly = df_weekly_full[df_weekly_full.index >= start_date].copy()
print(f"워밍업 제거 후 주봉 수: {len(df_weekly)}주")

display_cols = ['Open', 'High', 'Low', 'Close', 'Volume',
                'MFI_14', 'ATR_14', 'STOCHk_14_3_3', 'STOCHd_14_3_3']

df_valid = df_weekly[display_cols].dropna()
print(f"지표 산출 가능 주 수: {len(df_valid)}주")
print(f"첫 번째 행 날짜: {df_weekly.index[0].strftime('%Y-%m-%d')}")
print(f"지표 산출 시작일: {df_valid.index[0].strftime('%Y-%m-%d')}\n")

print("===== 전체 데이터 앞 5주 (NaN 없어야 함) =====\n")
print(df_weekly[display_cols].head(5).to_string())

print("\n===== 최근 10주 =====\n")
print(df_valid.tail(14).to_string())
