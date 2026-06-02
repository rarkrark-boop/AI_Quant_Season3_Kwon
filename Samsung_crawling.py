
"""
삼성전자(005930) · 코스피(^KS11) · 비트코인(BTC-USD) 주봉 데이터 수집 + MFI / ATR / Stochastic 계산
- OHLCV : yfinance — 티커별 자동 다운로드

- 지표   : 트레이딩뷰 기준 수식으로 재계산
- 기간   : 최근 6년 주봉 (2020-05-25 ~ 2026-05-25)

설치 (최초 1회):
    pip install yfinance pandas numpy matplotlib seaborn
"""
import sys
import subprocess
import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.dates as mdates

# ============================================================
# 0. 설정
# ============================================================

# ── 종목 정의: {티커: 표시명} ──────────────────────────────
SYMBOLS = {
    '005930.KS': '삼성전자',
    '^KS11':     '코스피',
    'BTC-USD':   '비트코인',
}

# ── 공통 기간 (6년) ────────────────────────────────────────
END_DATE   = pd.Timestamp('2026-05-25')
START_DATE = END_DATE - pd.DateOffset(years=6)   # 2020-05-25

# 지표 워밍업용 선행 데이터 (14주 기간 + 평활 여유 10주)  << Ratify 논문기준 10주 + 여유분 = 총 20주
WARMUP_WEEKS = 20
WARMUP_START = START_DATE - pd.DateOffset(weeks=WARMUP_WEEKS)

print(f"분석 기간 : {START_DATE.date()} ~ {END_DATE.date()} (6년)")
print(f"워밍업 시작: {WARMUP_START.date()}\n")


# ============================================================
# 1. yfinance 주봉 다운로드
# ============================================================

def fetch_yahoo_weekly(symbol: str,
                       label: str,
                       warmup_start: pd.Timestamp,
                       end_date: pd.Timestamp) -> pd.DataFrame:
    """yfinance로 주봉 OHLCV 다운로드 후 정제"""

    print(f"[{label}] yfinance 주봉 다운로드 시작: {symbol}")
    print(f"  요청 기간: {warmup_start.date()} ~ {end_date.date()}")

    # yfinance end는 exclusive → 하루 더 추가
    raw = yf.download(
        tickers     = symbol,
        start       = warmup_start.strftime('%Y-%m-%d'),
        end         = (end_date + pd.Timedelta(days=7)).strftime('%Y-%m-%d'),
        interval    = '1wk',
        auto_adjust = True,   # 수정 주가 자동 적용
        progress    = False,
    )

    if raw.empty:
        raise ValueError(f"[{label}] 데이터를 받아오지 못했습니다. 네트워크 연결 또는 티커를 확인하세요.")

    # yfinance 멀티인덱스 컬럼 처리 (최신 버전 대응)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # 필요한 컬럼만 추출 + 이름 정규화
    df = raw[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

    # 인덱스를 tz-naive Timestamp로 통일
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = 'Date'

    # END_DATE 이후 행 제거
    df = df[df.index <= end_date]

    # 결측 행 제거
    # ※ 코스피 지수(^KS11)는 Volume=0이 정상이므로 Volume 필터는 적용하지 않음
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])

    df['Weekly Return(%)'] = df['Close'].pct_change()*100 # 이번 주 수익률(주간 수익률[%])

    print(f"  다운로드 완료: {len(df)}주 (워밍업 포함)")
    print(f"  실제 기간: {df.index[0].date()} ~ {df.index[-1].date()}\n")
    return df
df_samsung = fetch_yahoo_weekly(
    symbol       = '005930.KS',
    label        = '삼성전자',
    warmup_start = WARMUP_START,
    end_date     = END_DATE
)

# 데이터가 잘 들어왔는지 확인용 출력
print(len(df_samsung[['Close', 'Weekly Return(%)']])) # 334

#%%
# =============================================================================
# [논문 수식 적용] 상장주식수 마디 분할 및 방향성 ATR 계산')
# 상장주식수 반영 및 거래회전율 계산
# =============================================================================
# 삼성전자 상장주식수 정의 [단위: 주]
shares_2021 = 5969782550
shares_2025 = 5919637922
shares_2026 = 5846278608

# 시계열 날짜 마디별로 슬라이싱 (복사본 생성)
df_period_1 = df_samsung.loc[:'2025-03-25'].copy()
df_period_2 = df_samsung.loc['2025-03-26':'2026-04-13'].copy()
df_period_3 = df_samsung.loc['2026-04-14':].copy()

# 각 마디별 거래량에 맞는 상장주식수 나누어 '거래회전율' 계산
df_period_1['거래회전율'] = (df_period_1['Volume'] / shares_2021)*100
df_period_2['거래회전율'] = (df_period_2['Volume'] / shares_2025)*100
df_period_3['거래회전율'] = (df_period_3['Volume'] / shares_2026)*100

# 분할 계산된 마디들을 다시 하나로 concat
df_final = pd.concat([df_period_1, df_period_2, df_period_3])

# 메인 데이터프레임에 거래회전율 추가
df_samsung['거래회전율'] = df_final['거래회전율']

# [사용 수식] 주간 거래회전율 * 주간 수익률의 부호 * 1000 스케일 반영 (논문 수식 3)
# np.sign(0) = 0 이므로, 수익률이 정확히 0인 주는 부호 정의 불가 → np.nan 처리
_sign_samsung = df_samsung['Weekly Return(%)'].apply(
    lambda r: np.nan if (pd.isna(r) or r == 0.0) else float(np.sign(r))
)
df_samsung['ATR(10^-3)'] = df_samsung['거래회전율'] * _sign_samsung * 1000

#결과 확인 출력
pd.set_option('display.float_format', '{:.4f}'.format)
print(df_samsung[['Close', 'Volume', 'Weekly Return(%)', '거래회전율', 'ATR(10^-3)']].dropna().head(5))
print("...")
print(df_samsung[['Close', 'Volume', 'Weekly Return(%)', '거래회전율', 'ATR(10^-3)']].tail(5))


#%%
# ============================================================
# 1-B. 상장주식 수 조회
#   - 주식/ETF : sharesOutstanding
#   - 비트코인  : circulatingSupply  (발행량 개념으로 대체)
#   - 코스피    : impliedSharesOutstanding → 없으면 NaN
# ============================================================

# =============================================================================
# def fetch_shares_outstanding(symbol: str, label: str) -> float:
#     """yfinance Ticker.info 에서 상장주식 수(또는 유통량)를 반환.
#     값이 없으면 float('nan')을 반환하므로 호출부에서 별도 예외 처리 불필요."""
#     try:
#         info = yf.Ticker(symbol).info # ← yfinance에서 종목 메타정보 딕셔너리 전체를 받아옴
#         # 우선순위대로 키를 탐색
#         for key in ('sharesOutstanding', 'impliedSharesOutstanding', 'circulatingSupply'):
#             val = info.get(key)
#             if val and val > 0:
#                 print(f"  [{label}] 상장주식 수 ({key}): {val:,.0f}")
#                 return float(val) # 처음으로 유효한 값이 나오면 즉시 반환
#         print(f"  [{label}] 상장주식 수 정보 없음 → NaN 처리")
#     except Exception as e:
#         print(f"  [{label}] 상장주식 수 조회 실패: {e}")
#     return float('nan') # 셋 다 없으면 NaN 반환
# =============================================================================
def fetch_shares_outstanding(symbol: str, label: str) -> float:
    try:
        info = yf.Ticker(symbol).info
        for key in ('sharesOutstanding', 'impliedSharesOutstanding', 'circulatingSupply'):
            val = info.get(key)
            if val and val > 0:
                print(f"  [{label}] 실시간 상장주식 수 조회 ({key}): {val:,.0f}")
                return float(val)
        print(f"  [{label}] 상장주식 수 정보 없음 → NaN 처리")
    except Exception as e:
        print(f"  [{label}] 상장주식 수 조회 실패: {e}")
    return float('nan')


# ============================================================
# 1-C. 발행주식수 시계열 취득 및 주봉 매핑
#   fetch_shares_history : get_shares_full()로 변경 이력 취득
#                          (삼성전자 자사주 소각 등 실제 변동 반영)
#   map_shares_to_weekly : 변경일 기준 forward-fill → 주봉 날짜에 정렬
# ============================================================

def fetch_shares_history(symbol: str, label: str,
                         warmup_start: pd.Timestamp,
                         end_date: pd.Timestamp):
    """
    yfinance get_shares_full()로 발행주식수 변경 이력 시계열 취득.
    삼성전자처럼 자사주 소각 등으로 주식수가 변경되는 종목에 유효.
    취득 실패 또는 데이터 없으면 None 반환.
    """
    try:
        raw = yf.Ticker(symbol).get_shares_full(
            start = warmup_start.strftime('%Y-%m-%d'),
            end   = (end_date + pd.Timedelta(days=7)).strftime('%Y-%m-%d'),
        )
        if raw is not None and not raw.empty:
            s = raw.copy().astype(float)
            s.index = pd.to_datetime(s.index).tz_localize(None)
            s = s[s > 0]                              # 0 이하 레코드 제거
            # get_shares_full()이 같은 날 여러 레코드를 반환하는 경우가 있음
            # → 중복 날짜는 마지막 값(최신 수치)만 유지
            s = s[~s.index.duplicated(keep='last')]
            s = s.sort_index()                        # 날짜 오름차순 정렬 보장
            print(f"  [{label}] 발행주식수 시계열 {len(s)}개 취득 "
                  f"({s.index[0].date()} ~ {s.index[-1].date()})")
            return s
    except Exception as e:
        print(f"  [{label}] 발행주식수 시계열 조회 실패: {e}")
    return None


def map_shares_to_weekly(shares_hist,
                         df_index: pd.DatetimeIndex,
                         scalar_shares: float) -> pd.Series:
    """
    발행주식수 시계열(shares_hist)을 주봉 인덱스(df_index)에 맞게 매핑.

    매핑 규칙:
      - 변경일 사이 구간은 직전 변경일의 주식수를 forward-fill로 유지
        → 삼성전자 자사주 소각 시점마다 해당 주봉 날짜에 자동 반영
      - 가장 이른 주봉보다 앞선 이력이 없으면 backward-fill로 가장
        가까운 이후 값으로 보완
      - 시계열 자체가 없으면 scalar_shares를 전 기간 동일 적용

    0 / NaN 방어:
      - 매핑 후에도 0 이하이거나 NaN이면 np.nan으로 처리
        → calc_atr 내부에서 회전율 = Volume / NaN 이 되어 NaN 전파됨
    """
    if shares_hist is not None and not shares_hist.empty:
        # reindex(..., method='ffill')은 중복 인덱스가 있으면 ValueError 발생
        # → fetch_shares_history에서 이미 제거했지만 이중 방어로 재처리
        s = shares_hist[~shares_hist.index.duplicated(keep='last')].sort_index()
        aligned = s.reindex(df_index, method='ffill')
        aligned = aligned.bfill()          # 앞 구간 보완
    else:
        aligned = pd.Series(float(scalar_shares), index=df_index, dtype=float)

    # 0 이하 또는 NaN → np.nan
    aligned = aligned.where((aligned > 0) & aligned.notna(), np.nan)
    return aligned


# ============================================================
# 2. 지표 계산 논문 기준 및 제로분모 완벽 첨삭 (코스피o, 삼성 다른방식필요, 비트코인 다른방식필요 )
# ============================================================

# ── 2-A. MFI (Money Flow Index, 논문 기준 10주로 변경)─────────────────────
def calc_mfi(df: pd.DataFrame, length: int = 10) -> pd.Series:
    tp     = (df['High'] + df['Low'] + df['Close']) / 3
    raw_mf = tp * df['Volume']

    pos_mf = raw_mf.where(tp > tp.shift(1), 0.0)
    neg_mf = raw_mf.where(tp < tp.shift(1), 0.0)

    pos_sum = pos_mf.rolling(window=length, min_periods=length).sum()
    neg_sum = neg_mf.rolling(window=length, min_periods=length).sum()

    mfr = pos_sum / neg_sum.replace(0, np.nan)
    return (100 - (100 / (1 + mfr))).rename(f'MFI_{length}')


# 백업용 표준 ATR (지수/데이터 누수 방지용) 
def calc_standard_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    h_l  = df['High'] - df['Low']
    h_pc = (df['High'] - df['Close'].shift(1)).abs()
    l_pc = (df['Low']  - df['Close'].shift(1)).abs()

    tr  = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    # 와일더 평활화(RMA) 반영 <--트레이딩뷰에서 정식사용
    atr = tr.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    return atr.rename(f'Standard_ATR_{length}')

# ── 2-B. ATR (Adjusted Turnover Rate, 논문 수식 3) ──────────
#   ATR_{i,t} = (V_{i,t} / OS_{i,t}) × (R_{i,t} / |R_{i,t}|) × 1000
#
#   V      : 거래량 (Volume)
#   OS     : 발행주식수 (map_shares_to_weekly로 주봉 날짜에 매핑된 시리즈)
#   R      : 종가 기준 전주 대비 수익률 ('Weekly Return(%)' 컬럼 사용)
#   × 1000 : 논문 표의 ATR(10⁻³) 단위 기준
#
#   0 나눗셈 / 부호 불가 방어 (논문 기준 미정의 구간 → np.nan):
#     - R = 0   → 부호 정의 불가            → sign = np.nan → ATR = np.nan
#     - R = NaN → 워밍업 첫 행 등           → sign = np.nan → ATR = np.nan
#     - OS = 0  → 회전율 정의 불가          → turnover = np.nan → ATR = np.nan
#     - OS = NaN → map_shares_to_weekly 처리 후 → ATR = np.nan
#
#   length 파라미터: 하위 호환성 유지용 (본 수식에서는 미사용)
def calc_atr(df: pd.DataFrame,
             shares: pd.Series,
             length: int = 10) -> pd.Series:

    # 수익률 (Weekly Return(%) 컬럼, 단위 % → 부호만 필요하므로 단위 무관)
    ret = df['Weekly Return(%)']

    # 수익률 부호: R=0 또는 NaN → np.nan
    sign_ret = ret.apply(
        lambda r: np.nan if (pd.isna(r) or r == 0.0) else float(np.sign(r))
    )

    # 발행주식수 0·NaN → np.nan (이중 방어; map_shares_to_weekly에서 이미 처리)
    os_safe = shares.where((shares > 0) & shares.notna(), np.nan)

    # 회전율 = 거래량 / 발행주식수
    turnover = df['Volume'] / os_safe

    # ATR = 회전율 × 수익률 부호 × 1000  (논문 ATR×10⁻³ 단위)
    atr = turnover * sign_ret * 1000

    return atr.rename(f'ATR_{length}')


# ── 2-C. Stochastic Oscillator (논문 기준 10, 3, 3 변경) ──────────────────
def calc_stoch(df: pd.DataFrame,        # 10주 간의 범위, %K를 만들기 위한 평활화 , %D를 만들기 위한 %K의 평활화
               length_k: int = 10,
               smooth_k: int = 3,
               smooth_d: int = 3) -> pd.DataFrame:
    lowest  = df['Low'].rolling(window=length_k, min_periods=length_k).min()
    highest = df['High'].rolling(window=length_k, min_periods=length_k).max()

    denom  = (highest - lowest).replace(0, np.nan)
    fast_k = 100 * (df['Close'] - lowest) / denom

    k = fast_k.rolling(window=smooth_k, min_periods=smooth_k).mean()
    d = k.rolling(window=smooth_d, min_periods=smooth_d).mean()

    tag = f'STOCH_{length_k}_{smooth_k}_{smooth_d}'
    return pd.DataFrame({f'{tag}_K': k, f'{tag}_D': d}, index=df.index)


# ============================================================
# 3. 종목별 데이터 수집 + 논문 수록 수식 지표 계산 통합 함수
# ============================================================
# =============================================================================
# 
# def build_weekly_df(symbol: str,
#                     label: str,
#                     start_date: pd.Timestamp,
#                     end_date: pd.Timestamp,
#                     warmup_start: pd.Timestamp) -> pd.DataFrame:
#     """다운로드 → 지표 계산 → 워밍업 제거 → 최종 DataFrame 반환"""
# 
#     # 1) 다운로드
#     df_full = fetch_yahoo_weekly(symbol, label, warmup_start, end_date)
# 
#     # 2) 지표 계산 (워밍업 포함 전체 구간)
#     df_full['MFI_14']        = calc_mfi(df_full, length=14)
#     df_full['ATR_14']        = calc_atr(df_full, length=14)
#     stoch_df                 = calc_stoch(df_full, 14, 3, 3)
#     df_full['STOCHk_14_3_3'] = stoch_df['STOCH_14_3_3_K']
#     df_full['STOCHd_14_3_3'] = stoch_df['STOCH_14_3_3_D']
# 
#     # 3) 워밍업 제거 → 목표 기간만 추출
#     df_weekly = df_full[
#         (df_full.index >= start_date) & (df_full.index <= end_date)
#     ].copy()
# 
#     # 4) 다음 주 변동률 (예측 분석용 타겟 변수)
#     df_weekly['Next_Return'] = df_weekly['Close'].pct_change().shift(-1) * 100
# 
#     # 5) 주가 성장률 (당주 종가 기준 전주 대비 등락률, %)
#     df_weekly['Close_Growth'] = df_weekly['Close'].pct_change() * 100
# 
#     # 6) 상장주식 수 및 성장률
#     #    - yfinance info 에서 현재 시점의 단일 수치를 조회해 전 행에 동일하게 기입
#     #    - 전 기간 동일값이므로 pct_change()는 첫 행만 NaN, 나머지는 0.00 으로 표시됨
#     shares = fetch_shares_outstanding(symbol, label) # 스칼라 1개
#     df_weekly['Shares_Outstanding'] = shares # 전 행에 동일한 값으로 채워짐
#     df_weekly['Shares_Growth']      = df_weekly['Shares_Outstanding'].pct_change() * 100
# 
#     print(f"[{label}] 워밍업 제거 후: {len(df_weekly)}주 "
#           f"({df_weekly.index[0].date()} ~ {df_weekly.index[-1].date()})")
# 
#     return df_weekly
#     #"이번 주가 지난주보다 몇 % 변했는가"를 나타냄.
# =============================================================================
def build_weekly_df(symbol: str,
                    label: str,
                    start_date: pd.Timestamp,
                    end_date: pd.Timestamp,
                    warmup_start: pd.Timestamp) -> pd.DataFrame:
    """다운로드 >> 논문/표준 가공 분기 >> 워밍업 제거 >> 최종 가독성 데이터셋 반환"""

    # 다운로드 (기존 구현된 fetch_yahoo_weekly 함수 호출)
    df_full = fetch_yahoo_weekly(symbol, label, warmup_start, end_date)

    # 주간 보조지표 계산 (논문 표준 10주 윈도우 반영)
    df_full['MFI_10']        = calc_mfi(df_full, length=10)
    stoch_df                 = calc_stoch(df_full, 10, 3, 3)
    df_full['STOCHk_10_3_3'] = stoch_df['STOCH_10_3_3_K']
    df_full['STOCHd_10_3_3'] = stoch_df['STOCH_10_3_3_D']

    #  공통 주간 수익률 및 주가 성장률 계산 (결측치 방지를 위해 워밍업 전 전체 구간 계산)
    df_full['Weekly Return(%)'] = df_full['Close'].pct_change() * 100
    df_full['Close_Growth']  = df_full['Close'].pct_change() * 100

    # 상장주식 수 취득 및 ATR_10 계산 (논문 수식 3 기반)
    # ─────────────────────────────────────────────────────────
    # 공통 흐름:
    #   1. fetch_shares_outstanding : 현재 시점 스칼라 (fallback)
    #   2. fetch_shares_history     : get_shares_full() 시계열 (우선)
    #   3. map_shares_to_weekly     : 주봉 날짜에 forward-fill 매핑
    #   4. calc_atr                 : 논문 수식(3) ATR 산출
    #
    # 코스피 지수(^KS11)는 총 상장주식수 데이터 미제공 → 표준 ATR 백업
    # ─────────────────────────────────────────────────────────
    scalar_shares = fetch_shares_outstanding(symbol, label)

    if symbol.startswith('^') or label == '코스피':
        print(f"  [{label}] 지수 데이터: 표준 와일더 ATR 적용 및 회전율 NaN 처리")
        df_full['거래회전율(%)']    = np.nan
        df_full['ATR_10']          = calc_standard_atr(df_full, length=14)
        df_full['Shares_Outstanding'] = np.nan

    else:
        # 시계열 발행주식수 취득 → 없으면 스칼라 fallback으로 map_shares_to_weekly 처리
        shares_hist   = fetch_shares_history(symbol, label, warmup_start, end_date)
        shares_mapped = map_shares_to_weekly(shares_hist, df_full.index, scalar_shares)

        df_full['Shares_Outstanding'] = shares_mapped

        # 거래회전율(%) = 거래량 / 발행주식수 × 100
        # shares_mapped에 NaN이 있는 행은 자동으로 NaN 전파
        df_full['거래회전율(%)'] = (df_full['Volume'] / shares_mapped) * 100

        # ATR = 회전율 × sign(수익률) × 1000  [논문 수식 3]
        # 수익률=0 또는 shares=NaN 인 구간은 calc_atr 내부에서 np.nan 처리
        df_full['ATR_10'] = calc_atr(df_full, shares_mapped, length=10)

    # 주식수 변동률 계산
    df_full['Shares_Growth'] = df_full['Shares_Outstanding'].pct_change() * 100

    # 워밍업 제거, 목표 기간(6년)만 정밀 추출
    df_weekly = df_full[
        (df_full.index >= start_date) & (df_full.index <= end_date)
    ].copy()

    # [타겟 변수] 다음 주 변동률 (예측 분석용 정답지 t+1)
    df_weekly['Next_Return'] = df_weekly['Close'].pct_change().shift(-1) * 100

    print(f"[{label}] 가공 및 워밍업 제거 완료: {len(df_weekly)}주 "
          f"({df_weekly.index[0].date()} ~ {df_weekly.index[-1].date()})")

    return df_weekly

# ============================================================
# 4. 메인 실행
# ============================================================
if __name__ == '__main__':

    # ── 4-1. 전 종목 수집 ─────────────────────────────────
    datasets: dict[str, pd.DataFrame] = {}

    for ticker, name in SYMBOLS.items():
        datasets[name] = build_weekly_df(
            symbol      = ticker,
            label       = name,
            start_date  = START_DATE,
            end_date    = END_DATE,
            warmup_start= WARMUP_START,
        )

    # ── 4-2. 공통 컬럼 정의 ───────────────────────────────
    display_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'MFI_10', 'ATR_10', 'STOCHk_10_3_3', 'STOCHd_10_3_3', 'Next_Return',
        'Close_Growth', 'Shares_Outstanding', 'Shares_Growth',
    ]

    # ── 4-3. 종목별 간단 미리보기 출력 ────────────────────
    pd.set_option('display.float_format', '{:.2f}'.format)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 200)

    # 지표 계산에 필요한 핵심 컬럼만 기준으로 dropna
    # (Shares_Outstanding 등 종목에 따라 NaN이 정상인 컬럼은 제외)
    DROPNA_SUBSET = ['MFI_10', 'ATR_10', 'STOCHk_10_3_3', 'STOCHd_10_3_3']

    for name, df in datasets.items():
        df_valid = df[display_cols].dropna(subset=DROPNA_SUBSET)

        print(f"\n{'='*70}")
        if df_valid.empty:
            print(f" [{name}]  지표 산출 가능한 주봉 없음 (데이터 확인 필요)")
            print(f"{'='*70}")
            continue

        print(f"  [{name}]  지표 산출 가능: {len(df_valid)}주 | "
              f"시작: {df_valid.index[0].date()}")
        print(f"{'='*70}")
        print("▶ 앞 3주")
        print(df[display_cols].head(3).to_string())
        print("\n▶ 최근 5주")
        print(df_valid.tail(5).to_string())

    # ============================================================
    # 5. 시각화 (종목별 Heatmap + Pair Plot)
    # ============================================================
    import matplotlib.pyplot as plt
    import seaborn as sns
    import platform

    # ── 5-1. 한글 폰트 설정 ───────────────────────────────
    if platform.system() == 'Windows':
        plt.rc('font', family='Malgun Gothic')
    elif platform.system() == 'Darwin':
        plt.rc('font', family='AppleGothic')
    else:
        plt.rc('font', family='NanumGothic')

    plt.rcParams['axes.unicode_minus'] = False
    sns.set_theme(style='whitegrid', font=plt.rcParams['font.family'])

    # 분석에 사용할 컬럼 (지표 + 타겟)
    analysis_cols = ['Close', 'Next_Return', 'MFI_10', 'ATR_10', 
                     'STOCHk_10_3_3', 'STOCHd_10_3_3']

    col_rename = {
        'Close':          '종가',
        'Next_Return':    '다음 주 변동률(%)',
        'MFI_10':         'MFI (10)',
        'ATR_10':         'ATR (10)',
        'STOCHk_10_3_3':  'Stochastic %K',
        'STOCHd_10_3_3':  'Stochastic %D',
    }

    for name, df in datasets.items():
        # 데이터프레임에 실재하는 DROPNA_SUBSET(_10 변수들)로 변경
        df_valid = df[analysis_cols].dropna(subset=DROPNA_SUBSET)

        if df_valid.empty:
            print(f"\n[{name}] 시각화 가능한 데이터 없음, 건너뜀")
            continue

        # ── 5-2. Heatmap ──────────────────────────────────
        fig, ax = plt.subplots(figsize=(9, 7))
        corr = df_valid.corr(method='pearson')
        sns.heatmap(
            corr, annot=True, fmt='.2f', cmap='coolwarm',
            vmin=-1, vmax=1, linewidths=0.5, square=True, ax=ax
        )
        ax.set_title(f'[{name}] 주가 및 보조지표 상관관계 (Heatmap)',
                     fontsize=13, fontweight='bold', pad=16)
        plt.tight_layout()
        plt.show()

        # ── 5-3. Pair Plot ─────────────────────────────────
        print(f"\n[{name}] Pair Plot 생성 중...")
        df_plot = df_valid.rename(columns=col_rename)
        pg = sns.pairplot(
            df_plot,
            kind='reg',
            diag_kind='kde',
            plot_kws={'scatter_kws': {'alpha': 0.5, 'color': '#1f77b4'},
                      'line_kws': {'color': 'red', 'lw': 1.5}},
        )
        pg.fig.suptitle(f'[{name}] 지표별 산점도 행렬 (Pair Plot)',
                        y=1.02, fontsize=14, fontweight='bold')
        plt.show()

        ### 가정: df_work에 주가 데이터 및 계산 완료된 ATR_10, MFI_10, STOCHk_10_3_3 가 들어있음


### 논문 수록 심리/Stochastic 지표 Resampling (잔차 추출)


### 1. 제어 변수(X_control) 생성 (14주/10일차 기준 반영)
df['RET'] = df['Close'].pct_change()
df['MOM'] = df['RET'].shift(1).rolling(window=10).sum()      # 박스 기간동안 누적수익률
df['VOL'] = df['RET'].rolling(window=11).var()               # 박스 기간동안 수익률의 분산

### 제어 변수 및 지표들의 결측치 제거
control_features = ['RET', 'MOM', 'VOL']
target_indicators = ['ATR_10', 'MFI_10', 'STOCHk_10_3_3']

df_resample = df[control_features + target_indicators].dropna()

X_control = df_resample[control_features]

### 2. 각 지표별 선형 회귀 수행 후 잔차(순수 심리 성분) 추출
residuals_dict = {}

for indicator in target_indicators:
    y = df_resample[indicator]
    
    # OLS 회귀 모델 적합
    lr = LinearRegression()
    lr.fit(X_control, y)
    
    # 예측값 계산 및 잔차(실제값 - 예측값) 산출
    y_pred = lr.predict(X_control)
    residual = y - y_pred
    
    # 잔차 변수명 정의 (예: ATR_10_res)
    residuals_dict[f"{indicator}_res"] = residual

### 잔차 데이터프레임 빌드
df_residuals = pd.DataFrame(residuals_dict, index=df_resample.index)


### 3. 순수 잔차 성분을 활용한 StandardScaler 및 PCA

pca_features = [f"{indicator}_res" for indicator in target_indicators]

# 표준화(Scaling)
scaler = StandardScaler()
scaled_matrix = scaler.fit_transform(df_residuals[pca_features])

# PCA 수행 (제1주성분 1개 추출)
pca = PCA(n_components=1)
df_residuals['Investor_Sentiment'] = pca.fit_transform(scaled_matrix)

# 검증 지표 출력
explained_variance = pca.explained_variance_ratio_[0]
print(f"[검증 결과] 제1주성분(Investor_Sentiment)의 정보 설명력: {explained_variance * 100:.2f}%")

# 기여도(Weights) 확인
for feature, weight in zip(pca_features, pca.components_[0]):
    print(f" - {feature}: {weight:.4f}")

# 4. 최종 메인 데이터프레임에 결합
df = df.join(df_residuals['Investor_Sentiment'])
print("\n=== [Resampling & PCA 완료] df_work에 'Investor_Sentiment'가 성공적으로 결합되었습니다. ===")


### 가정: df_residuals에 가격 요인이 통제된 잔차 변수들이 들어있음
### pca_features = ['ATR_10_res', 'MFI_10_res', 'STOCHk_10_3_3_res']

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

### 1. 피어슨 상관계수 Heatmap (선형 관계 확인 - 회귀 후라 값이 낮게 나오는 것이 정상)
corr_pearson = df_residuals[pca_features].corr(method='pearson')
sns.heatmap(
    corr_pearson, annot=True, fmt='.2f', cmap='coolwarm',
    vmin=-1, vmax=1, linewidths=0.5, square=True, ax=axes[0]
)
axes[0].set_title('잔차 간 피어슨(Pearson) 상관관계\n(선형적 연관성 통제 확인)', fontsize=12, fontweight='bold')

### 2. 스피어만 상관계수 Heatmap (비선형/순위 관계 확인 - 심리적 동조화 포착)
corr_spearman = df_residuals[pca_features].corr(method='spearman')
sns.heatmap(
    corr_spearman, annot=True, fmt='.2f', cmap='viridis',
    vmin=-1, vmax=1, linewidths=0.5, square=True, ax=axes[1]
)
axes[1].set_title('잔차 간 스피어만(Spearman) 상관관계\n(비선형적 심리 동조화 포착)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

# ============================================================
# 6. 통합 데이터셋 구성 (StandardScaler 적용) + ML 파이프라인
# ============================================================

# ── 6-1. 삼성전자 df_work 구성 ────────────────────────────────────────────────
#   · datasets['삼성전자'] 를 명시적으로 사용 (loop 변수 df 의존 제거)
df_work = datasets['삼성전자'].copy()

# ── 6-2. RSI 계산 추가 (Wilder's Smoothing, 14주) ─────────────────────────────
def calc_rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """트레이딩뷰 기본 방식과 동일한 Wilder 지수이동평균 RSI"""
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).rename(f'RSI_{length}')

df_work['RSI_14'] = calc_rsi(df_work['Close'])

# ── 6-3. Investor_Sentiment(PC1) 재계산 (삼성전자 전용) ──────────────────────
#   · 기존 섹션 5의 Resampling 로직을 삼성전자 데이터에 명시적으로 재적용
#   · 제어 변수(RET, MOM, VOL)로 지표 3개를 회귀한 뒤 잔차를 PCA 압축
df_work['RET'] = df_work['Close'].pct_change()
df_work['MOM'] = df_work['RET'].shift(1).rolling(window=10).sum()   # 누적 수익률 (10주)
df_work['VOL'] = df_work['RET'].rolling(window=11).var()             # 수익률 분산 (11주)

_ctrl_cols = ['RET', 'MOM', 'VOL']
_ind_cols  = ['ATR_10', 'MFI_10', 'STOCHk_10_3_3']
_df_rs     = df_work[_ctrl_cols + _ind_cols].dropna()
_X_ctrl    = _df_rs[_ctrl_cols]

_res_dict = {}
for _ind in _ind_cols:
    _lr = LinearRegression()
    _lr.fit(_X_ctrl, _df_rs[_ind])
    _res_dict[f'{_ind}_res'] = _df_rs[_ind] - _lr.predict(_X_ctrl)

_df_res    = pd.DataFrame(_res_dict, index=_df_rs.index)
_pca_cols  = list(_res_dict.keys())
_sc_pca    = StandardScaler()
_sc_matrix = _sc_pca.fit_transform(_df_res[_pca_cols])
_pca_s     = PCA(n_components=1)
_pc1_vals  = _pca_s.fit_transform(_sc_matrix).flatten()

df_work.loc[_df_res.index, 'Investor_Sentiment(PC1)'] = _pc1_vals
print(f"\n[삼성전자 PCA 재계산] 제1주성분 설명력: {_pca_s.explained_variance_ratio_[0]*100:.2f}%")

# ── 6-4. 컬럼명 표준화 (논문 기준 단축 명칭으로 통일) ─────────────────────────
df_work = df_work.rename(columns={
    'STOCHk_10_3_3': 'STOCH',
    'ATR_10':        'ATR',
    'RSI_14':        'RSI',
})

feature_cols = ['Close', 'STOCH', 'RSI', 'ATR', 'Investor_Sentiment(PC1)']

# ── 6-5. 타겟 변수 정의 및 결측치 제거 ────────────────────────────────────────
#   y = 다음 주 종가 (현재 Close 를 한 칸 앞으로 이동)
#   shift(-1) 로 인해 마지막 행이 NaN → dropna 로 일괄 제거
df_work['Target'] = df_work['Close'].shift(-1)
df_work.dropna(inplace=True)

print(f"\n[df_work] 최종 행 수: {len(df_work)}주 "
      f"({df_work.index[0].date()} ~ {df_work.index[-1].date()})")

# ── 6-6. X / y 정의 ──────────────────────────────────────────────────────────
#   X : 현재 시점 피처 5개
#   y : 다음 주 종가 (Target)
#   shift(-1) 결측은 dropna 로 이미 제거됐으나 명시적으로 마지막 행 제외
X = df_work[feature_cols].copy()
y = df_work['Target'].copy()
X = X[:-1]
y = y[:-1]

print(f"  X shape : {X.shape}  /  y shape : {y.shape}")

# ── 6-7. StandardScaler 적용 (피처 표준화) ────────────────────────────────────
scaler_X = StandardScaler()
X_scaled = pd.DataFrame(
    scaler_X.fit_transform(X),
    columns=feature_cols,
    index=X.index,
)

# ── 6-8. Train / Validation / Test Split (70 / 15 / 15) ──────────────────────
# 1차 분할 : Train(70%) vs 나머지(30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.30, shuffle=False
)
# 2차 분할 : 나머지(X_temp, y_temp) 를 반반 → Validation(15%) / Test(15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, shuffle=False
)

n_total = len(X_scaled)
print(f"\n{'─'*56}")
print(f"  {'구분':<14}  {'샘플 수':>6}  {'비율':>6}  {'기간'}")
print(f"  {'─'*52}")
print(f"  {'전체':<14}  {n_total:>6}   100.0%  "
      f"{X_scaled.index[0].date()} ~ {X_scaled.index[-1].date()}")
print(f"  {'Train':<14}  {len(X_train):>6}  {len(X_train)/n_total*100:>5.1f}%  "
      f"{X_train.index[0].date()} ~ {X_train.index[-1].date()}")
print(f"  {'Validation':<14}  {len(X_val):>6}  {len(X_val)/n_total*100:>5.1f}%  "
      f"{X_val.index[0].date()} ~ {X_val.index[-1].date()}")
print(f"  {'Test':<14}  {len(X_test):>6}  {len(X_test)/n_total*100:>5.1f}%  "
      f"{X_test.index[0].date()} ~ {X_test.index[-1].date()}")
print(f"{'─'*56}\n")

# ── 6-9. 모델 학습 (Linear Regression) ────────────────────────────────────────
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

y_pred_train = model_lr.predict(X_train)
y_pred_val   = model_lr.predict(X_val)
y_pred_test  = model_lr.predict(X_test)

# ── 6-10. 평가 함수 및 결과 출력 ──────────────────────────────────────────────
def evaluate_model(y_true, y_pred, label: str) -> tuple:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"  [{label:<12}]  RMSE: {rmse:>10,.0f}원  |  MAE: {mae:>10,.0f}원  |  R²: {r2:>7.4f}")
    return rmse, mae, r2

print("=" * 62)
print("   모델 평가 결과  ─  Linear Regression  (삼성전자 주봉)")
print("=" * 62)
m_train = evaluate_model(y_train, y_pred_train, 'Train')
m_val   = evaluate_model(y_val,   y_pred_val,   'Validation')
m_test  = evaluate_model(y_test,  y_pred_test,  'Test')
print("=" * 62)

# ── 6-11. 시각화 (2×2 서브플롯) ──────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('[삼성전자] 주봉 종가 예측 – Linear Regression 결과',
             fontsize=15, fontweight='bold')

date_fmt = mdates.DateFormatter('%Y-%m')

# (A) 전체 구간 실제 vs 예측 (Train / Val / Test 색상 구분)
ax = axes[0, 0]
ax.plot(y_train.index, y_train.values,
        label='실제 (Train)', color='steelblue',  lw=1.5)
ax.plot(y_train.index, y_pred_train,
        label='예측 (Train)', color='steelblue',  lw=1, ls='--', alpha=0.75)
ax.plot(y_val.index,   y_val.values,
        label='실제 (Val)',   color='darkorange',  lw=1.5)
ax.plot(y_val.index,   y_pred_val,
        label='예측 (Val)',   color='darkorange',  lw=1, ls='--', alpha=0.75)
ax.plot(y_test.index,  y_test.values,
        label='실제 (Test)',  color='seagreen',   lw=1.5)
ax.plot(y_test.index,  y_pred_test,
        label='예측 (Test)',  color='seagreen',   lw=1, ls='--', alpha=0.75)
ax.set_title('전체 구간 실제 종가 vs 예측 종가', fontsize=11, fontweight='bold')
ax.set_ylabel('주가 (원)')
ax.legend(fontsize=7, ncol=2, loc='upper left')
ax.xaxis.set_major_formatter(date_fmt)

# (B) Test 구간 확대 (실제 vs 예측 + 오차 음영)
ax = axes[0, 1]
ax.plot(y_test.index, y_test.values,
        label='실제 종가', color='seagreen', lw=2)
ax.plot(y_test.index, y_pred_test,
        label='예측 종가', color='tomato',   lw=1.5, ls='--')
ax.fill_between(y_test.index, y_test.values, y_pred_test,
                alpha=0.15, color='tomato', label='오차 구간')
ax.set_title('[Test 구간] 실제 vs 예측 종가 (확대)', fontsize=11, fontweight='bold')
ax.set_ylabel('주가 (원)')
ax.legend(fontsize=9)
ax.xaxis.set_major_formatter(date_fmt)

# (C) Test 잔차 분포 (히스토그램)
ax = axes[1, 0]
residuals_test = y_test.values - y_pred_test
ax.hist(residuals_test, bins=20, color='mediumpurple',
        edgecolor='white', alpha=0.85)
ax.axvline(0, color='red', lw=1.5, ls='--', label='잔차=0 기준선')
ax.set_title('[Test 구간] 잔차(Residual) 분포', fontsize=11, fontweight='bold')
ax.set_xlabel('잔차 (실제 - 예측, 원)')
ax.set_ylabel('빈도')
ax.legend(fontsize=9)

# (D) 특성 중요도 (표준화 회귀 계수)
ax = axes[1, 1]
coef_series = pd.Series(model_lr.coef_, index=feature_cols).sort_values()
bar_colors  = ['tomato' if c < 0 else 'steelblue' for c in coef_series]
coef_series.plot(kind='barh', ax=ax, color=bar_colors, edgecolor='white')
ax.axvline(0, color='black', lw=0.8)
ax.set_title('특성 중요도 (표준화 회귀 계수)', fontsize=11, fontweight='bold')
ax.set_xlabel('계수 크기 (표준화 스케일)')

plt.tight_layout()
plt.show()

# ── 6-12. 최종 평가 요약표 ────────────────────────────────────────────────────
metrics_summary = pd.DataFrame(
    {
        'RMSE (원)': [m_train[0], m_val[0], m_test[0]],
        'MAE  (원)': [m_train[1], m_val[1], m_test[1]],
        'R²':        [m_train[2], m_val[2], m_test[2]],
    },
    index=['Train', 'Validation', 'Test'],
)
pd.set_option('display.float_format', '{:,.4f}'.format)
print("\n=== 최종 평가 지표 요약 ===")
print(metrics_summary.to_string())
