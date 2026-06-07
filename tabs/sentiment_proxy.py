from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from utils.table_utils import render_presentation_table
from visualization import (
    plot_correlation_heatmap,
    plot_pca_loading_bar,
    plot_sentiment_index,
)


# =========================================================
# 경로 및 공통 상수
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MISSING_SUMMARY_PATH = (
    PROJECT_ROOT / "삼성전자_missing_summary.csv"
)

RAW_INDICATORS_PATH = (
    PROJECT_ROOT / "삼성전자_raw_indicators.csv"
)

SENTIMENT_FEATURES_PATH = (
    PROJECT_ROOT / "삼성전자_sentiment_features.csv"
)

PCA_REFERENCE_PATH = (
    PROJECT_ROOT / "outputs" / "pca_reference.csv"
)

INDICATOR_COLUMNS = [
    "ATR_10",
    "MFI_10",
    "STOCHk_10_3_3",
]

RESIDUAL_COLUMNS = [
    "ATR_10_res",
    "MFI_10_res",
    "STOCHk_10_3_3_res",
]

SENTIMENT_COLUMN = "Investor_Sentiment_PC1"
DISPLAY_START_DATE = pd.Timestamp("2024-01-01")


# =========================================================
# 계산 함수
# 현재 대시보드의 기본 표시값은 루트 CSV를 사용한다.
# 아래 함수는 계산 구조 확인 및 재생성 참고용으로 유지한다.
# =========================================================
def extract_sentiment_residuals(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    가격 수익률·모멘텀·변동성으로 기술지표를 회귀한 뒤
    ATR·MFI·Stochastic residual을 생성한다.
    """
    result = df.sort_index().copy()

    result["RET"] = result["Close"].pct_change()
    result["MOM"] = (
        result["RET"]
        .shift(1)
        .rolling(window=10)
        .sum()
    )
    result["VOL"] = (
        result["RET"]
        .rolling(window=11)
        .var()
    )

    control_columns = [
        "RET",
        "MOM",
        "VOL",
    ]

    clean = result.dropna(
        subset=control_columns + INDICATOR_COLUMNS
    ).copy()

    controls = clean[control_columns]

    for indicator in INDICATOR_COLUMNS:
        model = LinearRegression()
        model.fit(
            controls,
            clean[indicator],
        )

        clean[f"{indicator}_res"] = (
            clean[indicator]
            - model.predict(controls)
        )

    return clean


def create_sentiment_index(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    """
    residual 3개를 표준화한 뒤 PCA를 적용하여 PC1을 생성한다.

    보성님 최종 산출물과 대시보드 표시값은
    루트의 삼성전자_sentiment_features.csv를 기준으로 한다.
    """
    result = df.copy()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(
        result[RESIDUAL_COLUMNS]
    )

    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(
        scaled
    ).flatten()

    mfi_index = RESIDUAL_COLUMNS.index(
        "MFI_10_res"
    )

    if pca.components_[0][mfi_index] < 0:
        pc1 = -pc1

    result[SENTIMENT_COLUMN] = pc1

    explained_variance = float(
        pca.explained_variance_ratio_[0]
    )

    return result, explained_variance


# =========================================================
# 최신 산출물 로더
# =========================================================
@st.cache_data
def load_latest_sentiment_outputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    프로젝트 루트에 저장된 삼성전자 최신 산출물 3개를 불러온다.
    """
    missing_df = pd.read_csv(
        MISSING_SUMMARY_PATH
    )

    raw_indicators_df = pd.read_csv(
        RAW_INDICATORS_PATH,
        parse_dates=["Date"],
    )

    sentiment_features_df = pd.read_csv(
        SENTIMENT_FEATURES_PATH,
        parse_dates=["Date"],
    )

    raw_indicators_df = (
        raw_indicators_df
        .sort_values("Date")
        .reset_index(drop=True)
    )

    sentiment_features_df = (
        sentiment_features_df
        .sort_values("Date")
        .reset_index(drop=True)
    )

    return (
        missing_df,
        raw_indicators_df,
        sentiment_features_df,
    )


@st.cache_data
def load_pca_reference() -> pd.DataFrame | None:
    """
    발표용 PCA loading 참고 CSV가 있으면 불러온다.
    """
    if not PCA_REFERENCE_PATH.exists():
        return None

    return pd.read_csv(
        PCA_REFERENCE_PATH
    )


def _validate_required_files() -> list[str]:
    required_paths = [
        MISSING_SUMMARY_PATH,
        RAW_INDICATORS_PATH,
        SENTIMENT_FEATURES_PATH,
    ]

    return [
        path.name
        for path in required_paths
        if not path.exists()
    ]


def _prepare_timeseries_frame(
    sentiment_features_df: pd.DataFrame,
) -> pd.DataFrame:
    result = sentiment_features_df.copy()

    result = result.loc[
        result["Date"] >= DISPLAY_START_DATE
    ].copy()

    result = result.set_index("Date")
    result.index.name = "주봉 기준일"

    return result


def _build_latest_summary(
    sentiment_features_df: pd.DataFrame,
) -> pd.DataFrame:
    latest = (
        sentiment_features_df
        .sort_values("Date")
        .iloc[-1]
    )

    return pd.DataFrame(
        [
            {
                "항목": "최신 기준일",
                "값": latest["Date"].strftime("%Y-%m-%d"),
                "해석": "현재 연결된 최종 심리 proxy 산출물의 마지막 주봉 기준일",
            },
            {
                "항목": "ATR residual",
                "값": f"{latest['ATR_10_res']:.4f}",
                "해석": "수익률·모멘텀·변동성으로 설명되지 않은 ATR 잔차",
            },
            {
                "항목": "MFI residual",
                "값": f"{latest['MFI_10_res']:.4f}",
                "해석": "가격·거래량 기반 MFI에서 일반 가격요인을 통제한 잔차",
            },
            {
                "항목": "Stochastic residual",
                "값": f"{latest['STOCHk_10_3_3_res']:.4f}",
                "해석": "Stochastic 위치 지표에서 일반 가격요인을 통제한 잔차",
            },
            {
                "항목": "Investor Sentiment PC1",
                "값": f"{latest[SENTIMENT_COLUMN]:.6f}",
                "해석": "보성님 최종 산출물에 저장된 residual 3개의 통합 심리 proxy",
            },
        ]
    )


def _build_pca_reference_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "항목": [
                "PC1 설명분산비율",
                "ATR_10_res loading",
                "MFI_10_res loading",
                "STOCHk_10_3_3_res loading",
            ],
            "값": [
                "55.04%",
                "-0.0451",
                "0.7062",
                "0.7065",
            ],
            "해석": [
                "패널 학습 기준 residual 3개의 공통 변동 중 약 55%를 요약",
                "PC1 내 상대적 기여가 제한적으로 나타남",
                "PC1에 상대적으로 크게 반영됨",
                "PC1에 상대적으로 크게 반영됨",
            ],
        }
    )


# =========================================================
# Streamlit 화면
# =========================================================
def run() -> None:
    st.header("심리지표 계산 및 처리")

    # 이 탭은 삼성전자 최종 산출물만 사용한다.
    st.session_state.asset_name = "삼성전자"

    st.info(
        "본 탭은 삼성전자 ATR·MFI·Stochastic에서 "
        "수익률·모멘텀·변동성으로 설명되는 부분을 통제한 residual과, "
        "residual 3개를 통합한 Investor_Sentiment_PC1을 확인합니다. "
        "화면의 최종값은 프로젝트 루트의 보성님 산출 CSV를 기준으로 표시합니다."
    )

    missing_files = _validate_required_files()

    if missing_files:
        st.error(
            "필요한 최신 산출물 파일이 없습니다: "
            + ", ".join(missing_files)
        )
        return

    (
        missing_summary_df,
        raw_indicators_df,
        sentiment_features_df,
    ) = load_latest_sentiment_outputs()

    latest = (
        sentiment_features_df
        .sort_values("Date")
        .iloc[-1]
    )

    st.subheader("최신 심리 proxy 산출 결과")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "ATR residual",
        f"{latest['ATR_10_res']:.4f}",
    )

    col2.metric(
        "MFI residual",
        f"{latest['MFI_10_res']:.4f}",
    )

    col3.metric(
        "Stochastic residual",
        f"{latest['STOCHk_10_3_3_res']:.4f}",
    )

    col4.metric(
        "PC1 심리지수",
        f"{latest[SENTIMENT_COLUMN]:.6f}",
    )

    st.caption(
        "최신 기준일: "
        f"{latest['Date'].strftime('%Y-%m-%d')} · "
        "PC1은 삼성전자_sentiment_features.csv의 최종값을 사용합니다."
    )

    latest_summary_df = _build_latest_summary(
        sentiment_features_df
    )

    render_presentation_table(
        latest_summary_df,
        title="최신 심리 proxy 값과 해석",
        footnote=(
            "현재 표는 프로젝트 루트의 "
            "삼성전자_sentiment_features.csv 마지막 행을 기준으로 합니다."
        ),
        left_align_cols=[
            "항목",
            "해석",
        ],
        height=520,
    )

    st.markdown(
        "<div style='height: 42px;'></div>",
        unsafe_allow_html=True,
    )

    st.subheader("결측치 검증")

    missing_display_df = (
        missing_summary_df
        .rename(
            columns={
                "Price": "항목",
                "missing_count": "결측치 수",
            }
        )
    )

    render_presentation_table(
        missing_display_df,
        title="삼성전자 원본 주봉 및 기술지표 결측 검증",
        footnote=(
            "Close·High·Low·Open·Volume과 "
            "MFI·ATR·Stochastic 계산 결과의 결측 상태입니다."
        ),
        left_align_cols=["항목"],
        height=520,
    )

    with st.expander(
        "원본 기술지표 데이터 보기",
        expanded=False,
    ):
        raw_display_df = (
            raw_indicators_df
            .copy()
        )

        raw_display_df["Date"] = (
            raw_display_df["Date"]
            .dt.strftime("%Y-%m-%d")
        )

        st.dataframe(
            raw_display_df,
            use_container_width=True,
            hide_index=True,
        )

    st.markdown(
        "<div style='height: 42px;'></div>",
        unsafe_allow_html=True,
    )

    st.subheader("PCA 기준 참고값")

    pca_reference_df = _build_pca_reference_table()

    render_presentation_table(
        pca_reference_df,
        title="PCA 기준 참고값",
        footnote=(
            "모델 생성 과정에서 확인한 PCA 기준 참고값입니다. "
            "loading 부호는 주성분 방향 설정에 따라 달라질 수 있으므로 "
            "절대적 방향보다 상대적 기여도를 중심으로 해석합니다."
        ),
        left_align_cols=[
            "항목",
            "해석",
        ],
        height=430,
    )

    pca_reference_csv = load_pca_reference()

    if pca_reference_csv is not None:
        st.subheader("PCA loading 시각화")

        st.caption(
            "outputs/pca_reference.csv를 기반으로 한 "
            "발표용 PCA loading 시각화입니다."
        )

        left, center, right = st.columns(
            [0.15, 0.70, 0.15]
        )

        with center:
            st.pyplot(
                plot_pca_loading_bar(
                    pca_reference_csv
                ),
                use_container_width=True,
            )
    else:
        st.info(
            "outputs/pca_reference.csv가 없어 "
            "PCA loading 그래프를 표시하지 못했습니다."
        )

    timeseries_df = _prepare_timeseries_frame(
        sentiment_features_df
    )

    with st.expander(
        "심리 proxy 전체 데이터 보기",
        expanded=False,
    ):
        proxy_columns = [
            "Close",
            "RET",
            "MOM",
            "VOL",
            "ATR_10_res",
            "MFI_10_res",
            "STOCHk_10_3_3_res",
            SENTIMENT_COLUMN,
        ]

        available_columns = [
            col
            for col in proxy_columns
            if col in timeseries_df.columns
        ]

        st.dataframe(
            timeseries_df[available_columns],
            use_container_width=True,
        )

    st.markdown(
        "<div style='height: 48px;'></div>",
        unsafe_allow_html=True,
    )

    st.subheader("Investor Sentiment PC1 흐름")

    st.caption(
        "PC1은 residual 3개의 공통 흐름을 요약한 심리 proxy입니다. "
        "양수와 음수의 절대적 의미보다 시계열 변화와 상대적 위치를 중심으로 확인합니다."
    )

    st.pyplot(
        plot_sentiment_index(
            timeseries_df[SENTIMENT_COLUMN],
            title="Investor Sentiment PC1",
        ),
        use_container_width=True,
    )

    st.markdown(
        "<div style='height: 48px;'></div>",
        unsafe_allow_html=True,
    )

    st.subheader("Residual 및 PC1 상관관계")

    st.caption(
        "ATR·MFI·Stochastic residual과 PC1 사이의 "
        "선형 관계를 탐색적으로 확인합니다. "
        "상관관계는 인과관계를 의미하지 않습니다."
    )

    correlation_columns = (
        RESIDUAL_COLUMNS
        + [SENTIMENT_COLUMN]
    )

    st.pyplot(
        plot_correlation_heatmap(
            timeseries_df[correlation_columns],
            title="Residual · PC1 Correlation",
        ),
        use_container_width=True,
    )
