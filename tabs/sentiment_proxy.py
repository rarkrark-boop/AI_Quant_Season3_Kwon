import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from utils.data_utils import safe_filename, save_output_csv
from visualization import plot_correlation_heatmap, plot_sentiment_index


INDICATOR_COLUMNS = ["ATR_10", "MFI_10", "STOCHk_10_3_3"]
RESIDUAL_COLUMNS = ["ATR_10_res", "MFI_10_res", "STOCHk_10_3_3_res"]


def extract_sentiment_residuals(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["RET"] = result["Close"].pct_change()
    result["MOM"] = result["RET"].shift(1).rolling(window=10).sum()
    result["VOL"] = result["RET"].rolling(window=11).var()

    control_columns = ["RET", "MOM", "VOL"]
    clean = result.dropna(subset=control_columns + INDICATOR_COLUMNS).copy()
    controls = clean[control_columns]

    for indicator in INDICATOR_COLUMNS:
        model = LinearRegression()
        model.fit(controls, clean[indicator])
        clean[f"{indicator}_res"] = clean[indicator] - model.predict(controls)

    return clean


def create_sentiment_index(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    result = df.copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(result[RESIDUAL_COLUMNS])

    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(scaled).flatten()

    mfi_index = RESIDUAL_COLUMNS.index("MFI_10_res")
    if pca.components_[0][mfi_index] < 0:
        pc1 = -pc1

    result["Investor_Sentiment_PC1"] = pc1
    return result, float(pca.explained_variance_ratio_[0])


def run() -> None:
    st.header("심리지표 계산 및 처리")

    raw_data = st.session_state.get("raw_data")
    if raw_data is None:
        st.info("데이터 전처리 탭에서 먼저 데이터를 불러오세요.")
        return

    if st.button("OLS 잔차 및 PCA 심리지수 계산", type="primary"):
        with st.spinner("가격 요인을 통제한 잔차와 PCA 심리지수를 계산하는 중입니다."):
            residual_data = extract_sentiment_residuals(raw_data)
            sentiment_data, explained_variance = create_sentiment_index(residual_data)
            asset_name = st.session_state.get("asset_name", "asset")
            output_path = save_output_csv(
                sentiment_data,
                f"{safe_filename(asset_name)}_sentiment_features.csv",
            )
            st.session_state.sentiment_data = sentiment_data
            st.session_state.explained_variance = explained_variance
            st.session_state.sentiment_data_output_path = str(output_path)

    df = st.session_state.get("sentiment_data")
    if df is None:
        st.info("심리지수 계산을 실행하세요.")
        return

    st.metric("PC1 설명분산", f"{st.session_state.explained_variance:.1%}")

    st.subheader("심리지표 데이터")
    st.dataframe(df[RESIDUAL_COLUMNS + ["Investor_Sentiment_PC1"]].tail(20), width="stretch")
    if st.session_state.get("sentiment_data_output_path"):
        st.caption(f"CSV 저장 위치: {st.session_state.sentiment_data_output_path}")

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(plot_sentiment_index(df["Investor_Sentiment_PC1"]))
    with col2:
        st.pyplot(plot_correlation_heatmap(df[RESIDUAL_COLUMNS + ["Investor_Sentiment_PC1"]]))
