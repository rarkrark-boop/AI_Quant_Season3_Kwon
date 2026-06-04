import pandas as pd
import streamlit as st

from modules.data_fetcher import get_recent_data
from utils.data_utils import safe_filename, save_output_csv, validate_columns
from visualization import plot_price_history


REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _load_recent_data(asset_name: str) -> pd.DataFrame:
    df = get_recent_data(asset_name)
    validate_columns(df, REQUIRED_COLUMNS)
    return df


def run() -> None:
    st.header("데이터 전처리")

    asset_name = st.session_state.get("asset_name", "삼성전자")
    asset_key = safe_filename(asset_name)

    if st.button("최근 주봉 데이터 불러오기", type="primary"):
        with st.spinner(f"{asset_name} 데이터를 수집하고 지표를 계산하는 중입니다."):
            raw_data = _load_recent_data(asset_name)
            output_path = save_output_csv(raw_data, f"{asset_key}_raw_indicators.csv")
            st.session_state.raw_data = raw_data
            st.session_state.raw_data_output_path = str(output_path)

    df = st.session_state.get("raw_data")
    if df is None:
        st.info("먼저 최근 주봉 데이터를 불러오세요.")
        return

    st.subheader("데이터 미리보기")
    st.dataframe(df.tail(20), width="stretch")
    if st.session_state.get("raw_data_output_path"):
        st.caption(f"CSV 저장 위치: {st.session_state.raw_data_output_path}")

    st.subheader("결측치")
    missing_df = df.isna().sum().rename("missing_count").to_frame()
    missing_path = save_output_csv(missing_df, f"{asset_key}_missing_summary.csv")
    st.dataframe(missing_df, width="stretch")
    st.caption(f"CSV 저장 위치: {missing_path}")

    st.subheader("가격 추이")
    st.pyplot(plot_price_history(df, title=f"{asset_name} 종가 추이"))
