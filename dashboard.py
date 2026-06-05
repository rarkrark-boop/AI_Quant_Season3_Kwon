import streamlit as st

from log.introduction import render_introduction
from log.process_flow import render_process_flow
from tabs import data_preprocessing, modeling_validation, sentiment_proxy


st.set_page_config(
    page_title="Investor Sentiment Stock Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_sidebar() -> None:
    st.sidebar.header("분석 설정")
    st.session_state.asset_name = "삼성전자"
    st.sidebar.metric("최종 예측 대상", "삼성전자")
    st.sidebar.markdown("**시장학습 보조자료**")
    st.sidebar.write("코스피, 비트코인")
    st.sidebar.caption("코스피와 비트코인은 시장학습 보조자료로만 사용합니다.")


def main() -> None:
    render_sidebar()

    st.title("삼성전자 심리 Proxy 기반 방향성 예측 대시보드")
    st.caption("삼성전자 주봉을 target으로 두고, 코스피와 비트코인은 시장학습 보조자료로 사용합니다.")

    overview_tab, data_tab, sentiment_tab, modeling_tab = st.tabs(
        ["프로젝트 소개", "데이터 전처리", "심리지표", "모델링/검증"]
    )

    with overview_tab:
        render_introduction()
        render_process_flow()

    with data_tab:
        data_preprocessing.run()

    with sentiment_tab:
        sentiment_proxy.run()

    with modeling_tab:
        modeling_validation.run()


if __name__ == "__main__":
    main()
