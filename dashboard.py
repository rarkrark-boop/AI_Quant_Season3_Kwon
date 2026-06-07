import streamlit as st

from log.introduction import render_introduction
from log.process_flow import render_process_flow
from tabs import data_preprocessing, modeling_validation, sentiment_proxy


# =========================================================
# 프로젝트 공통 설정
# =========================================================
APP_TITLE = "투자자 심리지수 기반 주가 예측 대시보드"
PREDICTION_TARGET = "삼성전자"

DATA_ASSETS = [
    "삼성전자",
    "코스피",
    "비트코인",
]


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 공통 세션 상태
# =========================================================
def initialize_session_state() -> None:
    """
    대시보드 전체에서 공유하는 기본 session_state를 설정한다.

    데이터 전처리 탭에서는 삼성전자·코스피·비트코인을 선택할 수 있고,
    심리지표 및 모델링/검증 탭의 분석 대상은 삼성전자로 고정한다.
    """
    if "asset_name" not in st.session_state:
        st.session_state.asset_name = PREDICTION_TARGET

    st.session_state.prediction_target = PREDICTION_TARGET

    if "raw_data_by_asset" not in st.session_state:
        st.session_state.raw_data_by_asset = {}

    if "sentiment_data_by_asset" not in st.session_state:
        st.session_state.sentiment_data_by_asset = {}

    if "explained_variance_by_asset" not in st.session_state:
        st.session_state.explained_variance_by_asset = {}


# =========================================================
# 사이드바
# =========================================================
def render_default_sidebar() -> None:
    """
    프로젝트 소개 및 데이터 전처리 탭에서 사용하는
    자산 선택 사이드바입니다.
    """
    st.sidebar.header("주봉 데이터 준비")

    asset_name = st.sidebar.selectbox(
        "불러올 주봉 데이터",
        options=DATA_ASSETS,
        key="sidebar_asset_name",
    )

    st.session_state.asset_name = asset_name

    st.sidebar.caption(
        "선택한 자산의 주봉 데이터와 "
        "기술지표 계산 결과를 확인합니다."
    )


def render_sentiment_sidebar() -> None:
    """
    심리지표 탭에서 사용하는 삼성전자 고정 사이드바입니다.
    """
    st.session_state.asset_name = PREDICTION_TARGET

    st.sidebar.header("주봉 데이터 준비")
    st.sidebar.markdown("**분석 대상**")
    st.sidebar.markdown("### 삼성전자")

    st.sidebar.caption(
        "심리지표 탭에서는 삼성전자 주봉 데이터를 기준으로 "
        "ATR·MFI·Stochastic residual과 PC1을 확인합니다."
    )


def render_modeling_sidebar() -> None:
    """
    모델링/검증 탭에서 사용하는 고정 안내 사이드바입니다.
    """
    st.session_state.asset_name = PREDICTION_TARGET

    st.sidebar.header("분석 대상")

    st.sidebar.markdown("**최종 예측 대상**")
    st.sidebar.markdown("### 삼성전자")

    st.sidebar.markdown("**시장 보조 입력**")
    st.sidebar.write("KOSPI · Bitcoin")

    st.sidebar.caption(
        "삼성전자의 다음 주 로그수익률 방향을 예측하며, "
        "KOSPI와 Bitcoin은 시장 흐름을 반영하는 "
        "보조 입력자료로 사용합니다."
    )


# =========================================================
# 대시보드 상단
# =========================================================
def render_dashboard_header() -> None:
    """
    프로젝트의 목적을 간결하게 소개한다.
    """
    st.title(APP_TITLE)

    st.caption(
        "시장 데이터와 심리 proxy를 활용한 "
        "삼성전자 다음 주 로그수익률 방향 예측"
    )

    st.info(
        "본 프로젝트는 삼성전자를 최종 예측 대상으로 설정합니다. "
        "KOSPI와 Bitcoin의 주봉 데이터는 삼성전자 방향 예측을 위한 "
        "시장학습 보조자료로 사용합니다."
    )


# =========================================================
# 프로젝트 소개 탭의 실험 구성
# =========================================================
def render_experiment_summary() -> None:
    """
    프로젝트 소개 탭에서 전체 실험 구조를 요약한다.

    자세한 성능 결과와 해석은 모델링/검증 탭에서 표시한다.
    """
    st.subheader("분석 도구 및 실험 구성")

    st.markdown(
        """
        **공통 가격 feature 18개**  
        삼성전자·KOSPI·Bitcoin의 현재 주봉 종가 3개와  
        각 자산의 과거 1~5주 로그수익률 15개
        """
    )

    st.markdown(
        """
        **Price-only**  
        공통 가격 feature만 사용하는 기준모델
        """
    )

    st.markdown(
        """
        **Model A-1 · ATR 단독형**  
        공통 가격 feature에 ATR residual만 추가
        """
    )

    st.markdown(
        """
        **Model A-2 · MFI 단독형**  
        공통 가격 feature에 MFI residual만 추가
        """
    )

    st.markdown(
        """
        **Model A-3 · Stochastic 단독형**  
        공통 가격 feature에 Stochastic residual만 추가
        """
    )

    st.markdown(
        """
        **Model B · PC1 통합형**  
        공통 가격 feature에 residual 3개의 공통 성분인 PC1을 추가
        """
    )

    st.markdown(
        """
        **Model C · Residual 개별형**  
        공통 가격 feature에 ATR·MFI·Stochastic residual 3개를 추가
        """
    )

    st.markdown(
        """
        **Model D · 결합형**  
        공통 가격 feature에 PC1과 residual 3개를 함께 추가
        """
    )

    st.caption(
        "Model A-1~A-3은 개별 심리 proxy의 단독 효과를 확인하기 위한 "
        "추가 진단실험입니다. 모든 모델은 삼성전자 다음 주 로그수익률의 "
        "상승·하락 방향을 예측하며 Directional Accuracy를 중심으로 비교합니다."
    )


# =========================================================
# 메인 화면
# =========================================================
def main() -> None:
    initialize_session_state()
    render_dashboard_header()

    overview_tab, data_tab, sentiment_tab, modeling_tab = st.tabs(
        [
            "프로젝트 소개",
            "데이터 전처리",
            "심리지표",
            "모델링/검증",
        ],
        key="dashboard_tabs",
        on_change="rerun",
    )

    # 현재 열린 탭에 맞춰 사이드바를 한 번만 렌더링한다.
    if modeling_tab.open:
        render_modeling_sidebar()
    elif sentiment_tab.open:
        render_sentiment_sidebar()
    else:
        render_default_sidebar()

    # -----------------------------------------------------
    # 프로젝트 소개
    # -----------------------------------------------------
    with overview_tab:
        render_introduction()

        st.markdown(
            "<div style='height: 24px;'></div>",
            unsafe_allow_html=True,
        )

        render_process_flow()

        st.markdown(
            "<div style='height: 32px;'></div>",
            unsafe_allow_html=True,
        )

        render_experiment_summary()

    # -----------------------------------------------------
    # 데이터 전처리
    # -----------------------------------------------------
    with data_tab:
        data_preprocessing.run()

    # -----------------------------------------------------
    # 심리지표
    # -----------------------------------------------------
    with sentiment_tab:
        sentiment_proxy.run()

    # -----------------------------------------------------
    # 모델링 / 검증
    # -----------------------------------------------------
    with modeling_tab:
        modeling_validation.run()


if __name__ == "__main__":
    main()
