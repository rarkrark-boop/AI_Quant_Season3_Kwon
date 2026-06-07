import streamlit as st


def render_introduction() -> None:
    """프로젝트 소개 화면을 렌더링한다."""
    st.title("🎯 Samsung Electronics Sentiment Analysis")

    st.markdown(
        """
### 📊 프로젝트 개요
- **목표**: 시장 데이터에 남은 투자자 반응을 심리 proxy로 구성하고 삼성전자 다음 주 방향 예측에 활용
- **최종 예측 대상**: 삼성전자
- **외생 보조 입력**: KOSPI, Bitcoin
- **분석 주기**: 주봉

### 🔬 분석 도구
- **기술지표**: ATR_10, MFI_10, STOCHk_10_3_3
- **심리 proxy**: RET·MOM·VOL 통제 후 residual 생성
- **통합 심리지수**: residual 3개 표준화 후 PCA의 PC1 사용
- **예측모델**: 저장된 XGBoost 모델
- **평가 지표**: R², RMSE, MAE, Directional Accuracy

### 🎯 실험 구성
- **Price-only**: 공통 가격 feature 18개
- **Model A-1**: 공통 가격 feature + ATR residual
- **Model A-2**: 공통 가격 feature + MFI residual
- **Model A-3**: 공통 가격 feature + Stochastic residual
- **Model B**: 공통 가격 feature + PC1
- **Model C**: 공통 가격 feature + residual 3개
- **Model D**: 공통 가격 feature + PC1 + residual 3개

### 📈 현재 연결된 산출물
- **삼성전자 심리 feature 데이터**: 107개 주봉
- **최종 PC1 기준값**: 보성님 산출 CSV의 마지막 행을 화면에 표시
- **검증 방식**: 시간 순서를 유지한 테스트 구간 성능 비교
        """
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Sentiment Rows",
        "107",
    )

    col2.metric(
        "PCA Variance",
        "55.04%",
    )

    col3.metric(
        "Models",
        "7",
    )
