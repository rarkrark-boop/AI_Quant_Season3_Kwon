from pathlib import Path

import streamlit as st


def render_process_flow() -> None:
    st.header("프로세스 흐름")

    st.markdown(
        """
1. 삼성전자·KOSPI·Bitcoin 주봉 가격 데이터 수집
2. 각 자산의 현재 종가와 과거 1~5주 로그수익률 생성
3. 삼성전자 ATR·MFI·Stochastic 기술지표 계산
4. RET·MOM·VOL로 설명되는 부분을 통제하여 residual 3개 생성
5. residual 3개를 표준화하고 PCA로 PC1 통합 심리 proxy 생성
6. Price-only와 Model A-1~A-3, Model B/C/D 비교
7. 삼성전자 다음 주 로그수익률의 오차와 방향 정확도 평가
8. 저장 모델 B/C/D로 최근 1회 예측 및 입력값 진단
        """
    )


def log_process_flow(
    path: str = "process_flow.md",
) -> None:
    content = (
        "# 프로세스 흐름\n\n"
        "1. 주봉 데이터 수집\n"
        "2. 공통 가격 feature 생성\n"
        "3. 기술지표와 residual 생성\n"
        "4. PCA 기반 PC1 생성\n"
        "5. Price-only 및 A-1~D 비교\n"
        "6. 방향 정확도와 오차 평가\n"
        "7. 저장 모델 최근 예측\n"
    )

    Path(path).write_text(
        content,
        encoding="utf-8",
    )
