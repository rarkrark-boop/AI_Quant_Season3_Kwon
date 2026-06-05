from pathlib import Path

import pandas as pd
import streamlit as st

from modules.predictor import QuantPredictor
from utils.data_utils import safe_filename, save_output_csv
from visualization import plot_price_history, plot_sentiment_index


MODEL_FILES = {
    "Model A": Path("models/best_xgboost_panel_model_A.pkl"),
    "Model B": Path("models/best_xgboost_panel_model_B.pkl"),
    "Model C": Path("models/best_xgboost_panel_model_C.pkl"),
    "Scaler": Path("models/scaler.pkl"),
    "PCA": Path("models/pca.pkl"),
}
BACKTEST_SUMMARY_PATH = Path("data/output/model_backtest_summary.csv")
PERIOD_BACKTEST_PATH = Path("data/output/model_period_backtest.csv")

BACKTEST_DA = {
    "Zero-return": 0.00,
    "Train-mean": 67.24,
    "Price-only": 60.34,
    "A": 60.34,
    "B": 60.34,
    "C": 60.34,
}

FALLBACK_BACKTEST_SUMMARY = [
    {
        "Model": "Zero-return Baseline",
        "R2": -0.2677,
        "RMSE": 0.0619,
        "MAE": 0.0484,
        "DA": 0.00,
        "Correct": 0,
        "Test_N": 58,
        "Up_DA": 0.00,
        "Down_DA": 0.00,
        "Pred_Up_Ratio": 0.00,
    },
    {
        "Model": "Train-mean Baseline",
        "R2": -0.2591,
        "RMSE": 0.0617,
        "MAE": 0.0482,
        "DA": 67.24,
        "Correct": 39,
        "Test_N": 58,
        "Up_DA": 100.00,
        "Down_DA": 0.00,
        "Pred_Up_Ratio": 100.00,
    },
    {
        "Model": "Price-only Baseline",
        "R2": -0.2613,
        "RMSE": 0.0617,
        "MAE": 0.0480,
        "DA": 60.34,
        "Correct": 35,
        "Test_N": 58,
        "Up_DA": 53.85,
        "Down_DA": 73.68,
        "Pred_Up_Ratio": 44.83,
    },
    {
        "Model": "Model A (PCA 통합)",
        "R2": -0.2613,
        "RMSE": 0.0617,
        "MAE": 0.0480,
        "DA": 60.34,
        "Correct": 35,
        "Test_N": 58,
        "Up_DA": 53.85,
        "Down_DA": 73.68,
        "Pred_Up_Ratio": 44.83,
    },
    {
        "Model": "Model B (세부 잔차)",
        "R2": -0.2664,
        "RMSE": 0.0619,
        "MAE": 0.0480,
        "DA": 60.34,
        "Correct": 35,
        "Test_N": 58,
        "Up_DA": 56.41,
        "Down_DA": 68.42,
        "Pred_Up_Ratio": 48.28,
    },
    {
        "Model": "Model C (전체 혼용)",
        "R2": -0.2664,
        "RMSE": 0.0619,
        "MAE": 0.0480,
        "DA": 60.34,
        "Correct": 35,
        "Test_N": 58,
        "Up_DA": 56.41,
        "Down_DA": 68.42,
        "Pred_Up_Ratio": 48.28,
    },
]

FALLBACK_PERIOD_BACKTEST = [
    {"Period": "2025-03-26 ~ 2025-06-11", "Model": "Zero-return Baseline", "Weeks": 12, "Correct": 0, "DA": 0.00, "Up_DA": 0.00, "Down_DA": 0.00, "Pred_Up_Ratio": 0.00},
    {"Period": "2025-03-26 ~ 2025-06-11", "Model": "Train-mean Baseline", "Weeks": 12, "Correct": 6, "DA": 50.00, "Up_DA": 100.00, "Down_DA": 0.00, "Pred_Up_Ratio": 100.00},
    {"Period": "2025-03-26 ~ 2025-06-11", "Model": "Price-only Baseline", "Weeks": 12, "Correct": 8, "DA": 66.67, "Up_DA": 83.33, "Down_DA": 50.00, "Pred_Up_Ratio": 66.67},
    {"Period": "2025-03-26 ~ 2025-06-11", "Model": "Model A (PCA 통합)", "Weeks": 12, "Correct": 8, "DA": 66.67, "Up_DA": 83.33, "Down_DA": 50.00, "Pred_Up_Ratio": 66.67},
    {"Period": "2025-03-26 ~ 2025-06-11", "Model": "Model B (세부 잔차)", "Weeks": 12, "Correct": 7, "DA": 58.33, "Up_DA": 83.33, "Down_DA": 33.33, "Pred_Up_Ratio": 75.00},
    {"Period": "2025-03-26 ~ 2025-06-11", "Model": "Model C (전체 혼용)", "Weeks": 12, "Correct": 7, "DA": 58.33, "Up_DA": 83.33, "Down_DA": 33.33, "Pred_Up_Ratio": 75.00},
    {"Period": "2025-06-18 ~ 2025-09-03", "Model": "Zero-return Baseline", "Weeks": 12, "Correct": 0, "DA": 0.00, "Up_DA": 0.00, "Down_DA": 0.00, "Pred_Up_Ratio": 0.00},
    {"Period": "2025-06-18 ~ 2025-09-03", "Model": "Train-mean Baseline", "Weeks": 12, "Correct": 8, "DA": 66.67, "Up_DA": 100.00, "Down_DA": 0.00, "Pred_Up_Ratio": 100.00},
    {"Period": "2025-06-18 ~ 2025-09-03", "Model": "Price-only Baseline", "Weeks": 12, "Correct": 8, "DA": 66.67, "Up_DA": 50.00, "Down_DA": 100.00, "Pred_Up_Ratio": 33.33},
    {"Period": "2025-06-18 ~ 2025-09-03", "Model": "Model A (PCA 통합)", "Weeks": 12, "Correct": 8, "DA": 66.67, "Up_DA": 50.00, "Down_DA": 100.00, "Pred_Up_Ratio": 33.33},
    {"Period": "2025-06-18 ~ 2025-09-03", "Model": "Model B (세부 잔차)", "Weeks": 12, "Correct": 10, "DA": 83.33, "Up_DA": 75.00, "Down_DA": 100.00, "Pred_Up_Ratio": 50.00},
    {"Period": "2025-06-18 ~ 2025-09-03", "Model": "Model C (전체 혼용)", "Weeks": 12, "Correct": 10, "DA": 83.33, "Up_DA": 75.00, "Down_DA": 100.00, "Pred_Up_Ratio": 50.00},
    {"Period": "2025-09-10 ~ 2025-11-26", "Model": "Zero-return Baseline", "Weeks": 12, "Correct": 0, "DA": 0.00, "Up_DA": 0.00, "Down_DA": 0.00, "Pred_Up_Ratio": 0.00},
    {"Period": "2025-09-10 ~ 2025-11-26", "Model": "Train-mean Baseline", "Weeks": 12, "Correct": 9, "DA": 75.00, "Up_DA": 100.00, "Down_DA": 0.00, "Pred_Up_Ratio": 100.00},
    {"Period": "2025-09-10 ~ 2025-11-26", "Model": "Price-only Baseline", "Weeks": 12, "Correct": 8, "DA": 66.67, "Up_DA": 55.56, "Down_DA": 100.00, "Pred_Up_Ratio": 41.67},
    {"Period": "2025-09-10 ~ 2025-11-26", "Model": "Model A (PCA 통합)", "Weeks": 12, "Correct": 8, "DA": 66.67, "Up_DA": 55.56, "Down_DA": 100.00, "Pred_Up_Ratio": 41.67},
    {"Period": "2025-09-10 ~ 2025-11-26", "Model": "Model B (세부 잔차)", "Weeks": 12, "Correct": 7, "DA": 58.33, "Up_DA": 44.44, "Down_DA": 100.00, "Pred_Up_Ratio": 33.33},
    {"Period": "2025-09-10 ~ 2025-11-26", "Model": "Model C (전체 혼용)", "Weeks": 12, "Correct": 7, "DA": 58.33, "Up_DA": 44.44, "Down_DA": 100.00, "Pred_Up_Ratio": 33.33},
    {"Period": "2025-12-03 ~ 2026-03-04", "Model": "Zero-return Baseline", "Weeks": 12, "Correct": 0, "DA": 0.00, "Up_DA": 0.00, "Down_DA": 0.00, "Pred_Up_Ratio": 0.00},
    {"Period": "2025-12-03 ~ 2026-03-04", "Model": "Train-mean Baseline", "Weeks": 12, "Correct": 9, "DA": 75.00, "Up_DA": 100.00, "Down_DA": 0.00, "Pred_Up_Ratio": 100.00},
    {"Period": "2025-12-03 ~ 2026-03-04", "Model": "Price-only Baseline", "Weeks": 12, "Correct": 5, "DA": 41.67, "Up_DA": 33.33, "Down_DA": 66.67, "Pred_Up_Ratio": 33.33},
    {"Period": "2025-12-03 ~ 2026-03-04", "Model": "Model A (PCA 통합)", "Weeks": 12, "Correct": 5, "DA": 41.67, "Up_DA": 33.33, "Down_DA": 66.67, "Pred_Up_Ratio": 33.33},
    {"Period": "2025-12-03 ~ 2026-03-04", "Model": "Model B (세부 잔차)", "Weeks": 12, "Correct": 5, "DA": 41.67, "Up_DA": 33.33, "Down_DA": 66.67, "Pred_Up_Ratio": 33.33},
    {"Period": "2025-12-03 ~ 2026-03-04", "Model": "Model C (전체 혼용)", "Weeks": 12, "Correct": 5, "DA": 41.67, "Up_DA": 33.33, "Down_DA": 66.67, "Pred_Up_Ratio": 33.33},
    {"Period": "2026-03-11 ~ 2026-05-13", "Model": "Zero-return Baseline", "Weeks": 10, "Correct": 0, "DA": 0.00, "Up_DA": 0.00, "Down_DA": 0.00, "Pred_Up_Ratio": 0.00},
    {"Period": "2026-03-11 ~ 2026-05-13", "Model": "Train-mean Baseline", "Weeks": 10, "Correct": 7, "DA": 70.00, "Up_DA": 100.00, "Down_DA": 0.00, "Pred_Up_Ratio": 100.00},
    {"Period": "2026-03-11 ~ 2026-05-13", "Model": "Price-only Baseline", "Weeks": 10, "Correct": 6, "DA": 60.00, "Up_DA": 57.14, "Down_DA": 66.67, "Pred_Up_Ratio": 50.00},
    {"Period": "2026-03-11 ~ 2026-05-13", "Model": "Model A (PCA 통합)", "Weeks": 10, "Correct": 6, "DA": 60.00, "Up_DA": 57.14, "Down_DA": 66.67, "Pred_Up_Ratio": 50.00},
    {"Period": "2026-03-11 ~ 2026-05-13", "Model": "Model B (세부 잔차)", "Weeks": 10, "Correct": 6, "DA": 60.00, "Up_DA": 57.14, "Down_DA": 66.67, "Pred_Up_Ratio": 50.00},
    {"Period": "2026-03-11 ~ 2026-05-13", "Model": "Model C (전체 혼용)", "Weeks": 10, "Correct": 6, "DA": 60.00, "Up_DA": 57.14, "Down_DA": 66.67, "Pred_Up_Ratio": 50.00},
]

TEST_SET_SUMMARY = {
    "weeks": 58,
    "model_correct": 35,
    "up_weeks": 39,
    "down_weeks": 19,
}

MODEL_LABELS = {
    "A": "Model A (PC1 통합 심리지수)",
    "B": "Model B (잔차 3개)",
    "C": "Model C (PC1 + 잔차 3개)",
}

MODEL_SUMMARY = {
    "A": "삼성전자 기본 학습 데이터와 PC1 통합 심리지수를 함께 사용합니다.",
    "B": "삼성전자 기본 학습 데이터와 ATR, MFI, Stochastic 잔차 3개를 개별 proxy로 사용합니다.",
    "C": "삼성전자 기본 학습 데이터에 PC1과 잔차 3개를 모두 함께 사용합니다.",
}

BASE_FEATURES = [
    "Samsung_Close",
    "Samsung_Log_Return_lag1~5",
    "KOSPI_Close",
    "KOSPI_Log_Return_lag1~5",
    "Bitcoin_Close",
    "Bitcoin_Log_Return_lag1~5",
]

MODEL_PROXY_FEATURES = {
    "A": ["Investor_Sentiment_PC1"],
    "B": ["ATR_10_res", "MFI_10_res", "STOCHk_10_3_3_res"],
    "C": ["Investor_Sentiment_PC1", "ATR_10_res", "MFI_10_res", "STOCHk_10_3_3_res"],
}

SUMMARY_MODEL_LABELS = {
    "Model A (PCA 통합)": "A",
    "Model B (세부 잔차)": "B",
    "Model C (전체 혼용)": "C",
}


@st.cache_resource
def load_predictor() -> QuantPredictor:
    return QuantPredictor()


def validate_model_files() -> list[str]:
    return [label for label, path in MODEL_FILES.items() if not path.exists()]


def get_model_feature_names(model_choice: str) -> list[str]:
    predictor = load_predictor()
    model = predictor._select_model(model_choice)
    return list(getattr(model, "feature_names_in_", []))


@st.cache_data
def load_backtest_summary() -> pd.DataFrame:
    if BACKTEST_SUMMARY_PATH.exists():
        return pd.read_csv(BACKTEST_SUMMARY_PATH)
    return pd.DataFrame(FALLBACK_BACKTEST_SUMMARY)


@st.cache_data
def load_period_backtest() -> pd.DataFrame:
    if PERIOD_BACKTEST_PATH.exists():
        return pd.read_csv(PERIOD_BACKTEST_PATH)
    return pd.DataFrame(FALLBACK_PERIOD_BACKTEST)


def get_selected_model_summary(model_choice: str) -> dict:
    summary_df = load_backtest_summary()
    target_model = {value: key for key, value in SUMMARY_MODEL_LABELS.items()}[model_choice]
    row = summary_df[summary_df["Model"] == target_model]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


def render_experiment_summary() -> None:
    st.subheader("실험 구조")
    target_col, exog_col = st.columns(2)
    target_col.metric("Target", "삼성전자")
    exog_col.metric("시장학습 보조자료", "KOSPI + Bitcoin")
    st.caption("시장학습 보조자료는 방향성 예측을 보조하는 학습 자료이며, 지표 해석의 중심은 삼성전자 심리 proxy입니다.")

    comparison_df = pd.DataFrame(
        [
            {
                "Model": "Train-mean Baseline",
                "기본 학습 데이터": "학습 구간 평균 로그수익률 반복 예측",
                "심리 proxy": "사용하지 않음",
                "백테스트 DA": f"{BACKTEST_DA['Train-mean']:.2f}%",
                "해석": "테스트 구간 상승 편향을 반영하는 단순 기준선",
            },
            {
                "Model": "Zero-return Baseline",
                "기본 학습 데이터": "0% 수익률 반복 예측",
                "심리 proxy": "사용하지 않음",
                "백테스트 DA": f"{BACKTEST_DA['Zero-return']:.2f}%",
                "해석": "방향성이 없는 기준선",
            },
            {
                "Model": "Price-only Baseline",
                "기본 학습 데이터": "삼성전자 + KOSPI + Bitcoin 주봉 종가/수익률 lag",
                "심리 proxy": "사용하지 않음",
                "백테스트 DA": f"{BACKTEST_DA['Price-only']:.2f}%",
                "해석": "시장학습 보조자료 포함 가격 기반 모델",
            },
            {
                "Model": "Model A",
                "기본 학습 데이터": "삼성전자 + KOSPI + Bitcoin 주봉 종가/수익률 lag",
                "심리 proxy": "PC1 통합 심리지수",
                "백테스트 DA": f"{BACKTEST_DA['A']:.2f}%",
                "해석": "PC1 추가 효과가 현재 테스트셋에서는 분리되지 않음",
            },
            {
                "Model": "Model B",
                "기본 학습 데이터": "삼성전자 + KOSPI + Bitcoin 주봉 종가/수익률 lag",
                "심리 proxy": "ATR, MFI, Stochastic 잔차 3개",
                "백테스트 DA": f"{BACKTEST_DA['B']:.2f}%",
                "해석": "잔차 proxy를 개별 투입한 비교 모델",
            },
            {
                "Model": "Model C",
                "기본 학습 데이터": "삼성전자 + KOSPI + Bitcoin 주봉 종가/수익률 lag",
                "심리 proxy": "PC1 + 잔차 3개",
                "백테스트 DA": f"{BACKTEST_DA['C']:.2f}%",
                "해석": "B와 같은 방향 예측을 보여 추가 PC1 효과가 제한적",
            },
        ]
    )
    st.dataframe(comparison_df, hide_index=True, width="stretch")
    st.warning(
        "A/B/C는 테스트셋 58주 중 모두 35주를 맞춰 60.34%로 동일합니다. "
        "이는 심리 proxy 조합별 차이가 현재 테스트 구간에서 뚜렷하게 분리되지 않았다는 뜻이며, "
        "학습 평균 수익률 기준선 67.24%보다 낮습니다."
    )

    st.subheader("보조 평가 지표")
    summary_df = load_backtest_summary()
    display_summary = summary_df[
        ["Model", "RMSE", "MAE", "DA", "Correct", "Test_N", "Up_DA", "Down_DA", "Pred_Up_Ratio"]
    ].copy()
    st.dataframe(display_summary, hide_index=True, width="stretch")

    period_df = load_period_backtest()
    if not period_df.empty:
        st.subheader("12주 단위 구간별 방향 정확도")
        model_filter = st.multiselect(
            "표시할 모델",
            options=period_df["Model"].unique().tolist(),
            default=[
                "Price-only Baseline",
                "Model A (PCA 통합)",
                "Model B (세부 잔차)",
                "Model C (전체 혼용)",
            ],
        )
        filtered_period_df = period_df[period_df["Model"].isin(model_filter)]
        st.dataframe(filtered_period_df, hide_index=True, width="stretch")
    else:
        st.info("구간별 백테스트 CSV가 없습니다. `model_train.py`를 실행하면 구간별 결과가 생성됩니다.")


def run() -> None:
    st.header("모델 구축 및 검증")
    st.caption("최종 예측 대상은 삼성전자이며, 코스피와 비트코인은 시장학습 보조자료로 사용합니다.")

    missing = validate_model_files()
    if missing:
        st.error(f"필요한 모델 파일이 없습니다: {', '.join(missing)}")
        st.info("먼저 모델 학습 스크립트를 실행해 `models/` 폴더에 pkl 파일을 생성하세요.")
        return

    render_experiment_summary()

    model_choice = st.radio(
        "사용할 저장 모델",
        options=["A", "B", "C"],
        horizontal=True,
        format_func=lambda value: MODEL_LABELS[value],
    )

    st.info(MODEL_SUMMARY[model_choice])
    feature_df = pd.DataFrame(
        [
            {
                "구분": "기본 학습 데이터",
                "Feature": ", ".join(BASE_FEATURES),
                "사용 이유": "삼성전자 자체 흐름과 시장/대체자산 흐름을 방향성 예측의 기본 맥락으로 반영합니다.",
            },
            {
                "구분": "심리 proxy",
                "Feature": ", ".join(MODEL_PROXY_FEATURES[model_choice]),
                "사용 이유": "기술지표에서 가격 요인을 통제한 뒤 남은 잔차 또는 그 통합 축이 방향성 예측에 주는 효과를 비교합니다.",
            },
        ]
    )
    st.dataframe(feature_df, hide_index=True, width="stretch")

    asset_name = "삼성전자"
    asset_key = safe_filename(asset_name)

    if st.button("삼성전자 방향성 예측 실행", type="primary"):
        with st.spinner(f"삼성전자 타깃 데이터와 KOSPI/BTC 시장학습 보조자료를 준비하고 Model {model_choice}로 예측하는 중입니다."):
            predictor = load_predictor()
            result = predictor.get_prediction(model_type=model_choice)

            timeseries_path = save_output_csv(
                result["df_plot"],
                f"{asset_key}_model_{model_choice}_prediction_timeseries.csv",
            )
            input_path = save_output_csv(
                pd.DataFrame([result["current_data"]]),
                f"{asset_key}_model_{model_choice}_prediction_input.csv",
                index=False,
            )

            st.session_state.prediction_result = result
            st.session_state.prediction_model_choice = model_choice
            st.session_state.prediction_plot_output_path = str(timeseries_path)
            st.session_state.prediction_input_output_path = str(input_path)

    result = st.session_state.get("prediction_result")
    selected_model = st.session_state.get("prediction_model_choice", model_choice)
    if result is None:
        st.info("삼성전자 방향성 예측 실행 버튼을 누르세요.")
        return

    direction_label = "상승" if result["direction"] == "UP" else "하락"
    direction_delta = result["pred_log_return"] * 100
    selected_summary = get_selected_model_summary(selected_model)
    selected_da = selected_summary.get("DA", BACKTEST_DA[selected_model])
    selected_mae = selected_summary.get("MAE")
    selected_up_da = selected_summary.get("Up_DA")
    selected_down_da = selected_summary.get("Down_DA")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("사용 모델", f"Model {selected_model}")
    col2.metric("백테스트 방향 정확도", f"{selected_da:.2f}%")
    col3.metric("다음 주 삼성전자 예측 방향", direction_label, delta=f"{direction_delta:.2f}%")
    col4.metric("Train-mean 기준선 대비", f"{selected_da - BACKTEST_DA['Train-mean']:.2f}%p")

    aux_col1, aux_col2, aux_col3 = st.columns(3)
    if selected_mae is not None:
        aux_col1.metric("MAE", f"{selected_mae:.4f}")
    if selected_up_da is not None:
        aux_col2.metric("실제 상승 주간 정답률", f"{selected_up_da:.2f}%")
    if selected_down_da is not None:
        aux_col3.metric("실제 하락 주간 정답률", f"{selected_down_da:.2f}%")

    st.caption(
        f"테스트셋 {TEST_SET_SUMMARY['weeks']}주 중 A/B/C는 각각 {TEST_SET_SUMMARY['model_correct']}주를 맞췄습니다. "
        f"같은 기간 실제 방향은 상승 {TEST_SET_SUMMARY['up_weeks']}주, 하락 {TEST_SET_SUMMARY['down_weeks']}주였습니다."
    )

    df_plot = result["df_plot"]
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.pyplot(plot_price_history(df_plot, title="Samsung Electronics Close"))
    with chart_col2:
        st.pyplot(plot_sentiment_index(df_plot["Investor_Sentiment_PC1"], title="Investor Sentiment PC1"))

    st.subheader("최근 예측 입력값")
    feature_names = get_model_feature_names(selected_model)
    input_df = pd.DataFrame([result["current_data"]])
    display_cols = [col for col in feature_names if col in input_df.columns]
    if display_cols:
        st.dataframe(input_df[display_cols].T.rename(columns={0: "value"}), width="stretch")
    else:
        st.dataframe(input_df, width="stretch")
    if st.session_state.get("prediction_input_output_path"):
        st.caption(f"예측 입력 CSV 저장 위치: {st.session_state.prediction_input_output_path}")
    if st.session_state.get("prediction_plot_output_path"):
        st.caption(f"예측 시계열 CSV 저장 위치: {st.session_state.prediction_plot_output_path}")
