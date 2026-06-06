from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from modules.data_fetcher import get_recent_data
from modules.predictor import QuantPredictor
from visualization import plot_price_history, plot_sentiment_index,plot_correlation_heatmap, plot_directional_accuracy_bar,plot_actual_vs_prediction_series, plot_actual_vs_predicted_scatter
from utils.table_utils import render_presentation_table     
import streamlit.components.v1 as components

MODEL_FILES = {
    "Model A": Path("models/best_xgboost_panel_model_A.pkl"),
    "Model B": Path("models/best_xgboost_panel_model_B.pkl"),
    "Model C": Path("models/best_xgboost_panel_model_C.pkl"),
    "Scaler": Path("models/scaler.pkl"),
    "PCA": Path("models/pca.pkl"),
}

BACKTEST_DA = {
    "A": 52.94,
    "B": 56.30,
    "C": 55.46,
}


MODEL_DESCRIPTIONS = {
    "A": {
        "name": "Model A",
        "label": "PC1 통합형",
        "features": (
            "삼성전자·KOSPI·Bitcoin의 현재 주봉 종가와 "
            "각 자산의 과거 1~5주 로그수익률 "
            "+ Investor_Sentiment_PC1"
        ),
        "purpose": (
            "공통 기본 학습자료에 PCA로 압축한 "
            "단일 심리 proxy를 추가했을 때의 효과를 확인"
        ),
    },
    "B": {
        "name": "Model B",
        "label": "Residual 개별형",
        "features": (
            "삼성전자·KOSPI·Bitcoin의 현재 주봉 종가와 "
            "각 자산의 과거 1~5주 로그수익률 "
            "+ ATR·MFI·Stochastic residual 3개"
        ),
        "purpose": (
            "공통 기본 학습자료에 개별 residual 3개를 "
            "직접 추가했을 때의 효과를 확인"
        ),
    },
    "C": {
        "name": "Model C",
        "label": "PC1 + Residual 결합형",
        "features": (
            "삼성전자·KOSPI·Bitcoin의 현재 주봉 종가와 "
            "각 자산의 과거 1~5주 로그수익률 "
            "+ Investor_Sentiment_PC1 + residual 3개"
        ),
        "purpose": (
            "통합 심리 proxy와 개별 residual을 함께 추가했을 때의 "
            "결합 효과를 확인"
        ),
    },
}


BASE_PRICE_FEATURES = [
    "Samsung_Close",
    "Samsung_Log_Return_lag1",
    "Samsung_Log_Return_lag2",
    "Samsung_Log_Return_lag3",
    "Samsung_Log_Return_lag4",
    "Samsung_Log_Return_lag5",
    "KOSPI_Close",
    "KOSPI_Log_Return_lag1",
    "KOSPI_Log_Return_lag2",
    "KOSPI_Log_Return_lag3",
    "KOSPI_Log_Return_lag4",
    "KOSPI_Log_Return_lag5",
    "Bitcoin_Close",
    "Bitcoin_Log_Return_lag1",
    "Bitcoin_Log_Return_lag2",
    "Bitcoin_Log_Return_lag3",
    "Bitcoin_Log_Return_lag4",
    "Bitcoin_Log_Return_lag5",
]


MODEL_FEATURES = {
    "A": BASE_PRICE_FEATURES + [
        "Investor_Sentiment_PC1",
    ],
    "B": BASE_PRICE_FEATURES + [
        "ATR_10_res",
        "MFI_10_res",
        "STOCHk_10_3_3_res",
    ],
    "C": BASE_PRICE_FEATURES + [
        "Investor_Sentiment_PC1",
        "ATR_10_res",
        "MFI_10_res",
        "STOCHk_10_3_3_res",
    ],
}

#### export 
OUTPUT_DIR = Path("outputs")
PREDICTION_WORKFLOW_IMAGE = OUTPUT_DIR / "model_predicts_workflow.png"
MODEL_PREDICTION_COLUMNS = {
    "Price-only Baseline": {
        "pred_col": "Price_Only_Pred_Log_Return",
        "label": "Price-only",
        "match_col": "Price_Only_Match",
    },
    "Model A": {
        "pred_col": "Model_A_Pred_Log_Return",
        "label": "Model A",
        "match_col": "Model_A_Match",
    },
    "Model B": {
        "pred_col": "Model_B_Pred_Log_Return",
        "label": "Model B",
        "match_col": "Model_B_Match",
    },
    "Model C": {
        "pred_col": "Model_C_Pred_Log_Return",
        "label": "Model C",
        "match_col": "Model_C_Match",
    },
}

#######





@st.cache_resource
def load_predictor() -> QuantPredictor:
    return QuantPredictor()


def validate_model_files() -> list[str]:
    return [label for label, path in MODEL_FILES.items() if not path.exists()]


def load_input_data(asset_name: str):
    cached = st.session_state.get("raw_data")
    if cached is not None:
        return cached

    df = get_recent_data(asset_name)
    st.session_state.raw_data = df
    return df

def _build_model_input_preview(current_data, model_choice: str) -> pd.DataFrame:
    """
    선택한 Model A/B/C에 실제로 들어가는 feature만 추려서
    발표용 입력값 표로 구성한다.

    숫자형 입력값은 소수점 4~5자리까지 표시한다.
    """
    current_series = pd.Series(current_data)

    selected_features = [
        col for col in MODEL_FEATURES[model_choice]
        if col in current_series.index
    ]

    preview_df = (
        current_series[selected_features]
        .rename("입력값")
        .reset_index()
        .rename(columns={"index": "입력 feature"})
    )

    def format_value(value):
        try:
            return f"{float(value):.4f}"
        except Exception:
            return value

    preview_df["입력값"] = preview_df["입력값"].apply(format_value)

    return preview_df

def _build_prediction_date_summary(df_plot: pd.DataFrame) -> pd.DataFrame:
    """
    예측 실행 시점과 예측 대상 주봉 기준일을 요약한다.
    """
    today = pd.Timestamp.today().normalize()

    last_data_date = df_plot.index.max()

    if hasattr(last_data_date, "to_pydatetime"):
        prediction_target_date = last_data_date + pd.Timedelta(days=7)
        last_data_date_text = last_data_date.strftime("%Y-%m-%d")
        prediction_target_date_text = prediction_target_date.strftime("%Y-%m-%d")
    else:
        prediction_target_date = None
        last_data_date_text = str(last_data_date)
        prediction_target_date_text = "최근 데이터 기준 다음 주"

    summary_df = pd.DataFrame(
        [
            {
                "구분": "오늘의 실행 일자",
                "일자": today.strftime("%Y-%m-%d"),
                "의미": "(대시보드에서) 실행한 날짜",
            },
            {
                "구분": "최근 입력 데이터 기준일",
                "일자": last_data_date_text,
                "의미": "모델에 입력된 가장 최근 주봉 데이터의 기준일",
            },
            {
                "구분": "예측 대상 주봉 기준일",
                "일자": prediction_target_date_text,
                "의미": "모델이 다음 주 로그수익률 방향을 예측하는 대상 시점",
            },
        ]
    )

    return summary_df

def _plot_next_week_log_return_forecast(df_plot: pd.DataFrame, pred_log_return: float, asset_name: str, model_label: str):
    """
    최근 실제 로그수익률 흐름과 다음 주 예측 로그수익률을 함께 보여준다.
    result에 backtest 예측 시계열이 없으므로, 현재는 다음 주 단일 예측값을 강조한다.
    """
    plot_df = df_plot.copy()

    if "Log_Return" not in plot_df.columns:
        plot_df["Log_Return"] = 0.0

    plot_df = plot_df.tail(52).copy()

    last_date = plot_df.index.max()
    if hasattr(last_date, "to_pydatetime"):
        next_date = last_date + pd.Timedelta(days=7)
    else:
        next_date = len(plot_df)

    fig, ax = plt.subplots(figsize=(10, 4.8))

    ax.plot(
        plot_df.index,
        plot_df["Log_Return"],
        marker="o",
        linewidth=1.8,
        label="최근 실제 주봉 로그수익률",
    )

    ax.scatter(
        [next_date],
        [pred_log_return],
        s=120,
        marker="*",
        label="다음 주 예측 로그수익률",
        zorder=5,
    )

    ax.axhline(0, linewidth=1)
    ax.set_title(f"{asset_name} · {model_label} 다음 주 로그수익률 예측")
    ax.set_xlabel("주봉 기준일")
    ax.set_ylabel("로그수익률")
    ax.legend()
    fig.tight_layout()

    return fig


### export

def _load_output_csv(filename: str) -> pd.DataFrame | None:
    """
    outputs 폴더에 저장된 발표용 CSV를 불러온다.
    기존 Streamlit 예측 로직과는 분리해서 사용한다.
    """
    path = OUTPUT_DIR / filename

    if not path.exists():
        return None

    return pd.read_csv(path)





def _format_performance_table(performance_df: pd.DataFrame) -> pd.DataFrame:
    """
    성능 비교표를 발표 화면용 문자열로 정리한다.
    """
    display_df = performance_df.copy()

    rename_map = {
        "Model": "모델",
        "Feature_Set": "입력 구성",
        "Feature_Count": "feature 수",
        "R2": "R²",
        "RMSE": "RMSE",
        "MAE": "MAE",
        "DA": "방향 정확도",
    }

    display_df = display_df.rename(columns=rename_map)

    for col in ["R²", "RMSE", "MAE"]:
        if col in display_df.columns:
            display_df[col] = pd.to_numeric(display_df[col], errors="coerce").map(
                lambda value: "-" if pd.isna(value) else f"{value:.4f}"
            )

    if "방향 정확도" in display_df.columns:
        display_df["방향 정확도"] = pd.to_numeric(
            display_df["방향 정확도"],
            errors="coerce",
        ).map(lambda value: "-" if pd.isna(value) else f"{value:.2f}%")

    if "feature 수" in display_df.columns:
        display_df["feature 수"] = pd.to_numeric(
            display_df["feature 수"],
            errors="coerce",
        ).map(lambda value: "-" if pd.isna(value) else f"{int(value)}개")

    return display_df




def _build_direction_match_preview(
    prediction_df: pd.DataFrame,
    asset_name: str,
    selected_model_name: str,
    rows: int = 20,
) -> pd.DataFrame:
    """
    선택 자산과 선택 모델 기준의 최근 방향성 적중표를 만든다.
    """
    model_info = MODEL_PREDICTION_COLUMNS[selected_model_name]
    pred_col = model_info["pred_col"]
    match_col = model_info["match_col"]

    direction_col_map = {
        "Price-only Baseline": "Price_Only_Pred_Direction",
        "Model A": "Model_A_Pred_Direction",
        "Model B": "Model_B_Pred_Direction",
        "Model C": "Model_C_Pred_Direction",
    }

    pred_direction_col = direction_col_map[selected_model_name]

    preview_df = prediction_df.copy()
    preview_df["Date"] = pd.to_datetime(preview_df["Date"])

    if "Asset" in preview_df.columns:
        preview_df = preview_df[preview_df["Asset"] == asset_name].copy()

    preview_df = preview_df.sort_values("Date").tail(rows).copy()

    result_df = pd.DataFrame(
        {
            "기준일": preview_df["Date"].dt.strftime("%Y-%m-%d"),
            "실제 로그수익률": preview_df["Actual_Log_Return"].map(lambda x: f"{x:.4f}"),
            "예측 로그수익률": preview_df[pred_col].map(lambda x: f"{x:.4f}"),
            "실제 방향": preview_df["Actual_Direction"],
            "예측 방향": preview_df[pred_direction_col],
            "결과": preview_df[match_col].map(lambda x: "적중" if bool(x) else "실패"),
        }
    )

    return result_df
##########


#### export

def _render_prediction_workflow_image_card(image_path: Path) -> None:
    """
    예측 구조 도식화 이미지를 발표용 카드 형태로 표시한다.
    이미지는 카드 안에서 약 50% 너비로 축소해 가운데 정렬한다.
    """
    if not image_path.exists():
        st.info(
            f"{image_path.as_posix()} 파일이 없어 예측 구조 도식화를 표시하지 못했습니다."
        )
        return

    import base64

    image_bytes = image_path.read_bytes()
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")

    html_text = f"""
    <style>
    .workflow-card {{
        background-color: #fbf8f1;
        border: 1px solid #e3dacb;
        border-radius: 16px;
        padding: 18px 18px 16px 18px;
        margin: 8px 0 18px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
        font-family: "Malgun Gothic", "Apple SD Gothic Neo", "Noto Sans KR", sans-serif;
        box-sizing: border-box;
        text-align: center;
    }}

    .workflow-card-title {{
        font-size: 20px;
        font-weight: 800;
        color: #3d352d;
        margin-bottom: 12px;
        text-align: left;
    }}

    .workflow-card img {{
        width: 50%;
        max-width: 760px;
        min-width: 420px;
        height: auto;
        border-radius: 12px;
        display: block;
        margin: 0 auto;
    }}

    .workflow-card-caption {{
        margin-top: 12px;
        font-size: 14px;
        line-height: 1.6;
        color: #6e665e;
        text-align: left;
    }}
    </style>

    <div class="workflow-card">
        <div class="workflow-card-title">예측 구조 도식화</div>
        <img src="data:image/png;base64,{encoded_image}" />
        <div class="workflow-card-caption">
            입력 자료는 삼성전자·코스피·비트코인 패널 데이터이며,
            최종 출력은 삼성전자의 다음 주 로그수익률 방향 예측입니다.
        </div>
    </div>
    """

    components.html(html_text, height=700, scrolling=False)



######




### export
def _style_direction_match_table(df: pd.DataFrame):
    """
    방향성 적중표의 '결과' 열에만 조건부 스타일을 적용한다.
    적중: 연한 초록 배경 + 진한 초록 글자 + 굵게
    실패: 연한 빨강 배경 + 진한 빨강 글자 + 굵게
    """
    def style_result_cell(value):
        if str(value).strip() == "적중":
            return (
                "background-color: #eaf7ea; "
                "color: #166534; "
                "font-weight: 700;"
            )
        elif str(value).strip() == "실패":
            return (
                "background-color: #fdecec; "
                "color: #991b1b; "
                "font-weight: 700;"
            )
        return ""

    styler = (
        df.style
        .map(style_result_cell, subset=["결과"])
        .set_properties(subset=["결과"], **{
            "font-weight": "700",
            "text-align": "center",
        })
    )

    return styler

#############
def add_vertical_space(px: int = 48) -> None:
    st.markdown(
        f"<div style='height: {px}px;'></div>",
        unsafe_allow_html=True,
    )



def run() -> None:
    st.header("모델 구축 및 검증")
   

    st.info(
        "최종 예측 대상은 삼성전자입니다. "
        "삼성전자·KOSPI·Bitcoin의 현재 주봉 종가와 "
        "각 자산의 과거 1~5주 로그수익률을 공통 기본 학습자료로 사용합니다. "
        "KOSPI와 Bitcoin은 시장학습 보조자료이며, "
        "대시보드의 최근 예측 결과는 삼성전자 기준으로만 표시합니다."
    )

    

    st.caption(
        "심리 proxy를 넣지 않은 기준선과 Model A/B/C의 입력 구성을 비교합니다."
    )

    comparison_df = pd.DataFrame(
        [
            {
                "모델": "· Price-only",
                "공통 기본 입력": (
                    "삼성전자·KOSPI·Bitcoin 현재 주봉 종가\n"
                    "+ 각 자산의 과거 1~5주 로그수익률"
                ),
                "심리 proxy": "사용하지 않음",
                "모델별 추가 입력": "없음",
                "비교 목적": "심리 proxy가 없는 가격 기반 기준선",
            },
            {
                "모델": "· Model A",
                "공통 기본 입력": (
                    "삼성전자·KOSPI·Bitcoin 현재 주봉 종가\n"
                    "+ 각 자산의 과거 1~5주 로그수익률"
                ),
                "심리 proxy": "PC1 통합형",
                "모델별 추가 입력": "Investor_Sentiment_PC1",
                "비교 목적": "통합 심리 proxy의 추가 효과 확인",
            },
            {
                "모델": "· Model B",
                "공통 기본 입력": (
                    "삼성전자·KOSPI·Bitcoin 현재 주봉 종가\n"
                    "+ 각 자산의 과거 1~5주 로그수익률"
                ),
                "심리 proxy": "Residual 개별형",
                "모델별 추가 입력": (
                    "ATR_10_res\n"
                    "MFI_10_res\n"
                    "STOCHk_10_3_3_res"
                ),
                "비교 목적": "개별 심리 proxy 3개의 추가 효과 확인",
            },
            {
                "모델": "· Model C",
                "공통 기본 입력": (
                    "삼성전자·KOSPI·Bitcoin 현재 주봉 종가\n"
                    "+ 각 자산의 과거 1~5주 로그수익률"
                ),
                "심리 proxy": "PC1 + Residual 결합형",
                "모델별 추가 입력": (
                    "Investor_Sentiment_PC1\n"
                    "+ residual 3개"
                ),
                "비교 목적": "통합·개별 심리 proxy 결합 효과 확인",
            },
        ]
    )



    render_presentation_table(
        comparison_df,
        title="모델별 입력 구조",
        footnote=(
            "모든 모델은 삼성전자·KOSPI·Bitcoin의 현재 주봉 종가와 "
            "각 자산의 과거 1~5주 로그수익률을 공통 기본 입력으로 사용합니다. "
            "삼성전자는 최종 예측 대상이며, KOSPI와 Bitcoin은 삼성전자 주봉 기준일에 "
            "맞춰 정렬되는 시장학습 보조자료입니다. "
            "Model A/B/C는 심리 proxy의 구성만 달리하여 다음 주 삼성전자 "
            "로그수익률 방향 예측 성능을 비교합니다."
        ),
        left_align_cols=[
            "공통 기본 입력",
            "심리 proxy",
            "모델별 추가 입력",
            "비교 목적",
        ],
        height=600,
    )

        # ---------------------------------------------------------
    # 현재 동일 조건 백테스트 결과 시각화
    # ---------------------------------------------------------
    add_vertical_space(48)

    st.subheader("동일 조건 백테스트 성능 비교")
        # "아래 표와 그래프는 최신 train_pipeline.py에서 생성한 "
        # "`data/output/model_backtest_summary.csv`를 사용합니다. "
    st.caption(

        "삼성전자를 단일 예측 대상으로 두고, KOSPI와 Bitcoin을 "
        "시장학습 보조자료로 정렬한 동일 조건 실험 결과입니다. "
        "저장된 최적화 pkl 모델의 공식 DA와는 구분하여 확인합니다."
    )
       
    current_summary_path = Path(
        "data/output/model_backtest_summary.csv"
    )

    if current_summary_path.exists():
        performance_df = pd.read_csv(current_summary_path)

        core_model_order = [
            "Naive Persistence Baseline",
            "Price-only Baseline",
            "Model A (PCA 통합)",
            "Model B (세부 잔차)",
            "Model C (전체 혼용)",
        ]

        performance_df = performance_df[
            performance_df["Model"].isin(core_model_order)
        ].copy()

        performance_df["Model"] = pd.Categorical(
            performance_df["Model"],
            categories=core_model_order,
            ordered=True,
        )

        performance_df = (
            performance_df
            .sort_values("Model")
            .reset_index(drop=True)
        )

        performance_display_df = _format_performance_table(
            performance_df
        )

        render_presentation_table(
            performance_display_df,
            title="Naive·Price-only·Model A/B/C 성능 비교",
            footnote=(
                "Naive Persistence는 이번 주 삼성전자 로그수익률의 방향을 "
                "다음 주에도 유지한다고 가정한 단순 기준선입니다. "
                "Price-only와 Model A/B/C는 삼성전자·KOSPI·Bitcoin의 "
                "현재 주봉 종가와 각 자산의 과거 1~5주 로그수익률을 "
                "공통 기본 학습자료로 사용합니다. "
                "DA는 삼성전자 다음 주 실제 방향과 예측 방향이 일치한 비율입니다."
            ),
            left_align_cols=["모델"],
            height=700,
        )

        left, center, right = st.columns(
            [0.15, 0.70, 0.15]
        )

        with center:
            st.pyplot(
                plot_directional_accuracy_bar(
                    performance_df
                ),
                use_container_width=True,
            )

        st.info(
            "참고: 저장된 최적화 pkl 모델의 공식 백테스트 DA는 "
            "Model A 52.94%, Model B 56.30%, Model C 55.46%입니다. "
            "위 표의 수치는 최신 훈련-프로세스의 동일 조건으로 다시 실행하여 "
            "생성한 비교 실험 결과이므로 두 결과를 동일한 값으로 합치지 않습니다."
        )

    else:
        st.info(
            "`data/output/model_backtest_summary.csv`가 없어 "
            "동일 조건 백테스트 성능표를 표시하지 못했습니다."
        )

    # ---------------------------------------------------------
    # 삼성전자 단일 타깃 방향성 적중 시각화
    # ---------------------------------------------------------
    prediction_path = Path(
        "data/output/model_predictions.csv"
    )

    if prediction_path.exists():
        prediction_df = pd.read_csv(
            prediction_path
        )
    else:
        prediction_df = None

    if prediction_df is not None:
        samsung_prediction_df = prediction_df.copy()

        # 과거 CSV가 여러 자산을 포함한 경우에도
        # 삼성전자 행만 남겨 단일 타깃 원칙을 유지한다.
        if "Asset" in samsung_prediction_df.columns:
            samsung_prediction_df = samsung_prediction_df[
                samsung_prediction_df["Asset"] == "삼성전자"
            ].copy()

        if samsung_prediction_df.empty:
            st.warning(
                "`data/output/model_predictions.csv`에 "
                "삼성전자 예측 행이 없어 적중 시각화를 표시하지 못했습니다."
            )
        else:
            add_vertical_space(74)

            st.subheader("삼성전자 테스트 구간 방향성 적중 확인")

            st.caption(
                "아래 표는 삼성전자 테스트 구간의 실제 로그수익률 방향과 "
                "예측 방향이 일치했는지를 보여줍니다. "
                "KOSPI와 Bitcoin은 시장학습 보조자료이며, "
                "이 화면의 예측 타깃에는 포함하지 않습니다."
            )

            _render_prediction_workflow_image_card(
                PREDICTION_WORKFLOW_IMAGE
            )

            selected_plot_model = st.radio(
                "시각화할 예측 모델",
                options=[
                    "Price-only Baseline",
                    "Model A",
                    "Model B",
                    "Model C",
                ],
                index=2,
                horizontal=True,
                key="csv_visual_model",
            )

            model_info = MODEL_PREDICTION_COLUMNS[
                selected_plot_model
            ]

            direction_preview_df = (
                _build_direction_match_preview(
                    samsung_prediction_df,
                    asset_name="삼성전자",
                    selected_model_name=selected_plot_model,
                    rows=20,
                )
            )

            render_presentation_table(
                direction_preview_df,
                title="삼성전자 최근 방향성 적중표",
                footnote=(
                    "실제 로그수익률과 예측 로그수익률의 부호가 일치하면 "
                    "적중으로 표시합니다. 이 표는 Directional Accuracy의 "
                    "계산 방식을 직관적으로 설명하기 위한 보조 자료입니다."
                ),
                left_align_cols=[
                    "기준일",
                    "실제 방향",
                    "예측 방향",
                ],
                cell_style_rules={
                    "결과": {
                        "적중": (
                            "background-color:#eaf7ea !important; "
                            "color:#166534; "
                            "font-weight:800;"
                        ),
                        "실패": (
                            "background-color:#fdecec !important; "
                            "color:#991b1b; "
                            "font-weight:800;"
                        ),
                    }
                },
            )

            with st.expander(
                "보조자료: 실제값 vs 예측값 시계열 보기",
                expanded=False,
            ):
                st.caption(
                    "삼성전자 테스트 구간의 실제 로그수익률과 "
                    "예측 로그수익률 흐름을 비교합니다. "
                    "수익률 시계열의 노이즈가 크므로 발표에서는 "
                    "보조자료로 사용합니다."
                )

                st.pyplot(
                    plot_actual_vs_prediction_series(
                        samsung_prediction_df,
                        asset_name="삼성전자",
                        model_col=model_info["pred_col"],
                        model_label=model_info["label"],
                    ),
                    use_container_width=True,
                )

            with st.expander(
                "보조자료: 실제값-예측값 산점도 보기",
                expanded=False,
            ):
                st.caption(
                    "삼성전자 실제 로그수익률과 예측 로그수익률의 "
                    "분포 및 편향을 확인하기 위한 보조자료입니다."
                )

                st.pyplot(
                    plot_actual_vs_predicted_scatter(
                        samsung_prediction_df,
                        asset_name="삼성전자",
                        model_col=model_info["pred_col"],
                        model_label=model_info["label"],
                    ),
                    use_container_width=True,
                )

    else:
        st.info(
            "`data/output/model_predictions.csv`가 없어 "
            "삼성전자 방향성 적중표와 보조 시각화를 표시하지 못했습니다."
        )

    # ---------------------------------------------------------
    # 저장 모델 파일 확인 및 최근 1회 예측
    # ---------------------------------------------------------

    missing = validate_model_files()
    if missing:
        st.error(f"필요한 모델 파일이 없습니다: {', '.join(missing)}")
        st.info("먼저 모델 학습 스크립트를 실행해 `models/` 폴더에 pkl 파일을 생성하세요.")
        return

    add_vertical_space(48)

    model_choice = st.radio(
        "사용할 저장 모델",
        options=["A", "B", "C"],
        horizontal=True,
        format_func=lambda value: {
            "A": "Model A (PCA 심리지수)",
            "B": "Model B (잔차 3개)",
            "C": "Model C (PCA + 잔차)",
        }[value],
    )

    selected_info = MODEL_DESCRIPTIONS[model_choice]

    st.markdown(f"### {selected_info['name']} · {selected_info['label']}")

    feature_col, purpose_col = st.columns(2)

    with feature_col:
        st.info(
            f"**사용 feature**\n\n"
            f"{selected_info['features']}"
        )

    with purpose_col:
        st.info(
            f"**비교 목적**\n\n"
            f"{selected_info['purpose']}"
        )

    st.caption(
        "모델 선택을 변경한 뒤에는 아래의 '저장된 모델로 예측 실행' 버튼을 다시 눌러야 "
        "선택 모델 기준의 예측 결과가 화면에 반영됩니다."
    )

    # 최종 예측 대상은 삼성전자로 고정한다.
    asset_name = "삼성전자"

    if st.button("저장된 모델로 예측 실행", type="primary"):
        with st.spinner(f"{asset_name} 최근 데이터를 준비하고 Model {model_choice}로 예측하는 중입니다."):
            predictor = load_predictor()
            raw_data = load_input_data(asset_name)
            st.session_state.prediction_result = predictor.get_prediction(
                asset_name,
                raw_data,
                model_type=model_choice,
            )
            st.session_state.prediction_model_choice = model_choice

    result = st.session_state.get("prediction_result")
    selected_model = st.session_state.get("prediction_model_choice", model_choice)
    if result is None:
        st.info("저장된 모델로 예측 실행 버튼을 누르세요.")
        return

    result_info = MODEL_DESCRIPTIONS[selected_model]

    direction_label = "상승" if result["direction"] == "UP" else "하락"
    direction_delta = result["pred_log_return"] * 100

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "사용 모델",
        f"Model {selected_model}",
    )

    col2.metric(
        "저장 모델 공식 DA",
        f"{BACKTEST_DA[selected_model]:.2f}%",
    )

    col3.metric(
        "다음 주 예측 방향",
        direction_label,
        delta=f"{direction_delta:.2f}%",
    )

    st.caption(
        "저장 모델 공식 DA는 보성님이 최적화하여 저장한 pkl 모델의 "
        "공식 백테스트 결과입니다. "
        "위의 동일 조건 재실험 CSV에서 계산된 DA와는 실험 시점과 조건이 다르므로 "
        "별도의 결과로 구분하여 표시합니다."
    )

    st.subheader("최근 예측 입력값")
    st.caption(
        f"아래 표는 {result_info['name']}에 실제로 입력되는 feature만 추려 정리한 것입니다. "
        "모델 선택을 바꾸면 표시되는 입력 feature도 함께 달라집니다."
    )
    expected_feature_count = result.get(
        "expected_feature_count"
    )

    actual_feature_count = len(
        result["current_data"]
    )

    st.success(
        f"Model {selected_model} 입력 확인: "
        f"{actual_feature_count}개 feature"
    )

    if expected_feature_count != actual_feature_count:
        st.error(
            "저장 모델이 요구하는 feature 수와 "
            "실제 입력 feature 수가 일치하지 않습니다."
        )
        return




    input_preview_df = _build_model_input_preview(
        result["current_data"],
        selected_model,
    )

    render_presentation_table(
        input_preview_df,
        title=f"{result_info['name']} 예측 입력값",
        footnote=(
            "저장된 pkl 모델의 feature 구성에 맞춰 최근 주봉 기준 입력값을 표시합니다."
        ),
        left_align_cols=["입력 feature"],
    )

    df_plot = result["df_plot"]

    date_summary_df = _build_prediction_date_summary(df_plot)

    render_presentation_table(
        date_summary_df,
        title="예측 기준일 요약",
        footnote=(
            "오늘의 실행 일자와 모델 입력에 사용된 최근 주봉 기준일, "
            "그리고 다음 주 로그수익률 예측 대상 시점을 함께 표시합니다."
        ),
        left_align_cols=["구분", "의미"],
   
    )

    st.subheader("다음 주 예측 결과 요약")

    pred_log_return = result["pred_log_return"]
    pred_return_pct = pred_log_return * 100
    direction_text = direction_label

    prediction_summary_df = pd.DataFrame(
        [
            {
                "항목": "예측 대상",
                "값": f"{asset_name} 다음 주 주봉 로그수익률",
                "해석": "저장된 XGBoost 모델이 예측한 다음 주 수익률 방향입니다.",
            },
            {
                "항목": "예측 로그수익률",
                "값": f"{pred_log_return:.5f}",
                "해석": "모델이 산출한 다음 주 로그수익률 예측값입니다.",
            },
            {
                "항목": "예측 수익률(%)",
                "값": f"{pred_return_pct:.5f}%",
                "해석": "로그수익률 예측값을 백분율로 환산한 참고값입니다.",
            },
            {
                "항목": "예측 방향",
                "값": direction_text,
                "해석": "예측 로그수익률이 0보다 크면 상승, 0보다 작으면 하락으로 해석합니다.",
            },
            {
                "항목": "사용 모델",
                "값": f"Model {selected_model}",
                "해석": result_info["purpose"],
            },
        ]
    )

    render_presentation_table(
        prediction_summary_df,
        title="다음 주 예측 결과 요약",
        footnote=(
            "현재 화면은 저장된 pkl 모델이 산출한 최근 1회 예측 결과를 요약합니다. "
            "전체 테스트 구간의 방향성 적중표와 실제값-예측값 보조 시각화는 위의 CSV 기반 백테스트 영역에서 확인할 수 있습니다."
        ),
        left_align_cols=["항목", "해석"],
        height=520,
    )
