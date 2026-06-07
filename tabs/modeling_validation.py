from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from modules.predictor import QuantPredictor
from visualization import (
    plot_actual_vs_predicted_scatter,
    plot_actual_vs_prediction_series,
    plot_directional_accuracy_bar,
)
from utils.table_utils import render_presentation_table     
import streamlit.components.v1 as components
import numpy as np
import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DISPLAY_NAME_MAP = {
    "Zero-return Baseline": "Zero-return Baseline",
    "Train-mean Baseline": "Train-mean Baseline",
    "Price-only Baseline": "Price-only",
    "Model B-ATR (ATR 단일 심리지표)": "Model A-1 (ATR)",
    "Model B-MFI (MFI 단일 심리지표)": "Model A-2 (MFI)",
    "Model B-Stochastic (Stochastic 단일 심리지표)": (
        "Model A-3 (Stochastic)"
    ),
    "Model A (PCA 통합)": "Model B (PC1)",
    "Model B (세부 잔차)": "Model C (잔차 3개)",
    "Model C (전체 혼용)": "Model D (PC1 + 잔차 3개)",
}

MODEL_DISPLAY_ORDER = [
    "Zero-return Baseline",
    "Train-mean Baseline",
    "Price-only",
    "Model A-1 (ATR)",
    "Model A-2 (MFI)",
    "Model A-3 (Stochastic)",
    "Model B (PC1)",
    "Model C (잔차 3개)",
    "Model D (PC1 + 잔차 3개)",
]

BACKTEST_SUMMARY_PATH = (
    PROJECT_ROOT / "model_backtest_summary.csv"
)

PERIOD_BACKTEST_PATH = (
    PROJECT_ROOT / "model_period_backtest.csv"
)

PREDICTION_PATH = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "model_predictions.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"

PREDICTION_WORKFLOW_IMAGE = (
    OUTPUT_DIR / "model_predicts_workflow.png"
)

PREDICTOR_MODEL_TYPE_MAP = {
    "B": "A",
    "C": "B",
    "D": "C",
}

MODEL_FILES = {
    "Model B": (
        PROJECT_ROOT
        / "models"
        / "best_xgboost_panel_model_A.pkl"
    ),
    "Model C": (
        PROJECT_ROOT
        / "models"
        / "best_xgboost_panel_model_B.pkl"
    ),
    "Model D": (
        PROJECT_ROOT
        / "models"
        / "best_xgboost_panel_model_C.pkl"
    ),
    "Scaler": (
        PROJECT_ROOT
        / "models"
        / "scaler.pkl"
    ),
    "PCA": (
        PROJECT_ROOT
        / "models"
        / "pca.pkl"
    ),
}

BACKTEST_DA = {
    "B": 52.94,
    "C": 56.30,
    "D": 55.46,
}


MODEL_DESCRIPTIONS = {
    "B": {
        "name": "Model B",
        "label": "PC1 통합형",
        "features": (
            "공통 가격 feature 18개 "
            "+ Investor_Sentiment_PC1"
        ),
        "purpose": (
            "ATR·MFI·Stochastic residual의 공통 성분을 "
            "PC1 하나로 압축했을 때의 효과 확인"
        ),
    },
    "C": {
        "name": "Model C",
        "label": "Residual 개별형",
        "features": (
            "공통 가격 feature 18개 "
            "+ ATR·MFI·Stochastic residual 3개"
        ),
        "purpose": (
            "개별 심리 proxy 정보를 압축하지 않고 "
            "각각 유지했을 때의 효과 확인"
        ),
    },
    "D": {
        "name": "Model D",
        "label": "PC1 + Residual 결합형",
        "features": (
            "공통 가격 feature 18개 "
            "+ PC1 + residual 3개"
        ),
        "purpose": (
            "통합 심리 proxy와 개별 residual을 "
            "함께 사용했을 때의 효과 확인"
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
    "B": [
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
        "Investor_Sentiment_PC1",
    ],
    "C": [
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
        "ATR_10_res",
        "MFI_10_res",
        "STOCHk_10_3_3_res",
    ],
    "D": [
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
        "Investor_Sentiment_PC1",
        "ATR_10_res",
        "MFI_10_res",
        "STOCHk_10_3_3_res",
    ],
}

# 기존 model_predictions.csv 내부 열 이름과 화면 표시명 매핑
MODEL_PREDICTION_COLUMNS = {
    "Price-only Baseline": {
        "pred_col": "Price_Only_Pred_Log_Return",
        "label": "Price-only",
        "match_col": "Price_Only_Match",
        "display_name": "Price-only",
    },
    "Model A": {
        "pred_col": "Model_A_Pred_Log_Return",
        "label": "Model B (PC1)",
        "match_col": "Model_A_Match",
        "display_name": "Model B (PC1)",
    },
    "Model B": {
        "pred_col": "Model_B_Pred_Log_Return",
        "label": "Model C (잔차 3개)",
        "match_col": "Model_B_Match",
        "display_name": "Model C (잔차 3개)",
    },
    "Model C": {
        "pred_col": "Model_C_Pred_Log_Return",
        "label": "Model D (PC1 + 잔차 3개)",
        "match_col": "Model_C_Match",
        "display_name": "Model D (PC1 + 잔차 3개)",
    },
}

#######





@st.cache_resource
def load_predictor() -> QuantPredictor:
    return QuantPredictor()


def validate_model_files() -> list[str]:
    return [label for label, path in MODEL_FILES.items() if not path.exists()]


def _build_model_input_preview(current_data, model_choice: str) -> pd.DataFrame:
    """
    선택한 저장 모델 B/C/D에 실제로 들어가는 feature만 추려서
    발표용 입력값 표로 구성한다.

    숫자형 입력값은 소수점 4자리까지 표시한다.
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

def _plot_next_week_log_return_forecast(
    df_plot: pd.DataFrame,
    pred_log_return: float,
    asset_name: str,
    model_label: str,
):
    """최근 실제 주봉 로그수익률과 다음 주 단일 예측값을 표시한다."""
    plot_df = df_plot.copy().sort_index()

    if "Log_Return" in plot_df.columns:
        plot_df["Actual_Log_Return"] = pd.to_numeric(
            plot_df["Log_Return"], errors="coerce"
        )
    elif "Samsung_Log_Return" in plot_df.columns:
        plot_df["Actual_Log_Return"] = pd.to_numeric(
            plot_df["Samsung_Log_Return"], errors="coerce"
        )
    elif "Samsung_Close" in plot_df.columns:
        samsung_close = pd.to_numeric(
            plot_df["Samsung_Close"], errors="coerce"
        )
        plot_df["Actual_Log_Return"] = np.log(
            samsung_close / samsung_close.shift(1)
        )
    else:
        plot_df["Actual_Log_Return"] = np.nan

    plot_df = plot_df.dropna(subset=["Actual_Log_Return"]).tail(52)

    if plot_df.empty:
        return None

    last_date = plot_df.index.max()
    next_date = (
        last_date + pd.Timedelta(days=7)
        if hasattr(last_date, "to_pydatetime")
        else len(plot_df)
    )

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(
        plot_df.index,
        plot_df["Actual_Log_Return"],
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
        ).map(lambda value: "-" if pd.isna(value) else f"{value:.4f}%")

    if "feature 수" in display_df.columns:
        display_df["feature 수"] = pd.to_numeric(
            display_df["feature 수"],
            errors="coerce",
        ).map(lambda value: "-" if pd.isna(value) else f"{int(value)}개")


    percentage_columns = [
        "Up_DA",
        "Down_DA",
        "Pred_Up_Ratio",
    ]
    for col in percentage_columns:
        if col in display_df.columns:
            display_df[col] = pd.to_numeric(
                display_df[col],
                errors="coerce",
            ).map(
                lambda value: (
                    "-"
                    if pd.isna(value)
                    else f"{value:.4f}%"
                )
            )

    if "feature 수" in display_df.columns:
        display_df["feature 수"] = pd.to_numeric(
        display_df["feature 수"],
        errors="coerce",
    ).map(
        lambda value: (
            "-"
            if pd.isna(value)
            else f"{int(value)}개"
        )
    )
    return display_df



# 모델의 입력 feature 
# Feature importance 표와 그룸 요약 함수 추가 
def _build_feature_importance_df(
    model_choice: str,
) -> pd.DataFrame:
    """
    저장된 XGBoost 모델의 gain 기반 feature importance를
    발표용 표 형태로 구성한다.
    """
    model_label = f"Model {model_choice}"
    model_path = MODEL_FILES[model_label]
    model = joblib.load(model_path)

    importance_df = pd.DataFrame(
        {
            "Feature": model.feature_names_in_,
            "Importance": model.feature_importances_,
        }
    )

    def classify_feature(feature_name: str) -> str:
        if feature_name == "Investor_Sentiment_PC1":
            return "통합 심리 proxy"

        if feature_name in {
            "ATR_10_res",
            "MFI_10_res",
            "STOCHk_10_3_3_res",
        }:
            return "개별 심리 proxy"

        if feature_name.startswith("Samsung_"):
            return "삼성전자 가격 입력"

        if feature_name.startswith("KOSPI_"):
            return "KOSPI 보조 입력"

        if feature_name.startswith("Bitcoin_"):
            return "Bitcoin 보조 입력"

        return "기타"

    importance_df["Feature_Group"] = importance_df[
        "Feature"
    ].apply(classify_feature)

    importance_df["Importance_Percent"] = (
        importance_df["Importance"] * 100
    )

    importance_df["Used"] = importance_df[
        "Importance"
    ].map(
        lambda value: (
            "사용됨"
            if value > 0
            else "사용되지 않음"
        )
    )

    return importance_df.sort_values(
        "Importance",
        ascending=False,
    ).reset_index(drop=True)


def _build_feature_group_summary(
    importance_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    feature importance를 역할별로 합산한다.
    """
    summary_df = (
        importance_df
        .groupby(
            "Feature_Group",
            as_index=False,
        )["Importance_Percent"]
        .sum()
        .sort_values(
            "Importance_Percent",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    summary_df["Importance_Percent"] = summary_df[
        "Importance_Percent"
    ].map(
        lambda value: f"{value:.2f}%"
    )

    summary_df = summary_df.rename(
        columns={
            "Feature_Group": "입력 그룹",
            "Importance_Percent": "상대 중요도",
        }
    )

    return summary_df


def _format_feature_importance_table(
    importance_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    개별 feature importance를 대시보드 표시용으로 변환한다.
    """
    display_df = importance_df[
        [
            "Feature",
            "Feature_Group",
            "Importance_Percent",
            "Used",
        ]
    ].copy()

    display_df["Importance_Percent"] = display_df[
        "Importance_Percent"
    ].map(
        lambda value: f"{value:.2f}%"
    )

    return display_df.rename(
        columns={
            "Feature": "Feature",
            "Feature_Group": "입력 구분",
            "Importance_Percent": "상대 중요도",
            "Used": "모델 사용 여부",
        }
    )

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
def _build_live_official_sentiment_table(
    result: dict,
) -> pd.DataFrame | None:
    """
    실시간 계산값과 보성님 공식 CSV의 최신값을 나란히 표시한다.
    두 값은 서로를 대체하지 않으며, 기준일도 함께 표출한다.
    """
    live_info = result.get(
        "live_sentiment_latest"
    )
    official_info = result.get(
        "official_sentiment_latest"
    )

    if not live_info or not official_info:
        return None

    live_values = live_info.get(
        "values",
        {}
    )
    official_values = official_info.get(
        "values",
        {}
    )

    feature_labels = {
        "ATR_10_res": "ATR residual",
        "MFI_10_res": "MFI residual",
        "STOCHk_10_3_3_res": "Stochastic residual",
        "Investor_Sentiment_PC1": "Investor Sentiment PC1",
    }

    rows = []

    for feature_name, display_name in feature_labels.items():
        if (
            feature_name not in live_values
            or feature_name not in official_values
        ):
            continue

        rows.append(
            {
                "항목": display_name,
                "실시간 계산값": f"{live_values[feature_name]:.6f}",
                "실시간 기준일": pd.Timestamp(
                    live_info["date"]
                ).strftime("%Y-%m-%d"),
                "보성님 CSV 값": f"{official_values[feature_name]:.6f}",
                "CSV 기준일": pd.Timestamp(
                    official_info["date"]
                ).strftime("%Y-%m-%d"),
            }
        )

    if not rows:
        return None

    return pd.DataFrame(rows)


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
        "각 자산의 과거 1~5주 로그수익률을 공통 가격 feature로 사용합니다. "
        "KOSPI와 Bitcoin은 시장 흐름을 반영하는 외생 보조 입력자료입니다."
    )

    st.caption(
        "Price-only 기준모델, 개별 residual 단독 모델 A-1~A-3, "
        "그리고 PC1·residual 구성에 따른 Model B/C/D를 비교합니다."
    )

    common_price_input = (
        "삼성전자·KOSPI·Bitcoin 현재 주봉 종가\n"
        "+ 각 자산의 과거 1~5주 로그수익률"
    )

    comparison_df = pd.DataFrame(
        [
            {
                "모델": "Price-only",
                "공통 가격 입력": common_price_input,
                "심리 proxy 구성": "사용하지 않음",
                "추가 feature": "없음",
                "총 feature 수": 18,
                "비교 목적": "심리 proxy가 없는 가격 기준모델",
            },
            {
                "모델": "Model A-1 (ATR)",
                "공통 가격 입력": common_price_input,
                "심리 proxy 구성": "개별 residual 단독형",
                "추가 feature": "ATR_10_res",
                "총 feature 수": 19,
                "비교 목적": "ATR residual의 단독 예측효과 확인",
            },
            {
                "모델": "Model A-2 (MFI)",
                "공통 가격 입력": common_price_input,
                "심리 proxy 구성": "개별 residual 단독형",
                "추가 feature": "MFI_10_res",
                "총 feature 수": 19,
                "비교 목적": "MFI residual의 단독 예측효과 확인",
            },
            {
                "모델": "Model A-3 (Stochastic)",
                "공통 가격 입력": common_price_input,
                "심리 proxy 구성": "개별 residual 단독형",
                "추가 feature": "STOCHk_10_3_3_res",
                "총 feature 수": 19,
                "비교 목적": "Stochastic residual의 단독 예측효과 확인",
            },
            {
                "모델": "Model B (PC1)",
                "공통 가격 입력": common_price_input,
                "심리 proxy 구성": "PC1 통합형",
                "추가 feature": "Investor_Sentiment_PC1",
                "총 feature 수": 19,
                "비교 목적": "residual 3개의 공통 성분을 압축한 효과 확인",
            },
            {
                "모델": "Model C (잔차 3개)",
                "공통 가격 입력": common_price_input,
                "심리 proxy 구성": "Residual 개별형",
                "추가 feature": (
                    "ATR_10_res\nMFI_10_res\nSTOCHk_10_3_3_res"
                ),
                "총 feature 수": 21,
                "비교 목적": "개별 residual 정보를 모두 유지한 효과 확인",
            },
            {
                "모델": "Model D (PC1 + 잔차 3개)",
                "공통 가격 입력": common_price_input,
                "심리 proxy 구성": "PC1 + Residual 결합형",
                "추가 feature": (
                    "Investor_Sentiment_PC1\n"
                    "+ ATR_10_res\n+ MFI_10_res\n+ STOCHk_10_3_3_res"
                ),
                "총 feature 수": 22,
                "비교 목적": "통합형과 개별형을 함께 사용한 효과 확인",
            },
        ]
    )

    render_presentation_table(
        comparison_df,
        title="모델별 입력 구조",
        footnote=(
            "모든 모델은 현재 주봉 종가 3개와 과거 1~5주 로그수익률 "
            "15개를 합친 공통 가격 feature 18개를 사용합니다. "
            "Model A-1~A-3은 개별 residual의 단독 효과를 확인하는 "
            "추가 진단모델입니다."
        ),
        left_align_cols=[
            "모델",
            "공통 가격 입력",
            "심리 proxy 구성",
            "추가 feature",
            "비교 목적",
        ],
        height=1000,
    )

    # 동일 조건 백테스트 성능 비교
    add_vertical_space(48)
    st.subheader("동일 조건 백테스트 성능 비교")
    st.caption(
        "프로젝트 루트의 `model_backtest_summary.csv`를 사용합니다. "
        "모든 결과는 동일한 58주 테스트 구간에서 계산됐습니다."
    )

    if BACKTEST_SUMMARY_PATH.exists():
        performance_df = pd.read_csv(BACKTEST_SUMMARY_PATH)
        performance_df["Model"] = performance_df["Model"].replace(
            MODEL_DISPLAY_NAME_MAP
        )
        performance_df["Model_Order"] = pd.Categorical(
            performance_df["Model"],
            categories=MODEL_DISPLAY_ORDER,
            ordered=True,
        )
        performance_df = (
            performance_df.sort_values("Model_Order")
            .drop(columns=["Model_Order"])
            .reset_index(drop=True)
        )

        performance_display_df = _format_performance_table(performance_df)
        render_presentation_table(
            performance_display_df,
            title="기준선 및 Model A-1~D 성능 비교",
            footnote=(
                "Zero-return·Train-mean은 단순 비교 기준선입니다. "
                "Price-only와 Model A-1~D는 실제 실험모델입니다. "
                "DA는 다음 주 실제 방향과 예측 방향이 일치한 비율입니다."
            ),
            left_align_cols=["모델"],
            height=920,
        )

        left, center, right = st.columns([0.08, 0.84, 0.08])
        with center:
            st.pyplot(
                plot_directional_accuracy_bar(performance_df),
                use_container_width=True,
            )

        st.info(
            "Model A-1(ATR)과 Model A-2(MFI)는 각각 36/58로 "
            "DA 62.0690%를 기록했습니다. Price-only와 Model A-3·B·C·D는 "
            "각각 35/58로 DA 60.3448%였습니다. 한 건 차이는 약 1.7241%p입니다."
        )
    else:
        st.error(f"성능 CSV를 찾지 못했습니다: {BACKTEST_SUMMARY_PATH}")

    # 기간별 백테스트
    if PERIOD_BACKTEST_PATH.exists():
        with st.expander("기간별 방향 정확도 보기", expanded=False):
            period_df = pd.read_csv(PERIOD_BACKTEST_PATH)
            period_df["Model"] = period_df["Model"].replace(
                MODEL_DISPLAY_NAME_MAP
            )
            period_df["Model_Order"] = pd.Categorical(
                period_df["Model"],
                categories=MODEL_DISPLAY_ORDER,
                ordered=True,
            )
            period_df = (
                period_df.sort_values(["Period", "Model_Order"])
                .drop(columns=["Model_Order"])
                .reset_index(drop=True)
            )
            period_display_df = period_df.copy()
            for col in ["DA", "Up_DA", "Down_DA", "Pred_Up_Ratio"]:
                if col in period_display_df.columns:
                    period_display_df[col] = pd.to_numeric(
                        period_display_df[col], errors="coerce"
                    ).map(
                        lambda value: "-" if pd.isna(value) else f"{value:.4f}%"
                    )
            render_presentation_table(
                period_display_df,
                title="기간별 모델 방향 정확도",
                footnote=(
                    "전체 DA가 같더라도 기간별 적중 양상과 상승·하락 예측 성향은 "
                    "다를 수 있습니다."
                ),
                left_align_cols=["Period", "Model"],
                height=1200,
            )

    # 과거 테스트 구간 실제값·예측값 연결
    if PREDICTION_PATH.exists():
        prediction_df = pd.read_csv(PREDICTION_PATH)
        if "Asset" in prediction_df.columns:
            prediction_df = prediction_df[
                prediction_df["Asset"] == "삼성전자"
            ].copy()

        if not prediction_df.empty:
            add_vertical_space(64)
            st.subheader("삼성전자 테스트 구간 방향성 적중 확인")
            st.caption(
                "이 영역은 기존 `model_predictions.csv`의 실제 열 이름을 유지하면서, "
                "화면에서는 새 모델명 B/C/D로 표시합니다."
            )

            _render_prediction_workflow_image_card(
                PREDICTION_WORKFLOW_IMAGE
            )

            selected_plot_model = st.radio(
                "시각화할 예측 모델",
                options=list(MODEL_PREDICTION_COLUMNS),
                index=2,
                horizontal=True,
                key="csv_visual_model",
                format_func=lambda key: MODEL_PREDICTION_COLUMNS[key][
                    "display_name"
                ],
            )
            model_info = MODEL_PREDICTION_COLUMNS[selected_plot_model]

            direction_preview_df = _build_direction_match_preview(
                prediction_df,
                asset_name="삼성전자",
                selected_model_name=selected_plot_model,
                rows=20,
            )
            render_presentation_table(
                direction_preview_df,
                title="삼성전자 최근 방향성 적중표",
                footnote=(
                    "실제 로그수익률과 예측 로그수익률의 부호가 일치하면 "
                    "적중으로 표시합니다."
                ),
                left_align_cols=["기준일", "실제 방향", "예측 방향"],
                cell_style_rules={
                    "결과": {
                        "적중": (
                            "background-color:#eaf7ea !important; "
                            "color:#166534; font-weight:800;"
                        ),
                        "실패": (
                            "background-color:#fdecec !important; "
                            "color:#991b1b; font-weight:800;"
                        ),
                    }
                },
                height=900,
            )

            with st.expander(
                "실제값 vs 예측값 시계열 보기", expanded=False
            ):
                st.pyplot(
                    plot_actual_vs_prediction_series(
                        prediction_df,
                        asset_name="삼성전자",
                        model_col=model_info["pred_col"],
                        model_label=model_info["label"],
                    ),
                    use_container_width=True,
                )

            with st.expander(
                "실제값-예측값 산점도 보기", expanded=False
            ):
                st.pyplot(
                    plot_actual_vs_predicted_scatter(
                        prediction_df,
                        asset_name="삼성전자",
                        model_col=model_info["pred_col"],
                        model_label=model_info["label"],
                    ),
                    use_container_width=True,
                )
        else:
            st.warning("예측 CSV에 삼성전자 행이 없습니다.")
    else:
        st.info(f"예측 시계열 CSV를 찾지 못했습니다: {PREDICTION_PATH}")

    # 저장 모델 B/C/D 중요도와 최근 예측
    missing = validate_model_files()
    if missing:
        st.error(f"필요한 모델 파일이 없습니다: {', '.join(missing)}")
        return

    add_vertical_space(56)
    st.subheader("저장 모델 중요도 및 최근 1회 예측")

    model_choice = st.radio(
        "사용할 저장 모델",
        options=["B", "C", "D"],
        horizontal=True,
        format_func=lambda value: {
            "B": "Model B (PC1 통합형)",
            "C": "Model C (잔차 3개)",
            "D": "Model D (PC1 + 잔차 3개)",
        }[value],
        key="saved_model_choice",
    )

    selected_info = MODEL_DESCRIPTIONS[model_choice]
    st.markdown(
        f"### {selected_info['name']} · {selected_info['label']}"
    )
    feature_col, purpose_col = st.columns(2)
    with feature_col:
        st.info(f"**사용 feature**\n\n{selected_info['features']}")
    with purpose_col:
        st.info(f"**비교 목적**\n\n{selected_info['purpose']}")

    importance_df = _build_feature_importance_df(model_choice)
    group_summary_df = _build_feature_group_summary(importance_df)
    importance_display_df = _format_feature_importance_table(importance_df)

    add_vertical_space(24)
    st.subheader(f"Model {model_choice} 입력 feature 중요도")
    st.caption(
        "gain 기반 중요도는 XGBoost 트리 분할에서 각 feature가 오차 감소에 "
        "기여한 상대적 비중입니다. 예측값 자체를 분해한 비율은 아닙니다."
    )
    render_presentation_table(
        group_summary_df,
        title="입력 그룹별 상대 중요도",
        footnote=(
            "삼성전자 가격 입력, KOSPI·Bitcoin 보조 입력, 심리 proxy의 "
            "gain 기반 중요도를 그룹별로 합산했습니다."
        ),
        left_align_cols=["입력 그룹"],
        height=380,
    )

    with st.expander("개별 feature 중요도 상세 보기", expanded=False):
        render_presentation_table(
            importance_display_df,
            title=f"Model {model_choice} 개별 feature 중요도",
            footnote=(
                "중요도가 0인 feature는 현재 저장된 XGBoost 트리 분할에 "
                "사용되지 않은 변수입니다."
            ),
            left_align_cols=["Feature", "입력 구분", "모델 사용 여부"],
            height=1180,
        )

    st.caption(
        "모델 선택을 변경한 뒤에는 예측 실행 버튼을 다시 눌러야 "
        "선택 모델의 결과가 반영됩니다."
    )

    if st.button("저장된 모델로 예측 실행", type="primary"):
        with st.spinner(
            f"삼성전자 최근 시장 데이터를 준비하고 Model {model_choice}로 "
            "예측하는 중입니다."
        ):
            predictor = load_predictor()
            internal_model_type = PREDICTOR_MODEL_TYPE_MAP[model_choice]
            st.session_state.prediction_result = predictor.get_prediction(
                model_type=internal_model_type
            )
            st.session_state.prediction_model_choice = model_choice

    stored_model = st.session_state.get("prediction_model_choice")
    if stored_model not in MODEL_DESCRIPTIONS:
        st.session_state.pop("prediction_result", None)
        st.session_state.pop("prediction_model_choice", None)

    result = st.session_state.get("prediction_result")
    selected_model = st.session_state.get(
        "prediction_model_choice", model_choice
    )
    if result is None:
        st.info("저장된 모델로 예측 실행 버튼을 누르세요.")
        return

    result_info = MODEL_DESCRIPTIONS[selected_model]
    direction_label = "상승" if result["direction"] == "UP" else "하락"
    direction_delta = result["pred_log_return"] * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("사용 모델", f"Model {selected_model}")
    col2.metric(
        "저장 모델 공식 DA",
        f"{BACKTEST_DA[selected_model]:.2f}%",
    )
    col3.metric(
        "다음 주 예측 방향",
        direction_label,
        delta=f"{direction_delta:.2f}%",
    )

    expected_feature_count = result.get("expected_feature_count")
    actual_feature_count = len(result["current_data"])
    st.success(
        f"Model {selected_model} 입력 확인: {actual_feature_count}개 feature"
    )
    if (
        expected_feature_count is not None
        and expected_feature_count != actual_feature_count
    ):
        st.error(
            "저장 모델이 요구하는 feature 수와 실제 입력 feature 수가 "
            "일치하지 않습니다."
        )
        return

    st.subheader("최근 예측 입력값")
    input_preview_df = _build_model_input_preview(
        result["current_data"], selected_model
    )
    render_presentation_table(
        input_preview_df,
        title=f"{result_info['name']} 예측 입력값",
        footnote=(
            "저장된 pkl 모델의 feature 구성에 맞춘 최근 주봉 기준 입력값입니다."
        ),
        left_align_cols=["입력 feature"],
        height=980,
    )

    sentiment_compare_df = (
        _build_live_official_sentiment_table(
            result
        )
    )

    if sentiment_compare_df is not None:
        st.subheader(
            "실시간 계산값과 보성님 CSV 값 함께 보기"
        )

        st.caption(
            "왼쪽은 현재 대시보드가 최근 데이터를 이용해 다시 계산한 값이고, "
            "오른쪽은 보성님이 전달한 삼성전자_sentiment_features.csv의 "
            "최종값입니다. 두 값을 서로 대체하지 않고 기준일과 함께 표시합니다."
        )

        render_presentation_table(
            sentiment_compare_df,
            title="심리 proxy 최신값 비교 표출",
            footnote=(
                "값이 다르면 데이터 수집 시점, 기술지표 계산 구간, "
                "잔차 회귀 범위 또는 PCA 기준의 차이일 수 있습니다. "
                "이 표를 그대로 보성님께 확인 자료로 사용할 수 있습니다."
            ),
            left_align_cols=["항목"],
        )
    else:
        st.info(
            "실시간 계산값 또는 보성님 CSV 값을 불러오지 못해 "
            "두 값을 함께 표시하지 못했습니다."
        )

    df_plot = result["df_plot"]
    date_summary_df = _build_prediction_date_summary(df_plot)
    render_presentation_table(
        date_summary_df,
        title="예측 기준일 요약",
        footnote=(
            "실행일, 최근 입력 주봉 기준일, 다음 주 예측 대상 시점을 "
            "구분해 표시합니다."
        ),
        left_align_cols=["구분", "의미"],
        height=420,
    )

    pred_log_return = result["pred_log_return"]
    prediction_summary_df = pd.DataFrame(
        [
            {
                "항목": "예측 대상",
                "값": "삼성전자 다음 주 주봉 로그수익률",
                "해석": "저장된 XGBoost 모델의 최근 1회 예측 대상입니다.",
            },
            {
                "항목": "예측 로그수익률",
                "값": f"{pred_log_return:.5f}",
                "해석": "모델이 산출한 다음 주 로그수익률 예측값입니다.",
            },
            {
                "항목": "예측 수익률(%)",
                "값": f"{pred_log_return * 100:.5f}%",
                "해석": "로그수익률을 백분율로 환산한 참고값입니다.",
            },
            {
                "항목": "예측 방향",
                "값": direction_label,
                "해석": "0보다 크면 상승, 0보다 작으면 하락으로 해석합니다.",
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
            "저장된 pkl 모델이 산출한 최근 1회 예측을 요약합니다."
        ),
        left_align_cols=["항목", "해석"],
        height=520,
    )

    forecast_fig = _plot_next_week_log_return_forecast(
        df_plot=df_plot,
        pred_log_return=pred_log_return,
        asset_name="삼성전자",
        model_label=f"Model {selected_model}",
    )
    if forecast_fig is not None:
        st.subheader("최근 실제 로그수익률과 다음 주 예측값")
        st.pyplot(forecast_fig, use_container_width=True)

