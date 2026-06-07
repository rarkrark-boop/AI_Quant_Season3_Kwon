import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager

from utils.plotting_utils import (
    COLORS,
    new_figure,
    style_axis,
)


def configure_korean_font() -> None:
    """
    실행 환경에서 사용할 수 있는 한글 폰트를 찾아
    Matplotlib 전체 그래프에 적용한다.
    """
    available_fonts = {
        font.name
        for font in font_manager.fontManager.ttflist
    }

    candidate_fonts = [
        "Malgun Gothic",
        "AppleGothic",
        "Noto Sans CJK KR",
        "Noto Sans KR",
        "NanumGothic",
    ]

    selected_font = next(
        (
            font_name
            for font_name in candidate_fonts
            if font_name in available_fonts
        ),
        None,
    )

    if selected_font is not None:
        plt.rcParams["font.family"] = selected_font

    plt.rcParams["axes.unicode_minus"] = False


configure_korean_font()


def plot_price_history(
    df: pd.DataFrame,
    title: str = "Price History",
):
    fig, ax = new_figure()

    ax.plot(
        df.index,
        df["Close"],
        color=COLORS["primary"],
        linewidth=2,
        label="Close",
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Close")
    ax.legend()

    style_axis(
        ax,
        title,
    )

    fig.tight_layout()
    return fig


def plot_sentiment_index(
    sentiment: pd.Series,
    title: str = "Investor Sentiment PC1",
):
    fig, ax = new_figure()

    colors = [
        (
            COLORS["secondary"]
            if value >= 0
            else COLORS["danger"]
        )
        for value in sentiment
    ]

    ax.bar(
        sentiment.index,
        sentiment.values,
        color=colors,
        alpha=0.8,
    )

    ax.axhline(
        0,
        color=COLORS["muted"],
        linewidth=1,
    )

    ax.set_ylabel("PC1")

    style_axis(
        ax,
        title,
    )

    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_actual_vs_predicted(
    actual,
    predicted,
    title: str = "Actual vs Predicted",
):
    fig, ax = new_figure()

    ax.plot(
        actual.index,
        actual,
        label="Actual",
        color=COLORS["primary"],
        linewidth=2,
    )

    ax.plot(
        predicted.index,
        predicted,
        label="Predicted",
        color=COLORS["danger"],
        linewidth=2,
        linestyle="--",
    )

    ax.axhline(
        0,
        color=COLORS["muted"],
        linewidth=1,
    )

    ax.set_xlabel("Date")
    ax.legend()

    style_axis(
        ax,
        title,
    )

    fig.tight_layout()
    return fig


def plot_correlation_heatmap(
    data: pd.DataFrame,
    title: str = "Correlation Heatmap",
):
    fig, ax = plt.subplots(
        figsize=(6.2, 4.8)
    )

    corr = data.corr(
        numeric_only=True
    )

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax,
        square=True,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 8},
    )

    ax.set_title(
        title,
        pad=12,
        fontsize=11,
    )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(
        axis="x",
        rotation=45,
        labelsize=8,
    )
    ax.tick_params(
        axis="y",
        rotation=0,
        labelsize=8,
    )

    fig.tight_layout()
    return fig


def plot_heatmap(
    data: pd.DataFrame,
):
    return plot_correlation_heatmap(
        data
    )


def plot_directional_accuracy_bar(
    performance_df: pd.DataFrame,
):
    """
    모델별 Directional Accuracy를 가로 막대그래프로 표시한다.
    """
    fig, ax = new_figure(
        figsize=(9.0, 6.0)
    )

    plot_df = performance_df.copy()

    if "DA" in plot_df.columns:
        value_col = "DA"
    elif "Directional_Accuracy" in plot_df.columns:
        value_col = "Directional_Accuracy"
    else:
        raise ValueError(
            "방향 정확도 컬럼 DA 또는 Directional_Accuracy가 필요합니다."
        )

    plot_df[value_col] = pd.to_numeric(
        plot_df[value_col],
        errors="coerce",
    )

    plot_df = (
        plot_df
        .dropna(subset=[value_col])
        .sort_values(value_col, ascending=True)
    )

    fig.patch.set_facecolor("#fbf8f1")
    fig.patch.set_edgecolor("#e3dacb")
    fig.patch.set_linewidth(1.4)
    ax.set_facecolor("#fffdfa")

    bars = ax.barh(
        plot_df["Model"],
        plot_df[value_col],
        color=COLORS["primary"],
        height=0.62,
    )

    ax.set_xlabel(
        "Directional Accuracy (%)"
    )
    ax.set_ylabel("")

    upper = max(
        float(plot_df[value_col].max()) + 8,
        75,
    )
    ax.set_xlim(0, upper)

    for bar, value in zip(
        bars,
        plot_df[value_col],
    ):
        ax.text(
            value + 0.6,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}%",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
            color=COLORS["muted"],
        )

    style_axis(
        ax,
        "모델별 방향 정확도 비교",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(
        pad=1.4
    )
    return fig


def plot_actual_vs_prediction_series(
    prediction_df: pd.DataFrame,
    asset_name: str,
    model_col: str,
    model_label: str,
):
    plot_df = prediction_df.copy()
    plot_df["Date"] = pd.to_datetime(
        plot_df["Date"]
    )

    if "Asset" in plot_df.columns:
        plot_df = plot_df[
            plot_df["Asset"] == asset_name
        ].copy()

    plot_df = plot_df.sort_values(
        "Date"
    )

    if model_col not in plot_df.columns:
        raise ValueError(
            f"예측 컬럼이 없습니다: {model_col}"
        )

    fig, ax = new_figure(
        figsize=(8.5, 5.2)
    )

    ax.plot(
        plot_df["Date"],
        plot_df["Actual_Log_Return"],
        marker="o",
        linewidth=1.8,
        color=COLORS["primary"],
        label="Actual",
    )

    ax.plot(
        plot_df["Date"],
        plot_df[model_col],
        marker="o",
        linewidth=1.8,
        linestyle="--",
        color=COLORS["secondary"],
        label=model_label,
    )

    ax.axhline(
        0,
        linewidth=1,
        color=COLORS["muted"],
        alpha=0.7,
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Log return")
    ax.legend()

    style_axis(
        ax,
        f"{asset_name} 실제값 vs 예측값 로그수익률",
    )

    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_actual_vs_predicted_scatter(
    prediction_df: pd.DataFrame,
    asset_name: str,
    model_col: str,
    model_label: str,
):
    plot_df = prediction_df.copy()

    if "Asset" in plot_df.columns:
        plot_df = plot_df[
            plot_df["Asset"] == asset_name
        ].copy()

    if model_col not in plot_df.columns:
        raise ValueError(
            f"예측 컬럼이 없습니다: {model_col}"
        )

    x = pd.to_numeric(
        plot_df["Actual_Log_Return"],
        errors="coerce",
    )
    y = pd.to_numeric(
        plot_df[model_col],
        errors="coerce",
    )

    valid = x.notna() & y.notna()
    x = x[valid]
    y = y[valid]

    if x.empty or y.empty:
        raise ValueError(
            "산점도를 그릴 유효한 실제값·예측값이 없습니다."
        )

    fig, ax = new_figure(
        figsize=(6.5, 6.0)
    )

    ax.scatter(
        x,
        y,
        alpha=0.75,
        color=COLORS["primary"],
    )

    min_value = min(
        float(x.min()),
        float(y.min()),
    )
    max_value = max(
        float(x.max()),
        float(y.max()),
    )

    ax.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--",
        linewidth=1,
        color=COLORS["muted"],
        alpha=0.8,
    )

    ax.axhline(
        0,
        linewidth=1,
        color=COLORS["grid"],
    )
    ax.axvline(
        0,
        linewidth=1,
        color=COLORS["grid"],
    )

    ax.set_xlabel(
        "Actual log return"
    )
    ax.set_ylabel(
        "Predicted log return"
    )

    style_axis(
        ax,
        f"{asset_name} 실제값-예측값 산점도 ({model_label})",
    )

    fig.tight_layout()
    return fig


def plot_pca_loading_bar(
    pca_reference_df: pd.DataFrame,
):
    """
    PCA loading 비교 가로 막대그래프.
    """
    plot_df = pca_reference_df.copy()

    loading_df = plot_df[
        plot_df["Item"].str.contains(
            "loading",
            na=False,
        )
    ].copy()

    loading_df["Value"] = pd.to_numeric(
        loading_df["Value"],
        errors="coerce",
    )

    loading_df = loading_df.dropna(
        subset=["Value"]
    )

    if loading_df.empty:
        raise ValueError(
            "PCA loading 시각화에 사용할 행이 없습니다."
        )

    fig, ax = new_figure(
        figsize=(7.6, 4.4)
    )

    bar_colors = [
        (
            COLORS["danger"]
            if value < 0
            else COLORS["secondary"]
        )
        for value in loading_df["Value"]
    ]

    bars = ax.barh(
        loading_df["Display_Label"],
        loading_df["Value"],
        color=bar_colors,
        height=0.55,
    )

    ax.axvline(
        0,
        linewidth=1,
        color=COLORS["muted"],
        alpha=0.8,
    )

    ax.set_xlabel("Loading")
    ax.set_ylabel("")

    min_value = float(
        loading_df["Value"].min()
    )
    max_value = float(
        loading_df["Value"].max()
    )

    margin = max(
        abs(min_value),
        abs(max_value),
        0.1,
    ) * 0.18

    ax.set_xlim(
        min_value - margin,
        max_value + margin,
    )

    for bar, value in zip(
        bars,
        loading_df["Value"],
    ):
        offset = margin * 0.22

        if value >= 0:
            x_pos = value + offset
            ha = "left"
        else:
            x_pos = value - offset
            ha = "right"

        ax.text(
            x_pos,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}",
            va="center",
            ha=ha,
            fontsize=11,
            fontweight="bold",
            color=COLORS["muted"],
        )

    style_axis(
        ax,
        "PCA Loading 비교",
    )

    fig.tight_layout()
    return fig
