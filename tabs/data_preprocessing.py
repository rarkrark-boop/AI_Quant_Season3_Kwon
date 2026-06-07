from pathlib import Path

import pandas as pd
import streamlit as st

from modules.data_fetcher import get_recent_data
from utils.data_utils import validate_columns
from utils.table_utils import render_presentation_table
from visualization import plot_price_history


REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
CACHE_DIR = Path("data/cache")


def _get_cache_path(asset_name: str) -> Path:
    safe_name = asset_name.replace("/", "_").replace(" ", "_")
    return CACHE_DIR / f"{safe_name}_weekly.csv"


def _load_cached_data(asset_name: str) -> pd.DataFrame | None:
    cache_path = _get_cache_path(asset_name)

    if not cache_path.exists():
        return None

    df = pd.read_csv(
        cache_path,
        index_col=0,
        parse_dates=True,
    )
    validate_columns(df, REQUIRED_COLUMNS)
    return df.sort_index()


def _save_cached_data(
    asset_name: str,
    df: pd.DataFrame,
) -> None:
    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.sort_index().to_csv(
        _get_cache_path(asset_name),
        encoding="utf-8-sig",
    )


def _load_recent_data(
    asset_name: str,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    자산별 주봉 데이터를 불러온다.

    기본적으로 CSV 캐시를 사용하고,
    force_refresh=True이면 새 데이터를 수집한 뒤 캐시를 갱신한다.
    """
    if not force_refresh:
        cached_df = _load_cached_data(
            asset_name
        )

        if cached_df is not None:
            return cached_df

    df = get_recent_data(
        asset_name
    )
    validate_columns(
        df,
        REQUIRED_COLUMNS,
    )

    df = df.sort_index().copy()

    _save_cached_data(
        asset_name,
        df,
    )

    return df


def _format_preview_table(
    df: pd.DataFrame,
    rows: int = 20,
) -> pd.DataFrame:
    """
    최근 주봉 데이터 중 발표 화면에 필요한 열만 정리한다.
    """
    preview = df.tail(rows).copy()

    display_columns = [
        "Close",
        "Volume",
        "ATR_10",
        "MFI_10",
        "STOCHk_10_3_3",
        "STOCHd_10_3_3",
        "Next_Return",
    ]

    available_columns = [
        col
        for col in display_columns
        if col in preview.columns
    ]

    preview = preview[available_columns]

    if preview.index.name is None:
        preview.index.name = "주봉 기준일"

    return preview


def _build_missing_table(
    df: pd.DataFrame,
) -> pd.DataFrame:
    return (
        df.isna()
        .sum()
        .rename("총 결측치 수")
        .reset_index()
        .rename(columns={"index": "컬럼"})
    )


def run() -> None:
    st.header("데이터 전처리")

    st.info(
        "본 탭은 사이드바에서 선택한 자산의 최근 주봉 OHLCV와 "
        "기술지표 계산 결과를 확인하는 영역입니다. "
        "최종 예측 대상은 삼성전자이며, KOSPI와 Bitcoin은 "
        "시장 흐름을 반영하는 외생 보조 입력자료로 사용합니다."
    )

    asset_name = st.session_state.get(
        "asset_name",
        "삼성전자",
    )

    st.subheader(
        f"{asset_name} 주봉 데이터 확인"
    )

    st.caption(
        "캐시된 데이터를 불러오거나 최신 데이터를 다시 수집할 수 있습니다."
    )

    if "raw_data_by_asset" not in st.session_state:
        st.session_state.raw_data_by_asset = {}

    load_col, refresh_col = st.columns(2)

    with load_col:
        load_clicked = st.button(
            f"{asset_name} 최근 주봉 데이터 불러오기",
            type="primary",
        )

    with refresh_col:
        refresh_clicked = st.button(
            f"{asset_name} 데이터 새로고침",
        )

    if load_clicked or refresh_clicked:
        force_refresh = refresh_clicked

        action_text = (
            "새로 수집"
            if force_refresh
            else "불러오기"
        )

        with st.spinner(
            f"{asset_name} 데이터를 {action_text} 중입니다."
        ):
            try:
                df_loaded = _load_recent_data(
                    asset_name,
                    force_refresh=force_refresh,
                )
            except Exception as exc:
                st.error(
                    f"{asset_name} 데이터를 준비하지 못했습니다: {exc}"
                )
                return

        st.session_state.raw_data_by_asset[
            asset_name
        ] = df_loaded
        st.session_state.raw_data = df_loaded
        st.session_state.loaded_asset_name = asset_name

        if force_refresh:
            st.success(
                f"{asset_name} 데이터를 새로 수집하고 CSV 캐시를 갱신했습니다."
            )
        else:
            st.success(
                f"{asset_name} 데이터를 CSV 캐시에서 불러왔습니다."
            )

    if asset_name in st.session_state.raw_data_by_asset:
        df = st.session_state.raw_data_by_asset[
            asset_name
        ]

        st.session_state.raw_data = df
        st.session_state.loaded_asset_name = asset_name
    else:
        cached_df = _load_cached_data(
            asset_name
        )

        if cached_df is None:
            st.info(
                f"아직 {asset_name} 데이터가 없습니다. "
                "위 버튼을 눌러 데이터를 준비하세요."
            )
            return

        df = cached_df
        st.session_state.raw_data_by_asset[
            asset_name
        ] = df
        st.session_state.raw_data = df
        st.session_state.loaded_asset_name = asset_name

    st.success(
        f"{asset_name} 최근 주봉 데이터가 준비되었습니다."
    )

    start_date = df.index.min()
    end_date = df.index.max()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "데이터 개수",
        f"{len(df):,}개",
    )

    col2.metric(
        "시작일",
        (
            start_date.strftime("%Y-%m-%d")
            if hasattr(start_date, "strftime")
            else str(start_date)
        ),
    )

    col3.metric(
        "최근 기준일",
        (
            end_date.strftime("%Y-%m-%d")
            if hasattr(end_date, "strftime")
            else str(end_date)
        ),
    )

    st.subheader("최근 주봉 데이터 미리보기")

    st.caption(
        "최근 20개 주봉 중 Close·Volume과 주요 기술지표를 표시합니다."
    )

    preview_df = _format_preview_table(
        df,
        rows=20,
    )

    st.dataframe(
        preview_df,
        use_container_width=True,
        height=420,
    )

    st.subheader("결측치 점검")

    missing_df = _build_missing_table(
        df
    )

    render_presentation_table(
        missing_df,
        title=f"{asset_name} 결측치 점검",
        footnote=(
            "결측치는 이후 심리 proxy 계산과 모델 입력 feature 생성에 "
            "영향을 줄 수 있으므로 컬럼별로 확인합니다."
        ),
        left_align_cols=["컬럼"],
    )

    st.subheader(
        f"{asset_name} 주봉 종가 추이"
    )

    st.caption(
        "Close 기준 주봉 가격 흐름입니다. "
        "로그수익률과 심리 proxy 계산의 기초 가격 흐름을 확인합니다."
    )

    st.pyplot(
        plot_price_history(
            df,
            title=f"{asset_name} 주봉 종가 추이",
        ),
        use_container_width=True,
    )
