import html
import uuid

import pandas as pd
import streamlit as st


def render_presentation_table(
    df: pd.DataFrame,
    title: str | None = None,
    footnote: str | None = None,
    left_align_cols: list[str] | None = None,
    height: int | None = None,
    cell_style_rules: dict[str, dict[str, str]] | None = None,
) -> None:
    """
    발표용 카드형 HTML 표를 렌더링한다.

    st.html을 사용하므로 iframe 고정 높이로 인한 표 잘림을 줄이고,
    열이 많은 표는 카드 내부에서 가로 스크롤할 수 있게 한다.
    height 인자는 기존 호출부 호환을 위해 유지하지만 사용하지 않는다.
    """
    del height

    if df is None:
        st.warning(
            "표시할 데이터가 없습니다."
        )
        return

    if left_align_cols is None:
        left_align_cols = []

    if cell_style_rules is None:
        cell_style_rules = {}

    display_df = df.copy()
    table_id = (
        "presentation-table-"
        + uuid.uuid4().hex
    )

    headers = "".join(
        f"<th>{html.escape(str(col))}</th>"
        for col in display_df.columns
    )

    rows_html = ""

    for _, row in display_df.iterrows():
        row_html = ""

        for col in display_df.columns:
            val = row[col]
            text = (
                ""
                if pd.isna(val)
                else str(val)
            )

            safe_text = html.escape(
                text
            ).replace(
                "\n",
                "<br>",
            )

            align_class = (
                "left-cell"
                if col in left_align_cols
                else ""
            )

            extra_style = ""

            if col in cell_style_rules:
                extra_style = (
                    cell_style_rules[col]
                    .get(text, "")
                )

            style_attr = (
                f' style="{html.escape(extra_style, quote=True)}"'
                if extra_style
                else ""
            )

            row_html += (
                f'<td class="{align_class}"'
                f'{style_attr}>{safe_text}</td>'
            )

        rows_html += (
            f"<tr>{row_html}</tr>"
        )

    title_html = (
        f'<div class="table-card-title">'
        f'{html.escape(str(title))}</div>'
        if title
        else ""
    )

    footnote_html = (
        f'<div class="table-footnote">'
        f'{html.escape(str(footnote))}</div>'
        if footnote
        else ""
    )

    html_text = f"""
    <style>
    #{table_id} {{
        background-color: #fbf8f1;
        border: 1px solid #e3dacb;
        border-radius: 16px;
        padding: 14px 18px 16px 18px;
        margin: 0 0 14px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
        font-family: "Malgun Gothic", "Apple SD Gothic Neo",
                     "Noto Sans KR", sans-serif;
        box-sizing: border-box;
        width: 100%;
    }}

    #{table_id} .table-card-title {{
        font-size: 20px;
        font-weight: 800;
        color: #3d352d;
        margin-bottom: 12px;
    }}

    #{table_id} .table-scroll {{
        width: 100%;
        overflow-x: auto;
        overflow-y: visible;
    }}

    #{table_id} table {{
        width: 100%;
        min-width: 780px;
        border-collapse: collapse;
        table-layout: auto;
        font-size: 15px;
        color: #433c35;
    }}

    #{table_id} thead th {{
        background-color: #e9e1d3;
        color: #3d352d;
        font-weight: 800;
        text-align: center;
        padding: 11px 10px;
        border: 1px solid #ddd2c3;
        white-space: normal;
        word-break: keep-all;
        overflow-wrap: anywhere;
        min-width: 90px;
    }}

    #{table_id} tbody td {{
        background-color: #fffdfa;
        padding: 10px 10px;
        border: 1px solid #e7ddd0;
        text-align: center;
        vertical-align: middle;
        white-space: normal;
        word-break: keep-all;
        overflow-wrap: anywhere;
        line-height: 1.55;
    }}

    #{table_id} tbody tr:nth-child(even) td {{
        background-color: #fcf8f2;
    }}

    #{table_id} .left-cell {{
        text-align: left !important;
    }}

    #{table_id} .table-footnote {{
        margin-top: 12px;
        font-size: 14px;
        line-height: 1.65;
        color: #6e665e;
    }}
    </style>

    <div id="{table_id}">
        {title_html}
        <div class="table-scroll">
            <table>
                <thead>
                    <tr>{headers}</tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
        {footnote_html}
    </div>
    """

    st.html(
        html_text
    )
