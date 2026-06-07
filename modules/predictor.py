from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from modules.data_fetcher import get_recent_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
OFFICIAL_SENTIMENT_PATH = (
    PROJECT_ROOT / "삼성전자_sentiment_features.csv"
)

TARGET_INDICATORS = [
    "ATR_10",
    "MFI_10",
    "STOCHk_10_3_3",
]

RESIDUAL_COLS = [
    "ATR_10_res",
    "MFI_10_res",
    "STOCHk_10_3_3_res",
]

SENTIMENT_COL = "Investor_Sentiment_PC1"
N_LAGS = 5


class QuantPredictor:
    def __init__(self):
        self.model_A = joblib.load(
            MODEL_DIR
            / "best_xgboost_panel_model_A.pkl"
        )
        self.model_B = joblib.load(
            MODEL_DIR
            / "best_xgboost_panel_model_B.pkl"
        )
        self.model_C = joblib.load(
            MODEL_DIR
            / "best_xgboost_panel_model_C.pkl"
        )
        self.scaler = joblib.load(
            MODEL_DIR / "scaler.pkl"
        )
        self.pca = joblib.load(
            MODEL_DIR / "pca.pkl"
        )

    def _extract_residuals(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        삼성전자 주봉 데이터에서 RET·MOM·VOL을 계산하고
        ATR·MFI·Stochastic residual을 생성한다.
        """
        result = df.sort_index().copy()

        result["RET"] = (
            result["Close"]
            .pct_change()
        )

        result["MOM"] = (
            result["RET"]
            .shift(1)
            .rolling(window=10)
            .sum()
        )

        result["VOL"] = (
            result["RET"]
            .rolling(window=11)
            .var()
        )

        control_features = [
            "RET",
            "MOM",
            "VOL",
        ]

        clean = result.dropna(
            subset=(
                control_features
                + TARGET_INDICATORS
            )
        ).copy()

        if clean.empty:
            raise ValueError(
                "잔차 계산에 필요한 유효 데이터가 부족합니다. "
                "더 긴 기간의 주봉 데이터가 필요합니다."
            )

        controls = clean[
            control_features
        ]

        for indicator in TARGET_INDICATORS:
            model = LinearRegression()

            model.fit(
                controls,
                clean[indicator],
            )

            clean[
                f"{indicator}_res"
            ] = (
                clean[indicator]
                - model.predict(controls)
            )

        return clean

    def _select_model(
        self,
        model_type: str,
    ):
        """
        내부 저장 모델 타입 A/B/C를 선택한다.

        대시보드 화면에서는 B/C/D로 표시하지만
        기존 pkl 파일과의 연결을 위해 내부 타입은 A/B/C를 유지한다.
        """
        if model_type == "A":
            return self.model_A

        if model_type == "B":
            return self.model_B

        if model_type == "C":
            return self.model_C

        raise ValueError(
            f"지원하지 않는 내부 모델 타입입니다: {model_type}"
        )

    def _add_return_lags(
        self,
        df: pd.DataFrame,
        prefix: str,
    ) -> pd.DataFrame:
        """
        자산별 현재 주봉 종가와 과거 1~5주 로그수익률을 생성한다.
        """
        result = df.sort_index().copy()

        result[
            f"{prefix}_Log_Return"
        ] = np.log(
            result["Close"]
            / result["Close"].shift(1)
        )

        result[
            f"{prefix}_Close"
        ] = result["Close"]

        for lag in range(
            1,
            N_LAGS + 1,
        ):
            result[
                f"{prefix}_Log_Return_lag{lag}"
            ] = result[
                f"{prefix}_Log_Return"
            ].shift(lag)

        return result

    def _align_market_learning_to_target(
        self,
        target_df: pd.DataFrame,
        market_df: pd.DataFrame,
        prefix: str,
    ) -> pd.DataFrame:
        """
        KOSPI·Bitcoin 주봉 feature를 삼성전자 기준일에 맞춘다.
        """
        market_features = [
            f"{prefix}_Close",
        ] + [
            f"{prefix}_Log_Return_lag{i}"
            for i in range(
                1,
                N_LAGS + 1,
            )
        ]

        return (
            market_df[
                market_features
            ]
            .sort_index()
            .reindex(
                target_df.index,
                method="ffill",
            )
        )

    def _load_official_sentiment_latest(
        self,
    ) -> dict | None:
        """
        보성님 최종 CSV의 마지막 행을 외부 표출용 참고값으로 불러온다.

        이 값은 실시간 예측 입력을 대체하지 않는다.
        """
        if not OFFICIAL_SENTIMENT_PATH.exists():
            return None

        official_df = pd.read_csv(
            OFFICIAL_SENTIMENT_PATH,
            parse_dates=["Date"],
        )

        if official_df.empty:
            return None

        official_df = (
            official_df
            .sort_values("Date")
            .reset_index(drop=True)
        )

        latest = official_df.iloc[-1]

        return {
            "date": latest["Date"],
            "values": {
                col: float(latest[col])
                for col in (
                    RESIDUAL_COLS
                    + [SENTIMENT_COL]
                )
                if col in latest.index
            },
        }

    def _build_live_feature_frame(
        self,
    ) -> pd.DataFrame:
        """
        삼성전자·KOSPI·Bitcoin 최근 주봉을 불러와
        저장 모델에 입력할 실시간 feature frame을 만든다.
        """
        samsung_raw = get_recent_data(
            "삼성전자"
        )
        kospi_raw = get_recent_data(
            "코스피"
        )
        bitcoin_raw = get_recent_data(
            "비트코인"
        )

        samsung = (
            self._extract_residuals(
                samsung_raw
            )
        )

        scaled_residuals = (
            self.scaler.transform(
                samsung[RESIDUAL_COLS]
            )
        )

        sentiment_score = (
            self.pca.transform(
                scaled_residuals
            )
        )

        mfi_index = (
            RESIDUAL_COLS.index(
                "MFI_10_res"
            )
        )

        if (
            self.pca.components_[0][
                mfi_index
            ]
            < 0
        ):
            sentiment_score = (
                -sentiment_score
            )

        samsung[
            SENTIMENT_COL
        ] = sentiment_score

        samsung = self._add_return_lags(
            samsung,
            "Samsung",
        )
        kospi = self._add_return_lags(
            kospi_raw,
            "KOSPI",
        )
        bitcoin = self._add_return_lags(
            bitcoin_raw,
            "Bitcoin",
        )

        kospi_aligned = (
            self._align_market_learning_to_target(
                samsung,
                kospi,
                "KOSPI",
            )
        )

        bitcoin_aligned = (
            self._align_market_learning_to_target(
                samsung,
                bitcoin,
                "Bitcoin",
            )
        )

        feature_frame = pd.concat(
            [
                samsung,
                kospi_aligned,
                bitcoin_aligned,
            ],
            axis=1,
        )

        feature_frame = (
            feature_frame
            .dropna()
            .sort_index()
            .copy()
        )

        if feature_frame.empty:
            raise ValueError(
                "예측 feature 생성 후 남은 유효 데이터가 없습니다. "
                "더 긴 기간의 주봉 데이터가 필요합니다."
            )

        return feature_frame

    def get_prediction(
        self,
        model_type: str = "B",
    ) -> dict:
        """
        내부 저장 모델 A/B/C 중 하나로
        삼성전자 다음 주 로그수익률을 예측한다.
        """
        feature_frame = (
            self._build_live_feature_frame()
        )

        current_data = (
            feature_frame
            .iloc[-1:]
            .copy()
        )

        model = self._select_model(
            model_type
        )

        expected_cols = list(
            model.feature_names_in_
        )

        missing_features = [
            col
            for col in expected_cols
            if col not in current_data.columns
        ]

        if missing_features:
            raise ValueError(
                "최근 예측 입력에 필요한 feature가 누락되었습니다: "
                + ", ".join(missing_features)
            )

        prediction_input = (
            current_data[
                expected_cols
            ]
            .copy()
        )

        pred_log_return = float(
            model.predict(
                prediction_input
            )[0]
        )

        direction = (
            "UP"
            if pred_log_return > 0
            else "DOWN"
        )

        official_sentiment_latest = (
            self._load_official_sentiment_latest()
        )

        live_sentiment_latest = {
            "date": feature_frame.index.max(),
            "values": {
                col: float(feature_frame.iloc[-1][col])
                for col in (
                    RESIDUAL_COLS
                    + [SENTIMENT_COL]
                )
                if col in feature_frame.columns
            },
        }

        return {
            "pred_log_return": pred_log_return,
            "direction": direction,
            "df_plot": feature_frame,
            "current_data": (
                prediction_input
                .iloc[0]
                .to_dict()
            ),
            "expected_feature_count": len(
                expected_cols
            ),
            "latest_feature_date": (
                feature_frame.index.max()
            ),
            "live_sentiment_latest": (
                live_sentiment_latest
            ),
            "official_sentiment_latest": (
                official_sentiment_latest
            ),
        }
