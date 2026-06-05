from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from modules.data_fetcher import get_recent_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
TARGET_INDICATORS = ["ATR_10", "MFI_10", "STOCHk_10_3_3"]
RESIDUAL_COLS = ["ATR_10_res", "MFI_10_res", "STOCHk_10_3_3_res"]
SENTIMENT_COL = "Investor_Sentiment_PC1"
N_LAGS = 5


class QuantPredictor:
    def __init__(self):
        self.model_A = joblib.load(MODEL_DIR / "best_xgboost_panel_model_A.pkl")
        self.model_B = joblib.load(MODEL_DIR / "best_xgboost_panel_model_B.pkl")
        self.model_C = joblib.load(MODEL_DIR / "best_xgboost_panel_model_C.pkl")
        self.scaler = joblib.load(MODEL_DIR / "scaler.pkl")
        self.pca = joblib.load(MODEL_DIR / "pca.pkl")

    def _extract_residuals(self, df):
        result = df.sort_index().copy()
        result["RET"] = result["Close"].pct_change()
        result["MOM"] = result["RET"].shift(1).rolling(window=10).sum()
        result["VOL"] = result["RET"].rolling(window=11).var()

        control_features = ["RET", "MOM", "VOL"]
        clean = result.dropna(subset=control_features + TARGET_INDICATORS).copy()
        if clean.empty:
            raise ValueError("잔차 계산에 필요한 유효 데이터가 부족합니다. 더 긴 기간의 주봉 데이터가 필요합니다.")

        controls = clean[control_features]
        for indicator in TARGET_INDICATORS:
            model = LinearRegression()
            model.fit(controls, clean[indicator])
            clean[f"{indicator}_res"] = clean[indicator] - model.predict(controls)

        return clean

    def _select_model(self, model_type: str):
        if model_type == "A":
            return self.model_A
        if model_type == "B":
            return self.model_B
        if model_type == "C":
            return self.model_C
        raise ValueError(f"지원하지 않는 모델 타입입니다: {model_type}")

    def _add_return_lags(self, df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        result = df.sort_index().copy()
        result[f"{prefix}_Log_Return"] = np.log(result["Close"] / result["Close"].shift(1))
        result[f"{prefix}_Close"] = result["Close"]
        for lag in range(1, N_LAGS + 1):
            result[f"{prefix}_Log_Return_lag{lag}"] = result[f"{prefix}_Log_Return"].shift(lag)
        return result

    def _align_market_learning_to_target(self, target_df: pd.DataFrame, market_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        exog_features = [f"{prefix}_Close"] + [f"{prefix}_Log_Return_lag{i}" for i in range(1, N_LAGS + 1)]
        return market_df[exog_features].sort_index().reindex(target_df.index, method="ffill")

    def _build_live_feature_frame(self):
        samsung_raw = get_recent_data("삼성전자")
        kospi_raw = get_recent_data("코스피")
        bitcoin_raw = get_recent_data("비트코인")

        samsung = self._extract_residuals(samsung_raw)
        scaled_residuals = self.scaler.transform(samsung[RESIDUAL_COLS])
        sentiment_score = self.pca.transform(scaled_residuals)
        mfi_idx = RESIDUAL_COLS.index("MFI_10_res")
        if self.pca.components_[0][mfi_idx] < 0:
            sentiment_score = -sentiment_score
        samsung[SENTIMENT_COL] = sentiment_score
        samsung = self._add_return_lags(samsung, "Samsung")

        kospi = self._add_return_lags(kospi_raw, "KOSPI")
        bitcoin = self._add_return_lags(bitcoin_raw, "Bitcoin")

        features = pd.concat(
            [
                samsung,
                self._align_market_learning_to_target(samsung, kospi, "KOSPI"),
                self._align_market_learning_to_target(samsung, bitcoin, "Bitcoin"),
            ],
            axis=1,
        )
        features = features.dropna().copy()
        if features.empty:
            raise ValueError("예측 피처 생성 후 남은 유효 데이터가 없습니다. 더 긴 기간의 주봉 데이터가 필요합니다.")
        return features

    def get_prediction(self, model_type="B"):
        feature_frame = self._build_live_feature_frame()
        current_data = feature_frame.iloc[-1:].copy()

        model = self._select_model(model_type)
        expected_cols = model.feature_names_in_
        for col in expected_cols:
            if col not in current_data.columns:
                current_data[col] = 0

        prediction_input = current_data[expected_cols]
        pred_log_return = float(model.predict(prediction_input)[0])
        direction = "UP" if pred_log_return > 0 else "DOWN"

        return {
            "pred_log_return": pred_log_return,
            "direction": direction,
            "df_plot": feature_frame,
            "current_data": current_data.iloc[0].to_dict(),
        }
