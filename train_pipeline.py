import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

import Samsung_crawling as sc


warnings.filterwarnings("ignore")

TARGET_ASSET = "삼성전자"
EXOG_ASSETS = ["코스피", "비트코인"]
TARGET_INDICATORS = ["ATR_10", "MFI_10", "STOCHk_10_3_3"]
RESIDUAL_COLS = ["ATR_10_res", "MFI_10_res", "STOCHk_10_3_3_res"]
SENTIMENT_COL = "Investor_Sentiment_PC1"
N_LAGS = 5
OUTPUT_DIR = "data/output"


def extract_sentiment_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """Extract ATR/MFI/STOCH residuals after removing price factors per asset."""
    if "Ticker" not in df.columns:
        raise ValueError("Ticker column is required for residual extraction.")

    control_features = ["RET", "MOM", "VOL"]
    processed_assets = []

    for ticker, asset_df in df.groupby("Ticker", sort=False):
        df_res = asset_df.sort_index().copy()

        # Keep time-series operations inside each asset to avoid cross-asset leakage.
        df_res["RET"] = df_res["Close"].pct_change()
        df_res["MOM"] = df_res["RET"].shift(1).rolling(window=10).sum()
        df_res["VOL"] = df_res["RET"].rolling(window=11).var()

        df_clean = df_res.dropna(subset=control_features + TARGET_INDICATORS).copy()
        if df_clean.empty:
            print(f"  [{ticker}] no valid rows for residual extraction")
            continue

        controls = df_clean[control_features]
        for indicator in TARGET_INDICATORS:
            lr = LinearRegression()
            lr.fit(controls, df_clean[indicator])
            df_clean[f"{indicator}_res"] = df_clean[indicator] - lr.predict(controls)

        print(f"  [{ticker}] residual extraction complete: {len(df_clean)} weeks")
        processed_assets.append(df_clean)

    if not processed_assets:
        raise ValueError("No rows left after residual extraction.")

    return pd.concat(processed_assets).sort_index()


def create_sentiment_index(df: pd.DataFrame) -> pd.DataFrame:
    """Create PCA-based investor sentiment index from residuals."""
    result = df.copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(result[RESIDUAL_COLS])

    pca = PCA(n_components=1)
    sentiment_score = pca.fit_transform(scaled)

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(pca, "models/pca.pkl")

    mfi_idx = RESIDUAL_COLS.index("MFI_10_res")
    if pca.components_[0][mfi_idx] < 0:
        sentiment_score = -sentiment_score

    result[SENTIMENT_COL] = sentiment_score
    return result


def add_return_lags(df: pd.DataFrame, prefix: str, n_lags: int = N_LAGS) -> pd.DataFrame:
    result = df.sort_index().copy()
    result[f"{prefix}_Log_Return"] = np.log(result["Close"] / result["Close"].shift(1))
    result[f"{prefix}_Close"] = result["Close"]

    for lag in range(1, n_lags + 1):
        result[f"{prefix}_Log_Return_lag{lag}"] = result[f"{prefix}_Log_Return"].shift(lag)

    return result


def align_market_learning_to_target(target_df: pd.DataFrame, market_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Align market-learning weekly features to Samsung dates using the latest available prior week."""
    exog_features = [f"{prefix}_Close"] + [f"{prefix}_Log_Return_lag{i}" for i in range(1, N_LAGS + 1)]
    aligned = market_df[exog_features].sort_index().reindex(target_df.index, method="ffill")
    return aligned


def build_samsung_target_dataset(df_processed: pd.DataFrame) -> pd.DataFrame:
    """Build one-row-per-Samsung-week training data with KOSPI/BTC as market-learning aids."""
    by_asset = {name: asset_df.sort_index().copy() for name, asset_df in df_processed.groupby("Ticker")}
    missing_assets = [asset for asset in [TARGET_ASSET] + EXOG_ASSETS if asset not in by_asset]
    if missing_assets:
        raise ValueError(f"Missing assets in panel data: {missing_assets}")

    samsung = create_sentiment_index(by_asset[TARGET_ASSET])
    samsung = add_return_lags(samsung, "Samsung")
    samsung["Target_Log_Return"] = samsung["Samsung_Log_Return"].shift(-1)

    feature_frames = [samsung]
    for asset_name, prefix in [("코스피", "KOSPI"), ("비트코인", "Bitcoin")]:
        exog = add_return_lags(by_asset[asset_name], prefix)
        feature_frames.append(align_market_learning_to_target(samsung, exog, prefix))

    df_model = pd.concat(feature_frames, axis=1)
    df_model = df_model.dropna().copy()
    df_model = df_model.sort_index()
    return df_model


def calculate_da(y_true, y_pred):
    true_sign = np.sign(y_true)
    pred_sign = np.sign(y_pred)
    match = (true_sign == pred_sign) & (true_sign != 0)
    return np.sum(match) / len(y_true) * 100


def calculate_side_da(y_true, y_pred, side):
    true_sign = np.sign(y_true)
    pred_sign = np.sign(y_pred)
    mask = true_sign == side
    if np.sum(mask) == 0:
        return np.nan
    return np.sum(pred_sign[mask] == true_sign[mask]) / np.sum(mask) * 100


def summarize_predictions(label, y_true, y_pred):
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    true_sign = np.sign(y_true_array)
    pred_sign = np.sign(y_pred_array)
    correct = (true_sign == pred_sign) & (true_sign != 0)

    return {
        "Model": label,
        "R2": r2_score(y_true, y_pred_array),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred_array)),
        "MAE": mean_absolute_error(y_true, y_pred_array),
        "DA": calculate_da(y_true_array, y_pred_array),
        "Correct": int(np.sum(correct)),
        "Test_N": len(y_true_array),
        "Up_DA": calculate_side_da(y_true_array, y_pred_array, 1),
        "Down_DA": calculate_side_da(y_true_array, y_pred_array, -1),
        "Pred_Up_Ratio": np.sum(pred_sign > 0) / len(pred_sign) * 100,
    }


def evaluate_constant_baseline(label, y_test, constant_value):
    y_pred = np.full(len(y_test), constant_value)
    return summarize_predictions(label, y_test, y_pred), y_pred


def train_and_evaluate(model_name, X_train, y_train, X_test, y_test):
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.01,
        max_depth=3,
        min_child_weight=5,
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=42,
    )
    model.fit(X_train, y_train, verbose=False)
    y_pred = model.predict(X_test)

    metrics = summarize_predictions(model_name, y_test, y_pred)

    return model, metrics, y_pred


def build_period_backtest(y_test, prediction_map, window_size=12):
    records = []
    for start in range(0, len(y_test), window_size):
        end = min(start + window_size, len(y_test))
        y_window = y_test.iloc[start:end]
        period_label = f"{y_window.index[0].date()} ~ {y_window.index[-1].date()}"
        for model_name, y_pred in prediction_map.items():
            pred_window = np.asarray(y_pred)[start:end]
            records.append(
                {
                    "Period": period_label,
                    "Model": model_name,
                    "Weeks": len(y_window),
                    "Correct": int(np.sum((np.sign(y_window) == np.sign(pred_window)) & (np.sign(y_window) != 0))),
                    "DA": calculate_da(y_window, pred_window),
                    "Up_DA": calculate_side_da(np.asarray(y_window), pred_window, 1),
                    "Down_DA": calculate_side_da(np.asarray(y_window), pred_window, -1),
                    "Pred_Up_Ratio": np.sum(np.sign(pred_window) > 0) / len(pred_window) * 100,
                }
            )
    return pd.DataFrame(records)


def save_backtest_outputs(summary_df, period_df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_path = os.path.join(OUTPUT_DIR, "model_backtest_summary.csv")
    period_path = os.path.join(OUTPUT_DIR, "model_period_backtest.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    period_df.to_csv(period_path, index=False, encoding="utf-8-sig")
    print(f"\n백테스트 요약 CSV 저장 완료: {summary_path}")
    print(f"구간별 백테스트 CSV 저장 완료: {period_path}")


def print_feature_summary(price_features, residual_cols, pca_col):
    print("\n=========================================================")
    print(" [Feature groups and rationale]")
    print("=========================================================")
    print("Target")
    print(" - Samsung Target_Log_Return: Samsung Electronics next-week log return.")
    print("\nBase close/return features")
    print(f" - {price_features}")
    print("   Reason: Samsung is the target asset; KOSPI and Bitcoin are market-learning aids.")
    print("\nSentiment proxy features")
    print(f" - PCA index: {pca_col}")
    print("   Reason: tests whether one compressed Samsung sentiment proxy is useful.")
    print(f" - Individual residuals: {residual_cols}")
    print("   Reason: tests whether Samsung ATR/MFI/STOCH residual signals add useful micro information.")
    print("\nExperiment design")
    print(" - Price-only Baseline: Samsung + KOSPI + Bitcoin close/return features only.")
    print(" - Model A: base features + Samsung PCA sentiment index.")
    print(" - Model B: base features + Samsung individual residuals.")
    print(" - Model C: base features + Samsung PCA sentiment index + Samsung individual residuals.")


def main():
    print("=========================================================")
    print(" [1단계] 패널 데이터 로드 및 삼성전자 타깃 데이터셋 구성")
    print("=========================================================")
    panel_list = []
    for ticker, name in sc.SYMBOLS.items():
        df_asset = sc.build_weekly_df(
            symbol=ticker,
            label=name,
            start_date=sc.START_DATE,
            end_date=sc.END_DATE,
            warmup_start=sc.WARMUP_START,
        )
        df_asset["Ticker"] = name
        panel_list.append(df_asset)

    df_full = pd.concat(panel_list).sort_index()

    print("=========================================================")
    print(" [2단계] 자산별 OLS 잔차 추출")
    print("=========================================================")
    df_processed = extract_sentiment_residuals(df_full)

    print("=========================================================")
    print(" [3단계] 삼성전자 타깃 + KOSPI/BTC 시장학습 보조자료 Feature Engineering")
    print("=========================================================")
    df_final = build_samsung_target_dataset(df_processed)
    print(f"최종 학습 데이터: {len(df_final)}주 ({df_final.index[0].date()} ~ {df_final.index[-1].date()})")

    print("=========================================================")
    print(" [4단계] Model A/B/C 및 Baseline 대조 실험")
    print("=========================================================")

    y = df_final["Target_Log_Return"]

    price_features = (
        ["Samsung_Close"]
        + [f"Samsung_Log_Return_lag{i}" for i in range(1, N_LAGS + 1)]
        + ["KOSPI_Close"]
        + [f"KOSPI_Log_Return_lag{i}" for i in range(1, N_LAGS + 1)]
        + ["Bitcoin_Close"]
        + [f"Bitcoin_Log_Return_lag{i}" for i in range(1, N_LAGS + 1)]
    )
    residual_cols = RESIDUAL_COLS
    pca_col = [SENTIMENT_COL]
    print_feature_summary(price_features, residual_cols, pca_col)

    split_idx = int(len(df_final) * 0.8)
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    results = []
    prediction_map = {}

    zero_metrics, y_pred_zero = evaluate_constant_baseline("Zero-return Baseline", y_test, 0.0)
    train_mean_metrics, y_pred_train_mean = evaluate_constant_baseline("Train-mean Baseline", y_test, y_train.mean())
    results.extend([zero_metrics, train_mean_metrics])
    prediction_map["Zero-return Baseline"] = y_pred_zero
    prediction_map["Train-mean Baseline"] = y_pred_train_mean

    X_base = df_final[price_features]
    X_train_b, X_test_b = X_base.iloc[:split_idx], X_base.iloc[split_idx:]
    _, metrics_b, y_pred_b = train_and_evaluate("Price-only Baseline", X_train_b, y_train, X_test_b, y_test)
    results.append(metrics_b)
    prediction_map["Price-only Baseline"] = y_pred_b

    X_A = df_final[price_features + pca_col]
    X_train_A, X_test_A = X_A.iloc[:split_idx], X_A.iloc[split_idx:]
    model_A, metrics_A, y_pred_A = train_and_evaluate("Model A (PCA 통합)", X_train_A, y_train, X_test_A, y_test)
    results.append(metrics_A)
    prediction_map["Model A (PCA 통합)"] = y_pred_A
    joblib.dump(model_A, "models/best_xgboost_panel_model_A.pkl")

    X_B = df_final[price_features + residual_cols]
    X_train_B, X_test_B = X_B.iloc[:split_idx], X_B.iloc[split_idx:]
    model_B, metrics_B, y_pred_B = train_and_evaluate("Model B (세부 잔차)", X_train_B, y_train, X_test_B, y_test)
    results.append(metrics_B)
    prediction_map["Model B (세부 잔차)"] = y_pred_B
    joblib.dump(model_B, "models/best_xgboost_panel_model_B.pkl")

    X_C = df_final[price_features + pca_col + residual_cols]
    X_train_C, X_test_C = X_C.iloc[:split_idx], X_C.iloc[split_idx:]
    model_C, metrics_C, y_pred_C = train_and_evaluate("Model C (전체 혼용)", X_train_C, y_train, X_test_C, y_test)
    results.append(metrics_C)
    prediction_map["Model C (전체 혼용)"] = y_pred_C
    joblib.dump(model_C, "models/best_xgboost_panel_model_C.pkl")

    print("\n=========================================================")
    print(" [최종] Model 대조 실험 결과 요약 (Test Set)")
    print("=========================================================")
    res_df = pd.DataFrame(results)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(res_df.to_string(index=False))

    period_df = build_period_backtest(y_test, prediction_map)
    print("\n=========================================================")
    print(" [추가] 12주 단위 구간별 방향 정확도")
    print("=========================================================")
    print(period_df.to_string(index=False))
    save_backtest_outputs(res_df, period_df)

    print("\n훈련된 Model A, B, C 저장 완료: 'models/best_xgboost_panel_model_A.pkl', 'models/best_xgboost_panel_model_B.pkl', 'models/best_xgboost_panel_model_C.pkl'")


if __name__ == "__main__":
    main()
