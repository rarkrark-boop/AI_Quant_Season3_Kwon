from pathlib import Path

import pandas as pd


OUTPUT_DIR = Path("data/output")


def load_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def save_csv(df: pd.DataFrame, path: str | Path, **kwargs) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, **kwargs)
    return target


def save_output_csv(df: pd.DataFrame, filename: str, **kwargs) -> Path:
    data = df.copy()
    if data.index.name is None:
        data.index.name = "Date"
    return save_csv(data, OUTPUT_DIR / filename, **kwargs)


def safe_filename(value: str) -> str:
    safe = "".join(char if char.isalnum() else "_" for char in value)
    return safe.strip("_") or "output"


def validate_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {', '.join(missing)}")


def drop_missing_rows(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    validate_columns(df, columns)
    return df.dropna(subset=columns).copy()
