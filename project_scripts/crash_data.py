import os
import pandas as pd

from src.data import get_data

DATA_DIR = "../data_finance/comparison"

AMZN_CRASH_DATE = "2022-04-29"

def prepare_crash_dataset(
    period="6y",
    interval="1d",
    window_before=60,
    window_after=60
):
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Loading full dataset via get_data()...")

    data = get_data(period=period, interval=interval, volume=False)

    data.index = pd.to_datetime(data.index)

    event_date = pd.to_datetime(AMZN_CRASH_DATE)

    # --- split ---
    before_start = event_date - pd.Timedelta(days=window_before)
    before_end = event_date - pd.Timedelta(days=1)

    after_start = event_date
    after_end = event_date + pd.Timedelta(days=window_after)

    before = data.loc[before_start:before_end]
    after = data.loc[after_start:after_end]

    before_path = f"{DATA_DIR}/before_amzn_crash_{event_date.date()}.csv"
    after_path = f"{DATA_DIR}/after_amzn_crash_{event_date.date()}.csv"

    before.to_csv(before_path)
    after.to_csv(after_path)

    print(f"Saved BEFORE crash -> {before_path}")
    print(f"Saved AFTER crash  -> {after_path}")

    return before, after


if __name__ == "__main__":
    before, after = prepare_crash_dataset()