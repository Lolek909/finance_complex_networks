import os
import pandas as pd
import yfinance as yf

tickers = {
    'Tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'INTC', 'ADBE'],
    'Finance': ['JPM', 'BAC', 'GS', 'MS', 'WFC', 'C', 'V', 'MA', 'AXP', 'PYPL'],
    'Energy': ['XOM', 'CVX', 'SHEL', 'BP', 'TTE', 'COP', 'SLB', 'PBR', 'EQNR', 'VLO'],
    'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'LLY', 'MRK', 'TMO', 'AZN', 'NVO', 'DHR']
}

all_tickers = [item for sublist in tickers.values() for item in sublist]
sector_map = {ticker: sector for sector, t_list in tickers.items() for ticker in t_list}
DATA_DIR = "../data_finance"

def get_data(period="5d", interval="5m", volume=False):
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    vol_suffix = "_vol" if volume else ""
    file_name = f"{DATA_DIR}/stock_data_{interval}_{period}{vol_suffix}.csv"

    if not os.path.exists(file_name):
        print("Pobieranie danych z Yahoo Finance...")
        # Używamy auto_adjust=True, wtedy kolumna nazywa się po prostu 'Close' i jest już skorygowana
        raw_data = yf.download(all_tickers, period=period, interval=interval, auto_adjust=True)

        # Wybieramy ceny zamknięcia ('Close')
        # Przy wielu tickerach kolumny to MultiIndex: (Cena, Ticker)
        if volume:
            if 'Close' in raw_data.columns and 'Volume' in raw_data.columns:
                 data = raw_data[['Close', 'Volume']]
            else:
                raise KeyError("Nie znaleziono kolumny 'Close' lub 'Volume'")
        else:
            if 'Close' in raw_data.columns:
                data = raw_data['Close']
            else:
                print("Dostępne kolumny:", raw_data.columns)
                raise KeyError("Nie znaleziono kolumny 'Close'")

        data.to_csv(file_name)
        print(f"Dane zapisane do {file_name}")
    else:
        print(f"Wczytywanie danych z pliku {file_name}...")
        if volume:
            data = pd.read_csv(file_name, index_col=0, header=[0, 1], parse_dates=True)
        else:
            data = pd.read_csv(file_name, index_col=0, parse_dates=True)

    return data