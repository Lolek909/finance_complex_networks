import yfinance as yf
import pandas as pd
import os

# 1. Konfiguracja
tickers = {
    'Tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'INTC', 'ADBE'],
    'Finance': ['JPM', 'BAC', 'GS', 'MS', 'WFC', 'C', 'V', 'MA', 'AXP', 'PYPL'],
    'Energy': ['XOM', 'CVX', 'SHEL', 'BP', 'TTE', 'COP', 'SLB', 'PBR', 'EQNR', 'VLO'],
    'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'LLY', 'MRK', 'TMO', 'AZN', 'NVO', 'DHR']
}

all_tickers = [item for sublist in tickers.values() for item in sublist]
sector_map = {ticker: sector for sector, t_list in tickers.items() for ticker in t_list}
FILE_NAME = "../finance_complex_networks/data_finance/stock_data_1m.csv"

# 2. Pobieranie danych z obsługą błędów i zapisem do pliku
if not os.path.exists(FILE_NAME):
    print("Pobieranie danych z Yahoo Finance...")
    # Używamy auto_adjust=True, wtedy kolumna nazywa się po prostu 'Close' i jest już skorygowana
    raw_data = yf.download(all_tickers, period="5d", interval="1m", auto_adjust=True)

    # Wybieramy ceny zamknięcia ('Close')
    # Przy wielu tickerach kolumny to MultiIndex: (Cena, Ticker)
    if 'Close' in raw_data.columns:
        data = raw_data['Close']
    else:
        # Failsafe: wypisz kolumny, jeśli znowu coś się zmieni w API
        print("Dostępne kolumny:", raw_data.columns)
        raise KeyError("Nie znaleziono kolumny 'Close' lub 'Adj Close'")

    data.to_csv(FILE_NAME)
    print(f"Dane zapisane do {FILE_NAME}")
else:
    print(f"Wczytywanie danych z pliku {FILE_NAME}...")
    data = pd.read_csv(FILE_NAME, index_col=0, parse_dates=True)

