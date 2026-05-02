import time


def main():
    from src.data import get_data
    df1 = get_data(period="5d", interval="5m", volume=True)
    time.sleep(5)

    df2 = get_data(period="5d", interval="5m", volume=False)
    time.sleep(5)

    df3 = get_data(period="1mo", interval="1h", volume=True)
    time.sleep(5)

    df4 = get_data(period="1mo", interval="1h", volume=False)
    time.sleep(5)

    df5 = get_data(period="1y", interval="1d", volume=True)
    time.sleep(5)

    df6 = get_data(period="1y", interval="1d", volume=False)

if __name__ == "__main__":
    main()
