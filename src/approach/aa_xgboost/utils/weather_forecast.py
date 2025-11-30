from datetime import date, datetime, timedelta

import pandas as pd


class WeatherForecastClient:
    def __init__(self, file_path_list):
        self.df = pd.concat(
            [self._load_file_(file_path) for file_path in file_path_list]
        )
        # self.cols = ["temperature_2m_max (°C)", "temperature_2m_min (°C)", "temperature_2m_mean (°C)"]

        self.col_map = {
            "平均気温(℃)": "temperature_2m_mean (°C)",
            "最高気温(℃)": "temperature_2m_max (°C)",
            "最低気温(℃)": "temperature_2m_min (°C)",
        }
        print(self.df)

    def _load_file_(self, file_path):
        df = pd.read_csv(file_path, skiprows=2)
        df["time"] = df["time"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d").date())

        df.set_index("time", drop=True, inplace=True)
        print(df)

        return df

    def get_daily_forecast(self, datetime_idx, cols):
        date_idx = (datetime_idx - timedelta(minutes=10)).date()
        return [self.df.loc[date_idx, self.col_map[col]] for col in cols]
