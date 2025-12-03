from datetime import date, datetime, timedelta

import pandas as pd


class WeatherClient:
    def __init__(self, historical_file_list, forecast_file_list):
        self.df_actual = pd.concat(
            [self._load_file_(file_path) for file_path in historical_file_list]
        )

        self.df_forecast = pd.concat(
            [self._load_file_(file_path) for file_path in forecast_file_list]
        )

        self.col_map = {
            "T_mean": "temperature_2m_mean (°C)",
            "T_max": "temperature_2m_max (°C)",
            "T_min": "temperature_2m_min (°C)",
            "rain": "rain_sum (mm)",
            "snow": "snowfall_sum (cm)",
        }

        print("--------------- Openmeteo data ---------------")
        print("Actual (historical) data:")
        print(self.df_actual)
        print("Forecast data:")
        print(self.df_forecast)

    def _load_file_(self, file_path):
        df = pd.read_csv(file_path, skiprows=3)
        df["time"] = df["time"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d").date())

        df.set_index("time", drop=True, inplace=True)
        print(df)

        return df

    def get_numerical_value(self, datetime_idx, actual_or_forecast, cols):
        # date_idx = (datetime_idx - timedelta(minutes=10)).date()
        date_idx = datetime_idx.date()

        if actual_or_forecast == "actual":
            if date_idx in self.df_actual.index:
                return [self.df_actual.loc[date_idx, self.col_map[col]] for col in cols]
        elif actual_or_forecast == "forecast":
            if date_idx in self.df_forecast.index:
                return [
                    self.df_forecast.loc[date_idx, self.col_map[col]] for col in cols
                ]
        return [None for col in cols]

    def get_category_value(self, datetime_idx, actual_or_forecast, cols):
        date_idx = datetime_idx.date()

        out = []
        for col in cols:
            assert col in ["rain", "snow"]
            if actual_or_forecast == "actual" and date_idx in self.df_actual.index:
                if self.df_actual.loc[date_idx, self.col_map[col]] == 0:
                    out.append(0)
                else:
                    out.append(1)
            elif (
                actual_or_forecast == "forecast" and date_idx in self.df_forecast.index
            ):
                if self.df_forecast.loc[date_idx, self.col_map[col]] == 0:
                    out.append(0)
                else:
                    out.append(1)
            else:
                out.append(None)
        return out
