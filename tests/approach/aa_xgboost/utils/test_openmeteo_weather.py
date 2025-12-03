from datetime import date, datetime

from src.approach.aa_xgboost.utils.openmeteo_weather import WeatherClient


def test_openmeteo_weather():

    weather_forecast_client = WeatherClient(
        historical_file_list=[
            "data/外部データ/source_openmeteo/train data 20221001-20230930/(Actual)Historical weather data 20221001_20230930(Sapporo,daily).csv",
        ],
        forecast_file_list=[
            "data/外部データ/source_openmeteo/train data 20221001-20230930/(Forecast)Weather forecast data 20221001_20230930(Sapporo,daily).csv",
            "data/外部データ/source_openmeteo/test data 20231001-20240930/「Weather forecast data 20231001_20240930(Sapporo,daily).csv",
        ],
    )
    cols = ["T_max", "T_min", "T_mean"]

    # temperature
    assert weather_forecast_client.get_numerical_value(
        datetime_idx=datetime(2022, 10, 1, 0, 10, 0),
        actual_or_forecast="actual",
        cols=cols,
    ) == [25.6, 15.4, 19.5]

    assert weather_forecast_client.get_numerical_value(
        datetime_idx=datetime(2023, 9, 30, 0, 0, 0),
        actual_or_forecast="actual",
        cols=cols,
    ) == [21, 8.9, 16.1]

    assert weather_forecast_client.get_numerical_value(
        datetime_idx=datetime(2023, 10, 1, 0, 0, 0),
        actual_or_forecast="forecast",
        cols=cols,
    ) == [23.9, 14.7, 18.7]

    assert weather_forecast_client.get_numerical_value(
        datetime_idx=datetime(2024, 9, 30, 23, 50, 0),
        actual_or_forecast="forecast",
        cols=cols,
    ) == [20.6, 12.2, 16.5]

    # rain, snow
    cols = ["rain", "snow"]
    assert weather_forecast_client.get_category_value(
        datetime_idx=datetime(2022, 10, 1, 0, 10, 0),
        actual_or_forecast="actual",
        cols=cols,
    ) == [0, 0]

    assert weather_forecast_client.get_category_value(
        datetime_idx=datetime(2022, 10, 4, 0, 0, 0),
        actual_or_forecast="actual",
        cols=cols,
    ) == [1, 0]
