from datetime import datetime

from src.approach.aa_xgboost.utils.holiday import HolidayClient


def test_holiday():

    holiday_client = HolidayClient("data/外部データ/syukujitsu.csv")

    assert (
        holiday_client.get_holiday_flag(datetime_idx=datetime(2023, 10, 9, 0, 0, 0))
        == 1
    )  # Sports day
    assert (
        holiday_client.get_holiday_flag(datetime_idx=datetime(2023, 10, 10, 0, 0, 0))
        == 0
    )

    assert (
        holiday_client.get_holiday_name(datetime_idx=datetime(2023, 10, 9, 0, 0, 0))
        == "スポーツの日"
    )
