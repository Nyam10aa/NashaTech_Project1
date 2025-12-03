from datetime import date

from src.approach.aa_xgboost.utils.jma import JmaClient


def test_jma():

    jma_client = JmaClient("data/外部データ/jma_札幌.csv")

    assert jma_client.get_flag(date_idx=date(2022, 10, 1), col="降雪量合計(cm)") == 0
    assert jma_client.get_flag(date_idx=date(2022, 11, 30), col="降雪量合計(cm)") == 1

    assert jma_client.get_flag(date_idx=date(2022, 10, 1), col="降水量の合計(mm)") == 0
    assert jma_client.get_flag(date_idx=date(2022, 11, 30), col="降水量の合計(mm)") == 1
