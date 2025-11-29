from datetime import date, datetime

import pandas as pd


class JmaClient:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self.df["年月日"] = self.df["年月日"].apply(
            lambda x: datetime.strptime(x, "%Y/%m/%d").date()
        )

        self.df.set_index("年月日", drop=True, inplace=True)
        print(self.df)

        assert list(self.df.columns) == [
            "平均気温(℃)",
            "最高気温(℃)",
            "最低気温(℃)",
            "日照時間(時間)",
            "最深積雪(cm)",
            "降雪量合計(cm)",
            "平均雲量(10分比)",
            "降水量の合計(mm)",
            "平均風速(m/s)",
            "最大風速(m/s)",
            "平均蒸気圧(hPa)",
            "平均湿度(％)",
        ]

    def get_flag(self, date_idx, col):
        assert col in ["降雪量合計(cm)", "降水量の合計(mm)"]
        if self.df.loc[date_idx, col] == 0:
            return 0
        else:
            return 1

    def get_value(self, date_idx, col):
        return self.df.loc[date_idx, col]
