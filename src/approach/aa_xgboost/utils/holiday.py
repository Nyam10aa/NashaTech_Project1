from datetime import date, datetime

import pandas as pd


class HolidayClient:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path, encoding="shift_jis")
        self.holiday_name_col = "国民の祝日・休日名称"

        self.df["国民の祝日・休日月日"] = self.df["国民の祝日・休日月日"].apply(
            lambda x: datetime.strptime(x, "%Y/%m/%d").date()
        )
        self.df.set_index("国民の祝日・休日月日", drop=True, inplace=True)

        # self.df.to_csv("data/processed/shyukujitus.csv")

    def get_holiday_flag(self, datetime_idx):
        if datetime_idx.date() in self.df.index:
            return 1
        return 0

    def get_holiday_name(self, datetime_idx):
        return self.df.loc[datetime_idx.date(), self.holiday_name_col]
