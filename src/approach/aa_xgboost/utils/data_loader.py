import os
from datetime import datetime, timedelta

import pandas as pd


def filt(val):
    if (val.count(":") == 2 and len(val) == 8) or (val == "1 day, 0:00:00"):
        return "measurement"
    elif val in ["tag", "objid", "name", "unit", "type"]:
        return "metadata"
    else:
        return "other"


def to_datetime(timestamp_str):
    try:
        return datetime.strptime(timestamp_str, "%Y%m%d %H:%M:%S")
    except:
        try:
            return datetime.strptime(timestamp_str[:8], "%Y%m%d") + timedelta(days=1)
        except Exception as e:
            raise e


class DataClient:
    def __init__(self, data_path, sensor_list):
        self.data = None
        self.metadata = pd.read_csv(f"{data_path}/metadata.csv", index_col=0)
        self.metadata.set_index("index", drop=True, inplace=True)
        # self.cache_sensor_name_to_idx = dict()

        # for sensor_idx in self.metadata.columns:
        #     self.cache_sensor_name_to_idx[self.metadata.loc["name", sensor_idx]] = (
        #         sensor_idx
        #     )

        self.cache_sensor_name_to_idx = {
            # "供給先B冷水熱量": str(1058 - 1),
            "供給先A冷水熱量": str(1291 - 1),
            "供給先A温水熱量(MJ)": str(1294 - 1),
            "供給先A蒸気熱量": str(1288 - 1),
            "供給先B冷水熱量": str(1058 - 1),
            "供給先B温水熱量": str(1059 - 1),
            "供給先B蒸気熱量": str(1299 - 1),
            "供給先C冷水熱量": str(1313 - 1),
            "供給先C蒸気熱量": str(1310 - 1),
            "供給先D冷水熱量": str(1321 - 1),
            "供給先D蒸気熱量": str(1318 - 1),
            "供給先E冷水熱量": str(1218 - 1),
            "供給先E高温水熱量": str(1219 - 1),
            "供給先F冷水熱量": str(1220 - 1),
            "供給先F高温水熱量": str(1221 - 1),
            "供給先G冷水熱量積算": str(1264 - 1),
            "供給先H冷水熱量": str(1345 - 1),
            "供給先H高温水熱量": str(1348 - 1),
            "外気温度": str(938 - 1),
            "外気湿度": str(939 - 1),
        }
        for sensor_name, sensor_idx in self.cache_sensor_name_to_idx.items():
            assert (
                self.metadata.loc["name", sensor_idx] == sensor_name
            ), f"{self.metadata.loc["name", sensor_idx]} -> {sensor_name}"

        sensor_id_list = [
            self.cache_sensor_name_to_idx[sensor_name] for sensor_name in sensor_list
        ]

        count = 0
        for file_name in sorted(os.listdir(f"{data_path}/measurement")):
            data = pd.read_csv(f"{data_path}/measurement/{file_name}", index_col=0)[
                sensor_id_list
            ]
            if self.data is None:
                self.data = data
            else:
                self.data = pd.concat([self.data, data])

        self.data.columns = sensor_list
        self.data.index = pd.to_datetime(
            self.data.index.str.strip(), format="%Y-%m-%d %H:%M:%S"
        )

    def get_all_data_by_sensor_list(self, sensor_list):
        # sensor_id_list = [
        #     self.cache_sensor_name_to_idx[sensor_name] for sensor_name in sensor_list
        # ]
        return self.data[sensor_list]

    def get_data(self, sensor_list, from_date, to_date): ...
