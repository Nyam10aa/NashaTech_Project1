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
        self.cache_sensor_name_to_idx = dict()

        for sensor_idx in self.metadata.columns:
            self.cache_sensor_name_to_idx[self.metadata.loc["name", sensor_idx]] = (
                sensor_idx
            )

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

            count += 1
            if count == 7 * 5:
                break

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
