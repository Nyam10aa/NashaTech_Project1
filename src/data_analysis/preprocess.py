# import csv
# import os

# import pandas as pd


def f(args):
    print(args)
    if True:
        file_name = os.listdir(args.data_path)[0]
        # for file_name in sorted(os.listdir(args.data_path)):
        df = pd.read_csv(
            os.path.join(args.data_path, file_name), index_col=0, encoding="shift_jis"
        )
        print(file_name, df.shape)
        for index in ["tag", "objid", "name", "unit", "type"]:
            print("-" * 100)
            print(index, df.loc[index].value_counts())

        # print(df)
        # print(df.loc["objid"].unique())
        # print(df.to_dict())

    print("oka")


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
    def __init__(self, data_path):
        self.data = None
        self.metadata = None

        count = 0
        for folder in ["熱供給プラントA", "熱供給プラントA_2022年"]:
            for file_name in os.listdir(os.path.join(data_path, folder)):
                file_path = os.path.join(data_path, folder, file_name)
                df = pd.read_csv(file_path, index_col=0, encoding="shift_jis")
                df.reset_index(inplace=True)

                print("=" * 100)
                df["data_type"] = df["index"].apply(filt)

                meta_data = df[df["data_type"] == "metadata"].drop("data_type", axis=1)
                if self.metadata is None:
                    self.metadata = meta_data
                else:
                    assert self.metadata.equals(meta_data)
                # print(df[df["data_type"] == "other"])

                data = df[df["data_type"] == "measurement"].drop("data_type", axis=1)
                data["index"] = file_name[:8] + " " + data["index"]
                data["index"] = data["index"].apply(to_datetime)
                data.set_index("index", drop=True, inplace=True)
                # data.sort_index(inplace=True)
                # assert data.shape == (144, 1722)

                # if self.data is None:
                #     self.data = data.to_dict()
                # else:
                #     for k, v in data.to_dict().items():
                #         self.data[k].update(v)

                dt_list = []
                start = datetime.strptime(file_name[:8], "%Y%m%d")
                end = start + timedelta(days=1)
                delta = timedelta(minutes=10)
                current = start
                while current < end:
                    current += delta
                    dt_list.append(current)

                # print(len(dt_list), dt_list[0], dt_list[-1])
                data = data.reindex(dt_list)
                assert data.shape == (144, 1722)

                data.to_csv(f"data/processed/measurement/{file_name}")
                # if count % 2 == 0:
                #     break
                print(file_name)

        self.metadata.to_csv("data/processed/metadata.csv")
        # tmp_df = pd.DataFrame.from_dict(self.data)
        # tmp_df.sort_index(inplace=True)
        # tmp_df.to_csv("data/processed/measurement.csv")

    def get_sensor_value_by_timestamp(self, sensor, time_stamp):
        return self.data[sensor][time_stamp]

    def get_sensor_meta_data(self, sensor):
        return
