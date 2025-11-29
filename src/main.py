import argparse
import traceback

import mlflow

from approach.aa_xgboost.train import train as aa_train
from data_analysis.preprocess import f


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--use-mlflow", action="store_true", help="Enable MLflow logging"
    )
    parser.add_argument("--experiment", type=str, default="Default")
    parser.add_argument(
        "--no-save-model", action="store_false", help="Enable MLflow logging"
    )
    parser.add_argument("--data", type=str, default="data/processed")

    # subparser
    subparsers = parser.add_subparsers(dest="mode")

    # -------------------- visualize data --------------------
    parser_data = subparsers.add_parser("data_analysis", help="")
    parser_data.add_argument("--data_path", type=str, default="data/raw")

    # -------------------- approach: aa_xgboost --------------------
    parser_approach_aa = subparsers.add_parser("aa_xgboost_train", help="")
    parser_approach_aa.add_argument(
        "--train_start", type=str, default="2022-10-01 00:10:00"
    )
    parser_approach_aa.add_argument(
        "--train_end", type=str, default="2023-10-01 00:00:00"
    )
    parser_approach_aa.add_argument(
        "--test_start", type=str, default="2023-10-01 00:10:00"
    )
    parser_approach_aa.add_argument(
        "--test_end", type=str, default="2024-10-01 00:00:00"
    )

    parser_approach_aa.add_argument("--categorical_col", type=str, default="")
    parser_approach_aa.add_argument("--numerical_col", type=str, default="")
    parser_approach_aa.add_argument("--target_col", type=str, default="供給先B冷水熱量")

    parser_approach_aa.add_argument(
        "--jma_categorical_col", type=str, default=""
    )  # 降雪量合計(cm),降水量の合計(mm)
    parser_approach_aa.add_argument(
        "--jma_numerical_col", type=str, default="平均気温(℃),最高気温(℃),最低気温(℃)"
    )
    parser_approach_aa.add_argument("--model_n_estimators", type=int, default=500)
    parser_approach_aa.add_argument("--model_learning_rate", type=float, default=0.01)
    parser_approach_aa.add_argument("--model_max_depth", type=int, default=5)
    parser_approach_aa.add_argument("--model_subsample", type=float, default=0.9)
    parser_approach_aa.add_argument("--model_colsample_bytree", type=float, default=0.8)
    parser_approach_aa.add_argument(
        "--model_objective", type=str, default="reg:squarederror"
    )

    # --------------------  --------------------
    args = parser.parse_args()

    kwargs = vars(args)

    if args.use_mlflow:
        mlflow.set_experiment(args.experiment)
        mlflow.start_run()
        mlflow.log_params(kwargs)

    print(args)
    try:
        if args.mode == "data_analysis":
            f(args)
        elif args.mode == "aa_xgboost_train":
            aa_train(**kwargs)

    except Exception as e:
        if args.use_mlflow:
            # Log exception as a tag or param
            mlflow.log_param("error_type", type(e).__name__)
            mlflow.log_param("error_message", str(e))

            # Log full traceback
            mlflow.log_text(traceback.format_exc(), "error_traceback.txt")

            # Mark MLflow run as failed
            mlflow.set_tag("mlflow.runStatus", "FAILED")
        else:
            raise e
    finally:
        if args.use_mlflow:
            mlflow.end_run()


if __name__ == "__main__":
    main()
