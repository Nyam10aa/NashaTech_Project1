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
    parser.add_argument("--data", type=str, default="data/processed")

    # subparser
    subparsers = parser.add_subparsers(dest="mode")

    # -------------------- visualize data --------------------
    parser_data = subparsers.add_parser("data_analysis", help="")
    parser_data.add_argument("--data_path", type=str, default="data/raw")

    # -------------------- approach: aa_xgboost --------------------
    parser_approach_aa = subparsers.add_parser("aa_xgboost_train", help="")

    # --------------------  --------------------
    args = parser.parse_args()

    kwargs = vars(args)

    if args.use_mlflow:
        mlflow.set_experiment(args.mode)
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
