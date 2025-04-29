import argparse
import yaml
from src import utils
import src.validation as val
import datetime


def main():
    """
    Initiates the validation of the classifier specified in the constants file.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--network", 
        type=str, 
        help="name of the network that should be used"
    )
    args = parser.parse_args()
    network = args.network
    raw_val_data = utils.load_raw_data(validation=True)

    metrics_dict = {}
    metrics_dict["network"] = network
    metrics_dict["metrics"] = val.unsupervised_metrics(raw_val_data)
    metrics_dict["data stats"] = val.data_stats(raw_val_data)
    metrics_dict["parameters"] = yaml.full_load(open("params.yaml", "r"))

    model_name = "unsup_" + datetime.datetime.now().strftime("%m-%d_%H-%M")
    utils.save_metrics(metrics_dict, model_name, network)


if __name__ == "__main__":
    main()
