import argparse
from typing import Any, Dict, List, Tuple, Union
from simulator import Simulator


def parse_arguments() -> Any:
    """Arguments for running the baseline.

    Returns:
        parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, default="", type=str)

    # parser.add_argument("--mode", default="val", type=str, help="Mode, train/val/test")
    # parser.add_argument("--features_dir", required=True, default="", type=str, help="Path to the dataset")
    # parser.add_argument("--use_cuda", action="store_true", help="Use CUDA for acceleration")
    # parser.add_argument("--data_aug", action="store_true", help="Enable data augmentation")
    # parser.add_argument("--adv_cfg_path", required=True, default="", type=str)
    # parser.add_argument("--model_path", required=False, type=str, help="path to the saved model")
    # #
    # parser.add_argument("--seq_id", default=-1, type=int, help="Selected sequence ID")
    # parser.add_argument("--shuffle", action="store_true", help="Shuffle order")
    # parser.add_argument("--visualizer", default="", type=str, help="Type of visualizer")
    # parser.add_argument("--show_conditioned", action="store_true", help="Show missed sample only")
    return parser.parse_args()


def main():
    args = parse_arguments()
    simulator = Simulator(args.config)
    simulator.run()


if __name__ == "__main__":
    main()
