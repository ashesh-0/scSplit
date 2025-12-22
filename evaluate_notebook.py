import argparse
import papermill as pm
from datetime import datetime
import os
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a notebook")
    parser.add_argument(
        "--notebook", type=str, help="Notebook to run", default="notebooks/CoveragePlotSyntheticInput.ipynb"
    )
    parser.add_argument(
        "--outputdir",
        type=str,
        help="Output notebook directory",
        default="/group/jug/ashesh/UQResults/notebook_results/scSplit",
    )
    parser.add_argument(
        "--training_rootdir",
        type=str,
        help="Storage root dir for training",
        default="/group/jug/ashesh/training/diffsplit/",
    )
    parser.add_argument("--ckpt_dir", type=str, help="Checkpoint to use. eg. 2502/Hagen-joint_indi-l1/63")
    parser.add_argument(
        "--ckpt_time_predictor",
        type=str,
        help="Checkpoint to use. eg. 2502/Hagen-UnetClassifier-l2/11",
    )

    parser.add_argument("--data_split_type", type=str, help="Data split type: Val/Test", default="Test")
    parser.add_argument("--tag_date", action="store_true", help="Tag time flag", default=False)
    parser.add_argument(
        "--override_kwargs",
        type=json.loads,
        default="{}",
    )
    args = parser.parse_args()
    assert args.data_split_type in ["Val", "Test"]

    param_dict = args.override_kwargs
    keys = sorted(param_dict.keys())
    param_str = "_".join([f"{k}-{param_dict[k]}" for k in keys if k != "data_dir"])
    param_str += f"_{args.data_split_type}"
    ckpt_dir = args.ckpt_dir
    model_token = "-".join(ckpt_dir.strip("/").split("/")[-3:])
    outputdir = os.path.join(args.outputdir, model_token)
    fname = os.path.basename(args.notebook)
    fname = fname.replace(".ipynb", "")
    fname = f"{fname}_{param_str}"

    param_dict["ckpt_dir"] = args.ckpt_dir
    param_dict["ckpt_time_predictor"] = args.ckpt_time_predictor
    param_dict["training_rootdir"] = args.training_rootdir
    output_configdir = os.path.join(outputdir, "config")
    output_resultsdir = os.path.join(outputdir, "results")
    # output_config_fpath = os.path.join(outputdir, "config", fname.replace(".ipynb", ".txt"))
    # output_results_fpath = os.path.join(outputdir, "results", fname.replace(".ipynb", ".pkl"))
    os.makedirs(output_configdir, exist_ok=True)
    os.makedirs(output_resultsdir, exist_ok=True)
    # save the configuration
    # convert args to dict
    args_dict = vars(args)

    output_results_fpath = os.path.join(output_resultsdir, fname + ".pkl")
    if args.data_split_type == "Test":
        calibration_params_fpath = output_results_fpath.replace("_Test_", "_Val_").replace("_Test.", "_Val.")
        assert os.path.exists(calibration_params_fpath), f"Calibration params not found: {calibration_params_fpath}"
        param_dict["calibration_params_fpath"] = calibration_params_fpath
        print("Calibration Params:", calibration_params_fpath)
        output_results_fpath = None

    if args.tag_date:
        now = datetime.now().strftime("%Y%m%d")
        fname += f"_{now}.ipynb"
    else:
        fname += ".ipynb"
    output_fpath = os.path.join(outputdir, fname)

    # save as json
    output_config_fpath = os.path.join(output_configdir, fname.replace(".ipynb", ".txt"))
    with open(output_config_fpath, "w") as f:
        f.write(str(args_dict))

    param_dict["eval_datasplit_type"] = args.data_split_type
    param_dict["notebook_output_fpath"] = output_results_fpath
    print(output_fpath, "\n", output_config_fpath, "\n Data Split Evaluated:", args.data_split_type)
    pm.execute_notebook(args.notebook, output_fpath, parameters=param_dict)
