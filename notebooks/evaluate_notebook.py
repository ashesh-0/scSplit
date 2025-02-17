import argparse
import papermill as pm
from datetime import datetime
import os



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a notebook')
    parser.add_argument('--notebook', type=str, help='Notebook to run', default='/home/ashesh.ashesh/code/DiffSplitting/notebooks/EvaluateJointIndi.ipynb')
    parser.add_argument('--outputdir', type=str, help='Output notebook directory', default='/group/jug/ashesh/indiSplit/notebook_results/')
    # parser.add_argument('parameters', type=str, help='Parameters for the notebook')
    parser.add_argument('--ckpt', type=str, help='Checkpoint to use. eg. 2502/Hagen-joint_indi-l1/57')
    parser.add_argument('--num_epochs_normalization', type=int, help='Number of epochs for normalization', default=50)
    parser.add_argument('--ckpt_time_predictor', type=str, help='Checkpoint for time predictor', default=None)
    parser.add_argument('--mixing_t_ood', type=float, help='Mixing parameter for input generation', default=0.5)
    parser.add_argument('--mmse_count', type=int, help='Number of mmse values to generate', default=10)
    parser.add_argument('--num_timesteps', type=int, help='Number of timesteps to use', default=1)
    parser.add_argument('--enable_real_input', type=bool, help='Enable real input', default=False)
    parser.add_argument('--infer_time', type=bool, help='Infer time', default=False)
    parser.add_argument('--use_aggregated_inferred_time', type=bool, help='Use aggregated inferred time', default=False)
    parser.add_argument('--use_hardcoded_time_for_inference', type=float, help='Use hardcoded time for inference', default=None)

    args = parser.parse_args()

    # get a year-month-day hour-minute formatted string
    param_str = f"T-{args.mixing_t_ood}_MMSE-{args.mmse_count}_InferT-{int(args.infer_time)}_InferTAgg-{int(args.use_aggregated_inferred_time)}_FixedT-{args.use_hardcoded_time_for_inference}"
    now = datetime.now().strftime("%Y%m%d.%H.%M")
    outputdir = os.path.join(args.outputdir, args.ckpt.replace('/','_'))
    fname = os.path.basename(args.notebook)
    fname = fname.replace('.ipynb','')
    fname = f"{fname}_{param_str}_{now}.ipynb"
    output_fpath = os.path.join(outputdir, fname)
    output_config_fpath = os.path.join(outputdir,'config', fname.replace('.ipynb','.txt'))
    os.makedirs(os.path.dirname(output_config_fpath), exist_ok=True)
    # save the configuration
    # convert args to dict
    args_dict = vars(args)
    # save as json
    with open(output_config_fpath, 'w') as f:
        f.write(str(args_dict))

    print(output_fpath, '\n', output_config_fpath)
    pm.execute_notebook(
        args.notebook,
        output_fpath,
        parameters = {
            'ckpt': args.ckpt,
            'num_epochs_normalization': args.num_epochs_normalization,
            'ckpt_time_predictor': args.ckpt_time_predictor,
            'mixing_t_ood': args.mixing_t_ood,
            'mmse_count': args.mmse_count,
            'num_timesteps': args.num_timesteps,
            'enable_real_input': args.enable_real_input,
            'infer_time': args.infer_time,
            'use_aggregated_inferred_time': args.use_aggregated_inferred_time,
            'use_hardcoded_time_for_inference': args.use_hardcoded_time_for_inference
        }
    )
    
