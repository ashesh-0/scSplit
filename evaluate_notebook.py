import argparse
import papermill as pm
from datetime import datetime
import os



if __name__ == '__main__':
    # python notebooks/evaluate_notebook.py --ckpt=2502/HT_LIF24-joint_indi-l1/60 --mmse_count=10  --notebook=/home/ashesh.ashesh/code/DiffSplitting/notebooks/EvaluateJointIndiRealInput.ipynb --ckpt_time_predictor=2501/COSEM_jrc-hela-UnetClassifier-l2/6 --ckpt=2502/COSEM_jrc-hela-joint_indi-l1/33 --infer_time=true --use_aggregated_inferred_time=true
    # python notebooks/evaluate_notebook.py --ckpt=2502/HT_LIF24-joint_indi-l1/60 --mmse_count=1 --mixing_t_ood=0.5 --notebook=/home/ashesh.ashesh/code/DiffSplitting/notebooks/EvaluateInputvsTarget.ipynb
    parser = argparse.ArgumentParser(description='Run a notebook')
    parser.add_argument('--notebook', type=str, help='Notebook to run', default='/home/ashesh.ashesh/code/DiffSplitting/notebooks/EvaluateJointIndi.ipynb')
    parser.add_argument('--outputdir', type=str, help='Output notebook directory', default='/group/jug/ashesh/indiSplit/notebook_results/')
    # parser.add_argument('parameters', type=str, help='Parameters for the notebook')
    parser.add_argument('--ckpt', type=str, help='Checkpoint to use. eg. 2502/Hagen-joint_indi-l1/57')
    parser.add_argument('--num_steps_normalization', type=int, help='Number of epochs for normalization', default=10000)
    parser.add_argument('--ckpt_time_predictor', type=str, help='Checkpoint for time predictor', default=None)
    parser.add_argument('--mixing_t_ood', type=float, help='Mixing parameter for input generation', default=0.5)
    parser.add_argument('--mmse_count', type=int, help='Number of mmse values to generate', default=10)
    parser.add_argument('--num_timesteps', type=int, help='Number of timesteps to use', default=1)
    parser.add_argument('--enable_real_input', type=bool, help='Enable real input', default=False)
    parser.add_argument('--infer_time', type=bool, help='Infer time', default=True)
    parser.add_argument('--use_aggregated_inferred_time', type=bool, help='Use aggregated inferred time', default=True)
    parser.add_argument('--use_hardcoded_time_for_inference', type=float, help='Use hardcoded time for inference', default=None)
    parser.add_argument('--input_channel_idx', type=int, help='Input channel index', default=2)

    args = parser.parse_args()

    # get a year-month-day hour-minute formatted string
    param_str = f"T-{args.mixing_t_ood}_InpIdx-{args.input_channel_idx}_MMSE-{args.mmse_count}_InferT-{int(args.infer_time)}_InferTAgg-{int(args.use_aggregated_inferred_time)}_FixedT-{args.use_hardcoded_time_for_inference}"
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
            'num_steps_normalization': args.num_steps_normalization,
            'ckpt_time_predictor': args.ckpt_time_predictor,
            'mixing_t_ood': args.mixing_t_ood,
            'mmse_count': args.mmse_count,
            'num_timesteps': args.num_timesteps,
            'enable_real_input': args.enable_real_input,
            'infer_time': args.infer_time,
            'use_aggregated_inferred_time': args.use_aggregated_inferred_time,
            'use_hardcoded_time_for_inference': args.use_hardcoded_time_for_inference,
            'input_channel_idx': args.input_channel_idx
        }
    )
    
