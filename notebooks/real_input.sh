python notebooks/evaluate_notebook.py --enable_real_input=true --ckpt=2502/HT_LIF24-joint_indi-l1/60 --mmse_count=10  --notebook=/home/USERNAME.USERNAME/code/DiffSplitting/notebooks/EvaluateJointIndiRealInput.ipynb --ckpt_time_predictor=2502/HT_LIF24-UnetClassifier-l2/3  --infer_time=true --use_aggregated_inferred_time=true

python notebooks/evaluate_notebook.py --enable_real_input=true --ckpt=2502/HT_LIF24-joint_indi-l1/60 --mmse_count=10  --notebook=/home/USERNAME.USERNAME/code/DiffSplitting/notebooks/EvaluateJointIndiRealInput.ipynb --ckpt_time_predictor=2502/HT_LIF24-UnetClassifier-l2/3  --infer_time=true

python notebooks/evaluate_notebook.py --enable_real_input=true --ckpt=2502/HT_LIF24-joint_indi-l1/60 --mmse_count=10  --notebook=/home/USERNAME.USERNAME/code/DiffSplitting/notebooks/EvaluateJointIndiRealInput.ipynb  --use_hardcoded_time_for_inference=0.5

python notebooks/evaluate_notebook.py --enable_real_input=true --ckpt=2502/HT_LIF24-joint_indi-l1/69 --mmse_count=10  --notebook=/home/USERNAME.USERNAME/code/DiffSplitting/notebooks/EvaluateJointIndiRealInput.ipynb  --use_hardcoded_time_for_inference=0.5

# HT_T24
python notebooks/evaluate_notebook.py --enable_real_input=true --ckpt=2502/HT_T24-joint_indi-l1/15 --mmse_count=10  --notebook=/home/USERNAME.USERNAME/code/DiffSplitting/notebooks/EvaluateJointIndiRealInput.ipynb  --use_hardcoded_time_for_inference=0.5


# input vs tar
python notebooks/evaluate_notebook.py --ckpt=2502/HT_T24-joint_indi-l1/14 --mmse_count=1 --mixing_t_ood=0.5 --notebook=/home/USERNAME.USERNAME/code/DiffSplitting/notebooks/EvaluateInputvsTarget.ipynb --enable_real_input=true

python notebooks/evaluate_notebook.py --ckpt=2502/HT_LIF24-joint_indi-l1/60 --mmse_count=1 --mixing_t_ood=0.5 --notebook=/home/USERNAME.USERNAME/code/DiffSplitting/notebooks/EvaluateInputvsTarget.ipynb --enable_real_input=true
