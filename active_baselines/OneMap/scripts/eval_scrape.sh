#!/usr/bin/env bash
# Copyright [2023] Boston Dynamics AI Institute, Inc.

# Ensure you have 'export VLFM_PYTHON=<PATH_TO_PYTHON>' in your .bashrc, where
# <PATH_TO_PYTHON> is the path to the python executable for your conda env
# (e.g., PATH_TO_PYTHON=`conda activate <env_name> && which python`)


export CURR_PYTHON='/mnt/anaconda3/envs/one_map/bin/python'

session_name=onemap_scrape

# Create a detached tmux session
tmux new-session -d -s ${session_name}

# Split the window vertically
tmux split-window -v -t ${session_name}:0

# Run commands in each pane
tmux send-keys -t ${session_name}:0.0 \
    "${CURR_PYTHON} eval_habitat.py \
    --config config/mon/eval_conf.yaml  \
    --PlanningConf.using_ov \
    --EvalConf.results_dir RAL/ovon_equal/scrape/split_1 \
    --PlanningConf.multi_prompt \
    --EvalConf.run_split 1 \
    " C-m

tmux send-keys -t ${session_name}:0.1 \
    "${CURR_PYTHON} eval_habitat.py \
    --config config/mon/eval_conf.yaml  \
    --PlanningConf.using_ov \
    --EvalConf.results_dir RAL/ovon_equal/scrape/split_2 \
    --PlanningConf.multi_prompt \
    --EvalConf.run_split 2 \
    " C-m


# Attach to the tmux session to view the windows
echo "Created tmux session '${session_name}'"
echo "Run the following to monitor all the server commands:"
echo "tmux attach-session -t ${session_name}"


