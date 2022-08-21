#!/usr/bin/env bash

experiment="table1_contrastive"
notes="
**Goal**: compares contrastive methods (table 1 contrastive column).
"

# parses special mode for running the script
source "$(dirname $0)"/../utils.sh
source "$(dirname $0)"/base_tin.sh

time=10000

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
$base_kwargs_tin
timeout=$time
representor=cissl
downstream_task.all_tasks=[torchlogisticw1e-4_datarepr,torchlogisticw1e-5_datarepr,torchlogisticw1e-6_datarepr]
seed=1
"
# use seed=1,2,3 for three seeds

cell_baseline="
representor=simclr
"

cell_ours="
"

cell_dim="
encoder.z_dim=2048
encoder.kwargs.arch_kwargs.is_channel_out_dim=True
+encoder.kwargs.arch_kwargs.bottleneck_channel=512
"

cell_epoch="
$cell_dim
update_trainer_repr.max_epochs=1000
"

cell_aug="
$cell_epoch
representor=cissl_coarse
"


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in "$cell_aug" #"$cell_aug"   #"$cell_ours"  "$cell_dim"  "$cell_aug" "$cell_epoch" #"$cell_baseline" #
  do
    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m >> logs/"$experiment".log 2>&1 &

    sleep 10

  done
else
  python utils/aggregate.py \
       +collect_data.params_to_add.aug_strength="data_repr.kwargs.dataset_kwargs.simclr_aug_strength" \
       experiment=$experiment  \
       agg_mode=[summarize_metrics] \
       $add_kwargs


  python -c 'import pandas as pd; df=pd.read_csv("results/exp_table1_contrastive/summarized_metrics_predictor.csv"); print(df.groupby(["jid","repr","zdim","ep","aug_strength"]).max()["test/pred/acc_mean"])'
fi

