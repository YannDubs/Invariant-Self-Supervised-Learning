#!/usr/bin/env bash

experiment="table1_distillation"
notes="
**Goal**: compares distillation methods (table 1 distillation column).
"

# parses special mode for running the script
source  "$(dirname $0)"/../utils.sh
source  "$(dirname $0)"/base_tin.sh

time=10000

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
$base_kwargs_tin
timeout=$time
representor=dissl
downstream_task.all_tasks=[torchlogisticw1e-4_datarepr,torchlogisticw1e-5_datarepr,torchlogisticw1e-6_datarepr]
data_repr.kwargs.batch_size=256
seed=1
"
# use seed=1,2,3 for the three seeds

cell_baseline="
representor=dino
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
representor=dissl_coarse
"


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in "$cell_baseline" "$cell_ours" "$cell_dim" "$cell_epoch" "$cell_aug"
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

  python -c 'import pandas as pd; df=pd.read_csv("results/exp_table1_distillation/summarized_metrics_predictor.csv"); print(df.groupby(["jid","repr","zdim","ep","aug_strength"]).max()["test/pred/acc_mean"])'
fi


