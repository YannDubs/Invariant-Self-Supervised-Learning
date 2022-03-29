#!/usr/bin/env bash

experiment="dstl_mlp_tin_final"
notes="
**Goal**: distillation final on timy imagenet.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh
source `dirname $0`/base_tin.sh



time=10080

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
$base_kwargs_tin
representor=dstl_mlp
data_repr.kwargs.batch_size=256
timeout=$time
"
#TODO change to 1.9 for beta_pM_unif (ran with 1.8)

kwargs_multi="
seed=1,2,3
downstream_task.all_tasks=[torchmlp_datarepr,torchmlpw1e-5_datarepr,torchmlpw1e-4_datarepr,torchlogisticw1e-4_datarepr]
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in "encoder.kwargs.arch_kwargs.is_channel_out_dim=True encoder.z_shape=2048" ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs  -m &

    sleep 10

  done
fi

wait

# for representor
python utils/aggregate.py \
       experiment=$experiment  \
       $col_val_subset \
       agg_mode=[summarize_metrics]
