#!/usr/bin/env bash

experiment=$prfx"slfdstl_cluster_ablation"
notes="
**Goal**: Ablation study to understand cluster self distillation ISSL.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
trainer.max_epochs=50
checkpoint@checkpoint_repr=bestTrainLoss
architecture@encoder=resnet18
architecture@online_evaluator=linear
data@data_repr=mnist
data_pred.all_data=[data_repr_agg,data_repr_30,data_repr_100,data_repr_1000]
predictor=sk_logistic
optimizer@optimizer_issl=Adam_lr3e-4_w0
representor=slfdstl_cluster
timeout=$time
$add_kwargs
"


# every arguments that you are sweeping over
kwargs_multi="
representor=slfdstl_cluster,slfdstl_cluster_asym,slfdstl_cluster_grad,slfdstl_cluster_noqueue,slfdstl_cluster_queueL,slfdstl_cluster_Mx,slfdstl_cluster_supA,slfdstl_cluster_reg
decodability.kwargs.n_Mx=128
seed=1
"




if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &

    sleep 3

  done
fi