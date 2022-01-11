#!/usr/bin/env bash

experiment=$prfx"table_slfdstl"
notes="
**Goal**: check different ways of doing selfdistillation.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
+logger.wandb_kwargs.project=mnist
trainer.max_epochs=50
checkpoint@checkpoint_repr=bestTrainLoss
architecture@encoder=resnet18
architecture@online_evaluator=linear
data@data_repr=mnist
downstream_task.all_tasks=[sklogistic_datarepragg16,sklogistic_datarepr10,sklogistic_datarepr10test]
data_repr.kwargs.val_size=2
+data_pred.kwargs.val_size=2
+trainer.num_sanity_val_steps=0
+trainer.limit_val_batches=0
representor=std_gen
timeout=$time
"


# every arguments that you are sweeping over
kwargs_multi="
seed=1,2,3
"
#run seed 2,3

kwargs_multi="
seed=1
decodability.kwargs.n_Mx=100
decodability.kwargs.ema_weight_prior=null
representor=slfdstl_prior
decodability.kwargs.projector_kwargs.architecture=linear
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  "decodability.kwargs.n_Mx=500" "decodability.kwargs.beta_pM_unif=3" #"" "decodability.kwargs.n_Mx=10,50,1000" "decodability.kwargs.beta_pM_unif=3,10" "decodability.kwargs.is_pred_proj_same=True"
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m &

    sleep 3

  done
fi

#wait
#
#python utils/aggregate.py \
#       experiment=$experiment  \
#       +col_val_subset.datapred=["mnist_agg"] \
#       kwargs.prfx="agg_" \
#       agg_mode=[summarize_metrics]