#!/usr/bin/env bash

experiment=$prfx"table_dim"
notes="
**Goal**: run the dim part of the MNIST table.
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
data_pred.all_data=[data_repr_agg]
predictor=sk_logistic
data_repr.kwargs.val_size=2
+data_pred.kwargs.val_size=2
+trainer.num_sanity_val_steps=0
+trainer.limit_val_batches=0
timeout=$time

"


# every arguments that you are sweeping over
kwargs_multi="
representor=exact,exact_stdA,exact_mlp,std_gen_smallZ,gen,gen_stdA,gen_stdA_resnet,gen_stdA_reg,std_cntr,cntr,cntr_stdA,cntr_stdA_mlp,cntr_stdA_reg,cntr_stdA_stoch,slfdstl_cluster,slfdstl_prior,slfdstl_prior_Mx,slfdstl_prior_mlp,slfdstl_prior_reg,slfdstl_prior_stoch
encoder.z_shape=5,10,16,32,128,512,2048
"

# difference for gen: linear resnet / augmentations / larger dim


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m &

    sleep 3

  done
fi

wait

python utils/aggregate.py \
       experiment=$experiment  \
       +summarize_threshold.cols_to_sweep=["zdim"] \
       +summarize_threshold.metric="test/pred/accuracy_score_min_mean" \
       +summarize_threshold.operator="geq" \
       +summarize_threshold.threshold=0.98 \
       agg_mode=[summarize_metrics,summarize_threshold]