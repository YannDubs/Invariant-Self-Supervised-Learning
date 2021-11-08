#!/usr/bin/env bash

experiment=$prfx"table_main"
notes="
**Goal**: run the main part of the tinyimagenet table.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
+logger.wandb_kwargs.project=tinyimagenet
trainer.max_epochs=200
checkpoint@checkpoint_repr=bestTrainLoss
architecture@encoder=resnet18
architecture@online_evaluator=linear
data@data_repr=tinyimagenet
data_pred.all_data=[data_repr_agg,data_repr_10,data_repr_30,data_repr_p5_test,data_repr_p1_test,data_repr_p10_test]
predictor=sk_logistic
+data_repr.kwargs.val_size=2
+data_pred.kwargs.val_size=2
+trainer.num_sanity_val_steps=0
+trainer.limit_val_batches=0
timeout=$time
$add_kwargs
"


# every arguments that you are sweeping over
kwargs_multi="
representor=exact,std_gen_smallZ,gen,gen_stdA,gen_stdA_resnet,gen_stdA_reg,gen_stdA_dim,std_cntr,cntr,cntr_stdA,cntr_stdA_mlplin,cntr_stdA_mlp,cntr_stdA_reg,cntr_stdA_bs,cntr_stdA_dim,slfdstl_cluster,slfdstl_prior,slfdstl_prior_Mx,slfdstl_prior_mlp,slfdstl_prior_reg,slfdstl_prior_dim,slfdstl_prior_mlplin
seed=1,2,3
"


# difference for gen: linear resnet / augmentations / larger dim


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &

    sleep 3

  done
fi

wait

# for representor
python utils/aggregate.py \
       experiment=$experiment  \
       $col_val_subset \
       agg_mode=[summarize_metrics]


python utils/aggregate.py \
       experiment=$experiment  \
       +col_val_subset.datapred=["stl10_agg"] \
       kwargs.prfx="agg_" \
       agg_mode=[summarize_metrics]