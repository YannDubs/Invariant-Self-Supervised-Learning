#!/usr/bin/env bash

experiment=$prfx"generative_regularizer"
notes="
**Goal**: Understand effect of regularizer on Generative ISSL.
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
data_pred.all_data=[data_repr_agg,data_repr_30,data_repr_100,data_repr_100_test,data_repr_1000]
predictor=sk_logistic
timeout=$time
$add_kwargs
"
# all ran with optimizer@optimizer_issl=Adam_lr1e-2_w0 but probably want 3e-4 or 1e-3



# every arguments that you are sweeping over
kwargs_multi="
representor=std_gen_stdA
seed=1
"
# huber 1e-3 is quite good
# kl is bad


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  "representor=std_gen_stoch representor.loss.beta=1e-5,1e-4,1e-3,1e-2,0.1,1,10" "" "regularizer=cosine,rmse,huber representor.loss.beta=1e-5,1e-4,1e-3,1e-2,0.1,1,10"
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep -m &

    sleep 3

  done
fi
