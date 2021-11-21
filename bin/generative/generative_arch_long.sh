#!/usr/bin/env bash

experiment=$prfx"generative_arch_long"
notes="
**Goal**: effect of predictive family on generative ISSL. Trained for longer.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
trainer.max_epochs=100
checkpoint@checkpoint_repr=bestTrainLoss
architecture@encoder=resnet18
architecture@online_evaluator=linear
data@data_repr=mnist
data_pred.all_data=[data_repr_agg,data_repr_30,data_repr_100,data_repr_100_test,data_repr_1000]
predictor=sk_logistic
timeout=$time

"
# all ran with optimizer@optimizer_issl=Adam_lr1e-2_w0 but probably want 3e-4 or 1e-3


# every arguments that you are sweeping over
kwargs_multi="
representor=std_gen_resnet,std_gen_cnn,std_gen_resnetlin,std_gen_cnnlin,std_gen_V,std_gen_mlp
encoder.z_shape=32,128,1024
seed=1
"


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m &

    sleep 3

  done
fi
