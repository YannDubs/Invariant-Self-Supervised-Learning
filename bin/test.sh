#!/usr/bin/env bash

experiment=$prfx"test"
notes="
**Goal**: Checking that everything is working.
"

# parses special mode for running the script
source `dirname $0`/utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
mode=dev
trainer.max_epochs=1
architecture@encoder=mlp
data@data_repr=mnist
data@data_pred=data_repr
timeout=$time
$add_kwargs
"

# every arguments that you are sweeping over
kwargs_multi="
representor=base_contrastive
regularizer=kl
"

if [ "$is_plot_only" = false ] ; then
    # add kwargs if parameters have dependencies, so you cannot use Hydra's multirun (they will run in parallel in the background).
    # e.g. `for kwargs1 in "a=1,2 b=3,4" "a=11,12 b=13,14" `
  for kwargs_dep in  ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep

    wait

  done
fi
