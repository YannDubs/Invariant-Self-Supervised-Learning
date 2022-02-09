#!/usr/bin/env bash

experiment="imagenet_aug"
notes="
**Goal**: compare results with standard imagenet augmentations.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh
source `dirname $0`/base_tin.sh

time=10080

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
$base_kwargs_tin
seed=3
trainer.max_epochs=1000
scheduler_issl.kwargs.UniformMultiStepLR.k_steps=5
scheduler_issl.kwargs.UniformMultiStepLR.decay_per_step=3
timeout=$time
"

# every arguments that you are sweeping over
kwargs_multi="
data_repr.kwargs.dataset_kwargs.a_augmentations=\['simclr-imagenet'\]
"



if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in "representor=cntr,cntr_simclr decodability.kwargs.temperature=0.07" "representor=slfdstl decodability.kwargs.out_dim=7000 decodability.kwargs.ema_weight_prior=null decodability.kwargs.beta_pM_unif=1.7 data_repr.kwargs.batch_size=256" "representor=slfdstl_simsiam optimizer_issl.kwargs.weight_decay=1e-4"
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m &

    sleep 10

  done
fi

wait

# for representor
python utils/aggregate.py \
       experiment=$experiment  \
       agg_mode=[summarize_metrics]
