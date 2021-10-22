#!/usr/bin/env bash

experiment=$prfx"test"
notes="
**Goal**: Improve decodability of generative ISSL compared to standard for linear classification.
"

# parses special mode for running the script
source `dirname $0`/utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
trainer.max_epochs=50
checkpoint@checkpoint_feat=bestValLoss
architecture@encoder=resnet18
data@data_repr=mnist
data@data_pred=data_repr
z_shape=128
timeout=$time
$add_kwargs
"

# every arguments that you are sweeping over
kwargs_multi="
representor=std_gen,vae,gen,gen_no_norm,gen_no_V,gen_A_pred,gen_no_reg,gen_no_aug,gen_std_aug
seed=1
"
