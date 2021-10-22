#!/usr/bin/env bash

experiment=$prfx"test"
notes="
**Goal**: Comparing the effect of functional families on generative ISSL.
"

# parses special mode for running the script
source `dirname $0`/utils.sh

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
trainer.max_epochs=50
checkpoint@checkpoint_feat=bestTrainLoss
architecture@encoder=mlp
data@data_repr=mnist
data@data_pred=data_repr
timeout=$time
$add_kwargs
"