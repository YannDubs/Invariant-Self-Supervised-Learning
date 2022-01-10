#!/usr/bin/env bash

experiment=$prfx"table_generative"
notes="
**Goal**: check different ways of doing generative modeling for linear decodability: CNN, linear, CNN with softmax
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
representor=gen_stdA_clfresnet
decodability.kwargs.predecode_n_Mx=100
encoder.z_shape=128
decodability.kwargs.softmax_kwargs.temperature_mode='anneal'
"

if [ "$is_plot_only" = false ] ; then
  #noA stdA_no_switch stdA_and_optA softmax
  for kwargs_dep in  ""  "encoder.z_shape=10,1024" "decodability.kwargs.predecode_n_Mx=10,1000" "decodability.kwargs.softmax_kwargs.temperature_mode='train'" "representor.is_switch_x_aux_trgt=False" "representor=gen_stdA_resnet decodability.kwargs.predecode_n_Mx=null"#"representor=gen_optA_clfresnet,gen_stdA_clfresnet,std_gen_clfresnet decodability.kwargs.predecode_n_Mx=10,100,1000" "representor=gen_stdA_resnetlin,gen_stdA_clfresnet representor.is_switch_x_aux_trgt=False" #"representor=gen_stdA_resnetlin,gen_stdA_clfresnet representor.is_switch_x_aux_trgt=False" "representor=gen_stdA_resnet,gen_stdA_linear,gen_optA_resnet,gen_optA_linear,std_gen_resnet,std_gen_linear"
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m &

    sleep 3

  done
fi

wait

python utils/aggregate.py \
       experiment=$experiment  \
       agg_mode=[summarize_metrics]