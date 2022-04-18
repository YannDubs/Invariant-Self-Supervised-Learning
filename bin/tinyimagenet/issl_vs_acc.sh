#!/usr/bin/env bash

experiment="issl_vs_acc"
notes="
**Goal**: correlation between issl and accuracy.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh
source `dirname $0`/base_tin.sh

time=10000

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
representor=cntr
++logger.wandb_kwargs.project=tinyimagenet
architecture@encoder=resnet18
architecture@online_evaluator=linear
data_repr.kwargs.batch_size=512
encoder.z_shape=512
encoder.kwargs.arch_kwargs.is_no_linear=True
data@data_repr=tinyimagenet
data_repr.kwargs.is_force_all_train=True
data_repr.kwargs.is_val_on_test=True
checkpoint@checkpoint_repr=last
optimizer@optimizer_issl=AdamW
scheduler@scheduler_issl=warm_unifmultistep
regularizer=none
data_repr.kwargs.batch_size=512
downstream_task.all_tasks=[torchlogistic_datarepr,torchlogisticw1e-5_datarepr,torchlogisticw1e-4_datarepr]
timeout=$time
"

kwargs_multi="
hydra/sweeper=optuna
hydra/sweeper/sampler=random
hypopt=optuna
monitor_direction=[maximize]
monitor_return=[pred/torchlogistic_datarepr/acc]
hydra.sweeper.n_trials=10
hydra.sweeper.n_jobs=10
hydra.sweeper.study_name=v5
optimizer_issl.kwargs.lr=tag(log,interval(5e-4,1e-2))
optimizer_issl.kwargs.weight_decay=tag(log,interval(1e-7,1e-5))
scheduler_issl.kwargs.UniformMultiStepLR.decay_per_step=shuffle(range(2,7))
scheduler_issl.kwargs.UniformMultiStepLR.k_steps=shuffle(range(2,7))
scheduler_issl.kwargs.base.warmup_epochs=interval(0,0.3)
seed=2,3,4,5
decodability.kwargs.projector_kwargs.n_hid_layers=1,2,3
decodability.kwargs.projector_kwargs.hid_dim=512,1024,2048,4096
decodability.kwargs.projector_kwargs.architecture=mlp
update_trainer_repr.max_epochs=50,100,200
"


kwargs="
experiment=$experiment
$base_kwargs_tin
representor=cntr
data_repr.kwargs.batch_size=512
downstream_task.all_tasks=[torchlogistic_datarepr,torchlogisticw1e-5_datarepr,torchlogisticw1e-4_datarepr]
encoder.z_shape=512
timeout=$time
"

kwargs_multi="
decodability.kwargs.is_batchnorm_post=True,False
decodability.kwargs.is_batchnorm_pre=True,False
update_trainer_repr.max_epochs=100
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m >> logs/"$experiment".log 2>&1 &

    sleep 10

  done
fi

wait
# TODO
# make scatter plot. x = ISSL loss, y = acc
# => show that training for logner is a way of getting ISSL loss closer to 0
# then do that same with larger acrhctecture on sample plot (only DISSL)
#
python utils/aggregate.py \
       experiment=$experiment  \
       patterns.representor=["results/exp_issl_vs_acc/**/results_representor.csv","results/exp_proj_heads_tin/**/results_representor.csv","results/exp_proj_layers_tin/**/results_representor.csv"] \
       patterns.predictor=["results/exp_issl_vs_acc/**/results_predictor.csv","results/exp_proj_heads_tin/**/results_predictor.csv","results/exp_proj_layers_tin/**/results_predictor.csv"] \
       +col_val_subset.pred=["torch_logisticw1.0e-04"] \
       +col_val_subset.datapred=["data_repr"] \
       +col_val_subset.repr=["cntr"] \
       '+col_cond_subset={test/pred/acc: ">0.3"}' \
       +plot_scatter_lines.data="merged" \
       +plot_scatter_lines.mode="lmplot" \
       +plot_scatter_lines.x="test/repr/decodability" \
       +plot_scatter_lines.y="test/pred/acc" \
       +plot_scatter_lines.cols_to_max=["pred","optpred"] \
       +plot_scatter_lines.filename="acc_vs_loss_corr_w1e-4" \
       +plot_scatter_lines.multipy_y=100 \
       +plot_scatter_lines.y_tick_spacing=2 \
       +plot_scatter_lines.x_tick_spacing=0.05 \
       +plot_scatter_lines.is_invert_xaxis=true \
       '+plot_scatter_lines.scatter_kws={color: "gray"}' \
       +plot_scatter_lines.sharex=False \
        +plot_scatter_lines.legend=False \
       agg_mode=[plot_scatter_lines] \
       $add_kwargs

# only showing accuracy larger than 30% because makes zoomed in plots (but trend holds regardless) as shown below

#python utils/aggregate.py \
#       experiment=$experiment  \
#       patterns.representor=["results/exp_issl_vs_acc/**/results_representor.csv","results/exp_proj_heads_tin/**/results_representor.csv","results/exp_proj_layers_tin/**/results_representor.csv"] \
#       patterns.predictor=["results/exp_issl_vs_acc/**/results_predictor.csv","results/exp_proj_heads_tin/**/results_predictor.csv","results/exp_proj_layers_tin/**/results_predictor.csv"] \
#       +col_val_subset.pred=["torch_logisticw1.0e-05"] \
#       +col_val_subset.datapred=["data_repr"] \
#       +col_val_subset.repr=["cntr"] \
#       +plot_scatter_lines.data="merged" \
#       +plot_scatter_lines.mode="lmplot" \
#       +plot_scatter_lines.x="test/repr/decodability" \
#       +plot_scatter_lines.y="test/pred/acc" \
#       +plot_scatter_lines.cols_to_max=["pred","optpred"] \
#       +plot_scatter_lines.filename="acc_vs_loss_corr_w1e-5_full" \
#       +plot_scatter_lines.multipy_y=100 \
#       +plot_scatter_lines.y_tick_spacing=10 \
#       +plot_scatter_lines.x_tick_spacing=0.2 \
#       +plot_scatter_lines.is_invert_xaxis=true \
#       '+plot_scatter_lines.scatter_kws={color: "gray"}' \
#       +plot_scatter_lines.sharex=False \
#        +plot_scatter_lines.legend=False \
#       agg_mode=[plot_scatter_lines] \
#       $add_kwargs

#python utils/aggregate.py \
#       experiment=$experiment  \
#       +col_val_subset.pred=["torch_logisticw1.0e-05"] \
#       +col_val_subset.datapred=["data_repr"] \
#       +col_val_subset.repr=["cntr"] \
#       '+col_cond_subset={test/pred/acc: ">0.1"}' \
#       +plot_scatter_lines.data="merged" \
#       +plot_scatter_lines.mode="lmplot" \
#       +plot_scatter_lines.x="test/repr/decodability" \
#       +plot_scatter_lines.y="test/pred/acc" \
#       +plot_scatter_lines.cols_to_max=["pred","optpred"] \
#       +plot_scatter_lines.filename="acc_vs_loss_corr_w1e-5" \
#       +plot_scatter_lines.multipy_y=100 \
#       +plot_scatter_lines.y_tick_spacing=2 \
#       +plot_scatter_lines.x_tick_spacing=0.05 \
#       +plot_scatter_lines.is_invert_xaxis=true \
#       +plot_scatter_lines.sharex=False \
#        +plot_scatter_lines.legend=False \
#       agg_mode=[plot_scatter_lines] \
#       $add_kwargs


#python utils/aggregate.py \
#       experiment=$experiment  \
#       +col_val_subset.pred=["torch_logisticw1.0e-04"] \
#       +col_val_subset.datapred=["data_repr"] \
#       +col_val_subset.repr=["cntr"] \
#       '+col_cond_subset={test/pred/acc: ">0.1"}' \
#       +plot_scatter_lines.data="merged" \
#       +plot_scatter_lines.mode="lmplot" \
#       +plot_scatter_lines.x="test/repr/decodability" \
#       +plot_scatter_lines.y="test/pred/acc" \
#       +plot_scatter_lines.cols_to_max=["pred","optpred"] \
#       +plot_scatter_lines.filename="acc_vs_loss_corr_w1e-4" \
#       +plot_scatter_lines.multipy_y=100 \
#       +plot_scatter_lines.y_tick_spacing=2 \
#       +plot_scatter_lines.x_tick_spacing=0.05 \
#       +plot_scatter_lines.is_invert_xaxis=true \
#       +plot_scatter_lines.sharex=False \
#        +plot_scatter_lines.legend=False \
#       agg_mode=[plot_scatter_lines] \
#       $add_kwargs



#python utils/aggregate.py \
#       experiment=$experiment  \
#       +col_val_subset.pred=["torch_logisticw1.0e-05"] \
#       +col_val_subset.datapred=["data_repr"] \
#       +col_val_subset.repr=["cntr"] \
#       +plot_scatter_lines.data="merged" \
#       +plot_scatter_lines.mode="lmplot" \
#       +plot_scatter_lines.x="train/repr/decodability" \
#       +plot_scatter_lines.y="test/pred/acc" \
#       +plot_scatter_lines.cols_to_max=["pred","optpred"] \
#       +plot_scatter_lines.filename="acc_vs_trloss_corr_w1e-5" \
#       +plot_scatter_lines.multipy_y=100 \
#       +plot_scatter_lines.y_tick_spacing=2 \
#       +plot_scatter_lines.x_tick_spacing=0.1 \
#       +plot_scatter_lines.is_invert_xaxis=true \
#       +plot_scatter_lines.sharex=False \
#        +plot_scatter_lines.legend=False \
#       agg_mode=[plot_scatter_lines] \
#       $add_kwargs
#
#python utils/aggregate.py \
#       experiment=$experiment  \
#       +col_val_subset.pred=["torch_logisticw1.0e-05"] \
#       +col_val_subset.datapred=["data_repr"] \
#       +col_val_subset.repr=["cntr"] \
#       +plot_scatter_lines.data="merged" \
#       +plot_scatter_lines.mode="lmplot" \
#       +plot_scatter_lines.x="train/repr/decodability" \
#       +plot_scatter_lines.y="test/pred_train/acc" \
#       +plot_scatter_lines.cols_to_max=["pred","optpred"] \
#       +plot_scatter_lines.filename="tracc_vs_trloss_corr_w1e-5" \
#       +plot_scatter_lines.multipy_y=100 \
#       +plot_scatter_lines.y_tick_spacing=2 \
#       +plot_scatter_lines.x_tick_spacing=0.1 \
#       +plot_scatter_lines.is_invert_xaxis=true \
#       +plot_scatter_lines.sharex=False \
#        +plot_scatter_lines.legend=False \
#       agg_mode=[plot_scatter_lines] \
#       $add_kwargs
