#!/usr/bin/env bash

experiment="epochs_cos_tin_final"
notes="
**Goal**: effect of training for longer  cntr.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh
source `dirname $0`/base_tin.sh

time=10000

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
$base_kwargs_tin
representor=cntr
seed=1
data_repr.kwargs.batch_size=512
downstream_task.all_tasks=[torchlogistic_datarepr,torchlogisticw1e-5_datarepr,torchlogisticw1e-4_datarepr]
timeout=$time
"
# could also run DISSL

kwargs_multi="
update_trainer_repr.max_epochs=2000,1000,500,200,100,50
"

kwargs_multi="
update_trainer_repr.max_epochs=1500
"

kwargs_multi="
update_trainer_repr.max_epochs=500
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
# make line plot. x = ISSL loss, y = acc, hue =repr
# => show that training for logner is a way of getting ISSL loss closer to 0
# then do that same with larger acrhctecture on sample plot (only DISSL)
#
#python utils/aggregate.py \
#       experiment=$experiment  \
#       +col_val_subset.pred=["torch_logisticw1.0e-06"] \
#       +col_val_subset.datapred=["data_repr"] \
#       +col_val_subset.repr=["cntr"] \
#       +collect_data.params_to_create.epochs=\[\["jobnum"\],\{0:50,1:100,2:200,3:500,4:1000,5:3000\}\] \
#       +plot_scatter_lines.data="merged" \
#       +plot_scatter_lines.hue="repr" \
#       +plot_scatter_lines.x="test/repr/decodability" \
#       +plot_scatter_lines.y="test/pred/acc" \
#       +plot_scatter_lines.cols_to_max=["pred","optpred"] \
#       +plot_scatter_lines.filename="acc_vs_loss_epochs_w1e-6" \
#       +plot_scatter_lines.multipy_y=100 \
#       +plot_scatter_lines.y_tick_spacing=1 \
#       +plot_scatter_lines.x_tick_spacing=0.05 \
#       +plot_scatter_lines.is_invert_xaxis=true \
#       +plot_scatter_lines.sharex=False \
#        +plot_scatter_lines.legend=False \
#       job_id_to_rm=\[3614804\] \
#       agg_mode=[plot_scatter_lines] \
#       $add_kwargs


#python utils/aggregate.py \
#       experiment=$experiment  \
#       patterns.representor=null \
#       +col_val_subset.pred=["torch_logisticw1.0e-06","torch_logisticw1.0e-05","torch_logisticw1.0e-04"] \
#       +col_val_subset.datapred=["data_repr"] \
#       +collect_data.params_to_create.epochs=\[\["jobnum"\],\{0:50,1:100,2:200,3:500,4:1000,5:3000\}\] \
#       +plot_scatter_lines.x="epochs" \
#       +plot_scatter_lines.y="test/pred/acc" \
#       +plot_scatter_lines.cols_to_max=["pred","optpred"] \
#       +plot_scatter_lines.filename="lines_epochs" \
#       +plot_scatter_lines.hue="repr" \
#       +plot_scatter_lines.logbase_x=2 \
#       +plot_scatter_lines.legend_out=False \
#       job_id_to_rm=\[3614804\] \
#       agg_mode=[plot_scatter_lines] \
#       $add_kwargs
#
#python utils/aggregate.py \
#       experiment=$experiment  \
#       patterns.representor=null \
#       +col_val_subset.pred=["torch_logisticw1.0e-06","torch_logisticw1.0e-05","torch_logisticw1.0e-04"] \
#       +col_val_subset.datapred=["data_repr"] \
#       +collect_data.params_to_create.epochs=\[\["jobnum"\],\{0:50,1:100,2:200,3:500,4:1000,5:3000\}\] \
#       +plot_scatter_lines.x="epochs" \
#       +plot_scatter_lines.y="test/pred/acc" \
#       +plot_scatter_lines.cols_to_max=["pred","optpred"] \
#       +plot_scatter_lines.filename="lines_epochs_nolog" \
#       +plot_scatter_lines.hue="repr" \
#       +plot_scatter_lines.logbase_x=1 \
#       +plot_scatter_lines.legend_out=False \
#       job_id_to_rm=\[3614804\] \
#       agg_mode=[plot_scatter_lines] \
#       $add_kwargs
#
#python utils/aggregate.py \
#       experiment=$experiment  \
#       +collect_data.params_to_create.epochs=\[\["jobnum"\],\{0:50,1:100,2:200,3:500,4:1000,5:3000\}\] \
#        job_id_to_rm=\[3614804\] \
#       agg_mode=[summarize_metrics] \
#       $add_kwargs
