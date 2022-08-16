#!/usr/bin/env bash

experiment="dimension_long_tin"
notes="
**Goal**: effect of using larger dimension Z.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh
source `dirname $0`/base_tin.sh

time=10000

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
$base_kwargs_tin
seed=1
timeout=$time
representor=cntr
downstream_task.all_tasks=[torchlogisticw1e-4_datarepr,torchlogisticw1e-5_datarepr,torchlogisticw1e-6_datarepr]
encoder.kwargs.arch_kwargs.is_channel_out_dim=True
+encoder.kwargs.arch_kwargs.bottleneck_channel=512
update_trainer_repr.max_epochs=1000
"


#3773012_1,3773012_3
kwargs_multi="
encoder.z_dim=256,1024
"
#
#3751008
kwargs_multi="
encoder.z_dim=128,512,2048,8192,16384,4096
"

kwargs_multi="
encoder.z_dim=256
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m >> logs/"$experiment".log 2>&1 &

    sleep 10

  done
fi
# TODO
# line plot. x = dim , y = acc, hue = repr
# color code 512 to say what is typical


python utils/aggregate.py \
       experiment=$experiment  \
       patterns.representor=null \
       +collect_data.params_to_add.task="task" \
       +col_val_subset.repr=["cntr","cntr_mlp"] \
       +col_val_subset.task=["torchlogisticw1e-5_datarepr"] \
       +plot_scatter_lines.x="zdim" \
       +plot_scatter_lines.y="test/pred/acc" \
       +plot_scatter_lines.cols_to_max=["task","pred","optpred","schedpred","eppred","bspred","addpred"] \
       +plot_scatter_lines.filename="lines_zdim_cntrall_1e-5_new" \
       +kwargs.pretty_renamer.Cntr_Mlp="MLP" \
       +kwargs.pretty_renamer.Cntr="Linear" \
       +plot_scatter_lines.hue="repr" \
       +plot_scatter_lines.logbase_x=2 \
        +plot_scatter_lines.legend=True \
        +plot_scatter_lines.legend_out=False \
       agg_mode=[plot_scatter_lines] \
       $add_kwargs

python utils/aggregate.py \
       experiment=$experiment  \
       patterns.representor=null \
       +collect_data.params_to_add.task="task" \
       +col_val_subset.repr=["cntr","cntr_mlp"] \
       +col_val_subset.optpred=["SGD_lr3.0e-01_w1.0e-04"] \
       +plot_scatter_lines.x="zdim" \
       +plot_scatter_lines.y="test/pred/acc" \
       +plot_scatter_lines.cols_to_max=["task","pred","optpred","schedpred","eppred","bspred","addpred"] \
       +plot_scatter_lines.filename="lines_zdim_cntrall_1e-4" \
       +kwargs.pretty_renamer.Cntr_Mlp="MLP" \
       +kwargs.pretty_renamer.Cntr="Linear" \
       +plot_scatter_lines.hue="repr" \
       +plot_scatter_lines.logbase_x=2 \
        +plot_scatter_lines.legend=True \
        +plot_scatter_lines.legend_out=False \
       agg_mode=[plot_scatter_lines] \
       $add_kwargs

python utils/aggregate.py \
       experiment=$experiment  \
       patterns.representor=null \
       +collect_data.params_to_add.task="task" \
       +col_val_subset.repr=["cntr","cntr_mlp"] \
       +col_val_subset.optpred=["SGD_lr3.0e-01_w1.0e-05"] \
       +plot_scatter_lines.x="zdim" \
       +plot_scatter_lines.y="test/pred/acc" \
       +plot_scatter_lines.cols_to_max=["task","pred","optpred","schedpred","eppred","bspred","addpred"] \
       +plot_scatter_lines.filename="lines_zdim_cntrall_1e-5" \
       +kwargs.pretty_renamer.Cntr_Mlp="MLP" \
       +kwargs.pretty_renamer.Cntr="Linear" \
       +plot_scatter_lines.hue="repr" \
       +plot_scatter_lines.logbase_x=2 \
        +plot_scatter_lines.legend=True \
        +plot_scatter_lines.legend_out=False \
       agg_mode=[plot_scatter_lines] \
       $add_kwargs

python utils/aggregate.py \
       experiment=$experiment  \
       patterns.representor=null \
       +collect_data.params_to_add.task="task" \
       +col_val_subset.repr=["cntr","cntr_mlp"] \
       +col_val_subset.onlyjid=["3751008","3773012","3774642"] \
       +plot_scatter_lines.x="zdim" \
       +plot_scatter_lines.y="test/pred/acc" \
       +plot_scatter_lines.cols_to_max=["task","pred","optpred","schedpred","eppred","bspred","addpred"] \
       +plot_scatter_lines.filename="lines_zdim_cntrall" \
       +kwargs.pretty_renamer.Cntr_Mlp="MLP" \
       +kwargs.pretty_renamer.Cntr="Linear" \
       +plot_scatter_lines.hue="repr" \
       +plot_scatter_lines.logbase_x=2 \
         +plot_scatter_lines.legend=True \
        +plot_scatter_lines.legend_out=False \
       agg_mode=[plot_scatter_lines] \
       $add_kwargs

