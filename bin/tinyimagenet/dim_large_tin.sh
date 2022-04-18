#!/usr/bin/env bash

experiment="dim_large_tin"
notes="
**Goal**: effect of using very larger dimension Z.
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
+decodability.kwargs.projector_kwargs.kwargs_prelinear.bottleneck_size=64
data_repr.kwargs.batch_size=512
representor=cntr
downstream_task.all_tasks=[torchlogistic_datarepr,torchlogisticw1e-5_datarepr,torchlogisticw1e-4_datarepr]
encoder.kwargs.arch_kwargs.is_channel_out_dim=True
architecture@online_evaluator=linear_lowrank
+encoder.kwargs.arch_kwargs.bottleneck_channel=64
"

kwargs_multi="
encoder.z_shape=512,16384,32768,65536
"

kwargs_multi="
encoder.z_shape=65536
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in   ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m #>> logs/"$experiment".log 2>&1 &

    sleep 10

  done
fi

# TODO
# line plot. x = dim , y = acc
# color code 512 to say what is typical
#
python utils/aggregate.py \
       experiment=$experiment  \
       patterns.representor=null \
       +col_val_subset.pred=["torch_logisticw1.0e-06"] \
       +col_val_subset.datapred=["data_repr"] \
       +plot_scatter_lines.x="zdim" \
       +plot_scatter_lines.y="test/pred/acc" \
       +plot_scatter_lines.cols_to_max=["pred","optpred"] \
       +plot_scatter_lines.filename="lines_large_zdim" \
       +plot_scatter_lines.hue="repr" \
       +plot_scatter_lines.logbase_x=2 \
       +plot_scatter_lines.legend=False \
       agg_mode=[plot_scatter_lines] \
       $add_kwargs