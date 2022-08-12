#!/usr/bin/env bash

experiment="dim_card_tin"
notes="
**Goal**: effect of number of equiv on dimension
"

# parses special mode for running the script
source `dirname $0`/../utils.sh
source `dirname $0`/base_tin.sh

time=10000


kwargs="
experiment=$experiment
$base_kwargs_tin
seed=1
timeout=$time
+decodability.kwargs.projector_kwargs.kwargs_prelinear.bottleneck_size=256
representor=dstl
data_repr.kwargs.batch_size=256
downstream_task.all_tasks=[torchlogistic_datarepr,torchlogisticw1e-5_datarepr,torchlogisticw1e-4_datarepr]
encoder.kwargs.arch_kwargs.is_channel_out_dim=True
+encoder.kwargs.arch_kwargs.bottleneck_channel=256
"

kwargs_multi="
decodability.kwargs.out_dim=512,8192
encoder.z_shape=128,512,2048,8192,32768
"

kwargs_multi="
decodability.kwargs.out_dim=512
encoder.z_shape=32768
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in   ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m #>> logs/"$experiment".log 2>&1 &

    sleep 10

  done
fi

# TODO
# line plot. x = dim , y = acc, hue = out_dim

python utils/aggregate.py \
       experiment=$experiment  \
       patterns.representor=null \
       +collect_data.params_to_add.n_equiv="decodability.kwargs.out_dim" \
       +col_val_subset.pred=["torch_logisticw1.0e-06"] \
       +col_val_subset.datapred=["data_repr"] \
       +plot_scatter_lines.x="zdim" \
       +plot_scatter_lines.y="test/pred/acc" \
       +plot_scatter_lines.cols_to_max=["pred","optpred"] \
       +plot_scatter_lines.filename="lines_nequiv_zdim" \
       +plot_scatter_lines.hue="n_equiv" \
       +plot_scatter_lines.logbase_x=2 \
       +plot_scatter_lines.is_no_legend_title=False \
       +plot_scatter_lines.legend_out=True \
       agg_mode=[plot_scatter_lines] \
       $add_kwargs