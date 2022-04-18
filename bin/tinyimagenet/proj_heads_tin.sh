#!/usr/bin/env bash

experiment="proj_heads_tin"
notes="
**Goal**: effect of using larger projection heads.
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

kwargs_multi="
decodability.kwargs.projector_kwargs.hid_dim=512,1024,2048,4096
"

kwargs_multi="
decodability.kwargs.projector_kwargs.hid_dim=512,2048,4096
"
kwargs_multi="
decodability.kwargs.projector_kwargs.hid_dim=2048
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
# make line plot. x = ISSL loss, y = acc
# => show that larger projection heads is a way of achieving better ISSL loss



python utils/aggregate.py \
       experiment=$experiment  \
       +col_val_subset.pred=["torch_logisticw1.0e-04"] \
       +col_val_subset.datapred=["data_repr"] \
       +collect_data.params_to_add.projection_size="decodability.kwargs.projector_kwargs.hid_dim" \
       +plot_scatter_lines.data="merged" \
       +plot_scatter_lines.hue="repr" \
       +plot_scatter_lines.x="test/repr/decodability" \
       +plot_scatter_lines.y="test/pred/acc" \
       +plot_scatter_lines.cols_to_max=["pred","optpred"] \
       +plot_scatter_lines.filename="acc_vs_loss_projH_w1e-4" \
       +plot_scatter_lines.multipy_y=100 \
       +plot_scatter_lines.y_tick_spacing=0.5 \
       +plot_scatter_lines.x_tick_spacing=0.005 \
       +plot_scatter_lines.is_invert_xaxis=true \
       +plot_scatter_lines.sharex=False \
        +plot_scatter_lines.legend=False \
       agg_mode=[plot_scatter_lines] \
       $add_kwargs
