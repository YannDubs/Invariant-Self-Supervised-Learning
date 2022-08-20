#!/usr/bin/env bash

experiment="fig7c_dimensions"
notes="
**Goal**: effect of using larger dimension Z (fig7c).
"

# parses special mode for running the script
source `dirname $0`/../utils.sh
source `dirname $0`/base_tin.sh

time=10000

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
$base_kwargs_tin
timeout=$time
encoder.kwargs.arch_kwargs.is_channel_out_dim=True
+encoder.kwargs.arch_kwargs.bottleneck_channel=512
update_trainer_repr.max_epochs=1000
"

# use seed=1,2,3 for the three seeds
# in paper uses: encoder.z_dim=128,256,512,1024,2048,4096,8192,16384
kwargs_multi="
encoder.z_dim=128,512,2048,8192
seed=1
"


linear="
representor=cissl
downstream_task.all_tasks=[torchlogisticw1e-4_datarepr,torchlogisticw1e-5_datarepr,torchlogisticw1e-6_datarepr]
"

mlp="
representor=cissl_mlp
downstream_task.all_tasks=[torchmlpw1e-4_datarepr,torchmlpw1e-5_datarepr,torchmlpw1e-6_datarepr]
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  "$linear" #"$mlp"
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m >> logs/"$experiment".log 2>&1 &

    sleep 10

  done
fi


#TODO keep only the desired one
python utils/aggregate.py \
       experiment=$experiment  \
       patterns.representor=null \
       +collect_data.params_to_add.task="task" \
       +col_val_subset.repr=["cissl","cissl_mlp"] \
       +col_val_subset.task=["torchlogisticw1e-5_datarepr"] \
       +plot_scatter_lines.x="zdim" \
       +plot_scatter_lines.y="test/pred/acc" \
       +plot_scatter_lines.cols_to_max=["task","pred","optpred","schedpred","eppred","bspred","addpred"] \
       +plot_scatter_lines.filename="lines_zdim_1e-5" \
       +kwargs.pretty_renamer.Cntr_Mlp="MLP" \
       +kwargs.pretty_renamer.Cntr="Linear" \
       +plot_scatter_lines.hue="repr" \
       +plot_scatter_lines.logbase_x=2 \
        +plot_scatter_lines.legend=True \
        +plot_scatter_lines.legend_out=False \
       agg_mode=[plot_scatter_lines] \
       $add_kwargs
#
#python utils/aggregate.py \
#       experiment=$experiment  \
#       patterns.representor=null \
#       +collect_data.params_to_add.task="task" \
#       +col_val_subset.repr=["cissl","cissl_mlp"] \
#       +col_val_subset.task=["torchlogisticw1e-6_datarepr"] \
#       +plot_scatter_lines.x="zdim" \
#       +plot_scatter_lines.y="test/pred/acc" \
#       +plot_scatter_lines.cols_to_max=["task","pred","optpred","schedpred","eppred","bspred","addpred"] \
#       +plot_scatter_lines.filename="lines_zdim_1e-6" \
#       +kwargs.pretty_renamer.Cntr_Mlp="MLP" \
#       +kwargs.pretty_renamer.Cntr="Linear" \
#       +plot_scatter_lines.hue="repr" \
#       +plot_scatter_lines.logbase_x=2 \
#        +plot_scatter_lines.legend=True \
#        +plot_scatter_lines.legend_out=False \
#       agg_mode=[plot_scatter_lines] \
#       $add_kwargs
#
#python utils/aggregate.py \
#       experiment=$experiment  \
#       patterns.representor=null \
#       +collect_data.params_to_add.task="task" \
#       +col_val_subset.repr=["cissl","cissl_mlp"] \
#       +col_val_subset.task=["torchlogisticw1e-4_datarepr"] \
#       +plot_scatter_lines.x="zdim" \
#       +plot_scatter_lines.y="test/pred/acc" \
#       +plot_scatter_lines.cols_to_max=["task","pred","optpred","schedpred","eppred","bspred","addpred"] \
#       +plot_scatter_lines.filename="lines_zdim_1e-4" \
#       +kwargs.pretty_renamer.Cntr_Mlp="MLP" \
#       +kwargs.pretty_renamer.Cntr="Linear" \
#       +plot_scatter_lines.hue="repr" \
#       +plot_scatter_lines.logbase_x=2 \
#        +plot_scatter_lines.legend=True \
#        +plot_scatter_lines.legend_out=False \
#       agg_mode=[plot_scatter_lines] \
#       $add_kwargs
#
#python utils/aggregate.py \
#       experiment=$experiment  \
#       patterns.representor=null \
#       +collect_data.params_to_add.task="task" \
#       +col_val_subset.repr=["cissl","cissl_mlp"] \
#       +plot_scatter_lines.x="zdim" \
#       +plot_scatter_lines.y="test/pred/acc" \
#       +plot_scatter_lines.cols_to_max=["task","pred","optpred","schedpred","eppred","bspred","addpred"] \
#       +plot_scatter_lines.filename="lines_zdim_cisslall" \
#       +kwargs.pretty_renamer.Cntr_Mlp="MLP" \
#       +kwargs.pretty_renamer.Cntr="Linear" \
#       +plot_scatter_lines.hue="repr" \
#       +plot_scatter_lines.logbase_x=2 \
#         +plot_scatter_lines.legend=True \
#        +plot_scatter_lines.legend_out=False \
#       agg_mode=[plot_scatter_lines] \
#       $add_kwargs
