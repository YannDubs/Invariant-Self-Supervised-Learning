#!/usr/bin/env bash

experiment="epochs_tin"
notes="
**Goal**: effect of training for longer for dstl and cntr.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh
source `dirname $0`/base_tin.sh

time=10000

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
$base_kwargs_tin
representor=dstl
seed=1
data_repr.kwargs.batch_size=256
downstream_task.all_tasks=[torchlogistic_datarepr,torchlogisticw1e-5_datarepr,torchlogisticw1e-4_datarepr]
timeout=$time
"

# every arguments that you are sweeping over
kwargs_multi="
update_trainer_repr.max_epochs=50,100,200,500,1000,3000
"

kwargs_multi="
representor=cntr
update_trainer_repr.max_epochs=3000
data_repr.kwargs.batch_size=512
"


if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m >> logs/"$experiment".log 2>&1 &

    sleep 10

  done
fi

wait


python utils/aggregate.py \
       experiment=$experiment  \
       patterns.representor=null \
       +collect_data.params_to_create.epochs=\[\["jobnum"\],\{0:50,1:100,2:200,3:500,4:1000,5:3000\}\] \
       +plot_scatter_lines.x="epochs" \
       +plot_scatter_lines.y="test/pred/pytorch_datarepr/acc" \
       +plot_scatter_lines.filename="lines_epochs" \
       +plot_scatter_lines.hue="repr" \
       +plot_scatter_lines.logbase_x=2 \
       +plot_scatter_lines.legend_out=False \
       job_id_to_rm=\[3614804\] \
       agg_mode=[plot_scatter_lines] \
       $add_kwargs

python utils/aggregate.py \
       experiment=$experiment  \
       patterns.representor=null \
       +collect_data.params_to_create.epochs=\[\["jobnum"\],\{0:50,1:100,2:200,3:500,4:1000,5:3000\}\] \
       +plot_scatter_lines.x="epochs" \
       +plot_scatter_lines.y="test/pred/pytorch_datarepr/acc" \
       +plot_scatter_lines.filename="lines_epochs_nolog" \
       +plot_scatter_lines.hue="repr" \
       +plot_scatter_lines.logbase_x=1 \
       +plot_scatter_lines.legend_out=False \
       job_id_to_rm=\[3614804\] \
       agg_mode=[plot_scatter_lines] \
       $add_kwargs

python utils/aggregate.py \
       experiment=$experiment  \
       +collect_data.params_to_create.epochs=\[\["jobnum"\],\{0:50,1:100,2:200,3:500,4:1000,5:3000\}\] \
       agg_mode=[summarize_metrics] \
       $add_kwargs
