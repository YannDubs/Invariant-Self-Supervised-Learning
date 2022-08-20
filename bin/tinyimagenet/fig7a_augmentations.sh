#!/usr/bin/env bash

experiment="fig7a_augmentations"
notes="
**Goal**: effect of coarser augmentations (fig 7a).
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
representor=dissl
downstream_task.all_tasks=[torchlogistic_datarepr,torchlogisticw1e-5_datarepr,torchlogistic_datarepr01test,torchlogistic_datarepr001test,torchlogistic_datarepr0002test]
timeout=$time
"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  "data_repr.kwargs.dataset_kwargs.simclr_aug_strength=0.25 decodability.kwargs.n_equivalence_classes=24000"  "data_repr.kwargs.dataset_kwargs.simclr_aug_strength=0.5 decodability.kwargs.n_equivalence_classes=20000" "data_repr.kwargs.dataset_kwargs.simclr_aug_strength=1 decodability.kwargs.n_equivalence_classes=16384"
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m >> logs/"$experiment".log 2>&1 &

    sleep 10

  done
else
  # Makes the desired line plot
python utils/aggregate.py \
       experiment=$experiment  \
       patterns.representor=null \
       +collect_data.params_to_add.augmentation_strength="data_repr.kwargs.dataset_kwargs.simclr_aug_strength" \
       +collect_data.params_to_add.n_downstream_samples="data_pred.kwargs.subset_train_size" \
       +fillna.n_downstream_samples=1 \
       +apply.n_downstream_samples="lambda x : x * 100000" \
       +col_val_subset.pred=["torch_logisticw1.0e-06"] \
       +col_val_subset.repr=["dissl"] \
       +plot_scatter_lines.x="n_downstream_samples" \
       +plot_scatter_lines.y="test/pred/acc" \
       +plot_scatter_lines.cols_to_max=["pred","optpred"] \
       +plot_scatter_lines.filename="lines_acc_samples_aug_card" \
       +plot_scatter_lines.hue="augmentation_strength" \
       +plot_scatter_lines.logbase_x=10 \
       +plot_scatter_lines.legend_out=True \
       +plot_scatter_lines.is_legend=False \
       agg_mode=[plot_scatter_lines] \
       $add_kwargs
fi



