#!/usr/bin/env bash

experiment="augmentations_tin_final"
notes="
**Goal**: effect of coarser augmentations.
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
data_repr.kwargs.batch_size=256
representor=dstl
downstream_task.all_tasks=[torchlogistic_datarepr,torchlogisticw1e-5_datarepr,torchlogisticw1e-4_datarepr,torchlogistic_datarepr01test,torchlogistic_datarepr001test,torchlogistic_datarepr0002test,torchlogistic_datarepr05test]
timeout=$time
"

kwargs_multi="
data_repr.kwargs.dataset_kwargs.simclr_aug_strength=0.25,0.5,1.0,1.5,2.0
"

kwargs_multi="
data_repr.kwargs.dataset_kwargs.simclr_aug_strength=0.25
" # 3674327


kwargs_multi="
data_repr.kwargs.dataset_kwargs.simclr_aug_strength=0.5,1.0,1.5,2.0
" # 3672541



kwargs_multi="
data_repr.kwargs.dataset_kwargs.simclr_aug_strength=2.0
" # 3672541

kwargs_multi="
data_repr.kwargs.dataset_kwargs.simclr_aug_strength=0.25
" # 3674327


kwargs_multi="
data_repr.kwargs.dataset_kwargs.simclr_aug_strength=0.25,0.5,1.0
representor=cntr
data_repr.kwargs.batch_size=512
" # 3674330


kwargs_multi="
data_repr.kwargs.dataset_kwargs.simclr_aug_strength=1.5,2.0
representor=cntr
data_repr.kwargs.batch_size=512
" # 3675802

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in  "" # "representor=cntr data_repr.kwargs.batch_size=512" # ""
  do

    python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs -m >> logs/"$experiment".log 2>&1 &

    sleep 10

  done
fi

python utils/aggregate.py \
       experiment=$experiment  \
       patterns.representor=null \
       +collect_data.params_to_add.augmentation_strength="data_repr.kwargs.dataset_kwargs.simclr_aug_strength" \
       +col_val_subset.pred=["torch_logisticw1.0e-05"] \
       +col_val_subset.augmentation_strength=[0.25,0.5,1.0,2.0] \
       +col_val_subset.datapred=["data_repr"] \
       +plot_scatter_lines.x="augmentation_strength" \
       +plot_scatter_lines.y="test/pred/acc" \
       +plot_scatter_lines.cols_to_max=["pred","optpred"] \
       +plot_scatter_lines.filename="lines_aug_acc" \
       +plot_scatter_lines.hue="repr" \
       +plot_scatter_lines.logbase_x=null \
       +plot_scatter_lines.legend_out=False \
       agg_mode=[plot_scatter_lines] \
       $add_kwargs


# make lien plot x : number of samples, and y : accuracy, and hue: aug strength
python utils/aggregate.py \
       experiment=$experiment  \
       patterns.representor=null \
       +collect_data.params_to_add.augmentation_strength="data_repr.kwargs.dataset_kwargs.simclr_aug_strength" \
       +collect_data.params_to_add.n_samples="data_pred.kwargs.subset_train_size" \
       +fillna.n_samples=1 \
       +apply.n_samples="lambda x : x * 100000" \
       +col_val_subset.pred=["torch_logisticw1.0e-06"] \
       +col_val_subset.repr=["cntr"] \
       +plot_scatter_lines.x="n_samples" \
       +plot_scatter_lines.y="test/pred/acc" \
       +plot_scatter_lines.cols_to_max=["pred","optpred"] \
       +plot_scatter_lines.filename="lines_acc_samples_aug_cntr" \
       +plot_scatter_lines.hue="augmentation_strength" \
       +plot_scatter_lines.logbase_x=10 \
       +plot_scatter_lines.legend_out=False \
       agg_mode=[plot_scatter_lines] \
       $add_kwargs


python utils/aggregate.py \
       experiment=$experiment  \
       patterns.representor=null \
       +collect_data.params_to_add.augmentation_strength="data_repr.kwargs.dataset_kwargs.simclr_aug_strength" \
       +collect_data.params_to_add.n_samples="data_pred.kwargs.subset_train_size" \
       +fillna.n_samples=1 \
       +apply.n_samples="lambda x : x * 100000" \
       +col_val_subset.pred=["torch_logisticw1.0e-06"] \
       +col_val_subset.repr=["dstl"] \
       +plot_scatter_lines.x="n_samples" \
       +plot_scatter_lines.y="test/pred/acc" \
       +plot_scatter_lines.cols_to_max=["pred","optpred"] \
       +plot_scatter_lines.filename="lines_acc_samples_aug_dstl" \
       +plot_scatter_lines.hue="augmentation_strength" \
       +plot_scatter_lines.logbase_x=10 \
       +plot_scatter_lines.legend_out=False \
       agg_mode=[plot_scatter_lines] \
       $add_kwargs

#
# TODO
# make lien plot x : aug strength and y : accuracy
# show that ours are better at taking advantage of more augmentations

