#!/usr/bin/env bash

experiment="no_aug_tin"
notes="
**Goal**: testing without augmentations to understand inductive bias.
"

# parses special mode for running the script
source `dirname $0`/../utils.sh
source `dirname $0`/base_tin.sh



time=10080

# define all the arguments modified or added to `conf`. If they are added use `+`
kwargs="
experiment=$experiment
$base_kwargs_tin
representor=dstl
data_repr.kwargs.batch_size=256
seed=1
downstream_task.all_tasks=[torchlogistic_datarepr,torchlogisticw1e-5_datarepr,torchlogisticw1e-4_datarepr]
data_repr.kwargs.dataset_kwargs.aux_target=input
data_repr.kwargs.dataset_kwargs.a_augmentations=[]
data_repr.kwargs.dataset_kwargs.train_x_augmentations=[]
timeout=$time
"

path_dstl="/juice/scr/yanndubs/Invariant-Self-Supervised-Learning/pretrained/exp_no_aug_tin/datarepr_tinyimagenet/augrepr_data-standard/repr_dstl/dec_assign_self_distillation/enc_resnet18/reg_none/optrepr_AdamW_lr2.0e-03_w1.0e-06/schedrepr_warm1.0e-01_unifmultistep_d3/zdim_512/zs_1/beta_1.0e-01/seed_1/addrepr_None/jid_0_3636771/best_representor.ckpt"
path_cntr="/juice/scr/yanndubs/Invariant-Self-Supervised-Learning/pretrained/exp_no_aug_tin/datarepr_tinyimagenet/augrepr_data-standard/repr_cntr/dec_contrastive/enc_resnet18/reg_none/optrepr_AdamW_lr2.0e-03_w1.0e-06/schedrepr_warm1.0e-01_unifmultistep_d3/zdim_512/zs_1/beta_1.0e-01/seed_1/addrepr_None/jid_0_3636772/best_representor.ckpt"

if [ "$is_plot_only" = false ] ; then
  for kwargs_dep in   "representor.is_train=False representor.is_use_init=True" # "encoder.kwargs.arch_kwargs.is_channel_out_dim=True encoder.z_shape=2048 +decodability.kwargs.projector_kwargs.kwargs_prelinear.bottleneck_size=512"
  do
    for kwargs_multi in   "" # "paths.pretrained.init_enc=$path_dstl" #"paths.pretrained.init_repr=$path_dstl" "representor=cntr data_repr.kwargs.batch_size=512 paths.pretrained.init_repr=$path_cntr"
    do

      python "$main" +hydra.job.env_set.WANDB_NOTES="\"${notes}\"" $kwargs $kwargs_multi $kwargs_dep $add_kwargs  -m  >> logs/"$experiment".log 2>&1 &

      sleep 10

    done

  done
fi
