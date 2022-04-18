base_kwargs_tin="
++logger.wandb_kwargs.project=tinyimagenet
architecture@encoder=resnet18
architecture@online_evaluator=linear
downstream_task.all_tasks=[torchlogistic_datarepr,torchlogisticw1e-5_datarepr,torchlogisticw1e-4_datarepr,torchlogisticw1e-3_datarepr,torchlogisticw1e-5_datarepr001test,torchlogisticw1e-4_datarepr001test,torchlogisticw1e-3_datarepr001test,torchmlpw1e-5_datarepr,torchmlpw1e-4_datarepr,torchmlp_datarepr,sklogistic_datarepr,sklogisticreg01_datarepr,sklogisticreg001_datarepr]
data_repr.kwargs.batch_size=512
encoder.z_shape=512
encoder.kwargs.arch_kwargs.is_no_linear=True
data@data_repr=tinyimagenet
data_repr.kwargs.is_force_all_train=True
data_repr.kwargs.is_val_on_test=True
checkpoint@checkpoint_repr=last
optimizer@optimizer_issl=Adam
scheduler@scheduler_issl=cosine
optimizer_issl.kwargs.weight_decay=1e-6
optimizer_issl.kwargs.lr=2e-3
update_trainer_repr.max_epochs=500
regularizer=none
"
# NOW USUGIN ADAM and COSINE so should rerun many things

# running all sklearn in a row because using warm start to make it computationally more efficient (doesn't restart from scratch)
#  but have to be careful of fisrt running the models with the least data => does not leak data with warm start