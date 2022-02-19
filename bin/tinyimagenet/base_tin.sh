base_kwargs_tin="
++logger.wandb_kwargs.project=tinyimagenet
architecture@encoder=resnet18
architecture@online_evaluator=linear
downstream_task.all_tasks=[pytorch_datarepr,pytorch_datarepr001test,pytorch_datarepr01test,pytorch_bn_datarepr,pytorch_bn_datarepr01test,pytorch_bn_datarepr001test,sklogistic_datarepr,sklogistic_datarepr01test,sklogistic_datarepr001test]
data_repr.kwargs.batch_size=512
encoder.z_shape=512
encoder.kwargs.arch_kwargs.is_no_linear=True
data@data_repr=tinyimagenet
data_repr.kwargs.is_force_all_train=True
data_repr.kwargs.is_val_on_test=True
checkpoint@checkpoint_repr=bestTrainLoss
optimizer@optimizer_issl=AdamW
scheduler@scheduler_issl=warm_unifmultistep
optimizer_issl.kwargs.weight_decay=1e-6
optimizer_issl.kwargs.lr=2e-3
regularizer=none
"

# only chose one between bn and not
# only chose one for subset