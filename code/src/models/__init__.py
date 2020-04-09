from . import trainer, kornia_trainer
def get_model(exp_dict):
    name = exp_dict["model"]
    if name == "default":
        return trainer.Trainer(exp_dict)
    elif name == "kornia_trainer":
        return kornia_trainer.Trainer(exp_dict)