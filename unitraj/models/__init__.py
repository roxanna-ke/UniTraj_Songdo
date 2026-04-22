from importlib import import_module

MODEL_REGISTRY = {
    'autobot': ('unitraj.models.autobot.autobot', 'AutoBotEgo'),
    'wayformer': ('unitraj.models.wayformer.wayformer', 'Wayformer'),
    'MTR': ('unitraj.models.mtr.MTR', 'MotionTransformer'),
    'MAE': ('unitraj.models.fmae.trainer_mae', 'TrainerMAE'),
    'forecast': ('unitraj.models.fmae.trainer_forecast', 'TrainerForecast'),
    'EMP': ('unitraj.models.emp.trainer_forecast', 'TrainerEMP'),
    'SMART': ('unitraj.models.smart.smart', 'SMART'),
}


def get_model_class(model_name):
    module_name, class_name = MODEL_REGISTRY[model_name]
    module = import_module(module_name)
    return getattr(module, class_name)


def build_model(config):
    model_cls = get_model_class(config.method.model_name)
    model = model_cls(config=config)
    return model
