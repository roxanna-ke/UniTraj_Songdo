from importlib import import_module

DATASET_REGISTRY = {
    'autobot': ('unitraj.datasets.autobot_dataset', 'AutoBotDataset'),
    'wayformer': ('unitraj.datasets.wayformer_dataset', 'WayformerDataset'),
    'MTR': ('unitraj.datasets.MTR_dataset', 'MTRDataset'),
    'forecast': ('unitraj.datasets.fmae_dataset', 'FMAEDataset'),
    'MAE': ('unitraj.datasets.fmae_dataset', 'FMAEDataset'),
    'EMP': ('unitraj.datasets.EMP_dataset', 'EMPDataset'),
    'SMART': ('unitraj.datasets.SMART_dataset', 'SMARTDataset'),
}


def get_dataset_class(model_name):
    module_name, class_name = DATASET_REGISTRY[model_name]
    module = import_module(module_name)
    return getattr(module, class_name)


def build_dataset(config, val=False):
    dataset_cls = get_dataset_class(config.method.model_name)
    dataset = dataset_cls(config=config, is_validation=val)
    return dataset
