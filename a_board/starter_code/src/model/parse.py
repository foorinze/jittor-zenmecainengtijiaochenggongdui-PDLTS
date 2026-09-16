from .spec import ModelSpec
from .vm import VelocityModule
from .pdlts_light import PDLTSLight
from .pdlts_heavy import PDLTSHeavy

def get_model(model_config, **kwargs) -> ModelSpec:
    MAP = {
        'VelocityModule': VelocityModule,
        'PDLTSLight': PDLTSLight,
        'PDLTSHeavy': PDLTSHeavy,
    }
    __target__ = model_config['__target__']
    del model_config['__target__']
    assert __target__ in MAP, f"expect: [{','.join(MAP.keys())}], found: {__target__}"
    return MAP[__target__](model_config=model_config, **kwargs)
