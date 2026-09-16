from .spec import DummySystem, DummyWriter
from .vm import VMSystem, VMWriter
from .pdlts_light import (
    PDLTSLightPredictSystem,
    PDLTSLightShardRotationSystem,
    PDLTSLightSystem,
    PDLTSLightWriter,
)

def get_system(**kwargs) -> DummySystem:
    MAP = {
        'dummy': DummySystem,
        'vm': VMSystem,
        'pdlts_light': PDLTSLightSystem,
        'pdlts_light_shard_rotation': PDLTSLightShardRotationSystem,
        'pdlts_light_predict': PDLTSLightPredictSystem,
    }
    __target__ = kwargs['__target__']
    assert __target__ in MAP, f"expect: [{','.join(MAP.keys())}], found: {__target__}"
    del kwargs['__target__']
    return MAP[__target__](**kwargs)

def get_writer(**kwargs) -> DummyWriter:
    MAP = {
        'dummy': DummyWriter,
        'vm': VMWriter,
        'pdlts_light': PDLTSLightWriter,
    }
    __target__ = kwargs['__target__']
    assert __target__ in MAP, f"expect: [{','.join(MAP.keys())}], found: {__target__}"
    del kwargs['__target__']
    return MAP[__target__](**kwargs)
