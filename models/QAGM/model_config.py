from easydict import EasyDict as edict

__C = edict()

model_cfg = __C

# QAGM model options.
__C.QAGM = edict()
__C.QAGM.FEATURE_CHANNEL = 512
__C.QAGM.ALPHA = 0.4
__C.QAGM.DISTILL = True
__C.QAGM.WARMUP_STEP = 0
__C.QAGM.MOMENTUM = 0.995
__C.QAGM.SOFTMAXTEMP = 0.07
