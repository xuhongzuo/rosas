

def get_model_class(model_name):
    if model_name == 'rosas':
        from models.model_rosas.RoSAS import RoSAS
        return RoSAS
    elif model_name == 'aabigan':
        from models.model_aabigan.aabigan import AABiGAN
        return AABiGAN
    elif model_name == 'devnet':
        from models.model_devnet.DevNet import DevNet
        return DevNet
    elif model_name == 'dsad':
        from models.model_dsad.DeepSAD import DeepSAD
        return DeepSAD
    elif model_name == 'prenet':
        from models.model_prenet.prenet import PreNet
        return PreNet
    elif model_name == 'plsd':
        from models.model_plsd.plsd import PLSD
        return PLSD
    elif model_name == 'feawad':
        from models.model_feawad.feawad import FeaWAD
        return FeaWAD
    elif model_name == 'c-sas' or model_name == 'r-sas':
        from models.model_supervised.SupervisedAD import SupervisedAD
        return SupervisedAD
    elif model_name == 'tss':
        from models.model_pu.plain_pu import TSS
        return TSS
    elif model_name == 'wsif':
        from models.model_wsif.wsif import WSIF
        return WSIF
    elif model_name == 'dif':
        from models.model_unsupervised.dif import DIF
        return DIF
    elif model_name == 'iforest' or model_name == 'copod' or model_name == 'ocsvm' or model_name == 'ecod':
        from models.model_unsupervised.PyodDetector import PyodDetector
        return PyodDetector
    else:
        raise NotImplementedError('not implement')


def get_model_params(model_name):
    model_params = params['model_params'][model_name]
    return model_params
