def get_model(model_name, args):
    name = model_name.lower()
    if name == "simplecil":
        from models.simplecil import Learner
    elif name == "adam_finetune":
        from models.adam_finetune import Learner
    elif name == "adam_ssf":
        from models.adam_ssf import Learner
    elif name == "adam_vpt":
        from models.adam_vpt import Learner 
    elif name == "adam_adapter":
        from models.adam_adapter import Learner
    elif name == "l2p":
        from models.l2p import Learner
    elif name == "dualprompt":
        from models.dualprompt import Learner
    elif name == "coda_prompt":
        from models.coda_prompt import Learner
    elif name == "finetune":
        from models.finetune import Learner
    elif name == "icarl":
        from models.icarl import Learner
    elif name == "der":
        from models.der import Learner
    elif name == "coil":
        from models.coil import Learner
    elif name == "foster":
        from models.foster import Learner
    elif name == "memo":
        from models.memo import Learner
    elif name == 'ranpac_original':
        from models.ranpac_original import Learner
    elif name == 'ranpac':
        from models.ranpac import Learner
    elif name == 'ranpac_cholesky':
        from models.ranpac_cholesky import Learner
    elif name == 'ranpac_cholesky_opt_lambda':
        from models.ranpac_cholesky_opt_lambda import Learner
    elif name == 'ranpac_cholesky_online':
        from models.ranpac_cholesky_online import Learner
    elif name == 'ranpac_cholesky_online_fast':
        from models.ranpac_cholesky_online_fast import Learner
    elif name == 'ranpac_cholesky_online_opt':
        from models.ranpac_cholesky_online_opt import Learner
    elif name == 'ranpac_cholesky_online_diag_vect':
        from models.ranpac_cholesky_online_diag_vect import Learner
    elif name == 'ranpac_nystrom':
        from models.ranpac_nystrom import Learner
    elif name == "ease":
        from models.ease import Learner
    elif name == 'tsvd':
        from models.tsvd import Learner
    elif name == 'tsvd_adapter':
        from models.tsvd_adapter import Learner
    elif name == 'tsvd_adapter_ease':
        from models.tsvd_adapter import Learner
    else:
        assert 0
    
    return Learner(args)
