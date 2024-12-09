try:
    from . import kerbitat
    from .kerbitat import Kerbitat
    from .kerbitat import GPU_TYPE
except ImportError:
    import traceback
    traceback.print_exc()
    
__description__ = 'Cross-GPU performance predictions for DNN.'

__author__ = 'murez'
__email__ = 'zhangsy3@shanghaitech.edu.cn'

__license__ = 'Apache-2.0'

__all__ = [
    'GPU_TYPE',
    'Kerbitat'
]