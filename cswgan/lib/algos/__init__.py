from cswgan.lib.algos.gans import RCGAN, RCWGAN, TimeGAN, CWGAN
from cswgan.lib.algos.gmmn import GMMN
from cswgan.lib.algos.sigcwgan import SigCWGAN

ALGOS = dict(SigCWGAN=SigCWGAN, TimeGAN=TimeGAN, RCGAN=RCGAN, GMMN=GMMN, RCWGAN=RCWGAN, CWGAN=CWGAN)
