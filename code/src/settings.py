from torch import Tensor
from torch.nn.functional import mse_loss, cross_entropy
from base_model import BaseModel

S = 2
MODELS = {"base": BaseModel, "skipper": None}
LOSS_FUNCTIONS = {"mse": mse_loss}
