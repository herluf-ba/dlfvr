from torch import Tensor
from torch.nn.functional import mse_loss, cross_entropy
from base_model import BaseModel
from settings import S


def is_batched(tensor):
    return len(tensor.shape) == 4


def batch_items_matches_shape(tensor, correct_shape):
    tensor_shape = tensor.shape[1:] if is_batched(tensor) else tensor.shape
    return tensor.shape == correct_shape


def extract_confidence(tensor):
    output = tensor
    assert batch_items_matches_shape(output, (S * S))
    return output


def extract_class_vector(tensor):
    output = tensor
    assert batch_items_matches_shape(output, (S * S, 10))
    return output


def extract_bounding_boxes(tensor):
    output = tensor
    assert batch_items_matches_shape(output, (S * S, 4))
    return output


# https://www.dailydot.com/wp-content/uploads/eba/cb/skitched-20161229-112404.jpg
# Is this?
def custom_loss(input,
                target,
                size_average=None,
                reduce=None,
                reduction="mean"):

    input_bb = extract_bounding_boxes(input)
    input_conf = extract_confidences(input)
    input_classes = extract_class_vector(input)

    target_bb = extract_bounding_boxes(target)
    target_conf = extract_confidences(target)
    target_classes = extract_class_vector(target)

    CE_loss = 

    print(f"{target.shape=}", f"{input.shape=}")
    exit()
    return 0


S = 2
MODELS = {"base": BaseModel, "skipper": None}
LOSS_FUNCTIONS = {"mse": mse_loss, "custom_loss": custom_loss}
