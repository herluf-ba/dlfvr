# Seperate file to avoid circular import in settings 
from svhn import batch_extract_confidence, batch_extract_bounding_box, batch_extract_classes
import torch.nn.functional as F


def batch_items_matches_shape(tensor, correct_shape):
    tensor_shape = tensor.shape[1:] if is_batched(tensor) else tensor.shape
    return tensor.shape == correct_shape


# https://www.dailydot.com/wp-content/uploads/eba/cb/skitched-20161229-112404.jpg
# Is this?
def custom_loss(input_batch,
                target_batch,
                size_average=None,
                reduce=None,
                reduction="mean"):

    input_bb = batch_extract_bounding_box(input_batch)
    input_conf = batch_extract_confidence(input_batch)
    input_classes = batch_extract_classes(input_batch)


    target_conf = batch_extract_confidence(target_batch)
    target_bb = batch_extract_bounding_box(target_batch)
    target_classes = batch_extract_classes(target_batch)
    
    cross_entropy_loss = F.cross_entropy(input_classes, target_classes * 100)
    return cross_entropy_loss
