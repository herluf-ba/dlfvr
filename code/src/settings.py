import torch
from torch.nn.functional import mse_loss, cross_entropy
from base_model import BaseModel
import torch.nn.functional as F
from torchvision.ops import complete_box_iou_loss


# All of this jazz is located here to avoid circular imports. Too bad.
class attribute:
    CONFIDENCE = 0
    LEFT = 1
    TOP = 2
    WIDTH = 3
    HEIGHT = 4
    CLASSES = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]


def batch_extract(tensor_batch, indicies):
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    indicies = torch.tensor(indicies).to(device)
    extracted_tensor = torch.index_select(tensor_batch, 1, indicies)
    extracted_tensor = extracted_tensor.reshape(-1, S * S, S * S)
    return extracted_tensor.mT


def batch_extract_confidence(tensor_batch):
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    indicies = torch.tensor([attribute.CONFIDENCE]).to(device)
    extracted_confidences = torch.index_select(tensor_batch, 1, indicies)
    extracted_confidences = extracted_confidences.reshape(-1, S * S)
    return extracted_confidences


def batch_extract_bounding_box(tensor_batch):
    return batch_extract(
        tensor_batch,
        [attribute.LEFT, attribute.TOP, attribute.WIDTH, attribute.HEIGHT])


def batch_extract_classes(tensor_batch):
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    indicies = torch.tensor(attribute.CLASSES).to(device)
    extracted_classes = torch.index_select(tensor_batch, 1, indicies)
    extracted_classes = extracted_classes.reshape(-1, 10, S * S)
    return extracted_classes.mT


def batch_items_matches_shape(tensor, correct_shape):
    tensor_shape = tensor.shape[1:] if is_batched(tensor) else tensor.shape
    return tensor.shape == correct_shape


def batch_transform_box_coordinates(boxes):
    LEFT = 0
    TOP = 1
    WIDTH = 2
    HEIGHT = 3
    RIGHT = 2
    BOTTOM = 3

    boxes = torch.clamp(boxes, min=0.0)
    boxes[:, RIGHT] = torch.add(boxes[:, LEFT], boxes[:, WIDTH])
    boxes[:, BOTTOM] = torch.add(boxes[:, TOP], boxes[:, HEIGHT])
    return boxes


def custom_mse(input_batch,
               target_batch,
               size_average=None,
               reduce=None,
               reduction="mean",
               logger=None):
    loss = mse_loss(input_batch, target_batch, reduction=reduction)
    if logger:
        logger.add_loss_item("MSE loss", loss.item())

    return loss


# https://www.dailydot.com/wp-content/uploads/eba/cb/skitched-20161229-112404.jpg
# Is this?
def custom_loss(input_batch,
                target_batch,
                size_average=None,
                reduce=None,
                reduction="mean",
                logger=None):

    input_bb = batch_extract_bounding_box(input_batch)
    input_conf = batch_extract_confidence(input_batch)
    input_classes = batch_extract_classes(input_batch)

    target_conf = batch_extract_confidence(target_batch)
    target_bb = batch_extract_bounding_box(target_batch)
    target_classes = batch_extract_classes(target_batch)

    # Filter off predictions predictions:for labels that have zero confidence (Theres no ground truth label)
    conf_filter = target_conf > 0
    target_classes = target_classes[conf_filter]
    input_classes = input_classes[conf_filter]

    classes_loss = F.cross_entropy(input_classes, target_classes)
    bb_loss = F.mse_loss(input_bb, target_bb)

    confidence_loss = F.binary_cross_entropy(
        input_conf, target_conf)  #, weight=confidence_weights)

    if logger:
        logger.add_loss_item("Confidence", confidence_loss.item())
        logger.add_loss_item("Bounding box", bb_loss.item())
        logger.add_loss_item("Classes", classes_loss.item())

    return classes_loss + bb_loss + confidence_loss


def custom_loss_with_iou(input_batch,
                         target_batch,
                         size_average=None,
                         reduce=None,
                         reduction="mean",
                         logger=None):

    #print(f'{torch.isnan(input_batch).any()=}')
    #print(f'{torch.isnan(target_batch).any()=}')
    #print(f'{target_batch=}')

    input_bb = batch_extract_bounding_box(input_batch)
    input_conf = batch_extract_confidence(input_batch)
    input_classes = batch_extract_classes(input_batch)

    target_conf = batch_extract_confidence(target_batch)
    target_bb = batch_extract_bounding_box(target_batch)
    target_classes = batch_extract_classes(target_batch)

    #print(f'{input_bb=}')
    #print(f'{target_conf=}')
    #print(f'{target_bb=}')
    #print(f'{target_classes=}')

    # Filter off predictions for labels that have zero confidence (Theres no ground truth label)
    conf_filter = target_conf > 0
    target_classes = target_classes[conf_filter]
    input_classes = input_classes[conf_filter]

    # Transform left,top,width,height boxes to left,top,right,bottom boxes for iou loss
    target_bb = batch_transform_box_coordinates(target_bb[conf_filter])
    input_bb = batch_transform_box_coordinates(input_bb[conf_filter])

    #print(f'{input_bb=}')
    #print(f'{target_bb=}')

    classes_loss = F.cross_entropy(input_classes, target_classes)
    bb_loss = complete_box_iou_loss(input_bb, target_bb, reduction='mean')
    confidence_loss = F.binary_cross_entropy(
        input_conf, target_conf)  #, weight=confidence_weights)

    if logger:
        logger.add_loss_item("Confidence", confidence_loss.item())
        logger.add_loss_item("Bounding box", bb_loss.item())
        logger.add_loss_item("Classes", classes_loss.item())

    return classes_loss + bb_loss + confidence_loss


# Actual settings
S = 2
MODELS = {"base": BaseModel, "skipper": None}
LOSS_FUNCTIONS = {
    "mse": custom_mse,
    "custom_loss": custom_loss,
    "custom_loss_with_iou": custom_loss_with_iou
}
