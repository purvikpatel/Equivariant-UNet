import numpy as np
import torch
from torch.nn import functional as F

def compute_metrics(y_pred, y_target):
    y_pred = F.softmax(y_pred, dim=1)
    y_pred = torch.argmax(y_pred, dim=1)

    y_target = y_target.squeeze(1)
    batch_size = y_pred.shape[0]
    num_classes = 3
    class_wise_iou = torch.zeros(num_classes)
    class_wise_dice_score = torch.zeros(num_classes)

    smoothening_factor = 0.00001
    for j in range(num_classes):
        intersection = torch.sum((y_pred == j) & (y_target == j), dim=(1, 2))
        y_true_area = torch.sum((y_target == j), dim=(1, 2))
        y_pred_area = torch.sum((y_pred == j), dim=(1, 2))
        combined_area = y_true_area + y_pred_area

        total_intersection = torch.sum(intersection)
        total_combined_area = torch.sum(combined_area)

        iou = (total_intersection + smoothening_factor) / (total_combined_area - total_intersection + smoothening_factor)
        class_wise_iou[j] = iou

        dice_score = 2 * ((total_intersection + smoothening_factor) / (total_combined_area + smoothening_factor))
        class_wise_dice_score[j] = dice_score

    return class_wise_iou, class_wise_dice_score
