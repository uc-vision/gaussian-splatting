from dataclasses import dataclass
import numpy as np
from detectron2.structures import Instances
import torch


@dataclass
class InstanceMask:
  mask : np.array
  label : int
  offset : (int, int)
  score : float



def draw_instances(image, instances, colors, alpha=0.5):
  image = image.copy()

  for i, inst in enumerate(instances):
    x, y = inst.offset
    h, w = inst.mask.shape

    patch = image[y:y+h, x:x+w]
    color = np.array(colors[i % len(colors)] )

    patch[inst.mask] = color * (1 - alpha) + patch[inst.mask] * alpha
  return image


def one_hot(instances, image_size, device):
  masks = torch.zeros((len(instances), image_size, image_size), dtype=torch.bool, device=device)
  for i, inst in enumerate(instances):
    x, y = inst.offset
    h, w = inst.mask.shape

    mask = torch.from_numpy(inst.mask).to(device)
    masks[i, y:y+h, x:x+w] |= mask
  return masks

def extract_instances(outputs):
  predictions:Instances = outputs['instances']

  def to_instance(pred):
    box = pred.pred_boxes.tensor.cpu().squeeze()
    l, u = box[:2].floor().long(), box[2:].ceil().long()
    lx, ly = l.tolist()
    ux, uy = u.tolist()
    return InstanceMask(pred.pred_masks[0, ly:uy, lx:ux].cpu().numpy(), pred.pred_classes.item(), (lx, ly), pred.scores.item())

  return [to_instance(predictions[i]) for i in range(len(predictions))]
