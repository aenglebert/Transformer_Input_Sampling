import torch


class PointingGame:
    def __init__(self, model, max_overlap=0.5):
        self.model = model
        self.max_overlap = max_overlap

    def __call__(self, image, saliency_map, target):

        h, w = image.shape[-2:]

        bboxes_map = torch.zeros((w, h))

        for bbox in target:
            xmin, ymin, xmax, ymax, class_idx = bbox
            bboxes_map[int(xmin):int(xmax), int(ymin):int(ymax)] = 1

        if (bboxes_map.sum() / torch.ones((w, h)).sum()) < self.max_overlap:
            bboxes_saliency = saliency_map * bboxes_map

            energy_bbox = bboxes_saliency.sum()
            energy_total = saliency_map.sum()

            return torch.tensor(energy_bbox / energy_total)

        else:
            return torch.tensor(torch.nan)
