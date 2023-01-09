import torch


class EnergyBasedPointingGame:
    def __init__(self, model, max_overlap=0.5):
        self.model = model
        self.max_overlap = max_overlap

    def __call__(self, image, saliency_map, target):

        h, w = image.shape[-2:]

        assert (image.shape[-2:] == saliency_map.shape[-2:]) and (h * w) == saliency_map.flatten().shape[0], \
            "Image and saliency map should have the same resolution"

        bboxes_map = torch.zeros((w, h))

        for bbox in target:
            xmin, ymin, xmax, ymax, class_idx = bbox
            bboxes_map[int(ymin):int(ymax), int(xmin):int(xmax)] = 1

        if (bboxes_map.sum() / torch.ones((w, h)).sum()) < self.max_overlap:
            bboxes_saliency = saliency_map * bboxes_map

            energy_bbox = bboxes_saliency.sum()
            energy_total = saliency_map.sum()

            return torch.tensor(energy_bbox / energy_total)

        else:
            return torch.tensor(torch.nan)


class PointingGame:
    def __init__(self, model, max_overlap=0.5):
        self.model = model
        self.max_overlap = max_overlap

    def __call__(self, image, saliency_map, target):

        h, w = image.shape[-2:]

        assert (image.shape[-2:] == saliency_map.shape[-2:]) and (h * w) == saliency_map.flatten().shape[0], \
            "Image and saliency map should have the same resolution"

        bboxes_map = torch.zeros((w, h))

        for bbox in target:
            xmin, ymin, xmax, ymax, class_idx = bbox
            #bboxes_map[int(xmin):int(xmax), int(ymin):int(ymax)] = 1
            bboxes_map[int(ymin):int(ymax), int(xmin):int(xmax)] = 1

        if (bboxes_map.sum() / torch.ones((w, h)).sum()) < self.max_overlap:
            result = bboxes_map.flatten()[saliency_map.flatten().argmax()]
        else:
            result = torch.tensor(torch.nan)

        return result
