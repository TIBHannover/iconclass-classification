# import os
# import uuid
# import tempfile

# import torch
# import h5py
# import numpy as np

# from sklearn.metrics import average_precision_score
# from pytorch_lightning.metrics import Metric


# class AveragePrecision(Metric):
#     def __init__(self, num_classes, dist_sync_on_step=False):
#         super().__init__(dist_sync_on_step=dist_sync_on_step)

#         self.tmp_dir = tempfile.mkdtemp()
#         self.num_classes = num_classes
#         self.h5 = h5py.File(os.path.join(self.tmp_dir, f"{uuid.uuid4().hex}.h5"), "a")

#         self.feature_data[cache_name]["h5"].create_dataset(
#             "data", [self.length] + [len(c["value"])], dtype=float, maxshape=[None] + [len(c["value"])]
#         )

#     def update(self, preds: torch.Tensor, target: torch.Tensor):

#         preds, target = self._input_format(preds, target)
#         assert preds.shape == target.shape

#         self.correct += torch.sum(preds == target)
#         self.total += target.numel()

#     def compute(self):
#         return self.correct.float() / self.total
