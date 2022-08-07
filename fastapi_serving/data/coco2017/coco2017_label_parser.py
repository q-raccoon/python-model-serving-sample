import os
from typing import Any, Union
import numpy as np
from loguru import logger

class COCO2017LabelParser:
    def __init__(self, file_path: str) -> None:
        self.file_path_ = file_path
        self.label_ = {}
        self.zero_base_ = False

        if not os.path.exists(self.file_path_):
            raise RuntimeError("`{}` does not exists.".format(self.file_path_))

        with open(self.file_path_, "r") as fp:
            count = 0 if self.zero_base_ else 1
            while True:
                line = fp.readline().strip()
                if not line: break
                self.label_[count] = line
                count += 1

    def get_label(self, label_idx: Union[int, np.array, None]) -> Any:
        if isinstance(label_idx, (int, np.generic)):
            if label_idx not in self.label_.keys():
                logger.error("`` is not included in the range of coco2017 validation set.".format(label_idx))
                return "No label"
            return self.label_[label_idx]
        elif isinstance(label_idx, np.ndarray):
            label_list = []
            for label in label_idx:
                label_list.append(self.get_label(label))
            return np.array(label_list)
        else:
            raise RuntimeError("")