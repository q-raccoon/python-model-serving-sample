import os
from loguru import logger

class COCO2017LabelParser:
    def __init__(self, file_path: str) -> None:
        self.file_path_ = file_path
        self.label_ = {}

        if not os.path.exists(self.file_path_):
            raise RuntimeError("`{}` does not exists.".format(self.file_path_))

        with open(self.file_path_, "r") as fp:
            count = 0
            while True:
                line = fp.readline().strip()
                if not line: break
                self.label_[count] = line
                count += 1

    def __call__(self, label_idx: int) -> str:
        if label_idx not in self.label_.keys():
            logger.warning("`` is not included in the range of coco2017 validation set.".format(label_idx))
            return "No label"
        return self.label_[label_idx]
