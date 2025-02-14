from dataclasses import dataclass
from typing import Tuple

@dataclass
class DataLocation:
    fpath: str = ''
    channelwise_fpath: Tuple[str]= ()
    directory: str = ''
    datasplit_type: str = ''

    def __post_init__(self):
        assert self.fpath or len(self.channelwise_fpath) or self.directory, "At least one of the following must be provided: fpath, channelwise_fpath, directory"
        assert (self.fpath and not self.channelwise_fpath and not self.directory) or (not self.fpath and self.channelwise_fpath and not self.directory) or (not self.fpath and not self.channelwise_fpath and self.directory), "Only one of the following must be provided: fpath, channelwise_fpath, directory"
