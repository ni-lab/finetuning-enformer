import numpy as np
from lightning.pytorch.callbacks import BaseFinetuning


def create_seq_idx_embedder():
    embedder = np.zeros((256), dtype=np.int64)
    embedder[ord("a")] = 0
    embedder[ord("c")] = 1
    embedder[ord("g")] = 2
    embedder[ord("t")] = 3
    embedder[ord("n")] = 4
    embedder[ord("A")] = 0
    embedder[ord("C")] = 1
    embedder[ord("G")] = 2
    embedder[ord("T")] = 3
    embedder[ord("N")] = 4
    embedder[ord(".")] = -1
    return embedder


class BaseEnformerFreezeUnfreeze(BaseFinetuning):
    def __init__(
        self,
        unfreeze_at_epoch: int,
        initial_denom_lr: float = 10.0,
        train_bn: bool = True,
    ):
        super().__init__()
        self.unfreeze_at_epoch = unfreeze_at_epoch
        self.initial_denom_lr = initial_denom_lr
        self.train_bn = train_bn

    def freeze_before_training(self, pl_module):
        self.freeze(pl_module.base)

    def finetune_function(self, pl_module, current_epoch, optimizer):
        if current_epoch == self.unfreeze_at_epoch:
            self.unfreeze_and_add_param_group(
                modules=pl_module.base,
                optimizer=optimizer,
                initial_denom_lr=self.initial_denom_lr,
                train_bn=self.train_bn,
            )
