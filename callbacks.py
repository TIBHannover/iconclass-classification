import torch
import logging

import torch
import copy
import pytorch_lightning as pl
import torchvision

from pytorch_lightning.utilities.distributed import rank_zero_only


class ProgressPrinter(pl.callbacks.ProgressBarBase):
    def __init__(self, refresh_rate: int = 100):
        super().__init__()
        self.refresh_rate = refresh_rate
        self.enabled = True

    @property
    def is_enabled(self):
        return self.enabled and self.refresh_rate > 0

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        # print(f"TRAIN_BATCH_END {self.refresh_rate}")
        if self.is_enabled and self.trainer.global_step % self.refresh_rate == 0:
            progress_bar_dict = copy.deepcopy(trainer.progress_bar_dict)

            progress_bar_dict.pop("v_num", None)
            logging.info(
                f"Train {self.trainer.global_step} " + " ".join([f"{k}:{v}" for k, v in progress_bar_dict.items()])
            )

    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)
        # print("###################")
        # print("VAL_BATCH_END")
        progress_bar_dict = copy.deepcopy(trainer.progress_bar_dict)

        progress_bar_dict.pop("v_num", None)
        logging.info(
            f"Val {self.trainer.global_step+1} " + " ".join([f"{k}:{v}" for k, v in progress_bar_dict.items()])
        )


class ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def __init__(self, *args, checkpoint_save_interval=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_save_interval = checkpoint_save_interval
        # self.filename = "checkpoint_{global_step}"

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        # print("SSSSSSSSS")
        self.trainer = trainer
        super().on_validation_end(trainer, pl_module)

    def format_checkpoint_name(self, epoch, ckpt_name_metrics, **kwargs):
        ckpt_name_metrics.update({"global_step": self.trainer.global_step + 1})

        return super().format_checkpoint_name(epoch, ckpt_name_metrics, **kwargs).replace("global_step=", "")

    def save_checkpoint(self, trainer, pl_module):
        # print("SAFE")

        epoch = trainer.current_epoch
        global_step = trainer.global_step
        # print(self.save_top_k)
        # print(self.period)
        # print(trainer.running_sanity_check)
        # print(self.last_global_step_saved == global_step)
        if (
            self.save_top_k == 0  # no models are saved
            or self.period < 1  # no models are saved
            or (epoch + 1) % self.period  # skip epoch
            or trainer.running_sanity_check  # don't save anything during sanity check
            or self.last_global_step_saved == global_step  # already saved at the last step
        ):
            pass
            # print("SKIP")

        super().save_checkpoint(trainer, pl_module)

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        # print("SAVE")
        if self.checkpoint_save_interval is not None:
            if (trainer.global_step + 1) % self.checkpoint_save_interval == 0:
                # print(f"SAVE {self.checkpoint_save_interval} ")
                self.on_validation_end(trainer, pl_module)


class LogModelWeightCallback(pl.callbacks.Callback):
    def __init__(self, flush_logs_every_n_steps=None, nrow=2, **kwargs):
        super(LogModelWeightCallback, self).__init__(**kwargs)
        self.flush_logs_every_n_steps = flush_logs_every_n_steps
        self.nrow = nrow

    @rank_zero_only
    def on_batch_end(self, trainer, pl_module):
        if trainer.logger is None:
            return

        if self.flush_logs_every_n_steps is None:
            log_interval = trainer.flush_logs_every_n_steps
        else:
            log_interval = self.flush_logs_every_n_steps

        if (trainer.global_step + 1) % log_interval == 0:
            for k, v in pl_module.state_dict().items():
                try:
                    trainer.logger.experiment.add_histogram(f"weights/{k}", v, trainer.global_step + 1)
                except ValueError as e:
                    logging.info(f"LogModelWeightCallback: {e}")


class LogImageCallback(pl.callbacks.Callback):
    def __init__(self, flush_logs_every_n_steps=None, nrow=2, **kwargs):
        super(LogImageCallback, self).__init__(**kwargs)
        self.flush_logs_every_n_steps = flush_logs_every_n_steps
        self.nrow = nrow

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

        if trainer.logger is None:
            return

        if self.flush_logs_every_n_steps is None:
            log_interval = trainer.flush_logs_every_n_steps
        else:
            log_interval = self.flush_logs_every_n_steps

        if (trainer.global_step + 1) % log_interval == 0:

            if hasattr(pl_module, "image"):
                grid = torchvision.utils.make_grid(pl_module.image, normalize=True, nrow=self.nrow)
                trainer.logger.experiment.add_image(f"input/image", grid, trainer.global_step + 1)
                try:
                    trainer.logger.experiment.add_histogram(f"input/dist", pl_module.image, trainer.global_step + 1)
                except ValueError as e:
                    logging.info(f"LogImageCallback (source/dist): {e}")

