# type: ignore
import argparse
import csv
import logging
import os

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
from torch.utils.data import DataLoader

from time import gmtime, strftime # log 시간 기록용
import re

logger = logging.getLogger(__name__)


class LoggingCallback(pl.Callback): 
        
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args


    SKIP_KEYS = set(["log", "progress_bar"])

    def on_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Train batch loss
        global_step = pl_module.global_step
        verbose_step_count = pl_module.hparams.verbose_step_count

        if global_step != 0 and global_step % verbose_step_count == 0:
            batch_loss = trainer.logged_metrics["train/loss"]
            rank_zero_info(f"Step: {global_step} - Loss: {batch_loss}")
        
        # LR Scheduler
        lr_scheduler = trainer.lr_schedulers[0]["scheduler"]
        lrs = {f"lr_group_{i}": lr for i, lr in enumerate(lr_scheduler.get_last_lr())}
        pl_module.logger.log_metrics(lrs)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        global_step = pl_module.global_step
        if global_step == 0:
            return

        rank_zero_info("***** Validation results *****")
        metrics = trainer.callback_metrics

        # # total_log.csv 파일에 저장 (for klue)

        begin_time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
        tokenizer_dir = os.path.join(self.args.tokenizer_name)
        pretrained_bert_files = [file for file in os.listdir(tokenizer_dir) if file.endswith("pth")]
        pretrained_bert_file_name = pretrained_bert_files[0]
        total_log_keys=['valid/macro_f1', 'valid/las_macro_f1', 'valid/uas_macro_f1']
        
    
        # get metrics
        for k, v in metrics.items():
            if k in self.SKIP_KEYS:
                continue
            rank_zero_info(f"{k} = {v}")           
            
            # for total_log
            if k in total_log_keys:
                if os.path.isfile('./run_outputs/klue_total_log.csv') == False:
                    with open ('./run_outputs/klue_total_log.csv', 'w', newline="") as f:
                        wr = csv.writer(f)
                        dev_result = k + '_' + re.findall("\d+\.\d+", str(v).split(',')[0])[0] # example of v: tensor(81.8213, device='cuda:0', dtype=torch.float64)
                        wr.writerow(['time', 'task', 'model', 'tokenizer', 'batch_size', 'lr', 'epoch', 'dev'])
                        wr.writerow([begin_time, self.args.task, pretrained_bert_file_name, self.args.tokenizer_name.split('/')[-1], self.args.seed, self.args.train_batch_size, self.args.learning_rate, dev_result])
                        print("making total_log.csv...")
                        print("logging dev, test...")
                else:
                    with open ('./run_outputs/klue_total_log.csv', 'a', newline="") as f:
                        wr = csv.writer(f)
                        dev_result = k + '_' + re.findall("\d+\.\d+", str(v).split(',')[0])[0] 
                        wr.writerow([begin_time, self.args.task, pretrained_bert_file_name, self.args.tokenizer_name.split('/')[-1], self.args.seed, self.args.train_batch_size, self.args.learning_rate, dev_result])
                        print("logging dev, test...")
                                       

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        rank_zero_info("***** Test results *****")

        # Write Predictions
        try:
            self._write_predictions(trainer.test_dataloaders, pl_module)
        except BaseException:
            pass

        metrics = trainer.callback_metrics
        # Log and save results to file
        output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for k, v in metrics.items():
                if k in self.SKIP_KEYS:
                    continue
                rank_zero_info(f"{k} = {v}")
                writer.write(f"{k} = {v}\n")

    def _write_predictions(self, dataloaders: DataLoader, pl_module: pl.LightningModule) -> None:
        index = 0
        output_test_pred_file = os.path.join(pl_module.hparams.output_dir, "test_predictions.tsv")
        with open(output_test_pred_file, "w", newline="\n") as csvfile:
            one_example = dataloaders[0].dataset.examples[0]
            fieldnames = list(one_example.to_dict().keys()) + ["prediction"]

            writer = csv.DictWriter(csvfile, delimiter="\t", fieldnames=fieldnames)
            writer.writeheader()

            for dataloader in dataloaders:
                for example in dataloader.dataset.examples:
                    row = example.to_dict()
                    row["prediction"] = pl_module.predictions[index].item()

                    writer.writerow(row)
                    index += 1
