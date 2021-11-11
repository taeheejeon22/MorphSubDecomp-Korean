from logging import Logger

import torch
import os
import csv
from sklearn.metrics import accuracy_score, classification_report
from torch import nn
from torch.optim.adamw import AdamW
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from tasks.pc.config import TrainConfig
from tasks.pc.model import PCModel

from time import gmtime, strftime

class Trainer:
    def __init__(
        self,
        config: TrainConfig,
        model: PCModel,
        train_data_loader: DataLoader,
        dev_data_loader: DataLoader,
        test_data_loader: DataLoader,
        logger: Logger,
        summary_writer: SummaryWriter,
    ):
        self.config = config
        if config.use_tpu == "tpu":
    
            # 사전에 torch_xla 설치 필요
            import torch_xla
            import torch_xla.core.xla_model as xm # for using tpu
            import torch_xla.distributed.xla_multiprocessing as xmp
            import torch_xla.distributed.parallel_loader as pl # for using multiple tpu core
            self.device = xm.xla_device()
            self.model = model
            print('TPU running...')

        elif config.use_tpu == "gpu":
            # multi gpu(3)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if (self.device.type == 'cuda') and (torch.cuda.device_count() > 1):
                print('Multi GPU({}) activate'.format(torch.cuda.device_count()))
                self.model = nn.DataParallel(model, device_ids=[0,1,2,3])
            else:
                self.model = model

        self.model.to(self.device)
     
        self.train_data_loader = train_data_loader
        self.dev_data_loader = dev_data_loader
        self.test_data_loader = test_data_loader
        self.logger = logger
        self.summary_writer = summary_writer

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = AdamW(model.parameters(), lr=config.learning_rate)

        # total step 계산
        self.steps_per_epoch = len(train_data_loader)
        self.total_steps = self.steps_per_epoch * config.num_epochs
        self.warmup_steps = config.warmup_step_ratio * self.total_steps

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.total_steps
        )
        self.global_step = 0

    def train(self):
        # train
        self.logger.info("========== train ==========")
        self.logger.info(f"device                : {self.device}")
        self.logger.info(f"dataset length/ train : {len(self.train_data_loader.dataset)}")
        self.logger.info(f"dataset length/ dev   : {len(self.dev_data_loader.dataset)}")
        self.logger.info(f"dataset length/ test  : {len(self.test_data_loader.dataset)}")
        self.logger.info(f"batch size            : {self.config.batch_size}")
        self.logger.info(f"learning rate         : {self.config.learning_rate}")
        self.logger.info(f"dropout prob          : {self.config.dropout_prob}")
        self.logger.info(f"total epoch           : {self.config.num_epochs}")
        self.logger.info(f"steps per epoch       : {self.steps_per_epoch}")
        self.logger.info(f"total steps           : {self.total_steps}")
        self.logger.info(f"warmup steps          : {self.warmup_steps}\n")

        for epoch in range(self.config.num_epochs):
            running_loss = 0.0
            train_targets = []
            train_predictions = []

            for step, data in enumerate(tqdm(self.train_data_loader)):
                self.model.train()

                self.global_step += 1

                input_token_ids = data[0].to(self.device)
                attention_mask = data[1].to(self.device)
                token_type_ids = data[2].to(self.device)
                labels = data[3].to(self.device)

                loss, outputs = self._train_step(input_token_ids, attention_mask, token_type_ids, labels)

                running_loss += loss
                train_targets.extend(labels.tolist())
                train_predictions.extend(outputs.argmax(-1).tolist())

                if (step + 1) % self.config.logging_interval == 0:
                    train_loss = running_loss / self.config.logging_interval
                    train_acc = accuracy_score(train_targets, train_predictions)
                    self.logger.info(f"Epoch {epoch}, Step {step + 1}\t| Loss {train_loss:.4f}  Acc {train_acc:.4f}")

                    self.summary_writer.add_scalar("pc/train/loss", train_loss, self.global_step)
                    self.summary_writer.add_scalar("pc/train/accuracy", train_acc, self.global_step)

                    running_loss = 0.0
                    train_targets = []
                    train_predictions = []

            # dev every epoch
            dev_loss, dev_targets, dev_predictions = self._validation(self.dev_data_loader)
            dev_report = classification_report(dev_targets, dev_predictions, digits=4)
            self.logger.info(f"######### DEV REPORT #EP{epoch} #########")
            self.logger.info(f"Loss {dev_loss:.4f}")
            self.logger.info(f"\n{dev_report}")

            dev_acc = accuracy_score(dev_targets, dev_predictions)
            self.summary_writer.add_scalar("pc/dev/loss", dev_loss, self.global_step)
            self.summary_writer.add_scalar("pc/dev/accuracy", dev_acc, self.global_step)

            # test every epoch
            test_loss, test_targets, test_predictions = self._validation(self.test_data_loader)
            test_report = classification_report(test_targets, test_predictions, digits=4)
            self.logger.info(f"######### TEST REPORT #EP{epoch} #########")
            self.logger.info(f"Loss {test_loss:.4f}")
            self.logger.info(f"\n{test_report}")

            test_acc = accuracy_score(test_targets, test_predictions)
            self.summary_writer.add_scalar("pc/test/loss", test_loss, self.global_step)
            self.summary_writer.add_scalar("pc/test/accuracy", test_acc, self.global_step)

            # output_path = os.path.join(self.config.checkpoint_dir, f"model-epoch-{epoch}.pth")
            # torch.save(self.model.state_dict(), output_path)
            # self.logger.info(f"MODEL IS SAVED AT {output_path}\n")

            # dev,test 결과만 따로 저장
            self.begin_time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
            tokenizer_dir = os.path.join(self.config.resource_dir, self.config.tokenizer)
            self.pretrained_bert_files = [file for file in os.listdir(tokenizer_dir) if file.endswith("pth")]
            self.pretrained_bert_file_name = self.pretrained_bert_files[0]

            if os.path.isfile('./run_outputs/total_log.csv') == False:
                with open ('./run_outputs/total_log.csv', 'w', newline="") as f:
                    wr = csv.writer(f)
                    self.dev_result = round(dev_acc * 100, 2)
                    self.test_result = round(test_acc * 100, 2)
                    wr.writerow(['time', 'task', 'model', 'tokenizer', 'batch_size', 'lr', 'epoch', 'dev', 'test'])
                    wr.writerow([self.begin_time, 'pc', self.pretrained_bert_file_name, self.config.tokenizer, self.config.batch_size, self.config.learning_rate, epoch, self.dev_result, self.test_result])
                    print("making total_log.csv...")
                    print("logging dev, test...")
            
            else:
                with open ('./run_outputs/total_log.csv', 'a', newline="") as f:
                    wr = csv.writer(f)
                    self.dev_result = round(dev_acc * 100, 2)
                    self.test_result = round(test_acc * 100, 2)
                    wr.writerow([self.begin_time, 'pc', self.pretrained_bert_file_name, self.config.tokenizer, self.config.batch_size, self.config.learning_rate, epoch, self.dev_result, self.test_result])
                    print("logging dev, test...")


    def _train_step(self, input_token_ids, attention_mask, token_type_ids, labels):
        self.optimizer.zero_grad()

        outputs = self.model(input_token_ids, attention_mask, token_type_ids)

        loss = self.criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        if self.config.use_tpu == "tpu":
            import torch_xla
            import torch_xla.core.xla_model as xm # for using tpu
            # optimizer for TPU (Note: Cloud TPU-specific code!)
            xm.optimizer_step(self.optimizer, barrier=True) # multi core 사용 시 barrier=True 불필요
        else:
            self.optimizer.step()

        #self.optimizer.step()
        self.scheduler.step()

        return loss.item(), outputs

    def _validation(self, data_loader):
        self.model.eval()

        running_loss = 0.0
        targets = []
        predictions = []

        with torch.no_grad():
            for data in data_loader:
                input_token_ids = data[0].to(self.device)
                attention_mask = data[1].to(self.device)
                token_type_ids = data[2].to(self.device)
                labels = data[3].to(self.device)

                outputs = self.model(input_token_ids, attention_mask, token_type_ids)

                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                targets.extend(labels.tolist())
                predictions.extend(outputs.argmax(-1).tolist())

        assert len(targets) == len(predictions)

        mean_loss = running_loss / len(data_loader)

        return mean_loss, targets, predictions
