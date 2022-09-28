
import torch
import transformers
transformers.logging.set_verbosity_info()


class SBertTrainer(object):
    """
    PreTrainer make the pretrained BERT model with two STM training method.

        1. Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Next Sentence prediction : 3.3.2 Task #2: Next Sentence Prediction

    please check the details on README.md with simple example.

    """

    def __init__(self, train_dataloader=None, val_dataloader=None, tb_writer=None, args=None):
        """
        :param bert: BERT model which you want to train
        :param spatial_size: total spatial size to embed positional space
        :param train_dataloader: train gen data loader
        :param val_dataloader: test gen data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        # self.device = torch.device("cuda" if torch.cuda.is_available() and config.with_cuda else "cpu")

        # Setting the train and val data loader
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.tb_writer = tb_writer

        self.clip_grads = args.clip_grads
        self.optim = None
        self.lr_scheduler = None
        self.args = args


    def step_lr_scheduler(self, loss):
        if self.args.lr_scheduler == 'loss':
            self.lr_scheduler.step(loss)
        else:
            self.lr_scheduler.step()

    def get_lr(self):
        for param_group in self.optim.param_groups:
            return param_group['lr']

    def train(self, epoch):
        self.model.train()
        return self.train_iteration(epoch, self.train_dataloader)

    def val(self, epoch):
        self.model.eval()
        with torch.no_grad():
            return self.val_iteration(epoch, self.val_dataloader)

    def save(self, path):
        if self.parallel:
            self.model.module.save_pretrained(path)
        else:
            self.model.save_pretrained(path)

    def load(self, path):
        if self.parallel:
            self.model.module.from_pretrained(path)
        else:
            self.model.from_pretrained(path)

    # def save_bert_only(self, path):
    #     if self.parallel:
    #         self.model.module.bert.save_pretrained(path)
    #     else:
    #         self.model.bert.save_pretrained(path)
    #
    # def load_bert_only(self, path):
    #     if self.parallel:
    #         self.model.module.bert.from_pretrained(path)
    #     else:
    #         self.model.bert.from_pretrained(path)