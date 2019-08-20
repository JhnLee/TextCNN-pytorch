import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from multiprocessing import cpu_count
from model import BaseTextCNN
from datasets import datasets
import utils
from config import cnn_config
from tokenizer import SentencePieceTokenizer


# Load Config
config = cnn_config

# Load Tokenizer
tokenizer = SentencePieceTokenizer(config.tokenizer_path, config.vocab_path)

# Load weight from pretrained gensim w2v
pre_weight = utils.get_preweight(config.w2v_path, tokenizer)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# model
model = BaseTextCNN(num_classes=config.num_classes,
                    vocab_size=config.vocab_size,
                    embedding_size=config.embedding_size,
                    pre_weight=pre_weight
                    ).to(device)

# training dataset
train = DataLoader(datasets(config.trainpath, tokenizer),
                   batch_size=config.batch_size,
                   shuffle=True,
                   num_workers=0,       # cpu_count() - 1 로 하면 issue 존재
                   drop_last=True)

# dev set
dev = DataLoader(datasets(config.devpath, tokenizer),
                 batch_size=config.batch_size)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
# Reduce learning rate when a metric has stopped improving
scheduler = ReduceLROnPlateau(optimizer, patience=5)

writer = SummaryWriter(config.model_dir)
cm = utils.CheckpointManager(config.model_dir)
sm = utils.SummaryManager(config.model_dir)
best_val_loss = 1e+9

for epoch in tqdm(range(config.epoch), desc='epochs'):

    train_loss = 0
    train_acc = 0

    model.train()
    for step, batch in tqdm(enumerate(train), desc='steps', total=len(train)):

        x_train, y_train = map(lambda x: x.to(device), batch)

        optimizer.zero_grad()
        y_hat = model(x_train)
        loss = loss_fn(y_hat, y_train)
        loss.backward()
        clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()

        with torch.no_grad():
            batch_acc = utils.acc(y_hat, y_train)

        train_loss += loss.item()
        train_acc += batch_acc.item()

        # evaluation
        if (epoch * len(train) + step) % config.summary_step == 0:

            val_loss = 0
            val_acc = 0

            if model.training:
                model.eval()

            for val_step, batch in enumerate(dev):

                x_dev, y_dev = map(lambda x: x.to(device), batch)

                with torch.no_grad():
                    y_dev_hat = model(x_dev)
                    val_loss += loss_fn(y_dev_hat, y_dev).item() * y_dev.size()[0]
                    val_acc += utils.acc(y_dev_hat, y_dev).item() * y_dev.size()[0]

            val_loss /= len(dev.dataset)
            val_acc /= len(dev.dataset)
            writer.add_scalars('loss', {'train': train_loss / (step + 1),
                                        'val': val_loss}, epoch * len(train) + step)
            model.train()

    train_loss /= (step + 1)
    train_acc /= (step + 1)

    tr_summary = {'loss': train_loss, 'acc': train_acc}
    val_summary = {'loss': val_loss, 'acc': val_acc}
    scheduler.step(val_summary['loss'])
    tqdm.write('epoch : {}, tr_loss: {:.3f}, val_loss: '
               '{:.3f}, tr_acc: {:.2%}, val_acc: {:.2%}'.format(epoch + 1, tr_summary['loss'],
                                                                val_summary['loss'], tr_summary['acc'],
                                                                val_summary['acc']))

    val_loss = val_summary['loss']
    is_best = val_loss < best_val_loss

    if is_best:
        state = {'epoch': epoch + 1,
                 'model_state_dict': model.state_dict(),
                 'opt_state_dict': optimizer.state_dict()}
        summary = {'train': tr_summary, 'validation': val_summary}

        sm.update(summary)
        sm.save('summary.json')
        cm.save_checkpoint(state, 'best.tar')

        best_val_loss = val_loss


