import random
from argparse import ArgumentParser
import sys
sys.path.append('..')
sys.path.append('/mnt/intention_net')

import torch
from torch.optim import Adam,SGD,RMSprop
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torch import nn
from torchvision import transforms
from torch.nn import functional as F
from torch.backends import cudnn

from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint,Timer
from ignite.metrics import Loss,RunningAverage

from tensorboardX import SummaryWriter

from src.model import DepthIntentionEncodeModel
from src.dataset import MultiCamPioneerDataset

from utils.radam import RAdam

def check_manual_seed(seed):
    """ If manual seed is not specified, choose a random one and communicate it to the user.
    """

    seed = seed or random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)

    print('Using manual seed: {seed}'.format(seed=seed))
    
def get_dataloader(train_dir,val_dir=None,use_transform=False,num_workers=1,batch_size=16,shuffle=False):
    data_transform = transforms.Compose([transforms.ToTensor()])
    if use_transform:
        train_data = MultiCamPioneerDataset(train_dir,transform=data_transform)
    else:
        train_data = MultiCamPioneerDataset(train_dir,transform=None)

    train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)
    
    if val_dir: 
        val_data = MultiCamPioneerDataset(val_dir,data_transform)
        val_loader = DataLoader(val_data,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)
    else:
        val_loader = None 
    
    return train_loader,val_loader

def create_summary_writer(model,data_loader,log_dir):
    writer = SummaryWriter()
    data_loader_iter = iter(data_loader)
    x,y = next(data_loader_iter)
    try:
        writer.add_graph(model,x)
    except Exception as e:
        print('Failed to save graph: {}'.format(e))
    return writer

def run(train_dir,val_dir=None,learning_rate=1e-4,num_workers=1,num_epochs=100,batch_size=16,shuffle=False,num_controls=2,num_intentions=4,hidden_dim=256,log_interval=10,log_dir='./logs',seed=2605,accumulation_steps=4,save_model='models',resume=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True
    train_loader,val_loader = get_dataloader(train_dir,val_dir,num_workers=num_workers,batch_size=batch_size,shuffle=shuffle)
    if resume:
        model = torch.load(resume)
    else:
        model = DepthIntentionEncodeModel(num_controls=num_controls,num_intentions=num_intentions,hidden_dim=hidden_dim)
    model = model.to(device)
    writer = create_summary_writer(model,train_loader,log_dir)
    criterion = nn.MSELoss()
    check_manual_seed(seed)

    # optim = RAdam(model.parameters(),lr=learning_rate,betas=(0.9,0.999))
    optim = SGD(model.parameters(),lr=learning_rate)

    lr_scheduler = ExponentialLR(optim,gamma=0.95)
    checkpoints = ModelCheckpoint(save_model,'Model',save_interval=1,n_saved=3,create_dir=True,require_empty=False,save_as_state_dict=False)

    def update_fn(engine, batch):
        model.train()
        optim.zero_grad()

        x, y = batch
        x = list(map(lambda x: x.to(device),x))
        y = y.to(device)
        y_pred = model(*x)

        loss = criterion(y_pred, y) 
        loss.backward()
        optim.step()

        return loss.item()
    
    def evaluate_fn(engine,batch):
        engine.state.metrics = dict()
        model.eval()

        x,y = batch
        
        x = list(map(lambda x: x.to(device),x))
        y = y.to(device)

        y_pred = model(*x)
        mse_loss = F.mse_loss(y_pred,y)
        mae_loss = F.l1_loss(y_pred,y)

        engine.state.metrics['mse'] = mse_loss
        engine.state.metrics['mae'] = mae_loss

    trainer = Engine(update_fn)
    trainer.add_event_handler(Events.EPOCH_COMPLETED,checkpoints,{'model':model})
    avg_loss = RunningAverage(output_transform=lambda x: x,alpha=0.1)
    avg_loss.attach(trainer, 'running_avg_loss')
    pbar = ProgressBar()
    pbar.attach(trainer,['running_avg_loss'])

    evaluator = Engine(evaluate_fn)
    pbar.attach(evaluator)
    
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1)%len(train_loader)+1
        if iter % log_interval == 0:
            print("[Epoch: {}][Iteration: {}/{}] loss: {:.4f}".format(engine.state.epoch,iter,len(train_loader),engine.state.output))
            writer.add_scalar("training/loss",engine.state.output,engine.state.iteration)
            for name, param in model.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(), iter)
    
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        mse = metrics['mse']
        mae = metrics['mae']
        print("Training Results - Epoch: {}  mae: {:.5f} mse: {:.5f}".format(engine.state.epoch, mse, mae))
        writer.add_scalar("training/mse", mse, engine.state.epoch)
        writer.add_scalar("training/mae", mae, engine.state.epoch)

    # @trainer.on(Events.EPOCH_COMPLETED)
    # def log_validation_results(engine):
    #     evaluator.run(val_loader)
    #     metrics = evaluator.state.metrics
    #     mse = metrics['mse']
    #     mae = metrics['mae']
    #     print("Validation Results - Epoch: {}  mae: {:.2f} mse: {:.2f}".format(engine.state.epoch, mse, mae))
    #     writer.add_scalar("valid/mse", mse, engine.state.epoch)
    #     writer.add_scalar("valid/mae", mae, engine.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def update_lr_scheduler(engine):
        lr_scheduler.step()
        print('learning rate is: {:6f}'.format(lr_scheduler.get_lr()[0]))

    trainer.run(train_loader,max_epochs=num_epochs)
    writer.close()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--val_batch_size', type=int, default=32,
                        help='input batch size for validation (default: 100)')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--log_interval', type=int, default=2,
                        help='how many batches to wait before logging training status')
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="log directory for Tensorboard log output")
    parser.add_argument('--train_dir',type=str,default="/home/duong/Downloads/data_correct_intention/data",help="path to train data directory")
    parser.add_argument('--val_dir',type=str,default="/home/duong/Downloads/data_correct_intention/val",help="path to val data directory")
    parser.add_argument('--shuffle',type=bool,default=False,help="Choose to shuffle the training set")
    parser.add_argument('--num_intentions',type=int,default=4,help="number of intentions")
    parser.add_argument('--num_controls',type=int,default=2,help="number of controls")
    parser.add_argument('--hidden_dim',type=int,default=256,help="hidden size of image embedded")
    parser.add_argument('--save_model',type=str,default='models',help='directory to save checkpoint')
    parser.add_argument('--resume',type=str,default=None,help='path to saved model')
    parser.add_argument('--accumulation_steps',type=int,default=1,help="number of accumulation steps for gradient update")
    args = parser.parse_args()

    run(train_dir=args.train_dir,val_dir=None,learning_rate=args.lr,num_workers=4,
        num_epochs=args.num_epochs,batch_size=args.batch_size,
        shuffle=args.shuffle,num_controls=args.num_controls,num_intentions=args.num_intentions,
        hidden_dim=args.hidden_dim,log_interval=args.log_interval,log_dir=args.log_dir,seed=2605,
        accumulation_steps=args.accumulation_steps,save_model=args.save_model,resume=args.resume)
