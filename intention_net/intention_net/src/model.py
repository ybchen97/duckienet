import torch
import torch.nn as nn 
from torch.nn import functional as F 
import numpy as np 
from torch.nn.modules.loss import MSELoss
from torch.utils import model_zoo
from torch.optim import Adam
from tensorboardX import SummaryWriter
from keras.utils import to_categorical
from torchvision.models import resnet18 

class Resnet18Extractor(nn.Module):
    def __init__(self,hidden_dim=256,droprate=0.2):
        super(Resnet18Extractor,self).__init__()
        self.hidden_dim = hidden_dim
        resnet = resnet18(True)
        layers = list(resnet.children())[:-1]
        self.linear = nn.Linear(512,self.hidden_dim)
        self.model = torch.nn.Sequential(*layers)
        
    def forward(self,x):
        # check if the feated input is 1 dim or not
        if x.size(1) == 1:
            x = torch.cat([x,x,x],dim=1)
        
        # feed to resnet18
        x = self.model(x)
        x = x.view(-1,512)

        # last fc layer
        x = self.linear(x)

        return x

class OneChannelFeat(nn.Module):
    def __init__(self,hidden_dim=512,droprate=0.2):
        super(OneChannelFeat,self).__init__()
        self.droprate = droprate

        self.conv1 = nn.Conv2d(1,3,7,padding=3,bias=False)
        self.conv1_bn = nn.BatchNorm2d(3)
        self.conv2 = nn.Conv2d(3,8,5,padding=2,bias=False)
        self.conv2_bn = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8,16,3,padding=1,bias=False)
        self.conv3_bn = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16,32,3,padding=1,bias=False)
        self.conv4_bn = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32,64,3,padding=1,bias=False)
        self.conv5_bn = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64,64,5,bias=False)
        self.conv6_bn = nn.BatchNorm2d(64)
        self.linear = nn.Linear(576,hidden_dim)
        self.pool = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(p=self.droprate)

    def forward(self,x):
        x = self.pool(self.conv1_bn(F.leaky_relu(self.conv1(x))))
        x = self.dropout(x)
        x = self.pool(self.conv2_bn(F.leaky_relu(self.conv2(x))))
        x = self.dropout(x)
        x = self.pool(self.conv3_bn(F.leaky_relu(self.conv3(x))))
        x = self.dropout(x)
        x = self.pool(self.conv4_bn(F.leaky_relu(self.conv4(x))))
        x = self.dropout(x)
        x = self.pool(self.conv5_bn(F.leaky_relu(self.conv5(x))))
        x = self.dropout(x)
        x = self.conv6_bn(F.leaky_relu(self.conv6(x)))
        #x = torch.flatten(x)
        x = x.view(x.size()[0],-1)
        x = self.linear(x)
        return x

class PositionEmbedding(nn.Module):
    def __init__(self,num_pos=3,hidden_dim=256):
        super(PositionEmbedding,self).__init__()
        self.num_pose = 3
        self.hidden_dim = hidden_dim
        self.embedding = torch.nn.Embedding(self.num_pose,self.hidden_dim)
    
    def forward(self,x):
        return self.embedding(x)

class AttentionScore(nn.Module):
    def __init__(self,num_intentions=4,hidden_dim=512,feat_model_name='resnet18',pos_embedding=None):
        super(AttentionScore,self).__init__()
        self.num_intentions = num_intentions
        self.hidden_dim = hidden_dim
        self.feat_model_name = feat_model_name
        self.intention_embedding = torch.nn.Embedding(self.num_intentions,self.hidden_dim)
        self.pos_embedding = pos_embedding
        self.linear1 = nn.Linear(self.hidden_dim,self.hidden_dim)
        self.linear1_ln = nn.LayerNorm(self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim,self.hidden_dim)
        self.linear2_ln = nn.LayerNorm(self.hidden_dim)
        self.transform = torch.nn.Sequential(self.linear1,nn.ReLU(True),self.linear1_ln,\
                                            self.linear2,nn.ReLU(True),self.linear2_ln)

        # create feat model for each direction to reduce the ambigious 
        if feat_model_name == 'onechannelfeat':
            self.feat_model = OneChannelFeat(hidden_dim=self.hidden_dim)
        elif feat_model_name == 'resnet18':
            self.feat_model = Resnet18Extractor(self.hidden_dim)

    def forward(self,intention,dl,dm,dr):
        l_pose = torch.tensor([0]*intention.size(0))
        m_pose = torch.tensor([1]*intention.size(0))
        r_pose = torch.tensor([2]*intention.size(0))
        if intention.is_cuda:
            l_pose = l_pose.cuda()
            m_pose = m_pose.cuda()
            r_pose = r_pose.cuda()

        # embedding intention as a key for scoring the weigths
        intention = self.intention_embedding(intention)
        # features of 3 depth images
        dl_feat = self.feat_model(dl)
        l_pose = self.pos_embedding(l_pose)
        dl_feat = dl_feat+l_pose # camera position embedded
        dl_feat = self.transform(dl_feat)

        dm_feat = self.feat_model(dm)
        m_pose = self.pos_embedding(m_pose)
        dm_feat = dm_feat+m_pose # camera position embedded
        dm_feat = self.transform(dm_feat)

        dr_feat = self.feat_model(dr)
        r_pose = self.pos_embedding(r_pose)
        dr_feat = dr_feat+r_pose # camera position embedded
        dr_feat = self.transform(dr_feat)
        
        # calculate the weight for each side
        l_score = torch.sum(intention.mul(dl_feat),dim=1,keepdim=True)/torch.tensor(self.hidden_dim) #normalize attention score
        m_score = torch.sum(intention.mul(dm_feat),dim=1,keepdim=True)/torch.tensor(self.hidden_dim) #normalize attention score
        r_score = torch.sum(intention.mul(dr_feat),dim=1,keepdim=True)/torch.tensor(self.hidden_dim) #normalize attention score
        
        # concat & normalize
        feat = torch.cat((l_score,m_score,r_score),dim=1)
        score = F.softmax(feat,dim=-1)
        # also return depth images features as it will be used in predict velocity model
        return score,dl_feat,dm_feat,dr_feat

class Predictor(nn.Module):
    def __init__(self,hidden_dim=512,num_intentions=4,num_controls=2,feat_model_name='resnet18',pos_embedding=None):
        super(Predictor,self).__init__()
        self.num_intentions = num_intentions
        self.num_controls = num_controls
        self.hidden_dim = hidden_dim
        self.pos_embedding = pos_embedding

        # create feature extractor for each cameras
        # create feat model for each direction to reduce the ambigious 
        if feat_model_name == 'onechannelfeat':
            self.feat_model = OneChannelFeat(hidden_dim=self.hidden_dim)
        elif feat_model_name == 'resnet18':
            self.feat_model = Resnet18Extractor(self.hidden_dim)
        
        # transform after concat with position embedding
        self.transform = nn.Sequential(nn.Linear(self.hidden_dim,self.hidden_dim),nn.ReLU(True),nn.LayerNorm(self.hidden_dim),\
                                        nn.Linear(self.hidden_dim,self.hidden_dim),nn.ReLU(True),nn.LayerNorm(self.hidden_dim))
        
        # predict the velocity
        self.linear1 = nn.Linear(2*hidden_dim,256,bias=False)
        self.linear1_ln = nn.LayerNorm(256)
        self.linear2 = nn.Linear(256,64,bias=False)
        self.linear2_ln = nn.LayerNorm(64)
        self.linear3 = nn.Linear(64,32,bias=False)
        self.linear3_ln = nn.LayerNorm(32)
        self.linear4 = nn.Linear(32,self.num_controls*self.num_intentions)
    
    def forward(self,lbnw,mbnw,rbnw,score,dl_feat,dm_feat,dr_feat):
        # create position input
        l_pose = torch.tensor([0]*score.size(0))
        m_pose = torch.tensor([1]*score.size(0))
        r_pose = torch.tensor([2]*score.size(0))
        if lbnw.is_cuda:
            l_pose = l_pose.cuda()
            m_pose = m_pose.cuda()
            r_pose = r_pose.cuda()

        # computed feature for each camera regard to its position
        lbnw_feat = self.feat_model(lbnw).view(-1,self.hidden_dim)
        l_pose = self.pos_embedding(l_pose)
        lbnw_feat = lbnw_feat+l_pose # camera position embedded
        lbnw_feat = self.transform(lbnw_feat)

        mbnw_feat = self.feat_model(mbnw).view(-1,self.hidden_dim)
        m_pose = self.pos_embedding(m_pose)
        mbnw_feat = mbnw_feat+m_pose # camera position embedded
        mbnw_feat = self.transform(mbnw_feat)

        rbnw_feat = self.feat_model(rbnw).view(-1,self.hidden_dim)
        r_pose = self.pos_embedding(r_pose)
        rbnw_feat = rbnw_feat+r_pose # camera position embedded
        rbnw_feat = self.transform(rbnw_feat)

        # reshape depth features
        dl_feat = dl_feat.view(-1,self.hidden_dim)
        dm_feat = dm_feat.view(-1,self.hidden_dim)
        dr_feat = dr_feat.view(-1,self.hidden_dim)

        # cat the features of both depth and grayscale camera each direction
        l_feat = torch.cat((lbnw_feat,dl_feat),dim=1).unsqueeze(1)
        m_feat = torch.cat((mbnw_feat,dm_feat),dim=1).unsqueeze(1)
        r_feat = torch.cat((rbnw_feat,dr_feat),dim=1).unsqueeze(1)
        feat = torch.cat((l_feat,m_feat,r_feat),dim=1)

        score = score.unsqueeze(1)
        # calculate combined feature
        feat = torch.matmul(score,feat)
        feat = feat.squeeze()
        feat = self.linear1_ln(F.leaky_relu(self.linear1(feat)))
        feat = self.linear2_ln(F.leaky_relu(self.linear2(feat)))
        feat = self.linear3_ln(F.leaky_relu(self.linear3(feat)))
        feat = self.linear4(feat) # intention_0: velocity,angle = feat[0],feat[1]; intention_1: velocity,angle = feat[2],feat[3]
        return feat

class DepthIntentionEncodeModel(nn.Module):
    def __init__(self,num_intentions=4,hidden_dim=512,num_controls=2):
        super(DepthIntentionEncodeModel,self).__init__()
        self.num_intentions = num_intentions
        self.num_controls = num_controls
        self.hidden_dim = hidden_dim

        self.pos_embedding = PositionEmbedding(3,self.hidden_dim)
        self.attention_score = AttentionScore(num_intentions=self.num_intentions,hidden_dim=self.hidden_dim,pos_embedding=self.pos_embedding)
        self.predictor = Predictor(hidden_dim=self.hidden_dim,num_intentions=self.num_intentions,num_controls=self.num_controls,pos_embedding=self.pos_embedding)
        
        if torch.cuda.is_available():
            self.pos_embedding.cuda()
            self.attention_score.cuda()
            self.predictor.cuda()

    def forward(self,intention,dl,dm,dr,lbnw,mbnw,rbnw):
        score,dl_feat,dm_feat,dr_feat = self.attention_score(intention,dl,dm,dr)
        feat = self.predictor(lbnw,mbnw,rbnw,score,dl_feat,dm_feat,dr_feat)
        feat = feat.view(-1,self.num_controls,self.num_intentions)
        i = intention.tolist()
        one_hot = to_categorical(i,self.num_intentions)
        masked = one_hot
        for i in range(self.num_controls-1):
            masked = np.concatenate([masked,one_hot],axis=1)
        masked = torch.tensor(masked)
        if intention.is_cuda:
            masked = masked.cuda()
        masked = masked.view(-1,self.num_controls,self.num_intentions)
        feat = torch.sum(feat.mul(masked),dim=-1)
        return feat

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test():
    from PIL import Image
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter()
    dl = torch.tensor(np.array(Image.open("../test/depth_0.jpg"))).expand(1,1,224,224).float()/255.0
    dm = torch.tensor(np.array(Image.open("../test/depth_2.jpg"))).expand(1,1,224,224).float()/255.0
    dr = torch.tensor(np.array(Image.open("../test/depth_3.jpg"))).expand(1,1,224,224).float()/255.0
    lbnw = torch.tensor(np.array(Image.open("../test/rgb_5.jpg"))).expand(1,1,224,224).float()/255.0
    mbnw = torch.tensor(np.array(Image.open("../test/rgb_4.jpg"))).expand(1,1,224,224).float()/255.0
    rbnw = torch.tensor(np.array(Image.open("../test/rgb_7.jpg"))).expand(1,1,224,224).float()/255.0
    intention = torch.tensor([0]).long()
    net = DepthIntentionEncodeModel()
    feat = net(intention,dl,dm,dr,lbnw,mbnw,rbnw)
    opt = Adam(net.parameters())
    y = torch.tensor([100,-5]).float()
    criterion = nn.MSELoss()
    for _ in range(50):
        net.train()
        opt.zero_grad()
        loss = torch.sqrt(criterion(feat,y))
        loss.backward()
        opt.step()
        net.eval()
        feat = net(intention,dl,dm,dr,lbnw,mbnw,rbnw)
        print(feat)
    print(count_parameters(net))

if __name__ == '__main__':
    test()

