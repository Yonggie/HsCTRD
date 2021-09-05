import copy
# binfomax_intra_drop0.1_l10.07gcn_128
from itertools import product
from matplotlib.image import pil_to_array
from sklearn.metrics import f1_score
from torch_geometric.nn import GCNConv,DeepGraphInfomax
from models import *
from utils.drop import *
from utils.evaluate import run_kmeans,run_similarity_search
from datetime import datetime
import torch
import torch.optim as optim
from TCMDataSet import TCMDataSet
from torch.utils.tensorboard import SummaryWriter

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATA_SET='chp'
G_EPOCH=25
LR=0.005
PATIENCE=20
FEATURE_SIZE=2048
ENCODE='gcn'
EMBED_SIZE=128   # also GCN hidden channel

REG_TYPE='l1'
REG_COEF=0.07


# gcn drop out rate
DROPOUT_PROB=0.1
ACT='prelu'
ACTIVATION={'relu': F.relu, 'prelu': nn.PReLU()}

# corruption edge drop rate
EDGE_DROP_PROB=0.1
NORM_COEF=0.1

# downstream task
CLASSIFIER_MAX_ITER=6000



time_stamp = "{0:%Y-%m-%d %H-%M-%S} ".format(datetime.now())

data_set_type='tcm362' if DATA_SET=='tcm_extract' else DATA_SET

oop=f'{REG_TYPE}={REG_COEF}'
save_model_name=f'best_binformax_{oop}_{DATA_SET}'
print(f'{data_set_type} {ENCODE} {REG_TYPE}={REG_COEF}'+'='*30)


base = f"datasets/{DATA_SET}/"
dataset = TCMDataSet(base,DATA_SET,FEATURE_SIZE,metapath_name='all')
data = dataset.data.to(DEVICE)

node_nums=len(data.x[0])
nb_feature=len(data.x)

encoder=GCNEncoder1(FEATURE_SIZE, EMBED_SIZE, ACTIVATION[ACT], DROPOUT_PROB).to(DEVICE)
summary=AvgReadout()


HsCTRD_model=HsCTRD(
    hidden_channels=EMBED_SIZE, encoder=encoder, summary=summary,
    nb_nodes=node_nums, embed_size=EMBED_SIZE, nb_feature=nb_feature,num_project_hidden=256,
    edge_drop_prob=EDGE_DROP_PROB,norm_coef=NORM_COEF
).to(DEVICE)


optimizer = optim.Adam(list(HsCTRD_model.parameters()), lr=LR)


HsCTRD_model.train()
best = 1e9
cnt_wait=0
real_epoch=-1
save_model_path='./saved_models/'
if not os.path.exists(save_model_path):
    os.mkdir(save_model_path)
for epoch in range(G_EPOCH):
    optimizer.zero_grad()

    poss, negs, summaries,=HsCTRD_model(data.x, data.edge_indexes)

    loss = HsCTRD_model.loss(poss, negs, summaries)

    if REG_TYPE=='l1':
        loss=loss+l1_regularization(HsCTRD_model,REG_COEF)
    elif REG_TYPE=='l2':
        loss=loss+l2_regularization(HsCTRD_model,REG_COEF)
    else:
        pass
    print(f'Epoch: {epoch}, Loss: {loss.item():.4f}')
    if loss < best:
        best = loss
        best_t = epoch
        cnt_wait = 0
        torch.save(HsCTRD_model.state_dict(), save_model_path+f'{save_model_name}.pkl')
    else:
        cnt_wait += 1

    if cnt_wait == PATIENCE:
        print('Early stopping!')
        real_epoch=epoch
        break

    loss.backward()
    optimizer.step()

HsCTRD_model.load_state_dict(torch.load(save_model_path+f'{save_model_name}.pkl'))

# evaluation
with torch.no_grad():
    HsCTRD_model.eval()

    all_idx = data.y
    amount = len(all_idx)
    train_idx = range(int(amount * 0.8))
    train_last = train_idx[-1]

    test_idx = range(train_last,train_last + int(amount * 0.2))


    # graph level attention    
    pos_embeds, _, _,  = HsCTRD_model(data.x, data.edge_indexes)
    
    z = HsCTRD_model.attentioned_fusion(torch.vstack(pos_embeds))
    y = data.y

    # classification task        
    microf1,macrof1 = HsCTRD_model.test(z[train_idx], y[train_idx], z[test_idx], y[test_idx], max_iter=CLASSIFIER_MAX_ITER)
    print('\t[Classification] Accuracy: Micro {:.4f} Macro {:.4f} '.format(microf1,macrof1))
    
    # clustering task
    k = len(set([val.item() for val in y]))
    NMI,ARI=run_kmeans(z,y,k)
    print('\t[Clustering] NMI: {:.4f} ARI {:.4f}'.format(NMI,ARI))

    # similarity search
    sim_scores=run_similarity_search(z, y)
    sim_scores = ','.join(sim_scores)
    print("\t[Similarity] [5,10,20,50,100] : [{}]".format(sim_scores))