import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import numpy as np
np.random.seed(0)
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, pairwise,adjusted_rand_score
from sklearn.utils.class_weight import compute_sample_weight



def run_similarity_search(test_embs, test_lbls):
    if type(test_embs)!=np.ndarray:
        test_embs=test_embs.cpu()
    if type(test_lbls)!=np.ndarray:
        test_lbls = test_lbls.cpu().numpy()
    numRows = test_embs.shape[0]

    cos_sim_array = pairwise.cosine_similarity(test_embs) - np.eye(numRows)
    st = []
    for N in [5, 10, 20, 50, 100]:
        indices = np.argsort(cos_sim_array, axis=1)[:, -N:]
        tmp = np.tile(test_lbls, (numRows, 1))
        selected_label = tmp[np.repeat(np.arange(numRows), N), indices.ravel()].reshape(numRows, N)
        original_label = np.repeat(test_lbls, N).reshape(numRows,N)
        st.append(str(np.round(np.mean(np.sum((selected_label == original_label), 1) / N),4)))

    return st
    


def run_kmeans(x, y, k):
    if type(x)!=np.ndarray:
        x=x.cpu()
    if type(y)!=np.ndarray:
        y=y.cpu().numpy()
    estimator = KMeans(n_clusters=k)

    NMI_list = []
    ARI_list=[]
    sample_w=compute_sample_weight(class_weight='balanced', y=y)
    for i in range(10):
        estimator.fit(x,sample_weight=sample_w)
        y_pred = estimator.predict(x)

        NMI_score = normalized_mutual_info_score(y, y_pred, average_method='arithmetic')
        ARI_score=adjusted_rand_score(y,y_pred)
        NMI_list.append(NMI_score)
        ARI_list.append(ARI_score)

    NMI_score = sum(NMI_list) / len(NMI_list)
    ARI_score=sum(ARI_list) / len(ARI_list)

    return NMI_score,ARI_score


