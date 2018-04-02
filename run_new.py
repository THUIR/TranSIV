
# coding: utf-8

## Fit TranSIV with hierarchical per-item $\mu_i$ to the binarized taste profile dataset

# In[1]:

import glob
import os
# if you are using OPENBLAS, you might want to turn this option on. Otherwise, joblib might get stuck
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import new_semf_bx
import numpy as np
import scipy.sparse
import pandas as pd
import rec_new


# In[2]:


DATA_ROOT = 'fm'

unique_uid = list()
with open(os.path.join(DATA_ROOT, 'unique_uid_sub.txt'), 'r') as f:
    for line in f:
        unique_uid.append(line.strip())
unique_sid = list()
with open(os.path.join(DATA_ROOT, 'unique_sid_sub.txt'), 'r') as f:
    for line in f:
        unique_sid.append(line.strip())

n_songs = len(unique_sid)
n_users = len(unique_uid)


def load_data(csv_file, shape=(n_users, n_songs)):
    tp = pd.read_csv(csv_file)
    print shape
    rows, cols = np.array(tp['uid'], dtype=np.int32), np.array(tp['sid'], dtype=np.int32)
    count = tp['count']
    return scipy.sparse.csr_matrix((count,(rows, cols)), dtype=np.int16, shape=shape), rows, cols

train_data, rows, cols = load_data(os.path.join(DATA_ROOT, 'train.num.sub.csv'))
# binarize the data
train_data.data = np.ones_like(train_data.data)


vad_data, rows_vad, cols_vad = load_data(os.path.join(DATA_ROOT, 'test.num.sub.csv'))
# binarize the data
vad_data.data = np.ones_like(vad_data.data)


test_data, rows_test, cols_test = load_data(os.path.join(DATA_ROOT, 'test.num.sub.csv'))
# binarize the data
test_data.data = np.ones_like(test_data.data)

S,row_s,cols_s=load_data(os.path.join(DATA_ROOT, 'trust.num.sub.csv'),shape=(n_users,n_users))
S.data=np.ones_like(S.data)
print S


# In[11]:

n_components_1 = 7
n_components_2 = 2  #gamma_s
n_components_3 = 2  #beta_r

max_iter = 20
n_jobs = 1
# grid search on init_mu = {0.1, 0.05, 0.01, 0.005, 0.001} from validation set
init_mu = 0.1
lam = 1e-3
init_yita=0.01
save_dir="model"
coder = new_semf_bx.TranSIV(n_components_1=n_components_1,n_components_2=n_components_2,n_components_3=n_components_3, max_iter=max_iter, batch_size=1000, init_std=0.01,
                       save_params=True,n_jobs=3,
                      save_dir=save_dir, early_stopping=True, verbose=True,
                      lam_y=1., lam_s=0.1,lam_theta=lam, lam_beta=lam, lam_gamma=lam,a1=50, b1=1,a2=10,b2=1., init_mu=init_mu,init_yita=init_yita)


coder.fit(train_data, S,vad_data=vad_data, batch_users=500, k=10)

n_params = len(glob.glob(os.path.join(save_dir, '*.npz')))

params = np.load(os.path.join(save_dir, 'mod.npz' ))
U1, V1,U2,V2, mu = params['U1'], params['V1'], params['U2'],params['V2'],params['mu']
print  U1.shape
print V1.shape


#### Rank by $\theta_u^\top \beta_i$

# In[18]:
print 'Test prec@5: %.4f' % rec_new.prec_at_k(train_data, test_data, U1, V1,U2,V2, k=5)
print 'Test prec@10: %.4f' % rec_new.prec_at_k(train_data, test_data, U1, V1,U2,V2, k=10)
print 'Test prec@20: %.4f' % rec_new.prec_at_k(train_data, test_data, U1, V1,U2,V2, k=20)
print 'Test Recall@10: %.4f' % rec_new.recall_at_k(train_data, test_data, U1, V1,U2,V2, k=10)
print 'Test Recall@20: %.4f' % rec_new.recall_at_k(train_data, test_data, U1, V1,U2,V2, k=20)
print 'Test Recall@50: %.4f' % rec_new.recall_at_k(train_data, test_data, U1, V1,U2,V2, k=50)
print 'Test NDCG@100: %.4f' % rec_new.normalized_dcg_at_k(train_data, test_data, U1, V1,U2,V2, k=100)
print 'Test MAP@100: %.4f' % rec_new.map_at_k(train_data, test_data, U1, V1,U2,V2, k=100)




# In[19]:
print "mu:"
print 'Test prec@5: %.4f' % rec_new.prec_at_k(train_data, test_data,U1, V1,U2,V2,k=5, mu=mu)
print 'Test prec@10: %.4f' % rec_new.prec_at_k(train_data, test_data, U1, V1,U2,V2,k=10, mu=mu)
print 'Test prec@20: %.4f' % rec_new.prec_at_k(train_data, test_data, U1, V1,U2,V2, k=20, mu=mu)
print 'Test Recall@10: %.4f' % rec_new.recall_at_k(train_data, test_data, U1, V1,U2,V2,k=10, mu=mu)
print 'Test Recall@20: %.4f' % rec_new.recall_at_k(train_data, test_data, U1, V1,U2,V2, k=20, mu=mu)
print 'Test Recall@50: %.4f' % rec_new.recall_at_k(train_data, test_data, U1, V1,U2,V2, k=50, mu=mu)
print 'Test NDCG@100: %.4f' % rec_new.normalized_dcg_at_k(train_data, test_data, U1, V1,U2,V2,k=100, mu=mu)
print 'Test MAP@10: %.4f' % rec_new.map_at_k(train_data, test_data, U1, V1,U2,V2, k=10, mu=mu)
print 'Test MAP@100: %.4f' % rec_new.map_at_k(train_data, test_data, U1, V1,U2,V2, k=100, mu=mu)


# In[ ]:



