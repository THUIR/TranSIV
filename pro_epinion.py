
import os
import pandas as pd
TPS_DIR = 'fullepinion'
TP_file = os.path.join(TPS_DIR, 'ratings.txt')
test=os.path.join(TPS_DIR, 'test_ratings.txt')
train=os.path.join(TPS_DIR, 'train_ratings.txt')
trust=os.path.join(TPS_DIR, 'trust.txt')
tp = pd.read_table(TP_file,sep='\t',header=None, names=['uid', 'sid', 'count'])
test_tp= pd.read_table(test,sep='\t',header=None, names=['uid', 'sid', 'count'])
train_tp= pd.read_table(train,sep='\t',header=None, names=['uid', 'sid', 'count'])
trust_tp=pd.read_table(trust,sep='\t',header=None, names=['uid', 'sid', 'count'])
print train_tp
def get_count(tp, id):
    playcount_groupbyid = tp[[id, 'count']].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count
usercount, songcount = get_count(tp, 'uid'), get_count(tp, 'sid')
sparsity_level = float(tp.shape[0]) / (1+usercount.shape[0] * songcount.shape[0])
print "After filtering, there are %d triplets from %d users and %d songs (sparsity level %.3f%%)" % (tp.shape[0],
                                                                                                      usercount.shape[0],
                                                                                                      songcount.shape[0],
                                                                                                      sparsity_level * 100)
tp=tp.sort_index(by='uid')
unique_uid = usercount.index
unique_sid = songcount.index
print trust_tp
song2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
user2id = dict((uid, i) for (i, uid) in enumerate(unique_uid))
with open(os.path.join(TPS_DIR, 'unique_uid_sub.txt'), 'w') as f:
    for uid in unique_uid:
        f.write('%s\n' % uid)
with open(os.path.join(TPS_DIR, 'unique_sid_sub.txt'), 'w') as f:
    for sid in unique_sid:
        f.write('%s\n' % sid)
def numerize(tp):
    uid = map(lambda x: user2id[x], tp['uid'])
    sid = map(lambda x: song2id[x], tp['sid'])
    tp['uid'] = uid
    tp['sid'] = sid
    return tp
def numerize_trust(tp):
    uid = map(lambda x: user2id[x], tp['uid'])
    sid = map(lambda x: user2id[x], tp['sid'])
    tp['uid'] = uid
    tp['sid'] = sid
    return tp
train_tp = numerize(train_tp)
train_tp.to_csv(os.path.join(TPS_DIR, 'train.num.sub.csv'), index=False)

test_tp = numerize(test_tp)
test_tp.to_csv(os.path.join(TPS_DIR, 'test.num.sub.csv'), index=False)

trust_tp = numerize_trust(trust_tp)
trust_tp.to_csv(os.path.join(TPS_DIR, 'trust.num.sub.csv'), index=False)

print "Done!"