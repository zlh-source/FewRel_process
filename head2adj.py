import json
import numpy as np
from tree import head_to_tree, tree_to_adj

filename="/data0/zhanglonghui/HATT-Proto/data/fewrel/val_fewrel_stf.json"
save_adj_path="/data0/zhanglonghui/HATT-Proto/data/fewrel/val_fewrel_adj.npy"
maxlen=47
prune=-1 #保留距离最短依存路径为prune以内的路径，-1表示保留所有路径

with open(filename) as infile:
    ori_data = json.load(infile)


def inputs_to_tree_reps(head, l, prune, subj_pos, obj_pos):
    trees = [head_to_tree(head[i], l[i], prune, subj_pos[i], obj_pos[i]) for i in range(len(l))]
    adj = [tree_to_adj(maxlen, tree, directed=False, self_loop=False).reshape(1, maxlen, maxlen) for tree in trees]
    adj = np.concatenate(adj, axis=0)
    return adj

def get_positions(start_idx, end_idx, length):
    """ Get subj/obj position sequence. """
    return list(range(-start_idx, 0)) + [0]*(end_idx - start_idx + 1) + \
            list(range(1, length-end_idx))

head, l ,subj_pos, obj_pos=[],[],[],[]
for relation in ori_data:
        for ex in ori_data[relation]:  # 遍历某个关系中的样本
            head.append([int(i) for i in ex["stanford_head"]]+[0]*(maxlen-len(ex["stanford_head"])))
            l.append(len(ex["tokens"]))
            subj_pos.append(get_positions(ex['t'][2][0][0],ex['t'][2][0][-1],len(ex["tokens"]))+[0]*(maxlen-len(ex["stanford_head"])))
            obj_pos.append(get_positions(ex['h'][2][0][0],ex['h'][2][0][-1], len(ex["tokens"])) + [0] * (maxlen - len(ex["stanford_head"])))

head, l ,subj_pos, obj_pos=[np.array(i) for i in [head, l ,subj_pos, obj_pos]]
adj = inputs_to_tree_reps(head, l, prune, subj_pos, obj_pos)

np.save(save_adj_path, adj)
