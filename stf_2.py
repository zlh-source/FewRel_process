from stanfordcorenlp import StanfordCoreNLP
from tqdm import tqdm
import json


def restore_dep(dep):
    '''
    将StanfordCoreNLP按照tail进行排序
    Args:
        dep:

    Returns:

    '''
    def judge(dep_list,j):
        '''判断dep_liat从j往前是否出现过root'''
        for index in reversed(dep_list[:(j+1)]):
            if index[0]=='ROOT':
                return False
            if index[-1]==1:
                return True
        return True

    #将root恢复到正常的位置
    root_list,dep_list=[],[]
    for dp in dep:
        if dp[0]=="ROOT":
            root_list.append(dp)
        else:
            dep_list.append(dp)

    for i,root in enumerate(root_list):
        root_tail=root[-1]
        for j in range(len(dep_list)):
            dp_tail=dep_list[j][-1]
            if j==0: #头
                if dp_tail==2 and root_tail==1:
                    dep_list.insert(0,root)
                    break
                if dp_tail==1 and root_tail==2 and dep_list[j+1][-1]==3:
                    dep_list.insert(1,root)
                    break
            elif j==len(dep_list)-1: #尾
                if ( dp_tail+1==root_tail or root_tail==1 ):
                    dep_list.insert(j+1,root)
                    break
            else: #中间
                if dp_tail+1==root_tail and dep_list[j+1][-1]-1==root_tail and judge(dep_list,j):
                    dep_list.insert(j+1,root)
                    break
                if dp_tail+1==root_tail and dep_list[j+1][-1]==1 and judge(dep_list,j):
                    dep_list.insert(j+1,root)
                    break
                if root_tail==1 and dep_list[j+1][-1]==2 and dep_list[j][-1]!=1:
                    dep_list.insert(j+1,root)
                    break
        assert j!=len(dep_list) #没有找到合适的位置
    return dep_list


path = './fewrel_dataset/train.json'
save_path='./fewrel_dataset/train_fewrel_stf.json'

with open(path,"r") as f:
    ori_data=json.load(f)

nlp = StanfordCoreNLP('D:\PycharmProjects\deft_corpus\stanford-corenlp-full-2018-10-05\stanford-corenlp-full-2018-10-05', lang='en')

new_data={}

error=0

for k,r in enumerate(ori_data.keys()):
    new_data[r]=[]
    with tqdm(desc=r,ncols=100) as tq:
        for i in range(len(ori_data[r])):
            if r=='P156' and i==348: #过滤坏数据
                continue
            cur_ex=ori_data[r][i]
            text=" ".join(cur_ex["tokens"]).lower()
            tokens = nlp.word_tokenize(text)
            dep=nlp.dependency_parse(text)
            def count_root(dep):
                ROOT = 0
                for d, h, t in dep:
                    if d == "ROOT":
                        ROOT += 1
                return ROOT
            ROOT=count_root(dep)
            if ROOT>1:
                text=" ".join(tokens)
                prun_list=[".","!","?"]
                for prun in prun_list:
                    if prun in text:
                        text=text.replace(prun,",")
                tokens = nlp.word_tokenize(text)
                text=" ".join(tokens)
                dep = nlp.dependency_parse(text)
                ROOT = count_root(dep)
                assert ROOT==1

            if len(dep)!=len(tokens):
                print("分词与依存分析不匹配")
                continue
            sub=" ".join(cur_ex["tokens"][cur_ex["h"][2][0][0]:cur_ex["h"][2][0][-1]+1]).lower()
            obj=" ".join(cur_ex["tokens"][cur_ex["t"][2][0][0]:cur_ex["t"][2][0][-1]+1]).lower()

            sub_tokens = nlp.word_tokenize(sub)
            obj_tokens = nlp.word_tokenize(obj)

            def get_loc(token, sub_tokens):
                for i in range(len(token)):
                    if sub_tokens == token[i:len(sub_tokens) + i]:
                        return [" ".join(sub_tokens), "", [list(range(i, len(sub_tokens) + i))]]
                return -1

            h=get_loc(tokens,sub_tokens)
            t=get_loc(tokens,obj_tokens)
            if h==-1:
                tq.update(1)
                error+=1
                print(sub_tokens)
                continue
            if t==-1:
                tq.update(1)
                error+=1
                print(obj_tokens)
                continue
            #dep = restore_dep(dep)
            stf_pos=nlp.pos_tag(text)

            stanford_deprel, stanford_head=[-1]*len(tokens),[-1]*len(tokens)
            stanford_pos=[i[1] for i in nlp.pos_tag(text)]

            for j in range(len(dep)):
                deprel, head, tail = dep[j]
                assert stanford_head[tail - 1 ] == -1
                stanford_head[tail - 1 ] = head
                stanford_deprel[tail - 1 ] = deprel

            assert -1 not in stanford_deprel
            assert -1 not in stanford_head

            new_data[r].append({
                "tokens":tokens,
                "h":h,
                "t":t,
                "stanford_head":stanford_head,
                "stanford_deprel":stanford_deprel,
                "stanford_pos":stanford_pos
            })
            tq.update(1)

with open(save_path,"w") as f:
    json.dump(new_data,f)
print(error)