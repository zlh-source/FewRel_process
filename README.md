# FewRel_process
使用stanfordcorenlp生成fewrel数据集的句法树信息，以邻接矩阵的形式存储在npy文件中。

步骤：
1.python stf.py 生成包含head，pos等依存树信息的新的数据集。
2.head2adj.py 转化为邻接矩阵

注意：
1.fewrel数据集中的一个样本可能包含多个句子，因此要进行特殊处理。
2.关于依存树的介绍，参考https://zhuanlan.zhihu.com/p/35238303

举例：
对于句子"i love you. But you don't love me!"
分词生成：['i', 'love', 'you', '.', 'But', 'you', 'do', "n't", 'love', 'me', '!']
依存树生成：[('ROOT', 0, 2), ('nsubj', 2, 1), ('dobj', 2, 3), ('punct', 2, 4), ('ROOT', 0, 5), ('cc', 5, 1), ('nsubj', 5, 2), ('aux', 5, 3), ('neg', 5, 4), ('dobj', 5, 6), ('punct', 5, 7)]
其中，因为句子中包含两个句子，所以生成句法树的时候，临界矩阵的标号在不同的句子上的标号并没有联系，处理时需要注意这一点。
