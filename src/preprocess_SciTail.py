import random
from random import randint

def Scitail_2_ExtreamPosNeg():
    root_file="/save/wenpeng/datasets/SciTailV1/tsv_format/scitail_1.0_train.tsv"
    writefilename = "/save/wenpeng/datasets/SciTailV1/tsv_format/scitail_1.0_train_2_ExtreamPosNeg.txt"
    # files=['scitail_1.0_train.tsv', 'scitail_1.0_dev.tsv', 'scitail_1.0_test.tsv']
    'we creat 10 neg, 10 pos for each sentence'
    readfile = open(root_file, 'r')
    writefile = open(writefilename, 'w')
    for line in readfile:
        parts = line.strip().split('\t')
        sent1 = parts[0]
        sent2 = parts[1]
        sent1_wordlist = sent1.split()
        sent2_wordlist = sent2.split()
        sent1_len = len(sent1_wordlist)
        sent2_len = len(sent2_wordlist)
        'sent1 pos'
        sent1_pos_list = []
        sent1_pos_list.append(sent1)

        # for i in range(9):
        #     left = randint(0, sent1_len/2)
        #     right = randint(left + 1, sent1_len)
        #     sent1_pos_ins = sent1_wordlist[left:right]
        #     sent1_pos_list.append(' '.join(sent1_pos_ins))
        # assert len(sent1_pos_list) == 10
        'sent1 neg'
        sent1_neg_list = []

        sent1_neg_list.append(' '.join(sent1_wordlist[::-1])) #reverse
        for i in range(8):
            insert_point = randint(0, sent1_len - 1)
            sent1_neg_list.append(' '.join(sent1_wordlist[:insert_point]+['not']+sent1_wordlist[insert_point:]))
        random.Random(100).shuffle(sent1_wordlist)
        sent1_neg_list.append(' '.join(sent1_wordlist)) #shuffle
        assert len(sent1_neg_list) == 10
        'write sent1 into file'
        for sent in sent1_pos_list:
            writefile.write(sent1+'\t'+sent+'\tentails\n')
        for sent in sent1_neg_list:
            writefile.write(sent1+'\t'+sent+'\tneutral\n')


        'sent2 pos'
        sent2_pos_list = []
        sent2_pos_list.append(sent2)

        # for i in range(9):
        #     left = randint(0, sent2_len/2)
        #     right = randint(left + 1, sent2_len)
        #     sent2_pos_ins = sent2_wordlist[left:right]
        #     sent2_pos_list.append(' '.join(sent2_pos_ins))
        # assert len(sent2_pos_list) == 10
        'sent2 neg'
        sent2_neg_list = []

        sent2_neg_list.append(' '.join(sent2_wordlist[::-1])) #reverse
        for i in range(8):
            insert_point = randint(0, sent2_len - 1)
            sent2_neg_list.append(' '.join(sent2_wordlist[:insert_point]+['not']+sent2_wordlist[insert_point:]))
        random.Random(100).shuffle(sent2_wordlist)
        sent2_neg_list.append(' '.join(sent2_wordlist)) #shuffle
        assert len(sent2_neg_list) == 10
        'write sent2 into file'
        for sent in sent2_pos_list:
            writefile.write(sent2+'\t'+sent+'\tentails\n')
        for sent in sent2_neg_list:
            writefile.write(sent2+'\t'+sent+'\tneutral\n')
    readfile.close()
    writefile.close()
    print 'write over'


if __name__ == '__main__':
    Scitail_2_ExtreamPosNeg()
