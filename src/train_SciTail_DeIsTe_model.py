import cPickle
import gzip
import os
import sys
sys.setrecursionlimit(6000)
import time

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import random

from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from theano.tensor.signal import downsample
from random import shuffle
from scipy.stats import mode

from load_data import load_SciTailV1_dataset,load_word2vec, load_word2vec_to_init, extend_word2vec_lowercase
from common_functions import Conv_for_Pair,dropout_layer, store_model_to_file, elementwise_is_two,Conv_with_Mask_with_Gate, Conv_with_Mask, create_conv_para, L2norm_paraList, ABCNN, create_ensemble_para, cosine_matrix1_matrix2_rowwise, Diversify_Reg, Gradient_Cost_Para, GRU_Batch_Tensor_Input_with_Mask, create_LSTM_para


def evaluate_lenet5(learning_rate=0.01, n_epochs=10, L2_weight=0.000001, extra_size=4, emb_size=300, posi_emb_size=50,batch_size=50, filter_size=[3,3], maxSentLen=50, hidden_size=300):

    model_options = locals().copy()
    print "model options", model_options

    seed=1234
    np.random.seed(seed)
    rng = np.random.RandomState(seed)    #random seed, control the model generates the same results


    all_sentences_l, all_masks_l, all_sentences_r, all_masks_r, all_labels, word2id  =load_SciTailV1_dataset(maxlen=maxSentLen)  #minlen, include one label, at least one word in the sentence
    # test_sents_l, test_masks_l, test_sents_r, test_masks_r, test_labels, word2id  =load_ACE05_dataset(maxSentLen, word2id)

    train_sents_l=np.asarray(all_sentences_l[0], dtype='int32')
    dev_sents_l=np.asarray(all_sentences_l[1], dtype='int32')
    test_sents_l=np.asarray(all_sentences_l[2], dtype='int32')

    train_masks_l=np.asarray(all_masks_l[0], dtype=theano.config.floatX)
    dev_masks_l=np.asarray(all_masks_l[1], dtype=theano.config.floatX)
    test_masks_l=np.asarray(all_masks_l[2], dtype=theano.config.floatX)

    train_sents_r=np.asarray(all_sentences_r[0], dtype='int32')
    dev_sents_r=np.asarray(all_sentences_r[1]    , dtype='int32')
    test_sents_r=np.asarray(all_sentences_r[2]    , dtype='int32')

    train_masks_r=np.asarray(all_masks_r[0], dtype=theano.config.floatX)
    dev_masks_r=np.asarray(all_masks_r[1], dtype=theano.config.floatX)
    test_masks_r=np.asarray(all_masks_r[2], dtype=theano.config.floatX)


    train_labels_store=np.asarray(all_labels[0], dtype='int32')
    dev_labels_store=np.asarray(all_labels[1], dtype='int32')
    test_labels_store=np.asarray(all_labels[2], dtype='int32')

    train_size=len(train_labels_store)
    dev_size=len(dev_labels_store)
    test_size=len(test_labels_store)
    print 'train size: ', train_size, ' dev size: ', dev_size, ' test size: ', test_size

    vocab_size=len(word2id)+1


    rand_values=rng.normal(0.0, 0.01, (vocab_size, emb_size))   #generate a matrix by Gaussian distribution
    #here, we leave code for loading word2vec to initialize words
    rand_values[0]=np.array(np.zeros(emb_size),dtype=theano.config.floatX)
    id2word = {y:x for x,y in word2id.iteritems()}
    word2vec=load_word2vec()
    rand_values=load_word2vec_to_init(rand_values, id2word, word2vec)
    init_embeddings=theano.shared(value=np.array(rand_values,dtype=theano.config.floatX), borrow=True)   #wrap up the python variable "rand_values" into theano variable

    posi_rand_values=rng.normal(0.0, 0.01, (maxSentLen, posi_emb_size))   #generate a matrix by Gaussian distribution
    posi_embeddings=theano.shared(value=np.array(posi_rand_values,dtype=theano.config.floatX), borrow=True)   #wrap up the python variable "rand_values" into theano variable


    #now, start to build the input form of the model
    sents_ids_l=T.imatrix()
    sents_mask_l=T.fmatrix()
    sents_ids_r=T.imatrix()
    sents_mask_r=T.fmatrix()
    labels=T.ivector()
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    def embed_input(emb_matrix, sent_ids):
        return emb_matrix[sent_ids.flatten()].reshape((batch_size,maxSentLen, emb_size)).dimshuffle(0,2,1)

    embed_input_l=embed_input(init_embeddings, sents_ids_l)#embeddings[sents_ids_l.flatten()].reshape((batch_size,maxSentLen, emb_size)).dimshuffle(0,2,1) #the input format can be adapted into CNN or GRU or LSTM
    embed_input_r=embed_input(init_embeddings, sents_ids_r)#embeddings[sents_ids_r.flatten()].reshape((batch_size,maxSentLen, emb_size)).dimshuffle(0,2,1)



    '''create_AttentiveConv_params '''
    conv_W, conv_b=create_conv_para(rng, filter_shape=(hidden_size, 1, emb_size, filter_size[0]))
    conv_W_posi, conv_b_posi=create_conv_para(rng, filter_shape=(hidden_size, 1, emb_size+posi_emb_size, filter_size[0]))
    conv_W_context, conv_b_context=create_conv_para(rng, filter_shape=(hidden_size, 1, emb_size, 1))

    NN_para=[conv_W, conv_b,conv_W_posi, conv_b_posi,conv_W_context]

    '''
    attentive convolution function
    '''

    attentive_conv_layer = Conv_for_Pair(rng,
            origin_input_tensor3=embed_input_l,
            origin_input_tensor3_r = embed_input_r,
            input_tensor3=embed_input_l,
            input_tensor3_r = embed_input_r,
             mask_matrix = sents_mask_l,
             mask_matrix_r = sents_mask_r,
             image_shape=(batch_size, 1, emb_size, maxSentLen),
             image_shape_r = (batch_size, 1, emb_size, maxSentLen),
             filter_shape=(hidden_size, 1, emb_size, filter_size[0]),
             filter_shape_context=(hidden_size, 1,emb_size, 1),
             W=conv_W, b=conv_b,
             W_posi=conv_W_posi, b_posi=conv_b_posi,
             W_context=conv_W_context, b_context=conv_b_context,
             posi_emb_matrix = posi_embeddings,
             posi_emb_size = posi_emb_size)
    attentive_sent_embeddings_l = attentive_conv_layer.attentive_maxpool_vec_l
    attentive_sent_embeddings_r = attentive_conv_layer.attentive_maxpool_vec_r

    sent_embeddings_l = attentive_conv_layer.maxpool_vec_l
    sent_embeddings_r = attentive_conv_layer.maxpool_vec_r

    "form input to LR classifier"
    LR_input = T.concatenate([sent_embeddings_l,sent_embeddings_r,sent_embeddings_l*sent_embeddings_r,attentive_sent_embeddings_l,attentive_sent_embeddings_r,attentive_sent_embeddings_l*attentive_sent_embeddings_r],axis=1)
    LR_input_size=6*hidden_size

    U_a = create_ensemble_para(rng, 2, LR_input_size) # the weight matrix hidden_size*2
    LR_b = theano.shared(value=np.zeros((2,),dtype=theano.config.floatX),name='LR_b', borrow=True)  #bias for each target class
    LR_para=[U_a, LR_b]


    layer_LR=LogisticRegression(rng, input=LR_input, n_in=LR_input_size, n_out=2, W=U_a, b=LR_b) #basically it is a multiplication between weight matrix and input feature vector
    loss=layer_LR.negative_log_likelihood(labels)  #for classification task, we usually used negative log likelihood as loss, the lower the better.





    params = [init_embeddings,posi_embeddings]+NN_para+LR_para
    # L2_reg = (init_embeddings**2).sum()+(conv_W**2).sum()+(conv_W_context**2).sum()+(U_a**2).sum()

    cost=loss#+L2_weight*L2_reg

    updates =   Gradient_Cost_Para(cost,params, learning_rate)


    train_model = theano.function([sents_ids_l, sents_mask_l, sents_ids_r, sents_mask_r, labels], cost, updates=updates, allow_input_downcast=True, on_unused_input='ignore')
    dev_model = theano.function([sents_ids_l, sents_mask_l, sents_ids_r, sents_mask_r, labels], layer_LR.errors(labels), allow_input_downcast=True, on_unused_input='ignore')

    test_model = theano.function([sents_ids_l, sents_mask_l, sents_ids_r, sents_mask_r, labels], layer_LR.errors(labels), allow_input_downcast=True, on_unused_input='ignore')

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 50000000000  # look as this many examples regardless
    start_time = time.time()
    mid_time = start_time
    past_time= mid_time
    epoch = 0
    done_looping = False

    n_train_batches=train_size/batch_size
    train_batch_start=list(np.arange(n_train_batches)*batch_size)+[train_size-batch_size]
    n_dev_batches=dev_size/batch_size
    dev_batch_start=list(np.arange(n_dev_batches)*batch_size)+[dev_size-batch_size]
    n_test_batches=test_size/batch_size
    test_batch_start=list(np.arange(n_test_batches)*batch_size)+[test_size-batch_size]


    max_acc_dev=0.0
    max_acc_test=0.0
    max_f1=0.0

    cost_i=0.0
    train_indices = range(train_size)

    while epoch < n_epochs:
        epoch = epoch + 1

        random.Random(100).shuffle(train_indices) #shuffle training set for each new epoch, is supposed to promote performance, but not garrenteed

        iter_accu=0

        for batch_id in train_batch_start: #for each batch
            iter = (epoch - 1) * n_train_batches + iter_accu +1
            iter_accu+=1
            train_id_batch = train_indices[batch_id:batch_id+batch_size]
            cost_i+= train_model(
                                train_sents_l[train_id_batch],
                                train_masks_l[train_id_batch],
                                train_sents_r[train_id_batch],
                                train_masks_r[train_id_batch],
                                train_labels_store[train_id_batch])

            #after each 1000 batches, we test the performance of the model on all test data
            if iter%100==0:
                print 'Epoch ', epoch, 'iter '+str(iter)+' average cost: '+str(cost_i/iter), 'uses ', (time.time()-past_time)/60.0, 'min'
                past_time = time.time()
                dev_error_sum=0.0
                for dev_batch_id in dev_batch_start: # for each test batch
                    dev_error_i=dev_model(
                            dev_sents_l[dev_batch_id:dev_batch_id+batch_size],
                            dev_masks_l[dev_batch_id:dev_batch_id+batch_size],
                            dev_sents_r[dev_batch_id:dev_batch_id+batch_size],
                            dev_masks_r[dev_batch_id:dev_batch_id+batch_size],
                            dev_labels_store[dev_batch_id:dev_batch_id+batch_size])

                    dev_error_sum+=dev_error_i
                dev_acc=1.0-dev_error_sum/(len(dev_batch_start))


                if dev_acc > max_acc_dev:
                    max_acc_dev=dev_acc
                    print '\tcurrent dev_acc:', dev_acc,' ; ','\tmax_dev_acc:', max_acc_dev


                    error_sum=0.0
                    for idd, test_batch_id in enumerate(test_batch_start): # for each test batch
                        error_i=test_model(
                                test_sents_l[test_batch_id:test_batch_id+batch_size],
                                test_masks_l[test_batch_id:test_batch_id+batch_size],
                                test_sents_r[test_batch_id:test_batch_id+batch_size],
                                test_masks_r[test_batch_id:test_batch_id+batch_size],
                                test_labels_store[test_batch_id:test_batch_id+batch_size])

                        error_sum+=error_i
                    test_acc=1.0-error_sum/(len(test_batch_start))
                    if test_acc > max_acc_test:
                        max_acc_test=test_acc
                        store_model_to_file('/home/wenpeng/workspace/SciTail/src/model_para_'+str(max_acc_test), params)
                    print '\t\tcurrent acc:', test_acc,' ; ','\t\tmax_acc:', max_acc_test
                else:
                    print '\tcurrent dev_acc:', dev_acc,' ; ','\tmax_dev_acc:', max_acc_dev


        print 'Epoch ', epoch, 'uses ', (time.time()-mid_time)/60.0, 'min'
        mid_time = time.time()

        #print 'Batch_size: ', update_freq
    end_time = time.time()

    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    return max_acc_test



if __name__ == '__main__':
    evaluate_lenet5()
    # lr_list=[0.005,0.01,0.02,0.03,0.001]
    # batch_list=[10,20,30,40,50,60,70,80,100]
    # maxlen_list=[20,25,30,35,40,45,50,55]
    #
    # best_acc=0.0
    # best_lr=0.01
    # for lr in lr_list:
    #     acc_test= evaluate_lenet5(learning_rate=lr)
    #     if acc_test>best_acc:
    #         best_lr=lr
    #         best_acc=acc_test
    #     print '\t\t\t\tcurrent best_acc:', best_acc
    # best_batch=50
    # for batch in batch_list:
    #     acc_test= evaluate_lenet5(learning_rate=best_lr,  batch_size=batch)
    #     if acc_test>best_acc:
    #         best_batch=batch
    #         best_acc=acc_test
    #     print '\t\t\t\tcurrent best_acc:', best_acc
    #
    # best_maxlen=40
    # for maxlen in maxlen_list:
    #     acc_test= evaluate_lenet5(learning_rate=best_lr,  batch_size=best_batch, maxSentLen=maxlen)
    #     if acc_test>best_acc:
    #         best_maxlen=maxlen
    #         best_acc=acc_test
    #     print '\t\t\t\tcurrent best_acc:', best_acc
    # print 'Hyper tune finished, best test acc: ', best_acc, ' by  lr: ', best_lr, ' batch: ', best_batch, ' maxlen: ', best_maxlen
