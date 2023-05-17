from __future__ import print_function, division
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import datetime
import math
import time
import shutil
import pdb

# GPU settings
core_idx = 0
os.environ["CUDA_VISIBLE_DEVICES"] = "{0}".format(core_idx)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#sess = tf.Session(config=config)

###############################Input###############################################

#filename = "802_11n_N648_R56_z27"
#filename = "wman_N0576_R34_z24"
filename = "5G_Base1_R05_N552_z12"
code_Proto_input = np.loadtxt("./BaseGraph/{0}.txt".format(filename), int, delimiter='	')
M_proto_input,N_proto_input = code_Proto_input.shape
q_bit = 5

#0: No weights // 1: edge,iter // 2: node, iter // 3: iter // 4: edge  // 5: Node // 6: only one // 11: degree, iter
sharing = [1,0,0,0,2] #CN_Weight,UCN_Weight,CN_Bias,UCN_Bias, VN_Weight
sampling_type = 0 #0: Default, 1: Read_Uncor, 2: Collect_Uncor, 3: Active learning
shuffle_data = 0 # sampling_type = 1 일때 매 epoch 마다 data shuffle 2: start shuffle
decoding_type = 2 #0:SP, 1:MS,  2: QMS
dmax = 0 #for Active learning
d_in_const = 0 # for Active learning
z_value = 12 #802:27, wman: 24
M_proto_ind = 0 #(6,3): 10, (7,4): 10, (15,11): 13, Liva_ARA: 10, RM: 17, Wman_r075: 6, 802_11n_R56: 4
iters_max = 20   # number of iterations
fixed_iter = 0
fixed_init = 10 #fixed_init 만큼은 initializing만 이전까지 learning된 weight로 하고 learning을 함
iter_step = 10
loss_type = 2 #0: BCE, 1: Soft BER , 2: FER_last, 3: FER (적어도 한 iter에 대해 all zero면 성공으로 판정) (측정 FER은 Dec_LLR = 0 인 경우 오류라 판정, Loss FER은 Dec_LLR = 0인 경우 1/2 만큼 증가 => 둘사이 차이 발생)
opt_result_print = 3 #0: BER_last, 1: FER_last, 2:FER, 3: Loss
etha_start = 0 #1: Equal, 0: Last iter
etha_discount = 0 # for multi-loss ftn
etha_discount_step = 0
learn_rate_start = 0.001
learn_rate_discount = 0
learn_rate_step = 0
batch_size = 20
sample_num = 50000
epoch_input = 100
valid_ratio = 500 # <=1 : ratio, >1: number
training_ratio = -1 # <=1 : ratio, >1: number, -1: remaining all
test_flag = 1 #1: opt나올때만 test, 2: check epoch마다 test
num_of_batch = math.floor(sample_num / batch_size)
check_epoch = 1

init_from_file = 0
init_weight = 0.75  #-1: Xavier_init
init_VN_weight = 1.0
init_bias = 0
Max_weight = 2
Min_weight = 0
Max_bias = 7.5
Min_bias = -7.5
pruning = 0 #1: 실제 pruning, 2: pruning_node 기준으로, 3: pruning_node 기준으로, iter const, 4: adding_node 기준으로, 5: adding_node 기준으로, iter const
pruning_step = 1
seed_in = 1

restart_num = 0
restart_point = np.array([])


input_CN_surv = np.ones([iters_max,M_proto_input],dtype = np.int64)
#pruning_node = [10,2,6,0,12,11] #(6,3)
#pruning_node = [16,14,3,17,9,2,15,12,7,20,8,5,10,18,21,6,11,4,19,0,1] #(7,4)
#pruning_node = [18,21,10,9,6,15,12,5,19,20,26,29,35,31,13,33,16,37,45,4,22,44,34,7,38,23,39,24,2,41,8,25,36,11,42,28,32,14,17,40,1,27,30,3,0] #(15,11) iter 10
#pruning_node = [25,23,20,17,19,26,36,16,27,33,11,12,7,13,10,8,4,5,40,14,32,30,37,34,35,38,18,3,41,2,31,21,45,39,6,42,1,22,24,44,28,9,15,0,29]#(15,11) iter 15

#adding_node = [0,1,2,4,5,7,8,9,11,12,6,3,10]
#adding_node = [0,1,3,11,14,17,19,22,28,32,33,34,45,31,25,15,30,7,42,36,41,6,21,35,5,44,24,12,38,27,43,39,37,2,8,18,4,9,40,13,10,23,29,26,16,20]
#adding_node = [0,1,3,11,14,17,19,22,28,32,33,34,45,31,25,15,30,13,37,43,42,20,8,41,44,2,16,40,4,39,6,35,36,23,24,18,38,12,7,10,5,9,27,29,21,26]
#adding_node = [0,1,2,3,4,6,7,8,9,11,10,5]#ARA
#adding_node = [0,23,43,109,217,222,229,236,244,245,303,332,357,386,427,482,620,82,40,249,402,426,479,486,212,81,378,325,404,168,506,188,509,483,286,285,313,125,351,8,67,466,121,44,324,326,101,170,489,118,194,605,399,452,199,115,594,45,171,491,9,94]#RM

update_CN_skip_idx = np.array([0])
is_skip_CN = 0
clip_tanh = 10.0
clip_LLR = 50.0

#SNR_Matrix = np.array([4, 4, 4, 4, 4, 4, 4, 4, 4, 4])
#SNR_Matrix = np.array([6.5, 5.5, 5.0, 4.5, 4.5, 4, 4, 4, 4, 4])
#SNR_Matrix = np.array([2.5, 3.0, 3.5, 4.0, 4.5])
#SNR_Matrix = np.array([2.0, 2.5, 3.0, 3.5, 4.0])
#SNR_Matrix = np.array([2.25])
SNR_Matrix = np.array([1.0,1.5,2,2.5,3.0])
#SNR_Matrix = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
#SNR_Matrix = np.array([6.5, 5.5, 5.0, 4.7, 4.7, 4.5, 4.5, 4.5, 4.5, 4.5])


#### CHECKING_INPUT ####
if sampling_type == 1:        
    if len(SNR_Matrix) > 1:
        SNR_Matrix = np.array([0.0])
elif sampling_type == 2:
    if len(SNR_Matrix) > 1:
        print("sampling_type == 2 and len(SNR_Matrix) > 1")
        exit()
        
if np.sum(sharing)==0:
    print("np.sum(sharing)==0")
    exit()

if any(value in [4,5,6] for value in sharing) and (iters_max - fixed_iter) % iter_step > 0:
    print("any(value in [4,5,6] for value in sharing) and (iters_max - fixed_iter) % iter_step > 0")
    exit()
    
if sharing[4] in [1,4]:
    print("sharing[4] in [1,4]")
    exit()

if any(value in [4,5,6] for value in sharing):
    RNN_mode = 1
else:
    RNN_mode = 0
    
    
if (sharing[1] != 0 and sharing[0]!=sharing[1]) or (sharing[3]!=0 and sharing[2]!=sharing[3]):
    print("sharing[1] != 0 and sharing[0]!=sharing[1]) or (sharing[3]!=0 and sharing[2]!=sharing[3]")
    exit()
    

###################################################################################

out_filename = "Out{0}_{1}_I{2}_CW{3}_UCW{4}_CB{5}_UCB{6}_VW{7}_L{8}_U{9}_DEC{10}_S{11}".format(core_idx,filename,iters_max,*sharing,loss_type,sampling_type,decoding_type,seed_in)

    
# train settings
train_on_zero_word = True
code_GM = []

# ramdom seed
word_seed = 2042
noise_seed = 1074 + seed_in
wordRandom = np.random.RandomState(word_seed)  # word seed
noiseRandom = np.random.RandomState(noise_seed)  # noise seed


if sampling_type == 1:
    uncor_filename = "[Uncor]_{0}".format(filename)
    input_llr_total = np.loadtxt("./Input{0}/{1}.txt".format(q_bit,uncor_filename), dtype=np.float32, delimiter='	')
    if input_llr_total.ndim > 1:
        input_llr_total = np.delete(input_llr_total,[0,1,2],1)
        
    if input_llr_total.shape[0]<sample_num:
        print("Wrong input: input_llr_total.shape[0]<sample_num")
        exit()
    else:
        input_llr_total = input_llr_total[:sample_num,:]
    
    if shuffle_data == 2:
        np.random.shuffle(input_llr_total)
    
    if train_on_zero_word == False:
        codeword_filename = "[Codeword]_{0}".format(filename)
        input_codeword_total = np.loadtxt("./Input{0}/{1}.txt".format(q_bit,codeword_filename),dtype=np.int32, delimiter='	')
    else:
        input_codeword_total =  np.zeros(input_llr_total.shape, dtype=np.int64)
        
    if valid_ratio <= 1:
        valid_num = round(valid_ratio * sample_num)
    elif valid_ratio > 1:
        valid_num = valid_ratio
        
    if training_ratio <= 1 and training_ratio >= 0:
        training_num = round(training_ratio * sample_num)
    elif training_ratio > 1:
        training_num = training_ratio
    elif training_ratio == -1:
        training_num = sample_num - valid_num
        
    input_llr_valid = input_llr_total[-valid_num:,:]
    input_llr_training = input_llr_total[:training_num,:]
    
    input_codeword_valid = input_codeword_total[-valid_num:,:]
    input_codeword_training = input_codeword_total[:training_num,:]
    
    
    if test_flag >= 1:
        uncor_filename = "[Uncor]_{0}_Test".format(filename)
        input_llr_test = np.loadtxt("./Input{0}/{1}.txt".format(q_bit,uncor_filename), dtype=np.float32, delimiter='	')
        if input_llr_test.ndim > 1:
            input_llr_test = np.delete(input_llr_test,[0,1,2],1)
        test_num = input_llr_test.shape[0]
        input_codeword_test = np.zeros(input_llr_test.shape, dtype=np.int64)
        
    #pdb.set_trace()
        
else:
    training_num = sample_num
    if valid_ratio <= 1:
        valid_num = round(valid_ratio * sample_num)
    elif valid_ratio > 1:
        valid_num = valid_ratio
    test_num = sample_num
    

def hard_sigmoid(x):
    #return tf.clip_by_value((x+0.5), 0, 1)
    #return tf.clip_by_value((x + 1.)/2., 0, 1)
    return tf.clip_by_value(x, 0, 1)

def proxy_sign(x):
    return tf.clip_by_value(x, -1, 1)

def inv_exp(x):
    return 2./(1.+tf.exp(-x))-1.
    
def round_through(x):
    rounded = tf.round(x)
#    return x + tf.stop_gradient(rounded-x) #foward = rounded, gradient = 1
    return hard_sigmoid(x) + tf.stop_gradient(rounded-hard_sigmoid(x)) #foward = rounded, gradient = 1

def sign_through(x):
    sign_v = tf.sign(x)
    #return proxy_sign(x) + tf.stop_gradient(sign_v-proxy_sign(x))
    return inv_exp(x) + tf.stop_gradient(sign_v-inv_exp(x))


def QMS_clipping(x):
    global q_bit
    if q_bit == 5:
        return tf.clip_by_value(x, -7.5, 7.5)
    elif q_bit == 4:
        return tf.clip_by_value(x, -7, 7)
    
def Cal_MSA_Q_TF(x):
    '''Element-wise rounding to the closest integer with full gradient propagation.
    A trick from [Sergey Ioffe](http://stackoverflow.com/a/36480182)
    a op that behave as f(x) in forward mode,
    but as g(x) in the backward mode.
    '''
    global q_bit
    if q_bit == 5:
        q_value = tf.clip_by_value(tf.round(x * 2)/2,-7.5,7.5) #(-7.5 -7.0 -6.5 ... 6.5 7.0 7.5) Quantizer  
    elif q_bit == 4:
        q_value = tf.clip_by_value(tf.round(x),-7,7) #(-7.0 -6.0 ... 6.0 7.0) Quantizer  
        
    return QMS_clipping(x) + tf.stop_gradient(q_value-QMS_clipping(x)) #foward = q_value, gradient = 1

def Cal_MSA_Q(x):
    global q_bit
    if q_bit == 5:
        q_value = np.clip(np.round(x * 2)/2,-7.5,7.5) #(-7.5 -7.0 -6.5 ... 6.5 7.0 7.5) Quantizer  
    elif q_bit == 4:
        q_value = np.clip(np.round(x),-7,7) #(-7.0 -6.0 ... 6.0 7.0) Quantizer  
    
    return q_value


def init_parameter(code_Proto,SNR_Matrix,M_proto_ind):
    M_proto,N_proto = code_Proto.shape
    code_Base = code_Proto.copy()
    for i in range(0, code_Proto.shape[0]):
        for j in range(0, code_Proto.shape[1]):
            if (code_Proto[i, j] == -1):
                code_Base[i, j] = 0
            else:
                code_Base[i, j] = 1
                
                
    CN_deg_proto = np.sum(code_Base, axis=1)
    VN_deg_proto = np.sum(code_Base, axis=0)
    
    Num_VN_deg = (np.unique(VN_deg_proto)).size
    Max_VN_deg = np.max(VN_deg_proto)
    
    VN_deg_profile = np.zeros(Max_VN_deg,dtype = int)
    k = 1
    for d in range(1,Max_VN_deg + 1):
        if d in np.unique(VN_deg_proto):
            VN_deg_profile[d - 1] = k
            k = k + 1

    Num_edge_proto = np.sum(VN_deg_proto)
    if M_proto_ind == 0:
        M_proto_ind = M_proto
    code_rate = 1.0 * (N_proto - M_proto_ind) / (N_proto)
    
    # train SNR
    SNR_lin = 10.0 ** (SNR_Matrix / 10.0)
    SNR_sigma = np.sqrt(1.0 / (2.0 * SNR_lin * code_rate))
    
    return M_proto, N_proto, code_Base, CN_deg_proto, VN_deg_proto, Num_edge_proto, code_rate, SNR_sigma, Num_VN_deg, Max_VN_deg, VN_deg_profile

def read_uncor_llr(input_llr,input_codeword,batch_idx,batch_size,code_n,Z):
    X =  -np.reshape(input_llr[batch_idx * batch_size:(batch_idx + 1) * batch_size,...],[batch_size,code_n,Z]) # defined as p1/p0
    if decoding_type == 2:
        X = Cal_MSA_Q(X)
        
    Y =  input_codeword[batch_idx * batch_size:(batch_idx + 1) * batch_size,:]
    
    return X, Y


#get train samples by active learning
def create_active_input(SNR_set, wordRandom, noiseRandom, batch_size, N_proto, K_proto, z_value, is_zeros_word, net_dict, iters_max, dmax,fixed_iter):


    if fixed_iter>0:
        X = np.zeros([batch_size * 2, N_proto * z_value], dtype=np.float32)
        Y = np.zeros([batch_size * 2, N_proto * z_value], dtype=np.int64)
        curr_batch_size = 0
        while curr_batch_size <= batch_size:
            X_curr, Y_curr = create_mix_epoch(SNR_set, wordRandom, noiseRandom, batch_size,
                                                                   N_proto, K_proto, z_value, [],
                                                                   is_zeros_word)

            y_pred_fix = sess.run(fetches=net_dict["ya_output{0}".format(fixed_iter - 1)], feed_dict={xa: X_curr, ya: Y_curr, etha: 1,batch_prob: False})
            
            X_curr = np.reshape(X_curr, [batch_size, N_proto * z_value])
            Y_curr = np.reshape(Y_curr, [batch_size, N_proto * z_value])

            d_out_fixed = np.abs(((y_pred_fix > 0) - Y_curr)).sum(axis=1)
            pass_flag = (d_out_fixed > 0)
            num_new_sample = pass_flag.sum()

            X[curr_batch_size:curr_batch_size + num_new_sample,:] = X_curr[pass_flag,:]
            Y[curr_batch_size:curr_batch_size + num_new_sample,:] = Y_curr[pass_flag,:]

            curr_batch_size = curr_batch_size + num_new_sample
            
        X = X[:batch_size,:]
        Y = Y[:batch_size,:]
        X = np.reshape(X, [batch_size, N_proto, z_value])
    else:
        X, Y = create_mix_epoch(SNR_set, wordRandom, noiseRandom, batch_size,
                                                           N_proto, K_proto, z_value, [],
                                                           is_zeros_word)
        
        
        
    y_pred_max = sess.run(fetches=net_dict["ya_output{0}".format(iters_max - 1)], feed_dict={xa: X, ya: Y, etha: 1,learn_rate: 1,real_batch_size: batch_size, batch_prob: False})

    X = np.reshape(X, [batch_size, N_proto * z_value])
    Y = np.reshape(Y, [batch_size, N_proto * z_value])

    
    d_in = np.abs(((X > 0) - Y)).sum(axis=1)
    d_out_max = np.abs(((y_pred_max > 0) - Y)).sum(axis=1)
    if d_in_const:
        if dmax == 0:
            remove_flag = (d_out_max == 0) + (d_out_max >= d_in)
        else:
            remove_flag = (d_in > dmax) + (d_out_max == 0) + (d_out_max >= d_in)
    else:
        if dmax == 0:
            remove_flag = np.zeros(batch_size,dtype=np.int64)
        else:
            remove_flag = (d_in > dmax)
        
    
    pass_flag = (remove_flag == 0) 
    true_batch_size = batch_size - (remove_flag>0).sum()
    
    X[remove_flag>0,:] = 0
    Y[remove_flag>0,:] = 0

    X = np.vstack((X[pass_flag>0,:],X[remove_flag>0,:]))
    Y = np.vstack((Y[pass_flag>0,:],Y[remove_flag>0,:]))
    
    X = np.reshape(X, [batch_size, N_proto , z_value])
    Y = Y[:]
            
    return X,Y,true_batch_size

    
#get train samples
def create_mix_epoch(scaling_factor, wordRandom, noiseRandom, batch_size, code_n, code_k, Z, code_GM, is_zeros_word):
    X = np.zeros([1, code_n * Z], dtype=np.float32)
    Y = np.zeros([1, code_n * Z], dtype=np.int64)
    
    curr_batch_size = 0
    while curr_batch_size < batch_size:
        for sf_i in scaling_factor:
            if is_zeros_word:
                infoWord_i = 0 * wordRandom.randint(0, 2, size=(1, code_k * Z))
                Y_i = 0 * wordRandom.randint(0, 2, size=(1, code_n * Z))
            else:
                infoWord_i = wordRandom.randint(0, 2, size=(1, code_k * Z))
                Y_i = np.dot(infoWord_i, code_GM) % 2

             # pay attention to this 1->1 0->-1
            X_p_i = noiseRandom.normal(0.0, 1.0, Y_i.shape) * sf_i + (-1) ** (1 - Y_i) 
            x_llr_i = 2 * X_p_i / ((sf_i) ** 2)  # defined as p1/p0

            
            if decoding_type == 2:
                x_llr_i = Cal_MSA_Q(x_llr_i)
            
            
            X = np.vstack((X, x_llr_i))
            Y = np.vstack((Y, Y_i))
            curr_batch_size = curr_batch_size + 1
            if curr_batch_size == batch_size:
                break
            
            
    X = X[1:]
    Y = Y[1:]
    X = np.reshape(X, [batch_size, code_n, Z]) # [B,N,Z]
    return X, Y
    

def calc_ber_fer(y_pred_all, iters_max, Y_test,batch_size):
    
    #uncor_flag = np.abs(((Y_test_pred >= 0) - Y_test)).sum(axis=1) > 0
    uncor_flag = np.empty([0,batch_size])
    for i in range(iters_max):
        uncor_flag_curr_iter = np.abs(((y_pred_all[i*batch_size:(i+1)*batch_size,:] >= 0) - Y_test)).sum(axis=1) > 0
        uncor_flag = np.append(uncor_flag,uncor_flag_curr_iter.reshape(1,batch_size) ,axis = 0 )
   
    uncor_flag = np.min(uncor_flag,axis = 0)
    
    fer = (uncor_flag).sum() * 1.0 / batch_size
    error_num = ((y_pred_all[(iters_max - 1)*batch_size:((iters_max - 1)+1)*batch_size,:] >= 0) - Y_test).sum(axis=1)
    ber_last = np.abs(error_num.sum()) / (Y_test.shape[0] * Y_test.shape[1])
    
    uncor_last_flag = np.abs(((y_pred_all[(iters_max - 1)*batch_size:((iters_max - 1)+1)*batch_size,:] >= 0) - Y_test)).sum(axis=1) > 0
    fer_last = (uncor_last_flag).sum() * 1.0 / Y_test.shape[0]
    
    return ber_last, fer_last, fer, uncor_flag, error_num



############################     init the connecting matrix between network layers   #################################

def init_connecting_matrix(code_Proto,code_Base,N_proto,M_proto,Num_edge_proto,z_value,Num_VN_deg,VN_deg_profile,VN_deg_proto):
    Lift_Matrix1 = []
    Lift_Matrix2 = []
    W_odd2even = np.zeros((Num_edge_proto, Num_edge_proto), dtype=np.float32)
    W_skipconn2even = np.zeros((N_proto, Num_edge_proto), dtype=np.float32)
    W_even2odd = np.zeros((Num_edge_proto, Num_edge_proto), dtype=np.float32)
    W_even2odd_with_self = np.zeros((Num_edge_proto, Num_edge_proto), dtype=np.float32) #tau>0 일때 CN 기준 min 값을 구하기 위해서 도입
    W_output = np.zeros((Num_edge_proto, N_proto), dtype=np.float32)
    W_skipconn2odd = np.zeros((M_proto,Num_edge_proto),dtype=np.float32) #KHY (21.12.22)
    W_deg2odd = np.zeros((Num_VN_deg,Num_edge_proto),dtype=np.float32)
    W_deg2output = np.zeros((Num_VN_deg,N_proto), dtype=np.float32)

    # init lifting matrix for cyclic shift
    Lift_M1 = np.zeros((Num_edge_proto * z_value, Num_edge_proto * z_value), np.float32)
    Lift_M2 = np.zeros((Num_edge_proto * z_value, Num_edge_proto * z_value), np.float32)

    k = 0
    for j in range(0, code_Proto.shape[1]):
        for i in range(0, code_Proto.shape[0]):
            if (code_Proto[i, j] != -1):
                Lift_num = code_Proto[i, j] % z_value
                for h in range(0, z_value, 1):
                    Lift_M1[k * z_value + h, k * z_value + (h + Lift_num) % z_value] = 1
                k = k + 1
    k = 0
    for i in range(0, code_Proto.shape[0]):
        for j in range(0, code_Proto.shape[1]):
            if (code_Proto[i, j] != -1):
                Lift_num = code_Proto[i, j] % z_value
                for h in range(0, z_value, 1):
                    Lift_M2[k * z_value + h, k * z_value + (h + Lift_num) % z_value] = 1
                k = k + 1
    Lift_Matrix1.append(Lift_M1)
    Lift_Matrix2.append(Lift_M2)



    # init W_odd2even  variable node updating
    k = 0
    vec_tmp = np.zeros((Num_edge_proto), dtype=np.float32)  # even layer index read with column
    for j in range(0, code_Base.shape[1], 1):  # run over the columns
        for i in range(0, code_Base.shape[0], 1):  # break after the first one
            if (code_Base[i, j] == 1):  # finding the first one is ok
                num_of_conn = int(np.sum(code_Base[:, j]))  # get the number of connection of the variable node
                idx = np.argwhere(code_Base[:, j] == 1)  # get the indexes
                for l in range(0, num_of_conn, 1):  # adding num_of_conn columns to W
                    vec_tmp = np.zeros((Num_edge_proto), dtype=np.float32)
                    for r in range(0, code_Base.shape[0], 1):  # adding one to the right place
                        if (code_Base[r, j] == 1 and idx[l][0] != r):
                            idx_row = np.cumsum(code_Base[r, 0:j + 1])[-1] - 1
                            odd_layer_node_count = 0
                            if r > 0:
                                odd_layer_node_count = np.cumsum(CN_deg_proto[0:r])[-1]
                            vec_tmp[idx_row + odd_layer_node_count] = 1  # offset index adding
                    W_odd2even[:, k] = vec_tmp.transpose()
                    k += 1
                break

    # init W_even2odd  parity check node updating
    k = 0
    for j in range(0, code_Base.shape[1], 1):
        for i in range(0, code_Base.shape[0], 1):
            if (code_Base[i, j] == 1):
                idx_row = np.cumsum(code_Base[i, 0:j + 1])[-1] - 1
                idx_col = np.cumsum(code_Base[0: i + 1, j])[-1] - 1
                odd_layer_node_count_1 = 0
                odd_layer_node_count_2 = np.cumsum(CN_deg_proto[0:i + 1])[-1]
                if i > 0:
                    odd_layer_node_count_1 = np.cumsum(CN_deg_proto[0:i])[-1]
                W_even2odd[k, odd_layer_node_count_1:odd_layer_node_count_2] = 1.0
                W_even2odd[k, odd_layer_node_count_1 + idx_row] = 0.0
                
                W_even2odd_with_self[k, odd_layer_node_count_1:odd_layer_node_count_2] = 1.0
                k += 1  # k is counted in column direction
                

    # init W_output odd to output
    k = 0
    for j in range(0, code_Base.shape[1], 1):
        for i in range(0, code_Base.shape[0], 1):
            if (code_Base[i, j] == 1):
                idx_row = np.cumsum(code_Base[i, 0:j + 1])[-1] - 1
                idx_col = np.cumsum(code_Base[0: i + 1, j])[-1] - 1
                odd_layer_node_count = 0
                if i > 0:
                    odd_layer_node_count = np.cumsum(CN_deg_proto[0:i])[-1]
                W_output[odd_layer_node_count + idx_row, k] = 1.0
        k += 1

    # init W_skipconn2even  channel input
    k = 0
    for j in range(0, code_Base.shape[1], 1):
        for i in range(0, code_Base.shape[0], 1):
            if (code_Base[i, j] == 1):
                W_skipconn2even[j, k] = 1.0
                k += 1

    # init W_skipconn2odd  channel input
    k = 0
    for i in range(0, code_Base.shape[0], 1):
        for j in range(0, code_Base.shape[1], 1):
            if (code_Base[i, j] == 1):
                W_skipconn2odd[i, k] = 1.0
                W_deg2odd[VN_deg_profile[VN_deg_proto[j] - 1] - 1,k] = 1.0 #khy 22.11.08
                k += 1            

    for j in range(0,code_Base.shape[1],1):
        W_deg2output[VN_deg_profile[VN_deg_proto[j] - 1] - 1,j] = 1.0
                
                
    return Lift_Matrix1,Lift_Matrix2,W_odd2even,W_skipconn2even,W_even2odd,W_output,W_skipconn2odd,W_even2odd_with_self,W_deg2odd,W_deg2output

    

                
##############################  bulid neural network ############################

def build_neural_network(net_dict,curr_iter,iters_max,fixed_iter,training_iter_start,training_iter_end,xa,ya,N_proto,M_proto,Num_edge_proto,z_value,batch_size,Lift_Matrix1,Lift_Matrix2,W_odd2even,W_skipconn2even,W_even2odd,W_output,W_skipconn2odd,W_even2odd_with_self,W_deg2odd,W_deg2output,CN_surv_edge,etha,learn_rate,real_batch_size):
    
# VN 연산부터 함, Standard Decoding 방식 (CN 연산부터함)과 다름
    
    xa_input = tf.transpose(xa, [0, 2, 1]) # [B,N,Z] -> [B,Z,N]
    
    
    #if curr_iter > 0:
    if sharing[4] == 2 or sharing[4] == 3:
        xa_input = tf.multiply(xa_input,net_dict[f'var_{4}_{curr_iter}'])
    elif sharing[4] == 5 or sharing[4] == 6:
        if curr_iter < fixed_iter:
            xa_input = tf.multiply(xa_input,net_dict[f'var_{4}_{curr_iter}'])
        else:
            xa_input = tf.multiply(xa_input,net_dict[f'var_{4}_{fixed_iter}'])
    elif sharing[4] == 11:
        VN_weight_per_node = tf.matmul(tf.reshape(net_dict[f'var_{4}_{curr_iter}'],[1,-1]),W_deg2output)
        xa_input = tf.multiply(xa_input,VN_weight_per_node)
        
    if decoding_type == 2:
        xa_input = Cal_MSA_Q_TF(xa_input)

        
    if (sharing[1]>0 or sharing[3]>0):
        if curr_iter == 0:
            VN_APP = xa_input  #[B,Z,N]
        else:
            VN_APP = tf.reshape(net_dict["ya_output{0}".format(curr_iter - 1)],[batch_size,N_proto, z_value]) #[B,N,Z]
            VN_APP = tf.transpose(VN_APP,[0,2,1]) #[B,Z,N]
            
        VN_APP = -VN_APP #[B,Z,N] (APP>0 <-> Dec=0)가 되게끔 sign change
        VN_APP_sign = tf.add(tf.to_float(VN_APP > 0),-tf.to_float(VN_APP <= 0)) #[B,Z,N]
        VN_APP_sign_edge = tf.matmul(VN_APP_sign, W_skipconn2even)  #[B,Z,E(V)] 
        VN_APP_sign_edge = tf.transpose(VN_APP_sign_edge, [0, 2, 1]) #[B,E(V),Z]
        VN_APP_sign_edge = tf.reshape(VN_APP_sign_edge, [batch_size, Num_edge_proto * z_value]) #[B,E(V)Z] (e1,1)(e1,2),...,(e1,z),(e2,1),...
        CN_in_sign = tf.matmul(VN_APP_sign_edge, Lift_Matrix1[0].transpose()) #[B,E(C)Z]
        CN_in_sign = tf.reshape(CN_in_sign, [batch_size, Num_edge_proto, z_value]) #[B,E(C),Z]
        CN_in_sign = tf.transpose(CN_in_sign, [0, 2, 1]) #[B,Z,E(C)], CN_input 형태로 완성
        CN_in_sign_tile = tf.tile(CN_in_sign, multiples=[1, 1, Num_edge_proto]) #[B,Z,E(C),E(C)]
        CN_in_sign_tile = tf.multiply(CN_in_sign_tile, tf.reshape(W_even2odd_with_self.transpose(), [-1]))
        CN_in_sign_tile = tf.reshape(CN_in_sign_tile, [batch_size, z_value, Num_edge_proto, Num_edge_proto])
        CN_in_sign_tile = tf.add(CN_in_sign_tile, 1.0 * (1 - tf.to_float(tf.abs(CN_in_sign_tile) > 0)))  
        CN_sign_edge = tf.reduce_prod(CN_in_sign_tile,axis=3)
        UCN_idx_edge = tf.to_float(CN_sign_edge < 0) #[B,Z,E(C)]        
        UCN_idx_edge = tf.transpose(UCN_idx_edge, [0, 2, 1])  #[B,Z,E(C)] -> [B,E(C),Z]
        UCN_idx_edge = tf.reshape(UCN_idx_edge, [batch_size, z_value * Num_edge_proto])
        UCN_idx_edge = tf.matmul(UCN_idx_edge, Lift_Matrix2[0])
        UCN_idx_edge = tf.reshape(UCN_idx_edge, [batch_size, Num_edge_proto, z_value]) #[B,E(V),Z]
        UCN_idx_edge = tf.transpose(UCN_idx_edge, [0, 2, 1]) #[B,Z,E(V)], Weighting 전의 C->V
        SCN_idx_edge = tf.add(-UCN_idx_edge,tf.ones((batch_size,z_value,Num_edge_proto), dtype=tf.float32))
    else:
        UCN_idx_edge = tf.zeros((batch_size, z_value, Num_edge_proto)) #[B,Z,E(V)]
        SCN_idx_edge = tf.add(-UCN_idx_edge,tf.ones((batch_size,z_value,Num_edge_proto), dtype=tf.float32))
    

            
    #variable node update

    x0 = tf.matmul(xa_input, W_skipconn2even)  #xa_input [B X Z X N], W_skipconn [N X E(V)] ==> x0 [B X Z X E(V)], Edge(VN 기준)별로 xa_input을 배치함, V->C
    x1 = tf.matmul(net_dict["LLRa{0}".format(curr_iter)], W_odd2even) #LLR [B,Z,E(C)]  C->V, W_odd2even [E(C),E(V)], x1[B,Z,E(V)] V->C
    x2 = tf.add(x0, x1)# x2 [B,Z,E(V)] V->C
    x2 = tf.transpose(x2, [0, 2, 1]) #[B,E(V),Z]
    x2 = tf.reshape(x2, [batch_size, Num_edge_proto * z_value]) #[B,E(V)Z] (e1,1)(e1,2),...,(e1,z),(e2,1),...
    x2 = tf.matmul(x2, Lift_Matrix1[0].transpose()) #x2 [B,E(V)Z], Lift_Matrix1 [E(V)Z,E(C)Z], x2 [B,E(C)Z] // Lift_Matrix1을 곱하는건 CN in 전에 Permute (?)
    x2 = tf.reshape(x2, [batch_size, Num_edge_proto, z_value]) #[B,E(C),Z]
    x2 = tf.transpose(x2, [0, 2, 1]) #[B,Z,E(C)], CN_input 완성
    
    if decoding_type == 2:
        x2 = Cal_MSA_Q_TF(x2)
    if decoding_type == 1 or decoding_type == 2:
        x2 = tf.add(x2, 0.0001 * (1 - tf.to_float(tf.abs(x2) > 0)))  #실제로 0인값을 0.1로 처리해서 뒤에 10000 으로 키우는 filter에 적용안되게 함
    
    x_tile = tf.tile(x2, multiples=[1, 1, Num_edge_proto]) #[B,Z,E(C)E(C)]
    W_input_reshape = tf.reshape(W_even2odd.transpose(), [-1]) #W_even2odd [E(V),E(C)], trans [E(C),E(V)]하고 1-dim [E(V,c1->v1),E(V,c1->v2),...]으로 바꿈 ?, 
    #check node update
    x_tile_mul = tf.multiply(x_tile, W_input_reshape)
    x2_1 = tf.reshape(x_tile_mul, [batch_size, z_value, Num_edge_proto, Num_edge_proto]) #각 Edge별로 extrinsic한 값을 4번째 차원에 배치한듯?

    if decoding_type == 0:
        x2_clip = 0.5 * tf.clip_by_value(x2_1, clip_value_min=-clip_tanh, clip_value_max=clip_tanh)
        x2_tanh = tf.tanh(-x2_clip)
        x2_abs = tf.add(x2_tanh, 1 - tf.to_float(tf.abs(x2_tanh) > 0))
        x3 = tf.reduce_prod(x2_abs, reduction_indices=3)
        x_output_0 = -tf.log(tf.div(1 + x3, 1 - x3))
    elif decoding_type == 1 or decoding_type == 2:
        x2_abs = tf.add(tf.abs(x2_1), 10000 * (1 - tf.to_float(tf.abs(x2_1) > 0))) # W_input_reshape과 곱하면서 생기는 zero가 min이 되지 않도록
        x3 = tf.reduce_min(x2_abs, axis=3)
        x3 = tf.add(x3, -0.0001 * (1 - tf.to_float(tf.abs(x3) > 0.0001))) #0->0.1로 변환한거 원래대로 돌림
        x2_2 = -x2_1 
        x4 = tf.add(tf.zeros((batch_size, z_value, Num_edge_proto, Num_edge_proto)), 1 - 2 * tf.to_float(x2_2 < 0))
        x4_prod = -tf.reduce_prod(x4, axis=3)
        x_output_0 = tf.multiply(x3, tf.sign(x4_prod))
        

#     if curr_iter == 0:
#         net_dict["point1"] = x_output_0[0,:,:]
#     if curr_iter == 1:
#         net_dict["point2"] = x_output_0[0,:,:]
#     if curr_iter == 2:
#         net_dict["point3"] = x_output_0[0,:,:]
#     if curr_iter == 3:
#         net_dict["point4"] = x_output_0[0,:,:]
#     if curr_iter == 4:
#         net_dict["point5"] = x_output_0[0,:,:] 
                
    
    x_output_0 = tf.transpose(x_output_0, [0, 2, 1])  #[B,Z,E(C)] -> [B,E(C),Z]
    x_output_0 = tf.reshape(x_output_0, [batch_size, z_value * Num_edge_proto])
    x_output_0 = tf.matmul(x_output_0, Lift_Matrix2[0])
    x_output_0 = tf.reshape(x_output_0, [batch_size, Num_edge_proto, z_value]) #[B,E(V),Z]
    x_output_0 = tf.transpose(x_output_0, [0, 2, 1]) #[B,Z,E(V)], Weighting 전의 C->V
    
    
    # revised by khy 22.01.01
    if sharing[0] == 0:             
        x_output_1 = tf.abs(x_output_0)
    elif sharing[0] == 1:   
        if sharing[1] == 1:
            W_per_edge_1 = net_dict[f'var_{0}_{curr_iter}']
            W_per_edge_2 = net_dict[f'var_{1}_{curr_iter}']
            x_output_11 = tf.multiply(tf.abs(x_output_0),W_per_edge_1) #[B,Z,E]
            x_output_12 = tf.multiply(tf.abs(x_output_0),W_per_edge_2) #[B,Z,E]
            x_output_1 = tf.add(tf.multiply(x_output_11, SCN_idx_edge),tf.multiply(x_output_12,UCN_idx_edge))
        else:            
            W_per_edge = net_dict[f'var_{0}_{curr_iter}']
            x_output_1 = tf.multiply(tf.abs(x_output_0),W_per_edge)
    elif sharing[0] == 2:
        if sharing[1] == 2:
            W_per_edge_1 = tf.matmul(tf.reshape(net_dict[f'var_{0}_{curr_iter}'],[1,M_proto]),W_skipconn2odd)
            W_per_edge_2 = tf.matmul(tf.reshape(net_dict[f'var_{1}_{curr_iter}'],[1,M_proto]),W_skipconn2odd)
            x_output_11 = tf.multiply(tf.abs(x_output_0),W_per_edge_1) #batch_size * Edge_num
            x_output_12 = tf.multiply(tf.abs(x_output_0),W_per_edge_2) #batch_size * Edge_num
            x_output_1 = tf.add(tf.multiply(x_output_11,-UCN_idx_edge + 1.0),tf.multiply(x_output_12,UCN_idx_edge))
        else:
            W_per_edge = tf.matmul(tf.reshape(net_dict[f'var_{0}_{curr_iter}'],[1,M_proto]),W_skipconn2odd) #[1,M] * [M,E(C)] = [1,E(C)]
            x_output_1 = tf.multiply(tf.abs(x_output_0),W_per_edge) #[B,Z,E]
    elif sharing[0] == 3:
        if sharing[1] == 3:
            W_per_edge_1 = tf.matmul(tf.reshape(tf.tile(net_dict[f'var_{0}_{curr_iter}'],[M_proto]),[1,M_proto]),W_skipconn2odd)
            W_per_edge_2 = tf.matmul(tf.reshape(tf.tile(net_dict[f'var_{1}_{curr_iter}'],[M_proto]),[1,M_proto]),W_skipconn2odd)
            x_output_11 = tf.multiply(tf.abs(x_output_0),W_per_edge_1) #[B,Z,E]
            x_output_12 = tf.multiply(tf.abs(x_output_0),W_per_edge_2) #[B,Z,E]
            x_output_1 = tf.add(tf.multiply(x_output_11, SCN_idx_edge),tf.multiply(x_output_12,UCN_idx_edge))
        else:
            W_per_edge = tf.matmul(tf.reshape(tf.tile(net_dict[f'var_{0}_{curr_iter}'],[M_proto]),[1,M_proto]),W_skipconn2odd)
            x_output_1 = tf.multiply(tf.abs(x_output_0),W_per_edge)
    elif sharing[0] == 4:
        if curr_iter < fixed_iter:
            W_per_edge = net_dict[f'var_{0}_{curr_iter}']
        else:
            W_per_edge = net_dict[f'var_{0}_{fixed_iter}']
        x_output_1 = tf.multiply(tf.abs(x_output_0),W_per_edge) #[B,Z,E]
    elif sharing[0] == 5:
        if curr_iter < fixed_iter:
            W_per_edge = tf.matmul(tf.reshape(net_dict[f'var_{0}_{curr_iter}'],[1,M_proto]),W_skipconn2odd)
        else:
            W_per_edge = tf.matmul(tf.reshape(net_dict[f'var_{0}_{fixed_iter}'],[1,M_proto]),W_skipconn2odd)
        x_output_1 = tf.multiply(tf.abs(x_output_0),W_per_edge) #[B,Z,E]
    elif sharing[0] == 6:
        if curr_iter < fixed_iter:
            W_per_edge = tf.matmul(tf.reshape(tf.tile(net_dict[f'var_{0}_{curr_iter}'],[M_proto]),[1,M_proto]),W_skipconn2odd)
        else:
            W_per_edge = tf.matmul(tf.reshape(tf.tile(net_dict[f'var_{0}_{fixed_iter}'],[M_proto]),[1,M_proto]),W_skipconn2odd) 
        x_output_1 = tf.multiply(tf.abs(x_output_0),W_per_edge)
    elif sharing[0] == 11:
        if sharing[1] == 11:
            W_per_edge_1 = tf.matmul(tf.reshape(net_dict[f'var_{0}_{curr_iter}'],[1,-1]),W_deg2odd)
            W_per_edge_2 = tf.matmul(tf.reshape(net_dict[f'var_{1}_{curr_iter}'],[1,-1]),W_deg2odd)
            x_output_11 = tf.multiply(tf.abs(x_output_0),W_per_edge_1) #[B,Z,E]
            x_output_12 = tf.multiply(tf.abs(x_output_0),W_per_edge_2) #[B,Z,E]
            x_output_1 = tf.add(tf.multiply(x_output_11, SCN_idx_edge),tf.multiply(x_output_12,UCN_idx_edge))
        else:
            W_per_edge = tf.matmul(tf.reshape(net_dict[f'var_{0}_{curr_iter}'],[1,-1]),W_deg2odd)
            x_output_1 = tf.multiply(tf.abs(x_output_0),W_per_edge)      
                          
                          
    if sharing[2] == 0:
        x_output_2 = x_output_1
    elif sharing[2] == 1:
        if sharing[3] == 1:
            B_per_edge_1 = net_dict[f'var_{2}_{curr_iter}']
            B_per_edge_2 = net_dict[f'var_{3}_{curr_iter}']
            x_output_21 = tf.add(x_output_1,-B_per_edge_1) #[B,Z,E]
            x_output_22 = tf.add(x_output_1,-B_per_edge_2) #[B,Z,E]
            x_output_2 = tf.add(tf.multiply(x_output_21, SCN_idx_edge),tf.multiply(x_output_22,UCN_idx_edge))
        else:            
            B_per_edge = net_dict[f'var_{2}_{curr_iter}']
            x_output_2 = tf.add(x_output_1,-B_per_edge)
    elif sharing[2] == 2:
        if sharing[3] == 2:
            B_per_edge_1 = tf.matmul(tf.reshape(net_dict[f'var_{2}_{curr_iter}'],[1,M_proto]),W_skipconn2odd)
            B_per_edge_2 = tf.matmul(tf.reshape(net_dict[f'var_{3}_{curr_iter}'],[1,M_proto]),W_skipconn2odd)
            x_output_21 = tf.add(x_output_1,-B_per_edge_1) #batch_size * Edge_num
            x_output_22 = tf.add(x_output_1,-B_per_edge_2) #batch_size * Edge_num
            x_output_2 = tf.add(tf.multiply(x_output_21,SCN_idx_edge),tf.multiply(x_output_22,UCN_idx_edge))
        else:
            B_per_edge = tf.matmul(tf.reshape(net_dict[f'var_{2}_{curr_iter}'],[1,M_proto]),W_skipconn2odd)
            x_output_2 = tf.add(x_output_1,-B_per_edge)
    elif sharing[2] == 3:
        if sharing[3] == 3:
            B_per_edge_1 = tf.matmul(tf.reshape(tf.tile(net_dict[f'var_{2}_{curr_iter}'],[M_proto]),[1,M_proto]),W_skipconn2odd)
            B_per_edge_2 = tf.matmul(tf.reshape(tf.tile(net_dict[f'var_{3}_{curr_iter}'],[M_proto]),[1,M_proto]),W_skipconn2odd)
            x_output_21 = tf.add(x_output_1,-B_per_edge_1) #[B,Z,E]
            x_output_22 = tf.add(x_output_1,-B_per_edge_2) #[B,Z,E]
            x_output_2 = tf.add(tf.multiply(x_output_21, SCN_idx_edge),tf.multiply(x_output_22,UCN_idx_edge))
        else:
            B_per_edge = tf.matmul(tf.reshape(tf.tile(net_dict[f'var_{2}_{curr_iter}'],[M_proto]),[1,M_proto]),W_skipconn2odd)
            x_output_2 = tf.add(x_output_1,-B_per_edge)
    elif sharing[2] == 4:
        if curr_iter < fixed_iter:
            B_per_edge = net_dict[f'var_{2}_{curr_iter}']
        else:
            B_per_edge = net_dict[f'var_{2}_{fixed_iter}']
            
        x_output_2 = tf.add(x_output_1,-B_per_edge)
    elif sharing[2] == 5:
        if curr_iter < fixed_iter:
            B_per_edge = tf.matmul(tf.reshape(net_dict[f'var_{2}_{curr_iter}'],[1,M_proto]),W_skipconn2odd)
        else:
            B_per_edge = tf.matmul(tf.reshape(net_dict[f'var_{2}_{fixed_iter}'],[1,M_proto]),W_skipconn2odd)
        x_output_2 = tf.add(x_output_1,-B_per_edge) #[B,Z,E]
    elif sharing[2] == 6:
        if curr_iter < fixed_iter:
            B_per_edge = tf.matmul(tf.reshape(tf.tile(net_dict[f'var_{2}_{curr_iter}'],[M_proto]),[1,M_proto]),W_skipconn2odd)
        else:
            B_per_edge = tf.matmul(tf.reshape(tf.tile(net_dict[f'var_{2}_{fixed_iter}'],[M_proto]),[1,M_proto]),W_skipconn2odd) 
        x_output_2 = tf.add(x_output_1,-B_per_edge)
        
    # Max( W * min(V->C), 0)
    x_output_2 = tf.multiply(x_output_2, tf.to_float(x_output_2 > 0))  
    
    if decoding_type == 2:
        x_output_2 = Cal_MSA_Q_TF(x_output_2)
        
        
    net_dict["LLRa{0}".format(curr_iter+1)] = tf.multiply(x_output_2, tf.sign(x_output_0)) # B X Z X E, C->V Message
    
    
    y_output_2 = tf.matmul(net_dict["LLRa{0}".format(curr_iter+1)], W_output) #B X Z X N, Sum_Input_LLR
    y_output_3 = tf.transpose(y_output_2, [0, 2, 1]) # B,Z,N -> B,N,Z
    
    #Decision
    if decoding_type == 2:
        xa = Cal_MSA_Q_TF(xa)
    
    y_output_4 = tf.add(xa, y_output_3) #B X N X Z, APP_LLR
    y_output_4 = tf.clip_by_value(y_output_4, clip_value_min=-clip_LLR, clip_value_max=clip_LLR)
    net_dict["ya_output{0}".format(curr_iter)] = tf.reshape(y_output_4, [batch_size, N_proto * z_value], name='ya_output'.format(curr_iter))
    
    # Loss
    if curr_iter == training_iter_end - 1 and sampling_type != 2 and fixed_iter != iters_max:
        if loss_type <= 2:
            loss_ftn = 0
            temp_coeff = 0
            for t in range(max(training_iter_start - fixed_init,fixed_iter),training_iter_end,1):
                if loss_type == 0:
                    loss_ftn = loss_ftn + pow(etha,(training_iter_end - 1 - t)) * tf.nn.sigmoid_cross_entropy_with_logits(labels=ya[0:real_batch_size],
                                                           logits=net_dict["ya_output{0}".format(t)][0:real_batch_size,:])
                elif loss_type == 1:
                    x_temp = net_dict["ya_output{0}".format(t)][0:real_batch_size,:]
                    loss_ftn = loss_ftn + pow(etha,(training_iter_end - 1 - t)) * tf.math.sigmoid(x_temp) #<-only for all zero codeword
                elif loss_type == 2:
                    x_temp = -net_dict["ya_output{0}".format(t)][0:real_batch_size,:]
                    x_temp = 1/2*(1-sign_through(tf.reduce_min(x_temp, axis=1))) #<-only for all zero codeword
                    loss_ftn = loss_ftn + pow(etha,(training_iter_end - 1 - t)) * x_temp
                    
                temp_coeff = temp_coeff + pow(etha,(training_iter_end - 1 - t))
            loss_ftn = loss_ftn / temp_coeff
            
        
        elif loss_type == 3:#<-only for all zero codeword
            x_temp = -net_dict["ya_output{0}".format(training_iter_start)][0:real_batch_size,:]
            loss_ftn = tf.reshape(1/2*(1-sign_through(tf.reduce_min(x_temp, axis=1))),[real_batch_size,1])
            for t in range(training_iter_start,training_iter_end,1):
                x_temp = -net_dict["ya_output{0}".format(t)][0:real_batch_size,:]
                curr_uncor_idx = tf.reshape(1/2*(1-sign_through(tf.reduce_min(x_temp, axis=1))),[real_batch_size,1])
                loss_ftn = tf.concat([loss_ftn,curr_uncor_idx],axis = 1)
            loss_ftn = tf.reduce_min(loss_ftn,axis = 1) 


        net_dict["lossa"] = 1.0 * tf.reduce_mean(loss_ftn, name='lossa')
            
        
        
        # var_list
        current_vars = []
        # Loop through the sharing list and add variables to the current variable list
        for i, share_type in enumerate(sharing):
            # Determine the iteration range based on the sharing type
            if share_type == 0:
                start_iter, end_iter = None, None
            elif share_type in [1,2,3,11]:
                start_iter, end_iter = max(training_iter_start - fixed_init,fixed_iter), training_iter_end
            elif share_type in [4,5,6]:
                start_iter, end_iter = fixed_iter, fixed_iter + 1
                
            # Add the variables to the current variable list
            if start_iter is not None and end_iter is not None:
                for j in range(start_iter, end_iter):
                    current_vars.append(net_dict[f'var_{i}_{j}'])

        net_dict["train_stepa"] = tf.train.AdamOptimizer(learning_rate=
                                                            learn_rate).minimize(net_dict["lossa"], var_list = current_vars)
        
    if curr_iter == 0:
        net_dict["ya_output_all"] = net_dict["ya_output{0}".format(curr_iter)]
    else:
        net_dict["ya_output_all"] = tf.concat([net_dict["ya_output_all"],net_dict["ya_output{0}".format(curr_iter)]],axis = 0)
        
    return net_dict


def weight_init(net_dict, training_iter_start, training_iter_end, fixed_iter, core_idx, filename, sharing, Num_edge_proto, M_proto, Num_VN_deg, Min_bias, Max_bias, init_bias, Min_weight, Max_weight, init_weight, init_VN_weight,init_from_file):
    
    if init_from_file == 1:
        input_weights_filename = f"./Weights/[Opt_Weight_NMS]_C{core_idx}_{filename}_End{training_iter_end}.txt"
    else:
        input_weights_filename = f"./Weights/[Opt_Weight_NMS]_C{core_idx}_{filename}_End{training_iter_start}.txt"
        
    temp_row = 0
    for i, share_type in enumerate(sharing):
        
        if share_type in [1, 4]:
            para_shape = Num_edge_proto
        elif share_type in [2, 5]:
            if i in [0,1,2,3]:
                para_shape = M_proto
            elif i in [4]:
                para_shape = N_proto
        elif share_type in [3, 6]:
            para_shape = 1
        elif share_type == 11:
            para_shape = Num_VN_deg
            
            
        if i in [0, 1]:
            para_min, para_max, para_init = Min_weight, Max_weight, init_weight
        elif i in [2, 3]:
            para_min, para_max, para_init = Min_bias, Max_bias, init_bias
        elif i in [4]:
            para_min, para_max, para_init = Min_weight, Max_weight, init_VN_weight
            
        if share_type in [1,2,3,11]:
            make_var_iter_end = training_iter_end
        elif share_type in [4,5,6]:
            make_var_iter_end = fixed_iter + 1
            
        if share_type > 0:
            for curr_iter in range(0, make_var_iter_end):
                if curr_iter < training_iter_start or init_from_file == 1:    
                    temp_row += 1
                    data = np.loadtxt(input_weights_filename, skiprows = 1 + temp_row, max_rows = 1, dtype=np.float32, delimiter='\t')
                    init = tf.constant_initializer(data)
                else:
                    if para_init == -1:
                        init = tf.truncated_normal_initializer(mean=(para_min + para_max) / 2, stddev=0.1, seed=i)
                    else:
                        init = tf.constant_initializer(para_init * np.ones(para_shape, dtype=np.float32))
                    
                var_name = f'var_{i}_{curr_iter}'
                net_dict[var_name] = tf.get_variable(var_name, dtype=tf.float32, shape=para_shape, initializer=init, constraint=lambda z: tf.clip_by_value(z, para_min, para_max))
                
            temp_row += 1
                    
    return net_dict



##################################  save weights and biases  ####################################
def print_info():
    out_file = open('./Weights/Performance_{0}.txt'.format(out_filename),'w')
    print('CN_weight_sharing = {0} UCW_weight_sharing = {1} CN_bias_sharing = {2} UCN_bias_sharing = {3} VN_weight_sharing = {4}'.format(*sharing),file=out_file)
    print('Init_CN_weight = {0} Max_weight = {1} Min_weight = {2} Init_CN_bias = {3} Max_bias = {4}, Min_bias = {5}, Init_VN_weight = {6}'.format(init_weight,Max_weight,Min_weight,init_bias,Max_bias,Min_bias,init_VN_weight),file=out_file)
    print('samping_type = {0}, shuffle = {1}, pruning = {2}, dmax = {3}, d_in_const = {4}'.format(sampling_type, shuffle_data, pruning, dmax,d_in_const),file=out_file)
    print('z_value = {0} M_proto_ind = {1} iters_max = {2} fixed_iter = {3} fixed_init = {4} iter_step = {5}'.format(z_value,M_proto_ind,iters_max,fixed_iter,fixed_init,iter_step),file=out_file)
    print('etha_start = {0} etha_discount = {1} etha_discount_step = {2}'.format(etha_start,etha_discount,etha_discount_step),file=out_file)
    print('loss_type = {0} learn_rate_start = {1} learn_rate_discount = {2} learn_rate_step = {3}'.format(loss_type,learn_rate_start,learn_rate_discount,learn_rate_step),file=out_file)
    print('batch_size = {0} num_of_batch = {1} epochs = {2} sample_num = {3} training_num = {4} valid_num = {5}'.format(batch_size,num_of_batch,epoch_input,sample_num,training_num,valid_num),file=out_file)
    print('restart_num = {0} restart_point = {1}'.format(restart_num,restart_point),file=out_file)
    print('update_CN_skip_idx = {0}'.format(update_CN_skip_idx), file=out_file)
    print('SNR_Matrix = {0}'.format(SNR_Matrix),file=out_file)
    if pruning == 2 or pruning == 3:
        print('Pruning_node = {0}'.format(pruning_node),file=out_file)
    if pruning == 4 or pruning == 6:
        print('Adding_node = {0}'.format(adding_node),file=out_file)
    print('',file=out_file)
    out_file.close()
    
    
def print_result(curr_epoch, opt_value, training_iter_start, training_iter_end, fixed_iter,etha_curr):

    out_file = open( "./Weights/[Weight_NMS]_C{0}_{1}_End{2}.txt".format(core_idx,filename,training_iter_end),'w')
    print("{0} {1} {2} {3} {4}\n".format(*sharing),file = out_file)
    

    #Weights print
    for i, share_type in enumerate(sharing):
        if share_type in [1,2,3,11]:
            for curr_iter in range(0, training_iter_end, 1):
                a = sess.run(fetches=[net_dict[f"var_{i}_{curr_iter}"]])
                np.savetxt(out_file,a,fmt = '%s', delimiter='	')
            print('',file=out_file)
        elif share_type in [4,5,6]:
            for curr_iter in range(0, training_iter_end, 1):
                if curr_iter < fixed_iter:
                    a = sess.run(fetches=[net_dict[f"var_{i}_{curr_iter}"]])
                else:
                    a = sess.run(fetches=[net_dict[f"var_{i}_{fixed_iter}"]])
                np.savetxt(out_file,a,fmt = '%s', delimiter='	')
            print('',file=out_file)
            
    out_file.close()
    
    ber_last_SNR = np.zeros((1,SNR_sigma.size),dtype=np.float32)
    fer_last_SNR = np.zeros((1,SNR_sigma.size),dtype=np.float32)
    fer_SNR = np.zeros((1,SNR_sigma.size),dtype=np.float32)
    loss_SNR = np.zeros((1,SNR_sigma.size),dtype=np.float32)
    
    valid_batch_num = math.floor(valid_num/batch_size)
    for valid_idx in range(0,valid_batch_num,1):
        for SNR_idx in range(0,SNR_sigma.size,1):
            SNR_point = np.array([SNR_sigma[SNR_idx]])
            if sampling_type == 0 or sampling_type == 2:
                training_received_data, training_coded_bits = create_mix_epoch(SNR_point, wordRandom, noiseRandom, batch_size,
                                                                       N_proto, N_proto - M_proto_ind, z_value,
                                                                       [],
                                                                       train_on_zero_word)
                true_batch_size = batch_size
            elif sampling_type == 1:
                training_received_data, training_coded_bits = read_uncor_llr(input_llr_valid,input_codeword_valid,valid_idx,batch_size,N_proto,z_value)
                true_batch_size = batch_size
            elif sampling_type == 3:
                training_received_data, training_coded_bits, true_batch_size = create_active_input(SNR_point, wordRandom, noiseRandom, batch_size, N_proto, N_proto - M_proto_ind, z_value, train_on_zero_word, net_dict, training_iter_end, dmax, fixed_iter)
           
            if sampling_type == 2 or iters_max == fixed_iter:
                y_pred_all = sess.run(fetches=net_dict["ya_output_all"], feed_dict={xa: training_received_data, ya: training_coded_bits, etha: etha_curr, learn_rate: 1, real_batch_size: batch_size, batch_prob: False})
                loss_batch = 0
            else:
                y_pred_all, loss_batch = sess.run(fetches=[net_dict["ya_output_all"],net_dict["lossa"]], feed_dict={xa: training_received_data, ya: training_coded_bits, etha: etha_curr, learn_rate: 0, real_batch_size: batch_size, batch_prob: False})


            
            ber_last_batch,fer_last_batch, fer_batch, uncor_flag,error_num = calc_ber_fer(y_pred_all,training_iter_end, training_coded_bits,batch_size)
            
            if sampling_type == 2 and np.sum(uncor_flag == 1)>0:
                out_uncor_filename = "Uncor_{0}.txt".format(core_idx)
                out_file = open(out_uncor_filename,'a')
                num_uncor = np.sum(uncor_flag == 1)
                uncor_received_data = -np.reshape(training_received_data[uncor_flag==1,:,:],[num_uncor,z_value * N_proto])
                np.savetxt(out_file,np.concatenate((np.zeros((num_uncor,3)),uncor_received_data),axis = 1),fmt='%.1f',delimiter = '	')
                out_file.close()
            
            
            
            ber_last_SNR[0,SNR_idx] = ber_last_SNR[0,SNR_idx] + ber_last_batch/valid_batch_num
            fer_last_SNR[0,SNR_idx] = fer_last_SNR[0,SNR_idx] + fer_last_batch/valid_batch_num
            fer_SNR[0,SNR_idx] = fer_SNR[0,SNR_idx] + fer_batch/valid_batch_num
            loss_SNR[0,SNR_idx] = loss_SNR[0,SNR_idx] + loss_batch/valid_batch_num
    
    print_flag = 0
    if opt_result_print == 0 and opt_value > np.sum(ber_last_SNR):
        opt_value = np.sum(ber_last_SNR)
        print_flag = 1
    elif opt_result_print == 1 and opt_value > np.sum(fer_last_SNR):
        opt_value = np.sum(fer_last_SNR)
        print_flag = 1
    elif opt_result_print == 2 and opt_value > np.sum(fer_SNR):
        opt_value = np.sum(fer_SNR)
        print_flag = 1
    elif opt_result_print == 3 and opt_value > np.sum(loss_SNR):
        opt_value = np.sum(loss_SNR)
        print_flag = 1
        
    
    out_file = open('./Weights/Performance_{0}.txt'.format(out_filename),'a')
    print('Out_iter: {0} training_iter_start: {1} training_iter_end: {2} epoch: [{3}/{4}]'.format(out_iter,training_iter_start,training_iter_end,curr_epoch + 1,epoch_input),file = out_file)
    print('Valid_Result',file = out_file)
    np.savetxt(out_file,ber_last_SNR,fmt='%s',delimiter = ' ')
    np.savetxt(out_file,fer_last_SNR,fmt='%s',delimiter = ' ')
    np.savetxt(out_file,fer_SNR,fmt='%s',delimiter = ' ')
    np.savetxt(out_file,loss_SNR,fmt='%s',delimiter = ' ')
    print('{0}'.format(opt_value),file=out_file)
    print('',file=out_file)
    
    
    if print_flag == 1:
        shutil.copyfile( "./Weights/[Weight_NMS]_C{0}_{1}_End{2}.txt".format(core_idx,filename,training_iter_end),
 "./Weights/[Opt_Weight_NMS]_C{0}_{1}_End{2}.txt".format(core_idx,filename,training_iter_end))
    

    print('Valid_Result \n BER_last: {0}\nFER_last: {1}\nFER: {2}\nloss: {3}'.format(ber_last_SNR,fer_last_SNR,fer_SNR,loss_SNR))
    print("opt_value: {0}\n".format(opt_value))
    
    out_file.close()
    
    if sampling_type == 1 and test_num > 0 and ((test_flag == 1 and print_flag == 1) or (test_flag == 2)):
        ber_last_SNR = np.zeros((1,SNR_sigma.size),dtype=np.float32)
        fer_last_SNR = np.zeros((1,SNR_sigma.size),dtype=np.float32)
        fer_SNR = np.zeros((1,SNR_sigma.size),dtype=np.float32)
        loss_SNR = np.zeros((1,SNR_sigma.size),dtype=np.float32)
        test_batch_num = math.floor(test_num/batch_size)
        for test_idx in range(0,test_batch_num,1):

            training_received_data, training_coded_bits = read_uncor_llr(input_llr_test,input_codeword_test,test_idx,batch_size,N_proto,z_value)

            y_pred_all, loss_batch = sess.run(fetches=[net_dict["ya_output_all"],net_dict["lossa"]], feed_dict={xa: training_received_data, ya: training_coded_bits, etha: 0, learn_rate: 0, real_batch_size: batch_size, batch_prob: False})
            ber_last_batch,fer_last_batch, fer_batch, uncor_flag,error_num = calc_ber_fer(y_pred_all,training_iter_end, training_coded_bits,batch_size)

            ber_last_SNR[0,SNR_idx] = ber_last_SNR[0,SNR_idx] + ber_last_batch/test_batch_num
            fer_last_SNR[0,SNR_idx] = fer_last_SNR[0,SNR_idx] + fer_last_batch/test_batch_num
            fer_SNR[0,SNR_idx] = fer_SNR[0,SNR_idx] + fer_batch/test_batch_num
            loss_SNR[0,SNR_idx] = loss_SNR[0,SNR_idx] + loss_batch/test_batch_num


        out_file = open('./Weights/Performance_{0}.txt'.format(out_filename),'a')
        print('Test_Result',file = out_file)
        np.savetxt(out_file,ber_last_SNR,fmt='%s',delimiter = ' ')
        np.savetxt(out_file,fer_last_SNR,fmt='%s',delimiter = ' ')
        np.savetxt(out_file,fer_SNR,fmt='%s',delimiter = ' ')
        np.savetxt(out_file,loss_SNR,fmt='%s',delimiter = ' ')
        print('',file=out_file)

        print('Test_Result \n BER_last: {0}\nFER_last: {1}\nFER: {2}\nloss: {3}\n\n'.format(ber_last_SNR,fer_last_SNR,fer_SNR,loss_SNR))

        out_file.close()
    
    
    return opt_value


##################################  init the learnable network parameters  ####################################

CN_surv = input_CN_surv.copy()

if pruning == 1:
    CN_org_idx = list(np.arange(0,M_proto_input))
elif pruning == 2 or pruning == 3:
    out_iter_max = np.int64(M_proto_input/pruning_step)
    iters_max_input = iters_max
elif pruning == 4 or pruning == 5:
    out_iter_max = np.int64((len(adding_node) - M_proto_ind + 1)/pruning_step)
    iters_max_input = iters_max
else:
    out_iter_max = 1
    
print_info()

# the decoding neural network 
for out_iter in range(0,out_iter_max,1):
    
    code_Proto = code_Proto_input
    M_proto_input,N_proto_input = code_Proto.shape
    M_proto,N_proto,code_Base,CN_deg_proto,VN_deg_proto,Num_edge_proto,code_rate,SNR_sigma, Num_VN_deg, Max_VN_deg, VN_deg_profile = init_parameter(code_Proto,SNR_Matrix,M_proto_ind)
    Lift_Matrix1,Lift_Matrix2,W_odd2even,W_skipconn2even,W_even2odd,W_output,W_skipconn2odd,W_even2odd_with_self,W_deg2odd,W_deg2output =           init_connecting_matrix(code_Proto,code_Base,N_proto,M_proto,Num_edge_proto,z_value,Num_VN_deg,VN_deg_profile,VN_deg_proto)
    
    ##################################  Pruning parameter  ####################################
    
    if pruning == 1:
        code_Proto = np.delete(code_Proto,pruning_node,0)
    elif pruning == 2 or pruning == 3:
        code_Proto = np.delete(code_Proto,pruning_node[0:out_iter],0)
    elif pruning == 4 or pruning == 5:
        code_Proto =code_Proto[sorted(adding_node[0:out_iter + M_proto_ind]),:]
    if pruning == 3 or pruning == 5:
        iters_max = np.int64(np.ceil( (iters_max_input * M_proto_ind)/(M_proto_input)))
    if pruning >= 1:
        CN_surv = np.ones([iters_max,M_proto_input],dtype = np.int64)
    CN_surv_edge = np.ones([iters_max, Num_edge_proto],dtype = np.int64)
    for i in range(0,iters_max,1):
        idx_temp = 0
        for ii in range(0,M_proto,1):
            if CN_surv[i,ii] == 0:
                CN_surv_edge[i,idx_temp:idx_temp + CN_deg_proto[ii]] = 0
            idx_temp = idx_temp + CN_deg_proto[ii]
    if sharing[0] == 1:
        CN_surv = CN_surv_edge
    ##################################  Pruning parameter  ####################################
    

    
    training_iter_start = fixed_iter
    training_iter_end = fixed_iter + iter_step
    while (training_iter_end <= iters_max):
        opt_value = 100000
        print('\nOut_iter: {0} training_iter_start: {1} training_iter_end: {2} epoch: [{3}/{4}]'.format(out_iter,training_iter_start,training_iter_end,0,epoch_input))
        
        ##################################  Build Network  ####################################
        tf.reset_default_graph()
        sess = tf.Session(config=config)
        xa = tf.placeholder(tf.float32, shape=[batch_size, N_proto, z_value], name='xa')
        ya = tf.placeholder(tf.float32, shape=[batch_size, N_proto * z_value], name='ya')
        etha = tf.placeholder(tf.float32, name='etha')
        learn_rate = tf.placeholder(tf.float32, name='learn_rate')
        real_batch_size = tf.placeholder(tf.int64, name='real_batch_size')
        batch_prob = tf.placeholder(tf.bool)
        net_dict = {}
        net_dict["LLRa0"] = tf.zeros((batch_size, z_value, Num_edge_proto), dtype=tf.float32)
        net_dict["infoM_lastlayera0"] = tf.zeros((batch_size,z_value,Num_edge_proto), dtype=tf.float32)
        weight_init(net_dict, training_iter_start, training_iter_end, fixed_iter, core_idx, filename, sharing, Num_edge_proto, M_proto, Num_VN_deg, Min_bias, Max_bias, init_bias, Min_weight, Max_weight, init_weight,init_VN_weight,init_from_file)
        
        for i in range(0, training_iter_end, 1):
            net_dict = build_neural_network(net_dict,i,iters_max,fixed_iter,training_iter_start,training_iter_end,xa,ya,N_proto,M_proto,Num_edge_proto,z_value,batch_size,Lift_Matrix1,Lift_Matrix2,W_odd2even,W_skipconn2even,W_even2odd,W_output,W_skipconn2odd,W_even2odd_with_self,W_deg2odd,W_deg2output,CN_surv_edge,etha,learn_rate,real_batch_size)    
            
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver() 
        ##################################  Build Network  ####################################

        etha_curr = etha_start
        opt_value = print_result(0,opt_value,training_iter_start,training_iter_end,fixed_iter,etha_curr)
        
        learn_rate_curr = learn_rate_start
        training_batch_num = math.floor(training_num/batch_size)
        for epoch_idx in range(0,epoch_input,1):
            if sampling_type == 1 and shuffle_data == 1:
                np.random.shuffle(input_llr_training)

            for batch_idx in range(0, training_batch_num, 1):

                if sampling_type == 0:
                    training_received_data, training_coded_bits = create_mix_epoch(SNR_sigma, wordRandom, noiseRandom, batch_size,
                                                                               N_proto, N_proto - M_proto_ind, z_value,
                                                                               [],
                                                                               train_on_zero_word)
                    true_batch_size = batch_size
                elif sampling_type == 1:
                    training_received_data, training_coded_bits = read_uncor_llr(input_llr_training,input_codeword_training,batch_idx,batch_size,N_proto,z_value)
                    true_batch_size = batch_size
                    
                if etha_discount !=0 and etha_discount_step != 0 and (epoch_idx + 1)%etha_discount_step == 0:
                    etha_curr = etha_curr * etha_discount

                if learn_rate_discount != 0 and learn_rate_step != 0 and (epoch_idx + 1)%learn_rate_step == 0:
                    learn_rate_curr = learn_rate_curr * learn_rate_discount

                ##################################  Training  ####################################
                if sampling_type != 2 and true_batch_size != 0:
                    y_pred, train_loss, _ = sess.run(fetches=[net_dict["ya_output{0}".format(training_iter_end - 1)], net_dict["lossa"],
                                                              net_dict["train_stepa"]],
                                                     feed_dict={xa: training_received_data, ya: training_coded_bits, etha: etha_curr,learn_rate: learn_rate_curr, real_batch_size: true_batch_size, batch_prob: True})
                ##################################  Training  ####################################
                

            if (epoch_idx + 1) % check_epoch == 0:
                print('Out_iter: {0} training_iter_start: {1} training_iter_end: {2} epoch: [{3}/{4}]'.format(out_iter,training_iter_start,training_iter_end,epoch_idx + 1,epoch_input))
                opt_value = print_result(epoch_idx,opt_value,training_iter_start,training_iter_end,fixed_iter,etha_curr)
            
        #매 while loop
        training_iter_start = training_iter_start + iter_step
        training_iter_end = training_iter_end + iter_step


                
    #pruning 부분 수정 필요
    if sharing[0] == 3 and pruning == 1: 
        a = sess.run(fetches=[net_dict["Weights_CN"]])[0]
        CN_surv[:,np.where(a == np.min(a[np.nonzero(a)]))] = 0
    elif sharing[0] == 1 and pruning == 1:
        a = np.zeros(M_proto_input, dtype=np.float32)
        for iters in range(0, iters_max, 1):
            a = a + sess.run(fetches=[net_dict["Weights_CN{0}".format(iters)]])[0]
            
    if sharing[2] == 3 and pruning == 1:
        a = sess.run(fetches=[net_dict["Bias_CN"]])[0]
        CN_surv[:,np.where(a == np.min(a[np.nonzero(a)]))] = 0
    elif sharing[2] == 1 and pruning == 1:
        a = np.zeros(M_proto_input, dtype=np.float32)
        for iters in range(0, iters_max, 1):
            a = a + sess.run(fetches=[net_dict["Bias_CN{0}".format(iters)]])[0]
        
        #code refine 필요
        a = np.reshape(a,(np.int64(M_proto_input/pruning_step),pruning_step))
        a = np.sum(a,axis=1)
        min_idx = np.where(a == np.min(a[np.nonzero(a)]))
        min_idx = min_idx[0]
        #min_idx = list(np.arange())
        pruning_node = np.append(pruning_node,CN_org_idx[min_idx[0]*pruning_step:(min_idx[0]+1)*pruning_step])
        pruning_node = np.int64(pruning_node)
        del CN_org_idx[min_idx[0]*pruning_step:(min_idx[0]+1)*pruning_step]
        print('pruning_node: {0}, CN_org_idx: {1}'.format(pruning_node,CN_org_idx))
        #CN_surv[:,min_idx[0]*pruning_step:(min_idx[0]+1)*pruning_step] = 0
        
    if pruning >= 1:
        tf.reset_default_graph()
        sess.close()
