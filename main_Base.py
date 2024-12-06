from __future__ import print_function, division
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import datetime
import math
import time
import pdb
import Main_Functions
import Print_Functions
import time

core_idx = 0
os.environ["CUDA_VISIBLE_DEVICES"] = "{0}".format(core_idx)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#sess = tf.Session(config=config)

###############################Input###############################################

filename = "wman_N0576_R34_z24"

#0: No weights // 1: edge,iter // 2: node, iter // 3: iter // 4: edge (Tempolar sharing)  // 5: Node (Tempolar sharing)
sharing     = [3,  0,  3] #CN_Weight,UCN_Weight, VN_Weight
sampling_type = 0 #0: Default, 1: Read_Uncor, 2: Collect_Uncor
decoding_type = 2 #0:SP, 1:MS,  2: QMS
q_bit = 5 #for QMS
systematic = 0 #0: non-systematic, 1: systematic (Only Info bit)
z_value = 24 #802:27, wman: 24
punct_start = 0
punct_end = 0
short_start = 0
short_end = 0
iters_max = 20
fixed_iter = 0
fixed_init = 0 #Delta_2
iter_step = 20 #Delta_1
loss_type = 2 #0: BCE, 1: Soft BER , 2: FER
opt_result_print = 1 #0: BER_last, 1: FER_last, 2:FER, 3: Loss
etha_start = 0 #1: Equal (Multi-loss), 0: Last iter
etha_discount = 0
etha_discount_step = 0
learn_rate_start = 0.001
learn_rate_discount = 0
learn_rate_step = 0
 
 
batch_size = 20
training_num = 10000
epoch_input = 200
valid_flag = 1 #0: No validation set, 1: with validation set
valid_num = 10000 #
test_flag = 0 #0: No test set, 1: with test set
test_num = 400
 
init_from_file = 0 #Transfer Weights
init_weight = 1 
init_VN_weight = 1
Max_weight = 2
Min_weight = 0
seed_in = 2
SNR_Matrix = np.array([2, 2.5, 3.0, 3.5, 4.0])

###############################Initial setting###############################################

code_Proto = np.loadtxt("./BaseGraph/{0}.txt".format(filename), int, delimiter='\t')
out_filename = f"C{core_idx}_{filename}"
clip_LLR = 20.0
train_on_zero_word = True
word_seed = 2042 + seed_in
noise_seed = 1074 + seed_in
wordRandom = np.random.RandomState(word_seed)  # word seed
noiseRandom = np.random.RandomState(noise_seed)  # noise seed


SNR_Matrix = Main_Functions.check_params(sampling_type, SNR_Matrix, sharing, iters_max, fixed_iter, iter_step)

input_llr_training, input_codeword_training, input_llr_valid, input_codeword_valid, input_llr_test, input_codeword_test = Main_Functions.process_data(sampling_type, filename, training_num, valid_flag, valid_num, test_flag, test_num)

M_proto, N_proto, code_Base, CN_deg_proto, VN_deg_proto, Num_edge_proto, code_rate, SNR_sigma = Main_Functions.init_parameter(code_Proto, SNR_Matrix,z_value, punct_start, punct_end,short_start,short_end)

if systematic == 1:
    target_node = N_proto - M_proto
else:
    target_node = N_proto

###############################Print_infomation###############################################

Perf_filename = f'./Weights/{out_filename}_Performance.txt'
with open(Perf_filename, 'w') as out_file:
    print(f"Decoding_type = {decoding_type} q_bit = {q_bit}", file=out_file)
    print(f'CN_weight_sharing = {sharing[0]} UCW_weight_sharing = {sharing[1]} VN_weight_sharing = {sharing[2]}', file=out_file)
    print(f'Init_CN_weight = {init_weight} Max_weight = {Max_weight} Min_weight = {Min_weight} Init_VN_weight = {init_VN_weight}, init_from_file = {init_from_file}', file=out_file)
    print(f'samping_type = {sampling_type} systematic = {systematic}', file=out_file)
    print(f'z_value = {z_value} iters_max = {iters_max} fixed_iter = {fixed_iter} fixed_init = {fixed_init} iter_step = {iter_step}', file=out_file)
    print(f'puncturing = {punct_start} ~ {punct_end}, shortening = {short_start} ~ {short_end}',file=out_file)
    print(f'etha_start = {etha_start} etha_discount = {etha_discount} etha_discount_step = {etha_discount_step}', file=out_file)
    print(f'loss_type = {loss_type} learn_rate_start = {learn_rate_start} learn_rate_discount = {learn_rate_discount} learn_rate_step = {learn_rate_step}', file=out_file)
    print(f'batch_size = {batch_size} epochs = {epoch_input} training_num = {training_num} valid_flag = {valid_flag} valid_num = {valid_num} test_flag = {test_flag} test_num = {test_num}', file=out_file)
    print(f'SNR_Matrix = {SNR_Matrix}', file=out_file)
    print(f'M_proto = {M_proto} N_proto = {N_proto} Num_edge_proto = {Num_edge_proto} code_rate = {code_rate}',file=out_file)
    print('', file=out_file)


##################################  init the learnable network parameters  ####################################
Lift_Matrix1,Lift_Matrix2,W_odd2even,W_skipconn2even,W_even2odd,W_output,W_skipconn2odd,W_even2odd_with_self = Main_Functions.init_connecting_matrix(code_Proto,code_Base,N_proto,M_proto,Num_edge_proto,z_value,VN_deg_proto,CN_deg_proto,punct_start, punct_end)
training_iter_start = fixed_iter
training_iter_end = fixed_iter + iter_step
while (training_iter_end <= iters_max):
    opt_valid,opt_test = 100000,100000
    time_train,time_valid,time_test = 0,0,0
    opt_print_flag = False
    print(f'\nTraining_iter_start: {training_iter_start} training_iter_end: {training_iter_end} epoch: [0/{epoch_input}]')

    ##################################  Init Network  ####################################
    tf.reset_default_graph()
    sess = tf.Session(config=config)


    net_dict = {}
    net_dict['etha'] = tf.placeholder(tf.float32, name='etha')
    net_dict['learn_rate'] = tf.placeholder(tf.float32, name='learn_rate')
    net_dict['xa'] = tf.placeholder(tf.float32, shape=[batch_size, N_proto, z_value], name='xa')
    net_dict['ya'] = tf.placeholder(tf.float32, shape=[batch_size, N_proto * z_value], name='ya')
    net_dict["LLRa0"] = tf.zeros((batch_size, z_value, Num_edge_proto), dtype=tf.float32)
    net_dict["infoM_lastlayera0"] = tf.zeros((batch_size,z_value,Num_edge_proto), dtype=tf.float32)
    net_dict = Main_Functions.weight_init(net_dict, init_from_file, out_filename, iters_max, training_iter_start, sharing, Num_edge_proto, M_proto, N_proto, Min_weight, Max_weight, init_weight, init_VN_weight, training_iter_end, fixed_iter)


    ##################################  Build Network  ####################################

    
    etha_curr = etha_start
    learn_rate_curr = learn_rate_start
    for i in range(0, training_iter_end, 1):
        net_dict = Main_Functions.build_neural_network(net_dict,sharing,decoding_type,sampling_type,loss_type,target_node,i,iters_max,fixed_iter,fixed_init,training_iter_start,training_iter_end,N_proto,M_proto,Num_edge_proto,z_value,batch_size,Lift_Matrix1,Lift_Matrix2,W_odd2even,W_skipconn2even,W_even2odd,W_output,W_skipconn2odd,W_even2odd_with_self,q_bit,clip_LLR)    

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver() 
    
    training_batch_num = math.floor(training_num/batch_size)
    for curr_epoch in range(0,epoch_input + 1,1):
        

        
        ##################################  Training  ####################################
        start_time = time.time()
        avg_train_loss = 0
        if curr_epoch > 0:
            for batch_idx in range(0, training_batch_num, 1):
                if sampling_type == 0:
                    training_received_data, training_coded_bits = Print_Functions.create_mix_epoch(SNR_sigma, wordRandom, noiseRandom, batch_size,
                                                                            N_proto, N_proto - M_proto, z_value, [], train_on_zero_word,decoding_type, punct_start, punct_end,short_start,short_end,q_bit,clip_LLR)
                elif sampling_type == 1:
                    training_received_data, training_coded_bits = Print_Functions.read_uncor_llr(input_llr_training,input_codeword_training,batch_idx,batch_size,N_proto,z_value)

                
                if sampling_type != 2:
                    y_pred, train_loss, _ = sess.run(fetches=[net_dict["ya_output{0}".format(training_iter_end - 1)], net_dict["lossa"],
                                                            net_dict["train_stepa"]],
                                                    feed_dict={net_dict['xa']: training_received_data, net_dict['ya']: training_coded_bits, net_dict['etha']: etha_curr,net_dict['learn_rate']: learn_rate_curr})
                    
                    avg_train_loss += train_loss/training_batch_num


        time_train = time.time() - start_time  # 걸린 시간 계산
        ######################################################################
        

        ##################################  Print  ####################################
        
        Print_Functions.print_weight(out_filename, training_iter_end, sharing, fixed_iter, sess, net_dict)
        Print_Functions.print_train_result(curr_epoch, Perf_filename, training_iter_start, training_iter_end, epoch_input, avg_train_loss)
        
        if valid_flag > 0:
            Results,time_valid = Print_Functions.compute_results(valid_num, input_llr_valid, input_codeword_valid, SNR_sigma, wordRandom, noiseRandom,  batch_size, sampling_type, N_proto, M_proto, z_value, train_on_zero_word, training_iter_end, sess, net_dict, etha_curr, decoding_type, punct_start, punct_end, short_start, short_end, q_bit,clip_LLR)
            opt_valid,opt_print_flag = Print_Functions.print_result (Results, opt_valid, Perf_filename, out_filename, training_iter_end, opt_result_print, opt_print_flag,False)
        

        if sampling_type == 1 and test_num > 0 and test_flag == 1:
            Results,time_test = Print_Functions.compute_results(test_num, input_llr_test, input_codeword_test, SNR_sigma, wordRandom, noiseRandom,  batch_size, sampling_type, N_proto, M_proto, z_value, train_on_zero_word, training_iter_end, sess, net_dict, etha_curr, decoding_type, punct_start, punct_end, short_start, short_end, q_bit,clip_LLR)
            opt_test,_ = Print_Functions.print_result (Results, opt_test, Perf_filename, out_filename, training_iter_end, opt_result_print, opt_print_flag,True)

        with open(Perf_filename,'a') as out_file:
            print(f'Running time (Train/Valid/Test): {time_train:.2f}/{time_valid:.2f}/{time_test:.2f}\n')
            print(f'Running time (Train/Valid/Test): {time_train:.2f}/{time_valid:.2f}/{time_test:.2f}\n',file=out_file)
        ######################################################################
            
            
        #Discount
        if etha_discount !=0 and etha_discount_step != 0 and (curr_epoch + 1)%etha_discount_step == 0:
            etha_curr = etha_curr * etha_discount
        if learn_rate_discount != 0 and learn_rate_step != 0 and (curr_epoch + 1)%learn_rate_step == 0:
            learn_rate_curr = learn_rate_curr * learn_rate_discount



        
    #while loop
    training_iter_start = training_iter_start + iter_step
    training_iter_end = training_iter_end + iter_step

