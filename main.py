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

###############################Input###############################################

#filename = "802_11n_N648_R56_z27"
filename = "wman_N0576_R34_z24"
#filename = "5G_rate_half_length552_z12"

q_bit = 5 #Quantization bit 

#0: No weights // 1: edge,iter // 2: node, iter // 3: iter // 4: edge (RNN) // 5: node (RNN)
sharing = [3,3,3] #CN_Weight,UCN_Weight,VN_Weight
sampling_type = 1 #0: Default, 1: Read_Uncor, 2: Collect_Uncor (Slow => recommended to use C code)
decoding_type = 2 #0:SP, 1:MS,  2: QMS
z_value = 24 #802:27, wman: 24, 5G: 12
punct_start = 0
punct_end = 0
iters_max = 50 
fixed_iter = 20 
fixed_init = 10 #Delta_2
iter_step = 5 #Delta_1
loss_type = 2 #0: BCE, 1: Soft BER , 2: FER
etha_value = 0 #1: Equal (Multi-loss), 0: Last iter
learn_rate_value = 0.001

batch_size = 20
training_num = 50000
epoch_input = 30
valid_flag = 1 #0: No validation set, 1: with validation set
valid_num = 5000 
test_flag = 1  #0: No test set, 1: with test set
test_num = 5000

init_weight = 1
init_VN_weight = 1
Max_weight = 2
Min_weight = 0
seed_in = 1
SNR_Matrix = np.array([3.0])

###############################Initial setting###############################################

code_Proto = np.loadtxt("./BaseGraph/{0}.txt".format(filename), int, delimiter='\t')
out_filename = f"C{core_idx}_{filename}"
clip_tanh = 10.0
clip_LLR = 20.0
word_seed = 2042 + seed_in
noise_seed = 1074 + seed_in
wordRandom = np.random.RandomState(word_seed)  # word seed
noiseRandom = np.random.RandomState(noise_seed)  # noise seed
train_on_zero_word = 1

SNR_Matrix = Main_Functions.check_params(sampling_type, SNR_Matrix, sharing, iters_max, fixed_iter, iter_step)

input_llr_training, input_codeword_training, input_llr_valid, input_codeword_valid, input_llr_test, input_codeword_test = Main_Functions.process_data(sampling_type, filename, training_num, valid_flag, valid_num, test_flag, test_num)

M_proto, N_proto, code_Base, CN_deg_proto, VN_deg_proto, Num_edge_proto, code_rate, SNR_sigma = Main_Functions.init_parameter(code_Proto, SNR_Matrix,z_value, punct_start, punct_end)


###############################Print_infomation###############################################

Perf_filename = f'./Weights/{out_filename}_Performance.txt'
with open(Perf_filename, 'w') as out_file:
    print(f'CN_weight_sharing = {sharing[0]} UCW_weight_sharing = {sharing[1]} VN_weight_sharing = {sharing[2]}', file=out_file)
    print(f'Init_CN_weight = {init_weight} Max_weight = {Max_weight} Min_weight = {Min_weight} Init_VN_weight = {init_VN_weight}', file=out_file)
    print(f'samping_type = {sampling_type}, punct_start = {punct_start} punct_end = {punct_end}', file=out_file)
    print(f'z_value = {z_value} iters_max = {iters_max} fixed_iter = {fixed_iter} fixed_init = {fixed_init} iter_step = {iter_step}', file=out_file)
    print(f'loss_type = {loss_type} learn_rate = {learn_rate_value} etha = {etha_value}', file=out_file)
    print(f'batch_size = {batch_size} epochs = {epoch_input} training_num = {training_num} valid_flag = {valid_flag} valid_num = {valid_num} test_flag = {test_flag} test_num = {test_num}', file=out_file)
    print(f'SNR_Matrix = {SNR_Matrix}', file=out_file)
    print(f'M_proto = {M_proto} N_proto = {N_proto} Num_edge_proto = {Num_edge_proto} code_rate = {code_rate}',file=out_file)
    print('', file=out_file)


##################################  init the learnable network parameters  ####################################
Lift_Matrix1,Lift_Matrix2,W_odd2even,W_skipconn2even,W_even2odd,W_output,W_skipconn2odd,W_even2odd_with_self = Main_Functions.init_connecting_matrix(code_Proto,code_Base,N_proto,M_proto,Num_edge_proto,z_value,VN_deg_proto,CN_deg_proto)

training_iter_start = fixed_iter
training_iter_end = fixed_iter + iter_step
while (training_iter_end <= iters_max):
    opt_valid = 100000
    opt_test = 100000
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
    net_dict = Main_Functions.weight_init(net_dict, out_filename, sharing, Num_edge_proto, M_proto, N_proto, Min_weight, Max_weight, init_weight, init_VN_weight, training_iter_start, training_iter_end, fixed_iter)


    ##################################  Build Network  ####################################

    
    for i in range(0, training_iter_end, 1):
        net_dict = Main_Functions.build_neural_network(net_dict,sharing,decoding_type,sampling_type,loss_type,i,iters_max,fixed_iter,fixed_init,training_iter_start,training_iter_end,N_proto,M_proto,Num_edge_proto,z_value,batch_size,Lift_Matrix1,Lift_Matrix2,W_odd2even,W_skipconn2even,W_even2odd,W_output,W_skipconn2odd,W_even2odd_with_self,q_bit,clip_LLR,clip_tanh)
    
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver() 
    
    training_batch_num = math.floor(training_num/batch_size)
    for curr_epoch in range(0,epoch_input + 1,1):
        
        ##################################  Print  ####################################
        if valid_flag > 0:
            print(f'training_iter_start: {training_iter_start} training_iter_end: {training_iter_end} epoch: [{curr_epoch}/{epoch_input}]')
            Print_Functions.print_weight(out_filename, training_iter_end, sharing, fixed_iter, sess, net_dict)
            opt_valid,opt_print_flag = Print_Functions.print_result (input_llr_valid, input_codeword_valid, curr_epoch, opt_valid, Perf_filename, out_filename, sharing, sampling_type, decoding_type, punct_start, punct_end, training_iter_start, training_iter_end, valid_num, batch_size, SNR_sigma,  wordRandom, noiseRandom,z_value, q_bit, N_proto, M_proto, train_on_zero_word, etha_value, sess, net_dict , epoch_input, opt_print_flag, False)

            if sampling_type == 1 and test_num > 0 and test_flag == 1:
                opt_test,_ = Print_Functions.print_result(input_llr_test, input_codeword_test, curr_epoch, opt_test, Perf_filename, out_filename, sharing, sampling_type, decoding_type, punct_start, punct_end, training_iter_start, training_iter_end, test_num, batch_size, SNR_sigma,  wordRandom, noiseRandom, z_value,q_bit, N_proto, M_proto, train_on_zero_word, etha_value, sess, net_dict , epoch_input, opt_print_flag, True)
            
        ######################################################################
        
        
        
        start_time = time.time()
        ##################################  Training  ####################################
        for batch_idx in range(0, training_batch_num, 1):
            if sampling_type == 0:
                training_received_data, training_coded_bits = Print_Functions.create_mix_epoch(SNR_sigma, wordRandom, noiseRandom, batch_size,
                                                                           N_proto, N_proto - M_proto, z_value, [], train_on_zero_word,decoding_type, punct_start, punct_end,q_bit)
            elif sampling_type == 1:
                training_received_data, training_coded_bits = Print_Functions.read_uncor_llr(input_llr_training,input_codeword_training,batch_idx,batch_size,N_proto,z_value)

            
            if sampling_type != 2:
                y_pred, train_loss, _ = sess.run(fetches=[net_dict["ya_output{0}".format(training_iter_end - 1)], net_dict["lossa"],
                                                          net_dict["train_stepa"]],
                                                 feed_dict={net_dict['xa']: training_received_data, net_dict['ya']: training_coded_bits, net_dict['etha']: etha_value,net_dict['learn_rate']: learn_rate_value})


                
        ######################################################################
        end_time = time.time()
        elapsed_time = end_time - start_time  # 걸린 시간 계산
        print(f'Epoch {curr_epoch}/{epoch_input} took {elapsed_time:.2f} seconds\n')
        trainable_variables = tf.global_variables()
        total_parameters = sum([sess.run(tf.reduce_prod(v.shape)) for v in trainable_variables])
        print(f"Total number of parameters: {total_parameters}")
            



        
    #매 while loop
    training_iter_start = training_iter_start + iter_step
    training_iter_end = training_iter_end + iter_step

