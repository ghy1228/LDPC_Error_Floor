import numpy as np
import math
import shutil
import time

def read_uncor_llr(input_llr,input_codeword,batch_idx,batch_size,code_n,Z):
    X =  -np.reshape(input_llr[batch_idx * batch_size:(batch_idx + 1) * batch_size,...],[batch_size,code_n,Z]) # defined as p1/p0
    Y =  input_codeword[batch_idx * batch_size:(batch_idx + 1) * batch_size,:]
    
    return X, Y

def Cal_MSA_Q(x,q_bit):
    
    if q_bit == 6:
        q_value = np.clip(np.round(x),-15.5,15.5) #(-15.5 -15.0  ...  15.0 15.5) Quantizer  
    elif q_bit == 5:
        q_value = np.clip(np.round(x * 2)/2,-7.5,7.5) #(-7.5 -7.0 -6.5 ... 6.5 7.0 7.5) Quantizer  
    elif q_bit == -5:
        q_value = np.clip(np.round(x),-15,15) #(-15 -14 ... 14 15) Quantizer  
    elif q_bit == 4:
        q_value = np.clip(np.round(x),-7,7) #(-7.0 -6.0 ... 6.0 7.0) Quantizer  
    elif q_bit == 3:
        q_value = np.clip(np.round(x / 2)*2,-6,6) #(-6 -4 -2.0 2 4 6) Quantizer  
    
    return q_value

    
#get train samples
def create_mix_epoch(scaling_factor, wordRandom, noiseRandom, batch_size, code_n, code_k, Z, code_GM, is_zeros_word, decoding_type, punct_start, punct_end, short_start, short_end,q_bit,clip_LLR):

    X = np.zeros([1, code_n * Z], dtype=np.float32)
    Y = np.zeros([1, code_n * Z], dtype=np.int64)
    
    curr_batch_size = 0
    while curr_batch_size < batch_size:
        for sf_i in scaling_factor:
            if is_zeros_word:
                #infoWord_i = 0 * wordRandom.randint(0, 2, size=(1, code_k * Z))
                Y_i = 0 * wordRandom.randint(0, 2, size=(1, code_n * Z))
            else:
                infoWord_i = wordRandom.randint(0, 2, size=(1, code_k * Z))
                Y_i = np.dot(infoWord_i, code_GM) % 2

             # pay attention to this 1->1 0->-1
            X_p_i = noiseRandom.normal(0.0, 1.0, Y_i.shape) * sf_i + (-1) ** (1 - Y_i) 
            x_llr_i = 2 * X_p_i / ((sf_i) ** 2)  # defined as p1/p0

            
            if decoding_type == 2:
                x_llr_i = Cal_MSA_Q(x_llr_i,q_bit)
            
            
            if punct_start > 0:
                if decoding_type == 0:
                    x_llr_i[0,punct_start - 1:punct_end] = 0.001 #For training of Sum-Product decoding
                else:
                    x_llr_i[0,punct_start - 1:punct_end] = 0

            if short_start > 0:
                x_llr_i[0,short_start - 1:short_end] = -clip_LLR
            
            X = np.vstack((X, x_llr_i))
            Y = np.vstack((Y, Y_i))
            curr_batch_size = curr_batch_size + 1
            if curr_batch_size == batch_size:
                break
            
            
    X = X[1:]
    Y = Y[1:]
    X = np.reshape(X, [batch_size, code_n, Z]) # [B,N,Z]
    return X, Y

def print_weight(out_filename, training_iter_end, sharing, fixed_iter, sess, net_dict):
    out_file = open( f"./Weights/{out_filename}_Weight_End{training_iter_end}.txt",'w')
    print("{0} {1} {2}\n".format(*sharing),file = out_file)
    

    #Weights print
    for i, share_type in enumerate(sharing):
        if share_type in [1,2,3]:
            for curr_iter in range(0, training_iter_end, 1):
                a = sess.run(fetches=[net_dict[f"var_{i}_{curr_iter}"]])
                np.savetxt(out_file,a,fmt = '%s', delimiter='	')
            print('',file=out_file)
        elif share_type in [4,5]:
            for curr_iter in range(0, training_iter_end, 1):
                if curr_iter < fixed_iter:
                    a = sess.run(fetches=[net_dict[f"var_{i}_{curr_iter}"]])
                else:
                    temp_iter = fixed_iter
                    a = sess.run(fetches=[net_dict[f"var_{i}_{temp_iter}"]])
                np.savetxt(out_file,a,fmt = '%s', delimiter='	')
            print('',file=out_file)
            
    out_file.close()
    
    

def calc_ber_fer(y_pred_all, iters_max, Y_test,batch_size):
    
    #uncor_flag = np.abs(((Y_test_pred >= 0) - Y_test)).sum(axis=1) > 0
    length = y_pred_all.shape[1]
    uncor_flag = np.empty([0,batch_size])
    for i in range(iters_max):
        uncor_flag_curr_iter = np.abs(((y_pred_all[i*batch_size:(i+1)*batch_size,:] >= 0) - Y_test[:,:length])).sum(axis=1) > 0
        uncor_flag = np.append(uncor_flag,uncor_flag_curr_iter.reshape(1,batch_size) ,axis = 0 )
   
    uncor_flag = np.min(uncor_flag,axis = 0)
    
    fer = (uncor_flag).sum() * 1.0 / batch_size
    error_num = ((y_pred_all[(iters_max - 1)*batch_size:((iters_max - 1)+1)*batch_size,:] >= 0) - Y_test[:,:length]).sum(axis=1)
    ber_last = np.abs(error_num.sum()) / (Y_test.shape[0] * Y_test.shape[1])
    
    uncor_last_flag = np.abs(((y_pred_all[(iters_max - 1)*batch_size:((iters_max - 1)+1)*batch_size,:] >= 0) - Y_test[:,:length])).sum(axis=1) > 0
    fer_last = (uncor_last_flag).sum() * 1.0 / Y_test.shape[0]
    
    return ber_last, fer_last, fer, uncor_flag, error_num

def write_uncor_file(uncor_flag,training_received_data, code_length):
    
    out_file = open("Uncor.txt", 'a')
    num_uncor = np.sum(uncor_flag == 1)
    uncor_received_data = -np.reshape(training_received_data[uncor_flag == 1, :, :],[num_uncor, code_length])
    np.savetxt(out_file, np.concatenate((np.zeros((num_uncor, 3)), uncor_received_data), axis=1), fmt='%.1f', delimiter='\t')
    out_file.close()
        


def compute_results(sample_num, input_llr, input_codeword, SNR_sigma, wordRandom, noiseRandom,  batch_size, sampling_type, N_proto, M_proto, z_value, train_on_zero_word, training_iter_end, sess, net_dict, etha_curr, decoding_type, punct_start, punct_end, short_start, short_end, q_bit,clip_LLR):
    start_time = time.time()  

    Results = np.zeros((4, SNR_sigma.size), dtype=np.float32) #BER, FER_last, FER, loss

    batch_num = math.floor(sample_num / batch_size)
    for batch_idx in range(batch_num):
        for SNR_idx in range(SNR_sigma.size):
            SNR_point = np.array([SNR_sigma[SNR_idx]])
            if sampling_type == 0 or sampling_type == 2:
                training_received_data, training_coded_bits = create_mix_epoch(SNR_point, wordRandom, noiseRandom, batch_size,
                                                                       N_proto, N_proto - M_proto, z_value,
                                                                       [],
                                                                       train_on_zero_word, decoding_type, punct_start, punct_end, short_start, short_end,q_bit,clip_LLR)
            elif sampling_type == 1:
                training_received_data, training_coded_bits = read_uncor_llr(input_llr,input_codeword,batch_idx,batch_size,N_proto,z_value)
            
            if sampling_type == 2:
                y_pred_all = sess.run(fetches=net_dict["ya_output_all"], feed_dict={net_dict['xa']: training_received_data, net_dict['ya']: training_coded_bits, net_dict['etha']: etha_curr, net_dict['learn_rate']: 0})
                loss_batch = 0
            else:
                y_pred_all, loss_batch = sess.run(fetches=[net_dict["ya_output_all"], net_dict["lossa"]], feed_dict={net_dict['xa']: training_received_data, net_dict['ya']: training_coded_bits, net_dict['etha']: etha_curr, net_dict['learn_rate']: 0})
                
            
            ber_last_batch, fer_last_batch, fer_batch, uncor_flag, error_num = calc_ber_fer(y_pred_all, training_iter_end, training_coded_bits, batch_size)
            if sampling_type == 2 and np.sum(uncor_flag == 1) > 0:
                write_uncor_file(uncor_flag,training_received_data, N_proto * z_value)

            Results[0, SNR_idx] += ber_last_batch / batch_num
            Results[1, SNR_idx] += fer_last_batch / batch_num
            Results[2, SNR_idx] += fer_batch / batch_num
            Results[3, SNR_idx] += loss_batch / batch_num

    time_took = time.time() - start_time

    return Results,time_took
    
def compute_opt_value(opt_value, opt_result_print, ber_last_SNR, fer_last_SNR, fer_SNR, loss_SNR):
    opt_print_flag = False
    if opt_result_print == 0 and opt_value > np.sum(ber_last_SNR):
        opt_value = np.sum(ber_last_SNR)
        opt_print_flag = True
    elif opt_result_print == 1 and opt_value > np.sum(fer_last_SNR):
        opt_value = np.sum(fer_last_SNR)
        opt_print_flag = True
    elif opt_result_print == 2 and opt_value > np.sum(fer_SNR):
        opt_value = np.sum(fer_SNR)
        opt_print_flag = True
    elif opt_result_print == 3 and opt_value > np.sum(loss_SNR):
        opt_value = np.sum(loss_SNR)
        opt_print_flag = True
    return opt_value, opt_print_flag



def print_result(Results, opt_value, Perf_filename, out_filename, training_iter_end, opt_result_print, opt_print_flag,test_time):

    with open(Perf_filename,'a') as out_file:

        if test_time == False:
            opt_value, opt_print_flag = compute_opt_value(opt_value, opt_result_print, Results[0,:], Results[1,:], Results[2,:], Results[3,:])

            print(f'Valid_Result\nBER_last: {FTE(Results[0,:])}\nFER_last: {FTE(Results[1,:])}\nFER: {FTE(Results[2,:])}\nloss: {FTE(Results[3,:])}')
            print(f"opt_value: {FTE([opt_value])}\n")

            print(f'Valid_Result\nBER_last: {FTE(Results[0,:])}\nFER_last: {FTE(Results[1,:])}\nFER: {FTE(Results[2,:])}\nloss: {FTE(Results[3,:])}',file=out_file)
            print(f"opt_value: {FTE([opt_value])}\n",file=out_file)

            if opt_print_flag == True:
                shutil.copyfile(f"./Weights/{out_filename}_Weight_End{training_iter_end}.txt",f"./Weights/{out_filename}_Opt_Weight_End{training_iter_end}.txt")
        else:
            if opt_print_flag == True:
                opt_value, _ = compute_opt_value(100000, opt_result_print, Results[0,:], Results[1,:], Results[2,:], Results[3,:])
            else:
                opt_value = opt_value
            print(f'Test_Result\nBER_last: {FTE(Results[0,:])}\nFER_last: {FTE(Results[1,:])}\nFER: {FTE(Results[2,:])}\nloss: {FTE(Results[3,:])}')
            print(f"opt_value: {FTE([opt_value])}\n")

            print(f'Test_Result\nBER_last: {FTE(Results[0,:])}\nFER_last: {FTE(Results[1,:])}\nFER: {FTE(Results[2,:])}\nloss: {FTE(Results[3,:])}',file=out_file)
            print(f"opt_value: {FTE([opt_value])}\n",file=out_file)


        
    
    return opt_value,opt_print_flag


def print_train_result(curr_epoch, Perf_filename, training_iter_start, training_iter_end, epoch_input, avg_train_loss):

        
    with open(Perf_filename,'a') as out_file:
        print(f'* Training_iter_start: {training_iter_start} training_iter_end: {training_iter_end} epoch: [{curr_epoch}/{epoch_input}]')
        print(f"Training loss: {FTE([avg_train_loss])}")
        
        print(f'* Training_iter_start: {training_iter_start} training_iter_end: {training_iter_end} epoch: [{curr_epoch}/{epoch_input}]', file=out_file)
        print(f"Training loss: {FTE([avg_train_loss])}",file=out_file)

def FTE(arr, precision=2): #format to exponential
    return [f"{val:.{precision}e}" for val in arr]

