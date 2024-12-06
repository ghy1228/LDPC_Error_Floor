import numpy as np
import math
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import sys
import pdb

def init_parameter(code_Proto, SNR_Matrix,z_value,punct_start,punct_end,short_start,short_end):
    M_proto, N_proto = code_Proto.shape
    code_Base = code_Proto.copy()
    for i in range(0, code_Proto.shape[0]):
        for j in range(0, code_Proto.shape[1]):
            if (code_Proto[i, j] == -1):
                code_Base[i, j] = 0
            else:
                code_Base[i, j] = 1
                
    CN_deg_proto = np.sum(code_Base, axis=1)
    VN_deg_proto = np.sum(code_Base, axis=0)
   
    
    Num_edge_proto = np.sum(VN_deg_proto)

    punct_num = punct_end - punct_start + 1
    short_num = short_end - short_start + 1

    n = N_proto * z_value - punct_num - short_num
    k = (N_proto - M_proto) * z_value - short_num
    code_rate = 1.0 * k/n

        
    
        
    # train SNR
    SNR_lin = 10.0 ** (SNR_Matrix / 10.0)
    SNR_sigma = np.sqrt(1.0 / (2.0 * SNR_lin * code_rate))

    return M_proto, N_proto, code_Base, CN_deg_proto, VN_deg_proto, Num_edge_proto, code_rate, SNR_sigma

    



############################     init the connecting matrix between network layers   #################################

def init_connecting_matrix(code_Proto,code_Base,N_proto,M_proto,Num_edge_proto,z_value,VN_deg_proto,CN_deg_proto,punct_start, punct_end):
    Lift_Matrix1 = []
    Lift_Matrix2 = []
    W_odd2even = np.zeros((Num_edge_proto, Num_edge_proto), dtype=np.float32)
    W_skipconn2even = np.zeros((N_proto, Num_edge_proto), dtype=np.float32)
    W_even2odd = np.zeros((Num_edge_proto, Num_edge_proto), dtype=np.float32)
    W_even2odd_with_self = np.zeros((Num_edge_proto, Num_edge_proto), dtype=np.float32)
    W_output = np.zeros((Num_edge_proto, N_proto), dtype=np.float32)
    W_skipconn2odd = np.zeros((M_proto,Num_edge_proto),dtype=np.float32)

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
    Lift_Matrix1.append(Lift_M1) #To convert #E(V)Z (VN output) into the permutation-applied E(V)Z (CN input), you need to multiply it by `Lift_Matrix1.T`.
    Lift_Matrix2.append(Lift_M2) #To convert #E(C)Z (CN output) into the permutation-applied E(C)Z (VN input), you need to multiply it by `Lift_Matrix2`.



    # init W_odd2even  variable node updating
    k = 0
    vec_tmp = np.zeros((Num_edge_proto), dtype=np.float32)  
    for j in range(0, code_Base.shape[1], 1): 
        for i in range(0, code_Base.shape[0], 1): 
            if (code_Base[i, j] == 1):  
                num_of_conn = int(np.sum(code_Base[:, j])) 
                idx = np.argwhere(code_Base[:, j] == 1)
                for l in range(0, num_of_conn, 1):
                    vec_tmp = np.zeros((Num_edge_proto), dtype=np.float32)
                    for r in range(0, code_Base.shape[0], 1):
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
                W_output[odd_layer_node_count + idx_row, k] = 1.0  #[E(C),N]
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
                W_skipconn2odd[i, k] = 1.0  #[M,E(C)]
                k += 1            
                
                
    return Lift_Matrix1,Lift_Matrix2,W_odd2even,W_skipconn2even,W_even2odd,W_output,W_skipconn2odd,W_even2odd_with_self

    

                
##############################  bulid neural network ############################

def build_neural_network(net_dict,sharing,decoding_type,sampling_type,loss_type,target_node,curr_iter,iters_max,fixed_iter,fixed_init,training_iter_start,training_iter_end,N_proto,M_proto,Num_edge_proto,z_value,batch_size,Lift_Matrix1,Lift_Matrix2,W_odd2even,W_skipconn2even,W_even2odd,W_output,W_skipconn2odd,W_even2odd_with_self,q_bit,clip_LLR):
    
#Starting with VN operations, which is different from the standard decoding method that begins with CN operations.
    
    xa = net_dict['xa']
    ya = net_dict['ya']
    
    xa_input = tf.transpose(xa, [0, 2, 1]) # [B,N,Z] -> [B,Z,N]
    
    
    #if curr_iter > 0:
    if sharing[2] == 2 or sharing[2] == 3:
        xa_input = tf.multiply(xa_input,net_dict[f'var_{2}_{curr_iter}'])
    elif sharing[2] == 4:
        if curr_iter < fixed_iter:
            xa_input = tf.multiply(xa_input,net_dict[f'var_{2}_{curr_iter}'])
        else:
            xa_input = tf.multiply(xa_input,net_dict[f'var_{2}_{fixed_iter}'])
        
    if decoding_type == 2:
        xa_input = Cal_MSA_Q_TF(xa_input,q_bit)

        
    if sharing[1]>0:
        if curr_iter == 0:
            VN_APP = xa_input  #[B,Z,N]
        else:
            VN_APP = tf.reshape(net_dict["ya_output{0}".format(curr_iter - 1)],[batch_size,N_proto, z_value]) #[B,N,Z]
            VN_APP = tf.transpose(VN_APP,[0,2,1]) #[B,Z,N]
            
        VN_APP = -VN_APP #[B,Z,N] Note that (APP>0 <-> Dec=0)
        VN_APP_sign = tf.add(tf.to_float(VN_APP > 0),-tf.to_float(VN_APP <= 0)) #[B,Z,N]
        VN_APP_sign_edge = tf.matmul(VN_APP_sign, W_skipconn2even)  #[B,Z,E(V)] 
        VN_APP_sign_edge = tf.transpose(VN_APP_sign_edge, [0, 2, 1]) #[B,E(V),Z]
        VN_APP_sign_edge = tf.reshape(VN_APP_sign_edge, [batch_size, Num_edge_proto * z_value]) #[B,E(V)Z] (e1,1)(e1,2),...,(e1,z),(e2,1),...
        CN_in_sign = tf.matmul(VN_APP_sign_edge, Lift_Matrix1[0].transpose()) #[B,E(V)Z] CN input
        CN_in_sign = tf.reshape(CN_in_sign, [batch_size, Num_edge_proto, z_value]) #[B,E(V),Z] CN input
        CN_in_sign = tf.transpose(CN_in_sign, [0, 2, 1]) #[B,Z,E(V)], CN input
        CN_in_sign_tile = tf.tile(CN_in_sign, multiples=[1, 1, Num_edge_proto]) #[B,Z,E(V),E(V)]
        CN_in_sign_tile = tf.multiply(CN_in_sign_tile, tf.reshape(W_even2odd_with_self.transpose(), [-1]))
        CN_in_sign_tile = tf.reshape(CN_in_sign_tile, [batch_size, z_value, Num_edge_proto, Num_edge_proto])
        CN_in_sign_tile = tf.add(CN_in_sign_tile, 1.0 * (1 - tf.to_float(tf.abs(CN_in_sign_tile) > 0)))  
        CN_sign_edge = tf.reduce_prod(CN_in_sign_tile,axis=3)
        UCN_idx_edge = tf.to_float(CN_sign_edge < 0) #[B,Z,E(C)]        
        UCN_idx_edge = tf.transpose(UCN_idx_edge, [0, 2, 1])  #[B,Z,E(C)] -> [B,E(C),Z]
        UCN_idx_edge = tf.reshape(UCN_idx_edge, [batch_size, z_value * Num_edge_proto]) #[B,E(C)Z]
        UCN_idx_edge = tf.matmul(UCN_idx_edge, Lift_Matrix2[0]) #[B,E(C)Z] VN input
        UCN_idx_edge = tf.reshape(UCN_idx_edge, [batch_size, Num_edge_proto, z_value]) #[B,E(C),Z]
        UCN_idx_edge = tf.transpose(UCN_idx_edge, [0, 2, 1]) #[B,Z,E(C)], VN input
        SCN_idx_edge = tf.add(-UCN_idx_edge,tf.ones((batch_size,z_value,Num_edge_proto), dtype=tf.float32))
    else:
        UCN_idx_edge = tf.zeros((batch_size, z_value, Num_edge_proto)) #[B,Z,E(C)]
        SCN_idx_edge = tf.add(-UCN_idx_edge,tf.ones((batch_size,z_value,Num_edge_proto), dtype=tf.float32))
    
    
    #variable node update
    x0 = tf.matmul(xa_input, W_skipconn2even)  #[B,Z,N] X [N,E(V)] = [B,Z,E(V)]
    x1 = tf.matmul(net_dict["LLRa{0}".format(curr_iter)], W_odd2even) #[B,Z,E(C)] X [E(C),E(V)] = [B,Z,E(V)] V->C
    x2 = tf.add(x0, x1)# x2 [B,Z,E(V)] V->C

    x2 = tf.transpose(x2, [0, 2, 1]) #[B,E(V),Z]
    x2 = tf.reshape(x2, [batch_size, Num_edge_proto * z_value]) #[B,E(V)Z] (e1,1)(e1,2),...,(e1,z),(e2,1),...
    x2 = tf.matmul(x2, Lift_Matrix1[0].transpose()) #x2 [B,E(V)Z] --> x2 [B,E(V)Z]
    x2 = tf.reshape(x2, [batch_size, Num_edge_proto, z_value]) #[B,E(V),Z]
    x2 = tf.transpose(x2, [0, 2, 1]) #[B,Z,E(V)]
    
    if decoding_type == 2:
        x2 = Cal_MSA_Q_TF(x2,q_bit)
    else:
        x2 = tf.clip_by_value(x2, clip_value_min=-clip_LLR, clip_value_max=clip_LLR)


    if decoding_type == 1 or decoding_type == 2:
        x2 = tf.add(x2, 0.0001 * (1 - tf.to_float(tf.abs(x2) > 0)))  # 0 -> 0.0001 temporaly
    x_tile = tf.tile(x2, multiples=[1, 1, Num_edge_proto]) #[B,Z,E(V)E(V)]
    W_input_reshape = tf.reshape(W_even2odd.transpose(), [-1]) 
    
    #check node update
    x_tile_mul = tf.multiply(x_tile, W_input_reshape) 
    x2_1 = tf.reshape(x_tile_mul, [batch_size, z_value, Num_edge_proto, Num_edge_proto]) 

    if decoding_type == 0:
        x2_tanh = tf.tanh(-0.5 * x2_1) 
        x2_abs = tf.add(x2_tanh, 1 - tf.to_float(tf.abs(x2_tanh) > 0))  #If `Punctured_LLR = 0`, there’s a possibility of malfunction because it becomes hard to distinguish between an initialized value of 0 and a punctured LLR. To address this, we set `punctured_LLR` to `-0.001` for training.
        x3 = tf.reduce_prod(x2_abs, reduction_indices=3)

        epsilon = 1e-7
        x3_clipped = tf.clip_by_value(x3, clip_value_min=-1 + epsilon, clip_value_max=1 - epsilon)
        x_output_0 = -2 * tf.atanh(x3_clipped)

    elif decoding_type == 1 or decoding_type == 2 or decoding_type == 3:
        x2_abs = tf.add(tf.abs(x2_1), 10000 * (1 - tf.to_float(tf.abs(x2_1) > 0))) # Make sure that the zeros resulting from multiplying with W_input_reshape do not become the minimum value.
        x3 = tf.reduce_min(x2_abs, axis=3)
        x3 = tf.add(x3, -0.0001 * (1 - tf.to_float(tf.abs(x3) > 0.0001))) # 0.0001 -> 0 
        x2_2 = -x2_1 
        x4 = tf.add(tf.zeros((batch_size, z_value, Num_edge_proto, Num_edge_proto)), 1 - 2 * tf.to_float(x2_2 < 0))
        x4_prod = -tf.reduce_prod(x4, axis=3)
        x_output_0 = tf.multiply(x3, tf.sign(x4_prod))
        

        
                
    x_output_0 = tf.transpose(x_output_0, [0, 2, 1])  #[B,Z,E(C)] -> [B,E(C),Z]
    x_output_0 = tf.reshape(x_output_0, [batch_size, z_value * Num_edge_proto])
    x_output_0 = tf.matmul(x_output_0, Lift_Matrix2[0]) #[B,Z,E(C)]
    x_output_0 = tf.reshape(x_output_0, [batch_size, Num_edge_proto, z_value]) #[B,E(C),Z]
    x_output_0 = tf.transpose(x_output_0, [0, 2, 1]) #[B,Z,E(C)]
    

    # revised by khy 22.01.01
    if sharing[0] == 0:             
        x_output_1 = tf.abs(x_output_0)
    elif sharing[0] == 1:   
        if sharing[1] == 1:
            W_per_edge_1 = net_dict[f'var_{0}_{curr_iter}'] #[B,Z,E(C)]
            W_per_edge_2 = net_dict[f'var_{1}_{curr_iter}'] #[B,Z,E(C)]
            x_output_11 = tf.multiply(tf.abs(x_output_0),W_per_edge_1) #[B,Z,E]
            x_output_12 = tf.multiply(tf.abs(x_output_0),W_per_edge_2) #[B,Z,E]
            x_output_1 = tf.add(tf.multiply(x_output_11, SCN_idx_edge),tf.multiply(x_output_12,UCN_idx_edge))
        else:            
            W_per_edge = net_dict[f'var_{0}_{curr_iter}'] #[B,Z,E(C)]
            x_output_1 = tf.multiply(tf.abs(x_output_0),W_per_edge)  #[B,Z,E(C)]
    elif sharing[0] == 2:
        if sharing[1] == 2:
            W_per_edge_1 = tf.matmul(tf.reshape(net_dict[f'var_{0}_{curr_iter}'],[1,M_proto]),W_skipconn2odd)
            W_per_edge_2 = tf.matmul(tf.reshape(net_dict[f'var_{1}_{curr_iter}'],[1,M_proto]),W_skipconn2odd)
            x_output_11 = tf.multiply(tf.abs(x_output_0),W_per_edge_1) #batch_size * Edge_num
            x_output_12 = tf.multiply(tf.abs(x_output_0),W_per_edge_2) #batch_size * Edge_num
            x_output_1 = tf.add(tf.multiply(x_output_11,-UCN_idx_edge + 1.0),tf.multiply(x_output_12,UCN_idx_edge))
        else:
            W_per_edge = tf.matmul(tf.reshape(net_dict[f'var_{0}_{curr_iter}'],[1,M_proto]),W_skipconn2odd) #[1,M] * [M,E(C)] = [1,E(C)]
            x_output_1 = tf.multiply(tf.abs(x_output_0),W_per_edge) #[B,Z,E(C)]
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
                          
        
    # Max( W * min(V->C), 0)
    x_output_2 = tf.multiply(x_output_1, tf.to_float(x_output_1 > 0))  #[B,Z,E(C)]
    
    if decoding_type == 2:
        x_output_2 = Cal_MSA_Q_TF(x_output_2,q_bit)
    else:
        x_output_2 = tf.clip_by_value(x_output_2, clip_value_min=-clip_LLR, clip_value_max=clip_LLR)


    net_dict["LLRa{0}".format(curr_iter+1)] = tf.multiply(x_output_2, tf.sign(x_output_0)) # [B,Z,E(C)], C->V Message
    y_output_2 = tf.matmul(net_dict["LLRa{0}".format(curr_iter+1)], W_output) #[B,Z,E(C)]X[E(C),N] = [B,Z,N], Sum_Input_LLR
    y_output_3 = tf.transpose(y_output_2, [0, 2, 1]) # B,Z,N -> B,N,Z
    
    #Decision
    if decoding_type == 2:
        xa = Cal_MSA_Q_TF(xa,q_bit)
    
    y_output_4 = tf.add(xa, y_output_3) #B X N X Z, APP_LLR
    y_output_4 = tf.clip_by_value(y_output_4, clip_value_min=-clip_LLR, clip_value_max=clip_LLR)
        
    net_dict["ya_output{0}".format(curr_iter)] = tf.reshape(y_output_4, [batch_size, N_proto * z_value], name='ya_output'.format(curr_iter))

    if target_node > 0:
        y_output_5 = y_output_4[:,:target_node,:]
    else:
        target_node = N_proto 
        y_output_5 = y_output_4
        
    net_dict["ya_output_target{0}".format(curr_iter)] = tf.reshape(y_output_5, [batch_size, target_node * z_value], name='ya_output_target'.format(curr_iter))
    
    # Loss    
    if curr_iter == training_iter_end - 1 and sampling_type != 2 and fixed_iter != iters_max:
        if loss_type <= 2:
            loss_ftn = 0
            temp_coeff = 0
            for t in range(training_iter_end - 1,max(training_iter_start - fixed_init,fixed_iter) - 1,-1):
                x_temp = net_dict["ya_output_target{0}".format(t)]
                if loss_type == 0:
                    loss_ftn = loss_ftn + pow(net_dict['etha'],(training_iter_end - 1 - t)) * tf.nn.sigmoid_cross_entropy_with_logits(labels=ya[:,:target_node* z_value], logits = x_temp)
                elif loss_type == 1:
                    loss_ftn = loss_ftn + pow(net_dict['etha'],(training_iter_end - 1 - t)) * tf.math.sigmoid(x_temp) #<-only for all zero codeword
                elif loss_type == 2:
                    x_temp = 1/2*(1-sign_through(tf.reduce_min(-x_temp, axis=1))) #<-only for all zero codeword
                    loss_ftn = loss_ftn + pow(net_dict['etha'],(training_iter_end - 1 - t)) * x_temp
                    
                temp_coeff = temp_coeff + pow(net_dict['etha'],(training_iter_end - 1 - t))
            loss_ftn = loss_ftn / temp_coeff
            

            net_dict["lossa"] = 1.0 * tf.reduce_mean(loss_ftn, name='lossa')
            
        
        
        # var_list
        current_vars = []
        # Loop through the sharing list and add variables to the current variable list
        for i, share_type in enumerate(sharing):
            # Determine the iteration range based on the sharing type
            if share_type == 0:
                start_iter, end_iter = None, None
            elif share_type in [1,2,3]:
                start_iter, end_iter = max(training_iter_start - fixed_init,fixed_iter), training_iter_end
            elif share_type in [4,5]:
                start_iter, end_iter = fixed_iter, fixed_iter + 1
                
            # Add the variables to the current variable list
            if start_iter is not None and end_iter is not None:
                for j in range(start_iter, end_iter):
                    current_vars.append(net_dict[f'var_{i}_{j}'])

        net_dict["train_stepa"] = tf.train.AdamOptimizer(learning_rate=
                                                            net_dict['learn_rate']).minimize(net_dict["lossa"], var_list = current_vars)
        
    if curr_iter == 0:
        net_dict["ya_output_all"] = net_dict["ya_output_target{0}".format(curr_iter)]
    else:
        net_dict["ya_output_all"] = tf.concat([net_dict["ya_output_all"],net_dict["ya_output_target{0}".format(curr_iter)]],axis = 0)
        
    return net_dict

def weight_init(net_dict, init_from_file, out_filename, iters_max, training_iter_start, sharing, Num_edge_proto, M_proto, N_proto, Min_weight, Max_weight, init_weight, init_VN_weight,  training_iter_end, fixed_iter):
    if init_from_file == 1:
        In_weight_file = f"./Weights/{out_filename}_In_Weight_End{iters_max}.txt"
    if training_iter_start > 0:
        Fixed_weight_file = f"./Weights/{out_filename}_Opt_Weight_End{training_iter_start}.txt"
        
    row_idx1 = 0
    row_idx2 = 0
    for i, share_type in enumerate(sharing):
        if share_type > 0:
            if share_type in [1, 4]:
                para_shape = Num_edge_proto
            elif share_type in [2, 5]:
                if i in [0,1]:
                    para_shape = M_proto
                elif i in [2]:
                    para_shape = N_proto
            elif share_type in [3]:
                para_shape = 1

            if i in [0, 1]:
                para_min, para_max, para_init = Min_weight, Max_weight, init_weight
            elif i in [2]:
                para_min, para_max, para_init = Min_weight, Max_weight, init_VN_weight

            if share_type in [1,2,3]:
                make_var_iter_end = training_iter_end
            elif share_type in [4,5]:
                make_var_iter_end = fixed_iter + 1
            
        
            for curr_iter in range(0, make_var_iter_end):
                if curr_iter < training_iter_start:    
                    row_idx1 += 1
                    data = np.loadtxt(Fixed_weight_file, skiprows = 1 + row_idx1, max_rows = 1, delimiter='\t')
                    init = tf.constant_initializer(data)
                elif init_from_file == 1:
                    row_idx2 += 1
                    data = np.loadtxt(In_weight_file, skiprows = 1 + row_idx2, max_rows = 1, delimiter='\t')
                    init = tf.constant_initializer(data)
                else:
                    if para_init == -1:
                        init = tf.truncated_normal_initializer(mean=(para_min + para_max) / 2, stddev=0.1)
                    else:
                        init = tf.constant_initializer(para_init * np.ones(para_shape, dtype=np.float32))
                
                var_name = f'var_{i}_{curr_iter}'
                net_dict[var_name] = tf.get_variable(var_name, dtype=tf.float32, shape=para_shape, initializer=init, constraint=lambda z: tf.clip_by_value(z, para_min, para_max))
            
            row_idx1 += 1
            row_idx2 += 1
                
    return net_dict

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


def QMS_clipping(x,q_bit):
    if q_bit == 6:
        return tf.clip_by_value(x, -15.5, 15.5)
    elif q_bit == 5:
        return tf.clip_by_value(x, -7.5, 7.5)
    elif q_bit == -5:
        return tf.clip_by_value(x, -15, 15)
    elif q_bit == 4:
        return tf.clip_by_value(x, -7, 7)
    elif q_bit == 3:
        return tf.clip_by_value(x,-6,6)
    
def Cal_MSA_Q_TF(x,q_bit):
    '''Element-wise rounding to the closest integer with full gradient propagation.
    A trick from [Sergey Ioffe](http://stackoverflow.com/a/36480182)
    a op that behave as f(x) in forward mode,
    but as g(x) in the backward mode.
    '''

    #q_value = QMS_clipping(tf.round(x / level)* level,q_bit)
    if q_bit == 6:
        q_value = tf.clip_by_value(tf.round(x),-15.5,15.5) #(-7.5 -7.0 -6.5 ... 6.5 7.0 7.5) Quantizer  
    elif q_bit == 5:
        q_value = tf.clip_by_value(tf.round(x * 2)/2,-7.5,7.5) #(-7.5 -7.0 -6.5 ... 6.5 7.0 7.5) Quantizer  
    elif q_bit == -5:
        q_value = tf.clip_by_value(tf.round(x),-15,15) #(-15 -14 -13 ... 13 14 15) Quantizer  
    elif q_bit == 4:
        q_value = tf.clip_by_value(tf.round(x),-7,7) #(-7.0 -6.0 ... 6.0 7.0) Quantizer  
    elif q_bit == 3:
        q_value = tf.clip_by_value(tf.round(x/2)*2,-6,6) #(-6,-4,-2,0,2,4,6) Quantizer
        
    return QMS_clipping(x,q_bit) + tf.stop_gradient(q_value-QMS_clipping(x,q_bit)) #foward = q_value, gradient = 1



def check_params(sampling_type, SNR_Matrix, sharing, iters_max, fixed_iter, iter_step):
    if sampling_type == 1:        
        if len(SNR_Matrix) > 1:
            SNR_Matrix = np.array([0.0])
    elif sampling_type == 2:
        if len(SNR_Matrix) > 1:
            print("sampling_type == 2 and len(SNR_Matrix) > 1")
            sys.exit()

    if np.sum(sharing) == 0:
        print("np.sum(sharing) == 0")
        sys.exit()

    if any(value in [4,5] for value in sharing) and (iters_max - fixed_iter) % iter_step > 0:
        print("any(value in [4,5] for value in sharing) and (iters_max - fixed_iter) % iter_step > 0")
        sys.exit()

    if sharing[2] in [1,4]:
        print("sharing[2] in [1,4]")
        sys.exit()

    if (sharing[1] != 0 and sharing[0]!=sharing[1]):
        print("sharing[1] != 0 and sharing[0]!=sharing[1])")
        sys.exit()
        
    return SNR_Matrix
        

def process_data(sampling_type, filename, training_num, valid_flag, valid_num, test_flag, test_num):
    if sampling_type == 1:

        uncor_filename = "[Uncor]_{0}".format(filename)
        input_llr_training = np.loadtxt(f"./Inputs/{uncor_filename}.txt", dtype=np.float32, delimiter='\t')
        if input_llr_training.ndim > 1:
            input_llr_training = np.delete(input_llr_training, [0,1,2], 1)

        if input_llr_training.shape[0] < training_num:
            print("Wrong input: input_llr_training.shape[0] < training_num")
            sys.exit()
        else:
            input_llr_training = input_llr_training[:training_num,:]
            input_codeword_training = np.zeros(input_llr_training.shape, dtype=np.int64)


        if valid_flag == 1:
            uncor_filename = "[Uncor]_{0}_Valid".format(filename)
            input_llr_valid = np.loadtxt(f"./Inputs/{uncor_filename}.txt", dtype=np.float32, delimiter='\t')
            if input_llr_valid.ndim > 1:
                input_llr_valid = np.delete(input_llr_valid, [0,1,2], 1)
            if input_llr_valid.shape[0] < valid_num:
                print("Wrong input: input_llr_valid.shape[0] < valid_num: < 0")
                sys.exit()

            input_llr_valid = input_llr_valid[:valid_num]
            input_codeword_valid = np.zeros(input_llr_valid.shape, dtype=np.int64)
        else:
            input_llr_valid = []
            input_codeword_valid = []
            valid_num = 0

        if test_flag == 1:
            uncor_filename = "[Uncor]_{0}_Test".format(filename)
            input_llr_test = np.loadtxt(f"./Inputs/{uncor_filename}.txt", dtype=np.float32, delimiter='\t')
            if input_llr_test.ndim > 1:
                input_llr_test = np.delete(input_llr_test, [0,1,2], 1)
            if input_llr_test.shape[0] < test_num:
                print("Wrong input: input_llr_test.shape[0] < test_num: < 0")
                sys.exit()

            input_llr_test = input_llr_test[:test_num]
            input_codeword_test = np.zeros(input_llr_test.shape, dtype=np.int64)
        else:
            input_llr_test = []
            input_codeword_test = []
            test_num = 0
            
        return input_llr_training, input_codeword_training, input_llr_valid, input_codeword_valid, input_llr_test, input_codeword_test
    else:
        return [],[],[],[],[],[]

        
    
        
    

def get_num(ratio, sample_num, valid_num=None):
    if ratio <= 1:
        num = round(ratio * sample_num)
    elif ratio > 1:
        num = ratio
    elif ratio == -1 and valid_num is not None:
        num = sample_num - valid_num
    return num
