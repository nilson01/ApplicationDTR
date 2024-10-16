
import os
import sys
from tqdm import tqdm
import json
from itertools import product
from utils import * # import utils and do utils.fn_name
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import time 
from datetime import datetime
import copy
from collections import defaultdict

# from sklearn import preprocessing
# import seaborn as sns
# from sklearn import metrics
from sklearn.model_selection import train_test_split
# import random
# import math

import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
numpy2ri.activate()

# Load the R script to avoid dynamic loading
ro.r.source("ACWL_tao.R")

# Generate Data
def load_and_preprocess_data(params, replication_seed, run='train'):

    # cutting off data points for faster testing
    df = pd.read_csv('final_data.csv').iloc[:2000, ]  #.iloc[:params["sample_size"], ] 
    print("df ==================> : ", df.shape, "Total data points: ",  df.shape[0]/2)

    # Shuffle
    #sample the rows creating a random order
    groups = [df for _, df in df.groupby('m:icustayid')]
    np.random.shuffle(groups)  # Shuffles the list of groups in place
    # Concatenate the shuffled groups back into a single DataFrame
    df = pd.concat(groups).reset_index(drop=True)

    #IV fluid is the treatment
    O1_df = df.copy()
    cols_to_drop = ['traj', 'm:presumed_onset', 'm:charttime', 'm:icustayid','o:input_4hourly']
    O1_df = O1_df.drop(cols_to_drop, axis=1)
    O1_df = O1_df[O1_df['step'] == 0]
    O1_df = O1_df.drop('step', axis = 1)
    O1_tens = torch.tensor(O1_df.values)
    O1 = O1_tens.t()
 
    #creating treatment levels
    A1_df = df.copy()
    A1_df = A1_df[A1_df['step'] == 0]
    A1_df = A1_df[['step', 'o:max_dose_vaso', 'o:input_4hourly']]
    A1_df = A1_df.drop('step', axis = 1)
    for index, row in A1_df.iterrows():
        if row['o:max_dose_vaso'] == 0:
            A1_df.at[index, 'o:max_dose_vaso'] = 1
        elif row['o:max_dose_vaso'] <= 0.18:
            A1_df.at[index, 'o:max_dose_vaso'] = 2
        elif row['o:max_dose_vaso'] > 0.18:
            A1_df.at[index, 'o:max_dose_vaso'] = 3

        if row['o:input_4hourly'] == 0:
            A1_df.at[index, 'o:input_4hourly'] = 1
        elif 0 < row['o:input_4hourly'] < 100:
            A1_df.at[index, 'o:input_4hourly'] = 2
        elif row['o:input_4hourly'] >= 100:
            A1_df.at[index, 'o:input_4hourly'] = 3

    A1_df = A1_df.drop('o:max_dose_vaso', axis = 1)   
    
   
    A1 = torch.tensor(A1_df.values).squeeze()

    
    probs1= M_propen(A1, O1.t(), stage=1)  


    Y1_df = df.copy()
    Y1_df = Y1_df[Y1_df['step'] == 0]
    Y1_df = Y1_df[['o:Arterial_lactate']]
    Y1_df = Y1_df['o:Arterial_lactate'].apply(lambda x:4 * (22-x))
    Y1 = torch.tensor(Y1_df.values).squeeze()
  
   
   
    O2_df = df.copy()
    cols_to_drop = ['traj', 'm:presumed_onset', 'm:charttime', 'm:icustayid','o:input_4hourly', 'o:gender', 'o:age', 'o:Weight_kg']
    O2_df = O2_df.drop(cols_to_drop, axis=1)
    O2_df = O2_df[O2_df['step'] == 1]
    O2_df = O2_df.drop('step', axis = 1)
    O2_tens = torch.tensor(O2_df.values)
    O2 = O2_tens.t()


    A2_df = df.copy()
    A2_df = A2_df[A2_df['step'] == 1]
    A2_df = A2_df[['o:max_dose_vaso', 'o:input_4hourly']]
    for index, row in A2_df.iterrows():
        if row['o:max_dose_vaso'] == 0:
            A2_df.at[index, 'o:max_dose_vaso'] = 1
        elif row['o:max_dose_vaso'] <= 0.18:
            A2_df.at[index, 'o:max_dose_vaso'] = 2
        elif row['o:max_dose_vaso'] > 0.18:
            A2_df.at[index, 'o:max_dose_vaso'] = 3

        if row['o:input_4hourly'] == 0:
            A2_df.at[index, 'o:input_4hourly'] = 1
        elif 0 < row['o:input_4hourly'] < 100:
            A2_df.at[index, 'o:input_4hourly'] = 2
        elif row['o:input_4hourly'] >= 100:
            A2_df.at[index, 'o:input_4hourly'] = 3

    A2_df = A2_df.drop('o:max_dose_vaso', axis = 1)       
    A2 = torch.tensor(A2_df.values).squeeze()

    combined_tensor = torch.cat((O1.t(),A1.unsqueeze(1), Y1.unsqueeze(1), O2.t()), dim=1)
 
    probs2 = M_propen(A2, combined_tensor, stage=2) 

    Y2_df = df.copy()
    Y2_df = Y2_df[Y2_df['step'] == 1]
    Y2_df = Y2_df[['o:Arterial_lactate']]
    Y2_df = Y2_df['o:Arterial_lactate'].apply(lambda x: 4 * (22-x))
    Y2 = torch.tensor(Y2_df.values).squeeze()


    if run != 'test':
      # transform Y for direct search 
      Y1, Y2 = transform_Y(Y1, Y2)


    # Propensity score stack
    pi_tensor_stack = torch.stack([probs1['pi_10'], probs1['pi_11'], probs1['pi_12'], probs2['pi_20'], probs2['pi_21'], probs2['pi_22']])
    # Adjusting A1 and A2 indices
    A1_indices = (A1 - 1).long().unsqueeze(0)  # A1 actions, Subtract 1 to match index values (0, 1, 2)
    A2_indices = (A2 - 1 + 3).long().unsqueeze(0)   # A2 actions, Add +3 to match index values (3, 4, 5) for A2, with added dimension

    # Gathering probabilities based on actions
    P_A1_given_H1_tensor = torch.gather(pi_tensor_stack, dim=0, index=A1_indices).squeeze(0)  # Remove the added dimension after gathering
    P_A2_given_H2_tensor = torch.gather(pi_tensor_stack, dim=0, index=A2_indices).squeeze(0)  # Remove the added dimension after gathering


    # #here the clipping starts with encoding A
    # label_encoder = preprocessing.LabelEncoder()
    # A2_enc = label_encoder.fit_transform(A2)
    # A1_enc = label_encoder.fit_transform(A1)

    # num_rows1 = len(probs1['pi_10'])
    # num_rows2 = len(probs2['pi_20'])

    # # Initialize encoded values list
    # encoded_values1 = []
    # encoded_values2 = []
    # #need for stage 1 and stage 2
    # # Iterate through each row
    # for i in range(num_rows1):
    #     pi_10_prob = probs1['pi_10'][i].item()
    #     pi_11_prob = probs1['pi_11'][i].item()
    #     pi_12_prob = probs1['pi_12'][i].item()
        
    #     # Determine the key with the highest probability
    #     max_prob_key = max(probs1.keys(), key=lambda key: probs1[key][i].item())
        
    #     # Encode based on the key with the highest probability
    #     if max_prob_key == 'pi_10':
    #         encoded_values1.append(0)
    #     elif max_prob_key == 'pi_11':
    #         encoded_values1.append(1)
    #     elif max_prob_key == 'pi_12':
    #         encoded_values1.append(2)

    # for i in range(num_rows2):
    #     # Get probabilities for each key for current row
    #     pi_20_prob = probs2['pi_20'][i].item()
    #     pi_21_prob = probs2['pi_21'][i].item()
    #     pi_22_prob = probs2['pi_22'][i].item()
        
    #     # Determine the key with the highest probability
    #     max_prob_key = max(probs2.keys(), key=lambda key: probs2[key][i].item())
        
    #     # Encode based on the key with the highest probability
    #     if max_prob_key == 'pi_20':
    #         encoded_values2.append(0)
    #     elif max_prob_key == 'pi_21':
    #         encoded_values2.append(1)
    #     elif max_prob_key == 'pi_22':
    #         encoded_values2.append(2)



    # encoded_values1 = np.array(encoded_values1)
    # encoded_values2 = np.array(encoded_values2)
    # cm0 = metrics.confusion_matrix(A1_enc, encoded_values1) #stage 1 confusion
    # cm1 = metrics.confusion_matrix(A2_enc, encoded_values2) #stage 2 confusion

    # # Plotting first confusion matrix
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm0, annot=True, cmap='Blues', fmt='d', cbar=False)
    # plt.xlabel('Predicted labels')
    # plt.ylabel('True labels')
    # plt.title('No Clipping Stage1')
    # plt.savefig('noclip_stage1.png')
    # plt.close()
    # #plot stage 2 no clip
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm1, annot=True, cmap='Blues', fmt='d', cbar=False)
    # plt.xlabel('Predicted labels')
    # plt.ylabel('True labels')
    # plt.title('No Clipping Stage2')
    # plt.savefig('noclip_stage2.png')
    # plt.close()



    #here is where I determine which indices to delete
    indices1 = torch.nonzero(P_A1_given_H1_tensor < 0.10)
    indices2 = torch.nonzero(P_A2_given_H2_tensor < 0.10)
    combined_indices_set = set(tuple(idx.tolist()) for idx in torch.cat((indices1, indices2)))
    combined_indices_tensor = torch.tensor(list(combined_indices_set))
    print("number of deletes", len(combined_indices_tensor))


    #then I have to go through every variable and delete those indices from them
    P_A2_given_H2_numpy = P_A2_given_H2_tensor.numpy()
    P_A2_given_H2_numpy = np.delete(P_A2_given_H2_numpy, combined_indices_tensor, axis=0)
    P_A2_given_H2_tensor_filtered = torch.tensor(P_A2_given_H2_numpy)


    print("P_A2_H2 max, min, avg", P_A2_given_H2_tensor_filtered.max(), P_A2_given_H2_tensor_filtered.min(), torch.mean(P_A2_given_H2_tensor_filtered))

    # encoded_values1 = np.delete(encoded_values1, combined_indices_tensor, axis=0)
    # encoded_values2 = np.delete(encoded_values2, combined_indices_tensor, axis=0)

    P_A1_given_H1_numpy = P_A1_given_H1_tensor.numpy()
    P_A1_given_H1_numpy = np.delete(P_A1_given_H1_numpy, combined_indices_tensor, axis=0)
    P_A1_given_H1_tensor_filtered = torch.tensor(P_A1_given_H1_numpy)
    print("P_A1_H1 max, min, avg", P_A1_given_H1_tensor_filtered.max(), P_A1_given_H1_tensor_filtered.min(), torch.mean(P_A1_given_H1_tensor_filtered))
  
    pi_tensor_stack_np = pi_tensor_stack.numpy()
    pi_tensor_stack_np = np.delete(pi_tensor_stack_np, combined_indices_tensor, axis=1)
    pi_tensor_filtered = torch.tensor(pi_tensor_stack_np)
    print("pi_tensor dimensions: ", pi_tensor_filtered.shape)

    O1_numpy = np.delete(O1.numpy(), combined_indices_tensor, axis=1)
    O1_filtered = torch.tensor(O1_numpy)

    O2_numpy = np.delete(O2.numpy(), combined_indices_tensor, axis=1)
    O2_filtered = torch.tensor(O2_numpy)

    A1_numpy = np.delete(A1.numpy(), combined_indices_tensor, axis=0)
    A1_filtered = torch.tensor(A1_numpy)

    A2_numpy = np.delete(A2.numpy(), combined_indices_tensor, axis=0)
    A2_filtered = torch.tensor(A2_numpy)

    Y1_numpy = np.delete(Y1.numpy(), combined_indices_tensor, axis=0)
    Y1_filtered = torch.tensor(Y1_numpy)

    Y2_numpy = np.delete(Y2.numpy(), combined_indices_tensor, axis=0)
    Y2_filtered = torch.tensor(Y2_numpy)


    # label_encoder = preprocessing.LabelEncoder()
    # A2filt_enc = label_encoder.fit_transform(A2_filtered)
    # A1filt_enc = label_encoder.fit_transform(A1_filtered)
    # #create and plot confusion matrices after clipping
    # cm2 = metrics.confusion_matrix(A1filt_enc, encoded_values1)
    # cm3 = metrics.confusion_matrix(A2filt_enc, encoded_values2)
  
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm2, annot=True, cmap='Blues', fmt='d', cbar=False)
    # plt.xlabel('Predicted labels')
    # plt.ylabel('True labels')
    # plt.title('After Clipping Stage1')
    # plt.savefig('clip_stage1.png')
    # plt.close()

    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm3, annot=True, cmap='Blues', fmt='d', cbar=False)
    # plt.xlabel('Predicted labels')
    # plt.ylabel('True labels')
    # plt.title('After Clipping Stage2')
    # plt.savefig('clip_stage2.png')
    # plt.close()
    #done with all clipping

    # Calculate Ci tensor
    Ci = (Y1_filtered + Y2_filtered) / (P_A1_given_H1_tensor_filtered * P_A2_given_H2_tensor_filtered)
    # # Input preparation
    input_stage1 = O1_filtered.t()
    input_stage2 = torch.cat([O1_filtered.t(), A1_filtered.unsqueeze(1), Y1_filtered.unsqueeze(1), O2_filtered.t()], dim=1) 

    # here I just need an updated set of indices (after clipping)
    # not necessary to sort by patient id since stages are split, also some tensors dont have ids (P_A1_H1)
    numpy_array = O1_filtered.numpy()
    df = pd.DataFrame(numpy_array)
    column_headings = df.columns
    unique_indexes = pd.unique(column_headings)


    #splitting the indices into test and train (not random)
    train_patient_ids, test_patient_ids = train_test_split(unique_indexes, test_size=0.5, shuffle = False)
    #print(train_patient_ids, test_patient_ids, unique_indexes)

    if run == 'test':
        #filter based on indices in test
        test_patient_ids = torch.tensor(test_patient_ids)
        Ci = Ci[test_patient_ids]
        O1_filtered = O1_filtered[:, test_patient_ids]
        O2_filtered = O2_filtered[:, test_patient_ids]
        Y1_filtered = Y1_filtered[test_patient_ids]
        Y2_filtered = Y2_filtered[test_patient_ids]
        A1_filtered = A1_filtered[test_patient_ids]
        A2_filtered = A2_filtered[test_patient_ids]

        #calculate input stages
        input_stage1 = O1_filtered.t()         
        params['input_dim_stage1'] = input_stage1.shape[1] #  (H_1)  
        input_stage2 = torch.cat([O1_filtered.t(), A1_filtered.unsqueeze(1), Y1_filtered.unsqueeze(1), O2_filtered.t()], dim=1) 
        params['input_dim_stage2'] = input_stage2.shape[1] # (H_2)

        P_A1_given_H1_tensor_filtered = P_A1_given_H1_tensor_filtered[test_patient_ids]
        P_A2_given_H2_tensor_filtered = P_A2_given_H2_tensor_filtered[test_patient_ids]
        #ensure proper data types
        input_stage1 = input_stage1.float()
        input_stage2 = input_stage2.float()
        Ci = Ci.float()
        Y1_filtered = Y1_filtered.float()
        Y2_filtered = Y2_filtered.float()
        A1_filtered = A1_filtered.float()
        A2_filtered = A2_filtered.float()
        return input_stage1, input_stage2, O2_filtered.t(), Y1_filtered, Y2_filtered, A1_filtered, A2_filtered, P_A1_given_H1_tensor_filtered, P_A2_given_H2_tensor_filtered
   
    #filter based on train ids
    train_patient_ids = torch.tensor(train_patient_ids)
    O1_filtered = O1_filtered[:, train_patient_ids]
    O2_filtered = O2_filtered[:, train_patient_ids]
    pi_tensor_filtered = pi_tensor_filtered[:, train_patient_ids]
    print("shape", pi_tensor_filtered.shape)
    Y1_filtered = Y1_filtered[train_patient_ids]
    Y2_filtered = Y2_filtered[train_patient_ids]
    A1_filtered = A1_filtered[train_patient_ids]
    A2_filtered = A2_filtered[train_patient_ids]
    Ci = Ci[train_patient_ids]

    input_stage1 = O1_filtered.t()
    params['input_dim_stage1'] = input_stage1.shape[1] #  (H_1)  
    print("dimesnions of input stage", len(input_stage1))
    input_stage2 = torch.cat([O1_filtered.t(), A1_filtered.unsqueeze(1), Y1_filtered.unsqueeze(1), O2_filtered.t()], dim=1)         
    params['input_dim_stage2'] = input_stage2.shape[1] # (H_2)

    input_stage1 = input_stage1.float()
    input_stage2 = input_stage2.float()
    Ci = Ci.float()
    Y1_filtered = Y1_filtered.float()
    Y2_filtered = Y2_filtered.float()
    A1_filtered = A1_filtered.float()
    A2_filtered = A2_filtered.float()
    # train_size = int(params['training_validation_prop'] * params['sample_size']) # this code is the main problem for divide by zero issue
    train_size = int(params['training_validation_prop'] * Y1_filtered.shape[0])
    # print(" train_size, params['training_validation_prop'],  params['sample_size'], Y1_filtered.shape ===================>>>>>>>>>>>>>>>>>>>> ", train_size, params['training_validation_prop'],  params['sample_size'], Y1_filtered.shape[0])

    train_tensors = [tensor[:train_size] for tensor in [input_stage1, input_stage2, Ci, Y1_filtered, Y2_filtered, A1_filtered, A2_filtered]]
    val_tensors = [tensor[train_size:] for tensor in [input_stage1, input_stage2, Ci, Y1_filtered, Y2_filtered, A1_filtered, A2_filtered]]

    # return tuple(train_tensors), tuple(val_tensors)
    return tuple(train_tensors), tuple(val_tensors), tuple([O1_filtered.t(), O2_filtered.t(), Y1_filtered, Y2_filtered, A1_filtered, A2_filtered, pi_tensor_filtered])



def surr_opt(tuple_train, tuple_val, params, config_number):
    
    sample_size = params['sample_size'] 
    
    train_losses, val_losses = [], []
    best_val_loss, best_model_stage1_params, best_model_stage2_params, epoch_num_model = float('inf'), None, None, 0

    nn_stage1 = initialize_and_prepare_model(1, params)
    nn_stage2 = initialize_and_prepare_model(2, params)

    optimizer, scheduler = initialize_optimizer_and_scheduler(nn_stage1, nn_stage2, params)

    #  Training and Validation data
    train_data = {'input1': tuple_train[0], 'input2': tuple_train[1], 'Ci': tuple_train[2], 'A1': tuple_train[5], 'A2': tuple_train[6]}
    val_data = {'input1': tuple_val[0], 'input2': tuple_val[1], 'Ci': tuple_val[2], 'A1': tuple_val[5], 'A2': tuple_val[6]}


    # Training and Validation loop for both stages  
    for epoch in range(params['n_epoch']):  

        train_loss = process_batches(nn_stage1, nn_stage2, train_data, params, optimizer, is_train=True)
        train_losses.append(train_loss)

        val_loss = process_batches(nn_stage1, nn_stage2, val_data, params, optimizer, is_train=False)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            epoch_num_model = epoch
            best_val_loss = val_loss
            best_model_stage1_params = nn_stage1.state_dict()
            best_model_stage2_params = nn_stage2.state_dict()

        # Update the scheduler with the current epoch's validation loss
        update_scheduler(scheduler, params, val_loss)

    model_dir = f"models/{params['job_id']}"
    # Check if the directory exists, if not, create it
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Define file paths for saving models
    model_path_stage1 = os.path.join(model_dir, f'best_model_stage_surr_1_{sample_size}_config_number_{config_number}.pt')
    model_path_stage2 = os.path.join(model_dir, f'best_model_stage_surr_2_{sample_size}_config_number_{config_number}.pt')
        
    # Save the models
    torch.save(best_model_stage1_params, model_path_stage1)
    torch.save(best_model_stage2_params, model_path_stage2)
    
    return ((train_losses, val_losses), epoch_num_model)



def DQlearning(tuple_train, tuple_val, params, config_number):
    train_input_stage1, train_input_stage2, _, train_Y1, train_Y2, train_A1, train_A2 = tuple_train
    val_input_stage1, val_input_stage2, _, val_Y1, val_Y2, val_A1, val_A2 = tuple_val

    nn_stage1, optimizer_1, scheduler_1 = initialize_model_and_optimizer(params, 1)
    nn_stage2, optimizer_2, scheduler_2 = initialize_model_and_optimizer(params, 2) 

    # print(" train_input_stage2.shape, val_input_stage2.shape: -------->>>>>>>>>>>>>> ", train_input_stage2.shape, val_input_stage2.shape)

    train_losses_stage2, val_losses_stage2, epoch_num_model_2 = train_and_validate(config_number, nn_stage2, optimizer_2, scheduler_2, 
                                                                                   train_input_stage2, train_A2, train_Y2, 
                                                                                   val_input_stage2, val_A2, val_Y2, params, 2)

    train_Y1_hat = evaluate_model_on_actions(nn_stage2, train_input_stage2, train_A2) + train_Y1
    val_Y1_hat = evaluate_model_on_actions(nn_stage2, val_input_stage2, val_A2) + val_Y1

    train_losses_stage1, val_losses_stage1, epoch_num_model_1 = train_and_validate(config_number, nn_stage1, optimizer_1, scheduler_1, 
                                                                                   train_input_stage1, train_A1, train_Y1_hat, 
                                                                                   val_input_stage1, val_A1, val_Y1_hat, params, 1)

    return (train_losses_stage1, train_losses_stage2, val_losses_stage1, val_losses_stage2)



def evaluate_tao(S1, S2, A1, A2, Y1, Y2, params_ds, config_number):

    # Convert test input from PyTorch tensor to numpy array
    S1 = S1.cpu().numpy()
    S2 = S2.cpu().numpy()

    # Load the R script that contains the required function
    ro.r('source("ACWL_tao.R")')

    # Call the R function with the parameters
    results = ro.globalenv['test_ACWL'](S1, S2, A1.cpu().numpy(), A2.cpu().numpy(), Y1.cpu().numpy(), Y2.cpu().numpy(), 
                                        config_number, params_ds['job_id'])

    # Extract the decisions and convert to PyTorch tensors on the specified device
    A1_Tao = torch.tensor(np.array(results.rx2('g1.a1')), dtype=torch.float32).to(params_ds['device'])
    A2_Tao = torch.tensor(np.array(results.rx2('g2.a1')), dtype=torch.float32).to(params_ds['device'])

    return A1_Tao, A2_Tao


# def eval_DTR(V_replications, num_replications, nn_stage1_DQL, nn_stage2_DQL, nn_stage1_DS, nn_stage2_DS, df_DQL, df_DS, df_Tao, params_dql, params_ds, config_number):
def eval_DTR(V_replications, num_replications, df_DQL, df_DS, df_Tao, params_dql, params_ds, tmp, config_number):

    # Generate and preprocess data for evaluation
    processed_result = load_and_preprocess_data(params_ds, replication_seed=num_replications, run='test')
    test_input_stage1, test_input_stage2, test_O2, Y1_tensor, Y2_tensor, A1_tensor_test, A2_tensor_test, P_A1_g_H1, P_A2_g_H2  = processed_result
    train_tensors = [test_input_stage1, test_input_stage2, Y1_tensor, Y2_tensor, A1_tensor_test, A2_tensor_test]

    # Append policy values for DS
    V_replications["V_replications_M1_behavioral"].append(torch.mean(Y1_tensor + Y2_tensor).cpu().item())  
    # Value function behavioral
    message = f'\nY1 beh mean: {torch.mean(Y1_tensor)}, Y2 beh mean: {torch.mean(Y2_tensor)}, Y1_beh+Y2_beh mean: {torch.mean(Y1_tensor + Y2_tensor)} '
    print(message)

    #######################################
    # Evaluation phase using Tao's method #
    #######################################
    if params_ds.get('run_adaptive_contrast_tao', True):
        start_time = time.time()  # Start time recording
        A1_Tao, A2_Tao = evaluate_tao(test_input_stage1, test_O2, A1_tensor_test, A2_tensor_test, Y1_tensor, Y2_tensor, params_ds, config_number)
        end_time = time.time()  # End time recording
        print(f"Total time taken to run evaluate_tao: { end_time - start_time} seconds")
        

        # Append to DataFrame
        new_row_Tao = {
            'Behavioral_A1': A1_tensor_test.cpu().numpy().tolist(),
            'Behavioral_A2': A2_tensor_test.cpu().numpy().tolist(),
            'Predicted_A1': A1_Tao.cpu().numpy().tolist(),
            'Predicted_A2': A2_Tao.cpu().numpy().tolist()
        }
        df_Tao = pd.concat([df_Tao, pd.DataFrame([new_row_Tao])], ignore_index=True)

        # Calculate policy values fn. using the estimator of Tao's method
        # print("Tao's method estimator: ")
        start_time = time.time()  # Start time recording
        V_rep_Tao = calculate_policy_values_W_estimator(train_tensors, params_ds, A1_Tao, A2_Tao, P_A1_g_H1, P_A2_g_H2, config_number)
        end_time = time.time()  # End time recording
        print(f"\n\nTotal time taken to run calculate_policy_values_W_estimator_tao: { end_time - start_time} seconds")
                
        # Append policy values for Tao
        V_replications["V_replications_M1_pred"]["Tao"].append(V_rep_Tao)     
        message = f'\nY1_tao+Y2_tao mean: {V_rep_Tao} \n'
        print(message)

    #######################################
    # Evaluation phase using DQL's method #
    #######################################
    if params_ds.get('run_DQlearning', True):
        start_time = time.time()  # Start time recording
        df_DQL, V_rep_DQL, param_W_DQL = evaluate_method('DQL', params_dql, config_number, df_DQL, test_input_stage1, A1_tensor_test, test_input_stage2, 
                                            A2_tensor_test, train_tensors, P_A1_g_H1, P_A2_g_H2, tmp)
        end_time = time.time()  # End time recording
        print(f"\n\nTotal time taken to run evaluate_method)W_estimator('DQL'): { end_time - start_time} seconds")
        # Append policy values for DQL
        V_replications["V_replications_M1_pred"]["DQL"].append(V_rep_DQL)     
        message = f'\nY1_DQL+Y2_DQL mean: {V_rep_DQL} '
        print(message)

    ########################################
    #  Evaluation phase using DS's method  #
    ########################################
    if params_ds.get('run_surr_opt', True):
        start_time = time.time()  # Start time recording
        df_DS, V_rep_DS, param_W_DS = evaluate_method('DS', params_ds, config_number, df_DS, test_input_stage1, A1_tensor_test, test_input_stage2, 
                                        A2_tensor_test, train_tensors, P_A1_g_H1, P_A2_g_H2, tmp)
        end_time = time.time()  # End time recording
        print(f"\n\nTotal time taken to run evaluate_method)W_estimator('DS'): { end_time - start_time} seconds\n\n")
                    
        # Append policy values for DS
        V_replications["V_replications_M1_pred"]["DS"].append(V_rep_DS)
        message = f'\nY1_DS+Y2_DS mean: {V_rep_DS} '
        print(message)

    return V_replications, df_DQL, df_DS, df_Tao, param_W_DQL, param_W_DS # {"df_DQL": df_DQL, "df_DS":df_DS, "df_Tao": df_Tao}




def adaptive_contrast_tao(all_data, contrast, config_number, job_id):
    S1, S2, train_Y1, train_Y2, train_A1, train_A2, pi_tensor_stack = all_data

    # Convert all tensors to CPU and then to NumPy
    A1 = train_A1.cpu().numpy()
    probs1 = pi_tensor_stack.T[:, :3].cpu().numpy()

    A2 = train_A2.cpu().numpy()
    probs2 = pi_tensor_stack.T[:, 3:].cpu().numpy()

    R1 = train_Y1.cpu().numpy()
    R2 = train_Y2.cpu().numpy()

    S1 = S1.cpu().numpy()
    S2 = S2.cpu().numpy()

    # Load the R script containing the function
    ro.r('source("ACWL_tao.R")')



    # print("probs1: --------> ",probs1, probs1.shape, "\n\n\n")
    # print("\n\n\n", "probs2: --------> ",probs2, probs2.shape, "\n\n")     
    # print("Max values of each row: ", np.min(np.max(probs2, axis=1)) , "\n\n")

    # Call the R function with the numpy arrays     
    ro.globalenv['train_ACWL'](job_id, S1, S2, A1, A2, probs1, probs2, R1, R2, config_number, contrast)



def simulations(V_replications, params, config_fixed, config_number):

    columns = ['Behavioral_A1', 'Behavioral_A2', 'Predicted_A1', 'Predicted_A2']

    # Initialize separate DataFrames for DQL and DS
    df_DQL = pd.DataFrame(columns=columns)
    df_DS = pd.DataFrame(columns=columns)
    df_Tao = pd.DataFrame(columns=columns)

    losses_dict = {'DQL': {}, 'DS': {}} 

    config_dict = {
        "trainingFixed": params['trainingFixed'],  
        "training_config": {'DQL': {}, 'DS': {}}, 
        "testing_config": {'DQL': {}, 'DS': {}}
    }

    # config_dict['trainingFixed'].append(params['trainingFixed'])

    epoch_num_model_lst = []
    

    # Clone the fixed config for DQlearning and surr_opt to load the correct trained model 
    if params['trainingFixed']:
        tmp = [params['num_layers'], params['hidden_dim_stage1'], params['hidden_dim_stage2'], params['activation_function'] ]
        # print(f"<<<<<<<<<<<<<--------------  {tmp} --------------->>>>>>>>>>>>>>>>>")
        params['num_layers'] = config_fixed['num_layers'] 
        params['hidden_dim_stage1'] = config_fixed['hidden_dim_stage1'] 
        params['hidden_dim_stage2'] = config_fixed['hidden_dim_stage2'] 
        params['activation_function'] = config_fixed['activation_function'] 

    else:         
        tmp = [config_fixed['num_layers'], config_fixed['hidden_dim_stage1'], config_fixed['hidden_dim_stage2'], config_fixed['activation_function'] ]
        config_fixed['num_layers'] = params['num_layers']
        config_fixed['hidden_dim_stage1'] = params['hidden_dim_stage1']
        config_fixed['hidden_dim_stage2'] = params['hidden_dim_stage2']
        config_fixed['activation_function'] = params['activation_function']


    # Clone the updated config for DQlearning and surr_opt
    params_DQL_u = copy.deepcopy(params)
    params_DS_u = copy.deepcopy(params)
    
    params_DS_u['f_model'] = 'surr_opt'
    params_DQL_u['f_model'] = 'DQlearning'
    params_DQL_u['num_networks'] = 1  

    params_DQL_f = copy.deepcopy(config_fixed)
    params_DS_f = copy.deepcopy(config_fixed)
    
    params_DS_f['f_model'] = 'surr_opt'
    params_DQL_f['f_model'] = 'DQlearning'
    params_DQL_f['num_networks'] = 1  

    # config_dict_training_config_DQL = {}
    # config_dict_training_config_DS = {}



    for replication in tqdm(range(params['num_replications']), desc="Replications_M1"):
        print(f"\nReplication # -------------->>>>>  {replication+1}")

        # config_dict['replications'].append(replication+1)

        # Generate and preprocess data for training
        tuple_train, tuple_val, adapC_tao_Data = load_and_preprocess_data(params, replication_seed=replication, run='train')

        # Estimate treatment regime : model --> surr_opt
        print("Training started!")
        
        # Run ALL models on the same tuple of data
        if params.get('run_adaptive_contrast_tao', True):
            start_time = time.time()  # Start time recording
            adaptive_contrast_tao(adapC_tao_Data, params["contrast"], config_number, params["job_id"])
            end_time = time.time()  # End time recording
            print(f"Total time taken to run adaptive_contrast_tao: { end_time - start_time} seconds")
            
        if params.get('run_DQlearning', True):
            # Run both models on the same tuple of data
            params_DQL_u['input_dim_stage1'] = params['input_dim_stage1'] + 1 # Ex. TAO: 5 + 1 = 6 # (H_1, A_1)
            params_DQL_u['input_dim_stage2'] = params['input_dim_stage2'] + 1 # Ex. TAO: 7 + 1 = 8 # (H_2, A_2)

            params_DQL_f['input_dim_stage1'] = params['input_dim_stage1'] + 1 # Ex. TAO: 5 + 1 = 6 # (H_1, A_1)
            params_DQL_f['input_dim_stage2'] = params['input_dim_stage2'] + 1 # Ex. TAO: 7 + 1 = 8 # (H_2, A_2)

            start_time = time.time()  # Start time recording

            if params['trainingFixed']:
                trn_val_loss_tpl_DQL = DQlearning(tuple_train, tuple_val, params_DQL_f, config_number)                 
                config_dict['training_config']['DQL'] = params_DQL_f  
                # config_dict_training_config_DQL = params_DQL_f  

            else:
                trn_val_loss_tpl_DQL = DQlearning(tuple_train, tuple_val, params_DQL_u, config_number)                 
                config_dict['training_config']['DQL'] = params_DQL_u 
                # config_dict_training_config_DQL = params_DQL_u 


            end_time = time.time()  # End time recording
            print(f"Total time taken to run DQlearning: { end_time - start_time} seconds")
            # Store losses 
            losses_dict['DQL'][replication] = trn_val_loss_tpl_DQL 
            
        if params.get('run_surr_opt', True):

            params_DS_u['input_dim_stage1'] = params['input_dim_stage1']  # Ex. TAO: 5  # (H_1, A_1)
            params_DS_u['input_dim_stage2'] = params['input_dim_stage2']  # Ex. TAO: 7  # (H_2, A_2)

            params_DS_f['input_dim_stage1'] = params['input_dim_stage1']  # Ex. TAO: 5  # (H_1, A_1)
            params_DS_f['input_dim_stage2'] = params['input_dim_stage2']  # Ex. TAO: 7  # (H_2, A_2)

            start_time = time.time()  # Start time recording

            if params['trainingFixed']:
                trn_val_loss_tpl_DS, epoch_num_model_DS = surr_opt(tuple_train, tuple_val, params_DS_f, config_number)                 
                config_dict['training_config']['DS'] = params_DS_f  # Store config for DS
                # config_dict_training_config_DS = params_DS_f 


            else:
                trn_val_loss_tpl_DS, epoch_num_model_DS = surr_opt(tuple_train, tuple_val, params_DS_u, config_number)
                config_dict['training_config']['DS'] = params_DS_u  # Store config for DS
                # config_dict_training_config_DS = params_DS_u 


            end_time = time.time()  # End time recording
            print(f"Total time taken to run surr_opt: { end_time - start_time} seconds")
            # Append epoch model results from surr_opt
            epoch_num_model_lst.append(epoch_num_model_DS)
            # Store losses 
            losses_dict['DS'][replication] = trn_val_loss_tpl_DS 

        # eval_DTR
        print("Evaluation started")
        start_time = time.time()  # Start time recording
        
        if params['trainingFixed']:            
            V_replications, df_DQL, df_DS, df_Tao, param_W_DQL, param_W_DS = eval_DTR(V_replications, replication, df_DQL, df_DS, df_Tao, params_DQL_u, params_DS_u, tmp, config_number)
        else:             
            V_replications, df_DQL, df_DS, df_Tao, param_W_DQL, param_W_DS = eval_DTR(V_replications, replication, df_DQL, df_DS, df_Tao, params_DQL_f, params_DS_f, tmp, config_number)
        
        config_dict['testing_config']['DS'] = param_W_DS  # Store config for DS
        # config_dict_training_config_DS = param_W_DS

        config_dict['testing_config']['DQL'] = param_W_DQL  # Store config for DQL
        # config_dict_training_config_DQL = param_W_DQL


        end_time = time.time()  # End time recording 
        print(f"Total time taken to run eval_DTR: { end_time - start_time} seconds \n\n")
                
    return V_replications, df_DQL, df_DS, df_Tao, losses_dict, epoch_num_model_lst, config_dict


def run_training(config, config_fixed, config_updates, V_replications, config_number, replication_seed):
    torch.manual_seed(replication_seed)
    local_config = {**config, **config_updates}  # Create a local config that includes both global settings and updates
    
    # Execute the simulation function using updated settings
    V_replications, df_DQL, df_DS, df_Tao, losses_dict, epoch_num_model_lst, config_dict = simulations(V_replications, local_config, config_fixed, config_number)
    
    if not any(V_replications[key] for key in V_replications):
        warnings.warn("V_replications is empty. Skipping accuracy calculation.")
    else:
        VF_df_DQL, VF_df_DS, VF_df_Tao, VF_df_Beh = extract_value_functions_separate(V_replications)
        return VF_df_DQL, VF_df_DS, VF_df_Tao, VF_df_Beh, df_DQL, df_DS, df_Tao, losses_dict, epoch_num_model_lst, config_dict
    
 
    
# parallelized 

def run_training_with_params(params):

    config, config_fixed, current_config, V_replications, i, config_number = params
    return run_training(config, config_fixed, current_config, V_replications, config_number, replication_seed=i)



def run_grid_search(config, config_fixed, param_grid):
    # Initialize for storing results and performance metrics
    results = {}
    all_configurations = []

    # Initialize separate cumulative DataFrames for DQL and DS
    all_dfs_DQL = pd.DataFrame()  # DataFrames from each DQL run
    all_dfs_DS = pd.DataFrame()   # DataFrames from each DS run
    all_dfs_Tao = pd.DataFrame()   # DataFrames from each Tao run

    all_losses_dicts = []  # Losses from each run
    all_epoch_num_lists = []  # Epoch numbers from each run 
    grid_replications = 1

    # Collect all parameter combinations
    param_combinations = [dict(zip(param_grid.keys(), param)) for param in product(*param_grid.values())]

    num_workers = 8 # multiprocessing.cpu_count()
    print(f'{num_workers} available workers for ProcessPoolExecutor.')

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_params = {}
        #for current_config in param_combinations:
        for config_number, current_config in enumerate(param_combinations):
            for i in range(grid_replications):          
                print(f"Grid replication: {i}, for config number: {config_number}")  # Debug print for replication number
                V_replications = {
                    "V_replications_M1_pred": defaultdict(list),
                    "V_replications_M1_behavioral": [],
                }

                params = (config, config_fixed, current_config, V_replications, i, config_number)
                future = executor.submit(run_training_with_params, params)
                future_to_params[future] = (current_config, i, config_number)

        for future in concurrent.futures.as_completed(future_to_params):
            current_config, i, config_number = future_to_params[future]            
            performance_DQL, performance_DS, performance_Tao, performance_Beh, df_DQL, df_DS, df_Tao, losses_dict, epoch_num_model_lst, config_dict = future.result()
            
            print(f'Configuration {current_config}, replication {i} completed successfully.')

            all_configurations.append( (config_number+1, config_dict)  )   
            
            # Processing performance DataFrame for both methods
            performances_DQL = pd.DataFrame()
            performances_DQL = pd.concat([performances_DQL, performance_DQL], axis=0)

            performances_DS = pd.DataFrame()
            performances_DS = pd.concat([performances_DS, performance_DS], axis=0)

            performances_Tao = pd.DataFrame()
            performances_Tao = pd.concat([performances_Tao, performance_Tao], axis=0)

            performances_Beh = pd.DataFrame()
            performances_Beh = pd.concat([performance_Beh, performance_Beh], axis=0)

            # Update the cumulative DataFrame for DQL with the current DataFrame results
            all_dfs_DQL = pd.concat([all_dfs_DQL, df_DQL], axis=0, ignore_index=True)

            # Update the cumulative DataFrame for DS with the current DataFrame results
            all_dfs_DS = pd.concat([all_dfs_DS, df_DS], axis=0, ignore_index=True)

            # Update the cumulative DataFrame for DS with the current DataFrame results
            all_dfs_Tao = pd.concat([all_dfs_Tao, df_Tao], axis=0, ignore_index=True)

            all_losses_dicts.append(losses_dict)
            all_epoch_num_lists.append(epoch_num_model_lst)
            
            # Store and log average performance across replications for each configuration
            config_key = json.dumps(current_config, sort_keys=True)

            # performances is a DataFrame with columns 'DQL' and 'DS'
            performance_DQL_mean = performances_DQL["Method's Value fn."].mean()
            performance_DS_mean = performances_DS["Method's Value fn."].mean()
            performance_Tao_mean = performances_Tao["Method's Value fn."].mean()
            performance_Beh_mean = performances_Beh["Method's Value fn."].mean()

            # Calculating the standard deviation for "Method's Value fn."
            performance_DQL_std = performances_DQL["Method's Value fn."].std()
            performance_DS_std = performances_DS["Method's Value fn."].std()
            performance_Tao_std = performances_Tao["Method's Value fn."].std()
            performance_Beh_std = performances_Beh["Method's Value fn."].std()

            # Check if the configuration key exists in the results dictionary
            if config_key not in results:
                # If not, initialize it with dictionaries for each model containing the mean values
                results[config_key] = {
                    'DQL': {"Method's Value fn.": performance_DQL_mean, 
                            "Method's Value fn. SD": performance_DQL_std, 
                            },
                    'DS': {"Method's Value fn.": performance_DS_mean, 
                           "Method's Value fn. SD": performance_DS_std,
                           },
                    'Tao': {"Method's Value fn.": performance_Tao_mean, 
                           "Method's Value fn. SD": performance_Tao_std,
                           },
                    'Behavioral': {"Method's Value fn.": performance_Beh_mean, 
                           "Method's Value fn. SD": performance_Beh_std,
                           }    
                }
            else:
                # Update existing entries with new means
                results[config_key]['DQL'].update({
                    "Method's Value fn.": performance_DQL_mean,                                 
                    "Method's Value fn. SD": performance_DQL_std, 
                })
                results[config_key]['DS'].update({
                    "Method's Value fn.": performance_DS_mean,
                    "Method's Value fn. SD": performance_DS_std,
                })
                results[config_key]['Tao'].update({
                    "Method's Value fn.": performance_Tao_mean, 
                    "Method's Value fn. SD": performance_Tao_std,
                })
                results[config_key]['Behavioral'].update({
                    "Method's Value fn.": performance_Beh_mean, 
                    "Method's Value fn. SD": performance_Beh_std,  
                })                

            print("Performances for configuration: %s", config_key)
            print("performance_DQL_mean: %s", performance_DQL_mean)
            print("performance_DS_mean: %s", performance_DS_mean)
            print("performance_Tao_mean: %s", performance_Tao_mean)
            print("\n\n")
        

        
    folder = f"data/{config['job_id']}"
    save_simulation_data(all_dfs_DQL, all_dfs_DS, all_losses_dicts, all_epoch_num_lists, results, all_configurations, folder)
    load_and_process_data(config, folder)

        
        
        
        
        

# # Sequential version  

# def run_grid_search(config, param_grid):
#     # Initialize for storing results and performance metrics
#     results = {}
#     all_dfs_DQL = pd.DataFrame()  # DataFrames from each DQL run
#     all_dfs_DS = pd.DataFrame()   # DataFrames from each DS run
    
#     all_losses_dicts = []  # Losses from each run
#     all_epoch_num_lists = []  # Epoch numbers from each run 
#     grid_replications = 1

#     for params in product(*param_grid.values()):
#         current_config = dict(zip(param_grid.keys(), params))
#         performances = pd.DataFrame()

#         for i in range(grid_replications): 
#             V_replications = {
#                     "V_replications_M1_pred": defaultdict(list),
#                     "V_replications_M1_behavioral": [],
#                 }
#             #performance, df, losses_dict, epoch_num_model_lst = run_training(config, current_config, V_replications, replication_seed=i)
#             performance_DQL, performance_DS, df_DQL, df_DS, losses_dict, epoch_num_model_lst = run_training(config, current_config, 
#                                                                                                             V_replications, replication_seed=i)

#             #performances = pd.concat([performances, performance], axis=0)
#             # Processing performance DataFrame for both methods
#             performances_DQL = pd.DataFrame()
#             performances_DQL = pd.concat([performances_DQL, performance_DQL], axis=0)

#             performances_DS = pd.DataFrame()
#             performances_DS = pd.concat([performances_DS, performance_DS], axis=0)

            
#             #all_dfs = pd.concat([all_dfs, df], axis=0)
#             # Update the cumulative DataFrame for DQL with the current DataFrame results
#             all_dfs_DQL = pd.concat([all_dfs_DQL, df_DQL], axis=0, ignore_index=True)

#             # Update the cumulative DataFrame for DS with the current DataFrame results
#             all_dfs_DS = pd.concat([all_dfs_DS, df_DS], axis=0, ignore_index=True)

                
#             all_losses_dicts.append(losses_dict)
#             all_epoch_num_lists.append(epoch_num_model_lst)
            
               
                

#         # Store and log average performance across replications for each configuration
#         config_key = json.dumps(current_config, sort_keys=True)
        
#         # This assumes performances is a DataFrame with columns 'DQL' and 'DS'
#         performance_DQL_mean = performances_DQL["Method's Value fn."].mean()
#         performance_DS_mean = performances_DS["Method's Value fn."].mean()
        
#         behavioral_DQL_mean = performances_DQL["Behavioral Value fn."].mean()  # Assuming similar structure
#         behavioral_DS_mean = performances_DS["Behavioral Value fn."].mean()

#         # Check if the configuration key exists in the results dictionary
#         if config_key not in results:
#             # If not, initialize it with dictionaries for each model containing the mean values
#             results[config_key] = {
#                 'DQL': {"Method's Value fn.": performance_DQL_mean, 'Behavioral Value fn.': behavioral_DQL_mean},
#                 'DS': {"Method's Value fn.": performance_DS_mean, 'Behavioral Value fn.': behavioral_DS_mean}
#             }
#         else:
#             # Update existing entries with new means
#             results[config_key]['DQL'].update({
#                 "Method's Value fn.": performance_DQL_mean,
#                 'Behavioral Value fn.': behavioral_DQL_mean
#             })
#             results[config_key]['DS'].update({
#                 "Method's Value fn.": performance_DS_mean,
#                 'Behavioral Value fn.': behavioral_DS_mean
#             })
                
#         print("Performances for configuration: %s", config_key)
#         print("performance_DQL_mean: %s", performance_DQL_mean)
#         print("performance_DS_mean: %s", performance_DS_mean)
#         print("\n\n")
     

#     folder = f"data/{config['job_id']}"
#     save_simulation_data(all_dfs_DQL, all_dfs_DS, all_losses_dicts, all_epoch_num_lists, results, folder)
#     load_and_process_data(config, folder)
    
    
    
    
    
        
def main():

    # Load configuration and set up the device
    config = load_config()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['device'] = device    
    
    # Get the SLURM_JOB_ID from environment variables
    job_id = os.getenv('SLURM_JOB_ID')

    # If job_id is None, set it to the current date and time formatted as a string
    if job_id is None:
        job_id = datetime.now().strftime('%Y%m%d%H%M%S')  # Format: YYYYMMDDHHMMSS
    
    config['job_id'] = job_id
    print("Job ID: ", job_id) 

    # training_validation_prop = config['training_validation_prop']
    # train_size = int(training_validation_prop * config['sample_size'])
    print("config['sample_size'] : %d", config['sample_size'])   

    config_fixed = copy.deepcopy(config)
    
    # Define parameter grid for grid search
    # only uncomment those params which has at least 2 values

    # param_grid = {}

    param_grid = {
        'num_layers': [5, 12], # 2,4
        'n_epoch':[60, 150]
    }

    # param_grid = {
    #     'activation_function': ['none', 'elu'], # elu, relu, sigmoid, tanh, leakyrelu, none
    #     'batch_size': [200, 500, 800], # 50
    #     'optimizer_lr': [0.07, 0.007], # 0.1, 0.01, 0.07, 0.001
    #     'num_layers': [2, 4], # 2,4
    #     'n_epoch':[60, 150], # 150
    #     'surrogate_num': [1],
    #     'option_sur': [2],
    #     'hidden_dim_stage1': [20],
    #     'hidden_dim_stage2':[20]
    # }
    
    # Perform operations whose output should go to the file
    run_grid_search(config, config_fixed, param_grid)
    

class FlushFile:
    """File-like wrapper that flushes on every write."""
    def __init__(self, f):
        self.f = f

    def write(self, x):
        self.f.write(x)
        self.f.flush()  # Flush output after write

    def flush(self):
        self.f.flush()


# if __name__ == '__main__':
#     start_time = time.time()
#     main()
#     end_time = time.time()
#     print(f'Total time taken: {end_time - start_time:.2f} seconds')
   
    
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    
    # Record the start time
    start_time = time.time()
    start_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
    print(f'Start time: {start_time_str}')
    
    sys.stdout = FlushFile(sys.stdout)
    main()
    
    # Record the end time
    end_time = time.time()
    end_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))
    print(f'End time: {end_time_str}')
    
    # Calculate and log the total time taken
    total_time = end_time - start_time
    print(f'Total time taken: {total_time:.2f} seconds')


    
