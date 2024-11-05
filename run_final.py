import os
import sys
import json
import time
import copy
import pickle
import yaml
import torch
import numpy as np
import pandas as pd
import warnings
import multiprocessing
import matplotlib.pyplot as plt
from datetime import datetime
from itertools import product
from collections import Counter, defaultdict
from tqdm import tqdm
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split

import rpy2.robjects as ro
from rpy2.robjects import numpy2ri


# # Activate NumPy to R conversion
# numpy2ri.activate()

# # Load the R script to avoid dynamic loading
# ro.r.source("ACWL_tao.R")

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 0. Simulation utils
def load_config(file_path='config.yaml'):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
    

def extract_unique_treatment_values(df, columns_to_process, name): 
    
    
    unique_values = {}

    for key, cols in columns_to_process.items():
        unique_values[key] = {}
        
        for col in cols:
            all_values = [item for sublist in df[col] for item in sublist]
            unique_values[key][col] = set(all_values)

    log_message = f"\nUnique values for {name}:\n" + "\n".join(f"{k}: {v}" for k, v in unique_values.items()) + "\n"
    print(log_message)
    
    return unique_values


def print_in_box(title, content):
    # Calculate the maximum width of the content
    max_content_length = max([len(line) for line in content]) + 2  # Add padding for borders
    box_width = max(len(title), max_content_length) + 4  # Add padding for borders

    # Print the top border of the box
    print("=" * box_width)
    print(f"| {title.center(box_width - 4)} |")
    print("=" * box_width)

    # Print the content inside the box
    for line in content:
        print(f"| {line.ljust(box_width - 4)} |")

    # Print the bottom border of the box
    print("=" * box_width)


def save_simulation_data(all_performances_Beh, all_performances_DQL, all_performances_DS,  all_performances_Tao, all_dfs_DQL, all_dfs_DS, all_losses_dicts, all_epoch_num_lists, results, all_configurations, folder):
    # Check if the folder exists, if not, create it
    if not os.path.exists(folder):
        os.makedirs(folder)

        
    # print()
    # # Print Beh value functions across all configurations
    # print("Behavioral Value Functions across all simulations:")
    # for idx, beh_values in enumerate(all_performances_Beh):
    #     # Convert each NumPy array to a Python float and print the result as a simple list
    #     simple_list = [val.item() if isinstance(val, np.ndarray) or isinstance(val, np.float32) else val for val in beh_values]
    #     print(f"Configuration {idx + 1}: {simple_list}")


    # print()
    # # Print DQL value functions across all configurations
    # print("DQL Value Functions across all simulations:")
    # for idx, dql_values in enumerate(all_performances_DQL):
    #     print(f"Configuration {idx + 1}: {dql_values}")

    # print()

    # # Print DS value functions across all configurations
    # print("\nDS Value Functions across all simulations:")
    # for idx, ds_values in enumerate(all_performances_DS):
    #     print(f"Configuration {idx + 1}: {ds_values}")

    # print()

    # # Print Tao value functions across all configurations
    # print("\nTao Value Functions across all simulations:")
    # for idx, tao_values in enumerate(all_performances_Tao):
    #     print(f"Configuration {idx + 1}: {tao_values}")

    # print()  


    # Combine the content for all four sections into one list
    content = ["Behavioral Value Functions across all simulations:"]
    for idx, beh_values in enumerate(all_performances_Beh):
        simple_list = [val.item() if isinstance(val, np.ndarray) or isinstance(val, np.float32) else val for val in beh_values]
        content.append(f"Configuration {idx + 1}: {simple_list}")

    content.append("")  # Add empty line for spacing
    content.append("DQL Value Functions across all simulations:")
    for idx, dql_values in enumerate(all_performances_DQL):
        content.append(f"Configuration {idx + 1}: {dql_values}")

    content.append("")  # Add empty line for spacing
    content.append("DS Value Functions across all simulations:")
    for idx, ds_values in enumerate(all_performances_DS):
        content.append(f"Configuration {idx + 1}: {ds_values}")

    content.append("")  # Add empty line for spacing
    content.append("Tao Value Functions across all simulations:")
    for idx, tao_values in enumerate(all_performances_Tao):
        content.append(f"Configuration {idx + 1}: {tao_values}")

    # Print everything in one box
    print_in_box("Value Functions Across All Simulations", content)
    

    # Define paths for saving files
    df_path_DQL = os.path.join(folder, 'simulation_data_DQL.pkl')
    df_path_DS = os.path.join(folder, 'simulation_data_DS.pkl')
    losses_path = os.path.join(folder, 'losses_dicts.pkl')
    epochs_path = os.path.join(folder, 'epoch_num_lists.pkl')
    results_path = os.path.join(folder, 'simulation_results.pkl')
    configs_path = os.path.join(folder, 'simulation_configs.pkl')

    df_sim_VF_path_DQL = os.path.join(folder, 'sim_VF_data_DQL.pkl')
    df_sim_VF_path_DS = os.path.join(folder, 'sim_VF_data_DS.pkl')
    df_sim_VF_path_Tao = os.path.join(folder, 'sim_VF_data_Tao.pkl')
    df_sim_VF_path_Beh = os.path.join(folder, 'sim_VF_data_Beh.pkl')

    # Save each DataFrame with pickle
    with open(df_sim_VF_path_DQL, 'wb') as f:
        pickle.dump(all_performances_DQL, f)
    with open(df_sim_VF_path_DS, 'wb') as f:
        pickle.dump(all_performances_DS, f)
    with open(df_sim_VF_path_Tao, 'wb') as f:
        pickle.dump(all_performances_Tao, f)
    with open(df_sim_VF_path_Beh, 'wb') as f:
        pickle.dump(all_performances_Beh, f)

    # Save each DataFrame with pickle
    with open(df_path_DQL, 'wb') as f:
        pickle.dump(all_dfs_DQL, f)
    with open(df_path_DS, 'wb') as f:
        pickle.dump(all_dfs_DS, f)
    
    # Save lists and dictionaries with pickle
    with open(losses_path, 'wb') as f:
        pickle.dump(all_losses_dicts, f)
    with open(epochs_path, 'wb') as f:
        pickle.dump(all_epoch_num_lists, f)
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    with open(configs_path, 'wb') as f:
        pickle.dump(all_configurations, f)

    print("Data saved successfully in the folder: %s", folder)


def save_results_to_dataframe(results, folder):
    # Expand the data dictionary to accommodate DQL and DS results separately
    data = {
        "Configuration": [],
        "Model": [],
        "Behavioral Value fn.": [],
        "Method's Value fn.": []
    }

    # Iterate through the results dictionary
    for config_key, performance in results.items():
        # Each 'performance' item contains a further dictionary for 'DQL' and 'DS'
        for model, metrics in performance.items():
            data["Configuration"].append(config_key)
            data["Model"].append(model)  # Keep track of which model (DQL or DS)
            # Safely extract metric values for each model
            data["Behavioral Value fn."].append(metrics.get("Behavioral Value fn.", None))
            data["Method's Value fn."].append(metrics.get("Method's Value fn.", None))

    # Create DataFrame from the structured data
    df = pd.DataFrame(data)

    # You might want to sort by 'Method's Value fn.' or another relevant column, if NaNs are present handle them appropriately
    df.sort_values(by=["Configuration", "Model"], ascending=[True, False], inplace=True)

    # Ensure the folder exists
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Save the DataFrame to a CSV file
    df.to_csv(f'{folder}/configurations_performance.csv', index=False)

    return df




def load_and_process_data(params, folder):
    # Check if the folder exists, if not, create it
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    # Define paths to the files for both DQL and DS
    df_path_DQL = os.path.join(folder, 'simulation_data_DQL.pkl')
    df_path_DS = os.path.join(folder, 'simulation_data_DS.pkl')
    losses_path = os.path.join(folder, 'losses_dicts.pkl')
    epochs_path = os.path.join(folder, 'epoch_num_lists.pkl')
    results_path = os.path.join(folder, 'simulation_results.pkl')
    configs_path = os.path.join(folder, 'simulation_configs.pkl')

    # Load DataFrames
    with open(df_path_DQL, 'rb') as f:
        global_df_DQL = pickle.load(f)
    with open(df_path_DS, 'rb') as f:
        global_df_DS = pickle.load(f)
        
    # Load lists and dictionaries with pickle
    with open(losses_path, 'rb') as f:
        all_losses_dicts = pickle.load(f)
    with open(epochs_path, 'rb') as f:
        all_epoch_num_lists = pickle.load(f)
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    with open(configs_path, 'rb') as f:
        configs = pickle.load(f)
    
    # Extract and process unique values for both DQL and DS
    columns_to_process = {
        'Predicted': ['Predicted_A1', 'Predicted_A2'],
    }

    unique_values_DQL = extract_unique_treatment_values(global_df_DQL, columns_to_process, name = "DQL")
    unique_values_DS = extract_unique_treatment_values(global_df_DS, columns_to_process, name = "DS")

    print("unique_values_DQL: ", unique_values_DQL)
    print("unique_values_DS: ", unique_values_DS)

    train_size = int(params['training_validation_prop'] * params['sample_size'])

    # Process and plot results from all simulations
    for i, method_losses_dicts in enumerate(all_losses_dicts):
        run_name = f"run_trainVval_{i}"
        selected_indices = [i for i in range(params['num_replications'])]  

        # Check if method_losses_dicts['DQL'] is not empty before plotting
        if method_losses_dicts.get('DQL'):
            plot_simulation_Qlearning_losses_in_grid(selected_indices, method_losses_dicts['DQL'], train_size, run_name, folder)

        # Check if method_losses_dicts['DS'] is not empty before plotting
        if method_losses_dicts.get('DS'):
            plot_simulation_surLoss_losses_in_grid(selected_indices, method_losses_dicts['DS'], train_size, run_name, folder)


    # Print results for each configuration
    print("\n\n")
    # print("configs: ", json.dumps(configs, indent=4))

    # Custom serializer to handle non-serializable objects like 'device'
    def custom_serializer(obj):
        if isinstance(obj, torch.device):  # Handle torch.device type, convert to string
            return str(obj)
        raise TypeError(f"Type {type(obj)} not serializable")

    # Pretty-print the configs with a custom serializer
    print("configs: ", json.dumps(configs, indent=4, default=custom_serializer))

    print("\n\n")
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<--------------------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print("<<<<<<<<<<<<<<<<<<<<<<<<-----------------------FINAL RESULTS------------------------>>>>>>>>>>>>>>>>>>>>>>>")
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<--------------------------------------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>")
    for config_key, performance in results.items():
        print("Configuration: %s\nAverage Performance:\n %s\n", config_key, json.dumps(performance, indent=4))
    
    # Call the function to plot value functions
    df = save_results_to_dataframe(results, folder)


    
        

# 1. DGP utils

def A_sim(matrix_pi, stage):
    N, K = matrix_pi.shape  # sample size and treatment options
    if N <= 1 or K <= 1:
        warnings.warn("Sample size or treatment options are insufficient! N: %d, K: %d", N, K)
        raise ValueError("Sample size or treatment options are insufficient!")
    if torch.any(matrix_pi < 0):
        warnings.warn("Treatment probabilities should not be negative!")
        raise ValueError("Treatment probabilities should not be negative!")

    # Normalize probabilities to add up to 1 and simulate treatment A for each row
    pis = matrix_pi.sum(dim=1, keepdim=True)
    probs = matrix_pi / pis
    A = torch.multinomial(probs, 1).squeeze()

    if stage == 1:
        col_names = ['pi_10', 'pi_11', 'pi_12']
    else:
        col_names = ['pi_20', 'pi_21', 'pi_22']
    
    probs_dict = {name: probs[:, idx] for idx, name in enumerate(col_names)}
    
    
    return {'A': A, 'probs': probs_dict}

def transform_Y(Y1, Y2):
    """
    Adjusts Y1 and Y2 values to ensure they are non-negative.
    """
    # Identify the minimum value among Y1 and Y2, only if they are negative
    min_negative_Y = torch.min(torch.cat([Y1, Y2])).item()
    if min_negative_Y < 0:
        Y1_trans = Y1 - min_negative_Y + 1
        Y2_trans = Y2 - min_negative_Y + 1
    else:
        Y1_trans = Y1
        Y2_trans = Y2

    return Y1_trans, Y2_trans



def M_propen(A, Xs, stage):
    """Estimate propensity scores using logistic or multinomial regression."""
    
    if isinstance(A, torch.Tensor):
        A = A.cpu().numpy()  # Convert to CPU and then to NumPy
    A = A.reshape(-1, 1)  # Ensure A is a column vector
    
    if isinstance(Xs, torch.Tensor):
        Xs = Xs.cpu().numpy()  # Convert tensor to NumPy if necessary

    # A = np.asarray(A).reshape(-1, 1)
    if A.shape[1] != 1:
        raise ValueError("Cannot handle multiple stages of treatments together!")
    if A.shape[0] != Xs.shape[0]:
        print("A.shape, Xs.shape: ", A.shape, Xs.shape)
        raise ValueError("A and Xs do not match in dimension!")
    if len(np.unique(A)) <= 1:
        raise ValueError("Treatment options are insufficient!")

    # Handle multinomial case using Logistic Regression
    # encoder = OneHotEncoder(sparse_output=False)  # Updated parameter name
    # A_encoded = encoder.fit_transform(A)
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)

    # Suppressing warnings from the solver, if not converged
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(Xs, A.ravel())

    # Predicting probabilities
    s_p = model.predict_proba(Xs)

    # similar to Laplace smoothing or additive smoothing used in statistical models to handle zero counts.
    #  in handling categorical data distributions, like in Naive Bayes classifiers, to avoid zero probability issues.

    # Add a small constant to all probabilities
    s_p += 1e-8
    # Normalize the rows to sum to 1
    s_p = s_p / s_p.sum(axis=1, keepdims=True)

    # print("s_p: ===============>>>>>>>>>>>>>>>>>>>>> ", s_p, "\n\n")
    print("Probs matrix Minimum of -> Max values over each rows: ", np.min(np.max(s_p, axis=1)) , "\n")

    if stage == 1:
        col_names = ['pi_10', 'pi_11', 'pi_12']
    else:
        col_names = ['pi_20', 'pi_21', 'pi_22']
        
    #probs_df = pd.DataFrame(s_p, columns=col_names)
    #probs_df = {name: s_p[:, idx] for idx, name in enumerate(col_names)}
    probs_dict = {name: torch.tensor(s_p[:, idx], dtype=torch.float32) for idx, name in enumerate(col_names)}

    return probs_dict


# Neural networks utils
def initialize_nn(params, stage):

    nn = NNClass(
        input_dim=params[f'input_dim_stage{stage}'],
        hidden_dim=params[f'hidden_dim_stage{stage}'],
        output_dim=params[f'output_dim_stage{stage}'],
        num_networks=params['num_networks'],
        dropout_rate=params['dropout_rate'],
        activation_fn_name=params['activation_function'],
        num_hidden_layers=params['num_layers'] - 1,  # num_layers is the number of hidden layers
        add_ll_batch_norm=params['add_ll_batch_norm']

    ).to(params['device'])
    return nn



def batches(N, batch_size, seed=0):
    # Set the seed for PyTorch random number generator for reproducibility
    # torch.manual_seed(seed)
    
    # Create a tensor of indices from 0 to N-1
    indices = torch.arange(N)
    
    # Shuffle the indices
    indices = indices[torch.randperm(N)]
    
    # Yield batches of indices
    for start_idx in range(0, N, batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size]
        yield batch_indices

# The Identity class acts as a no-operation (no-op) activation function.
#  It simply returns the input it receives without any modification. 
class Identity(nn.Module):
    def forward(self, x):
        return x
    
class NNClass(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_networks, dropout_rate, activation_fn_name, num_hidden_layers, add_ll_batch_norm):
        super(NNClass, self).__init__()
        self.networks = nn.ModuleList()

        # Map the string name to the actual activation function class
        activation_fn_name = activation_fn_name.lower()
        if activation_fn_name == 'elu':
            activation_fn = nn.ELU
        elif activation_fn_name == 'relu':
            activation_fn = nn.ReLU
        elif activation_fn_name == 'sigmoid':
            activation_fn = nn.Sigmoid
        elif activation_fn_name == 'tanh':
            activation_fn = nn.Tanh
        elif activation_fn_name == 'leakyrelu':
            activation_fn = nn.LeakyReLU
        elif activation_fn_name == 'none': # Check for 'none' and use the Identity class
            activation_fn = Identity
        else:
            raise ValueError(f"Unsupported activation function: {activation_fn_name}")

        for _ in range(num_networks):
            layers = []
            layers.append(nn.Linear(input_dim, hidden_dim))
            if activation_fn is not Identity:  # Only add activation if it's not Identity
                layers.append(activation_fn())            
            layers.append(nn.Dropout(dropout_rate))
            
            for _ in range(num_hidden_layers):  # Adjusting the hidden layers count
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if activation_fn is not Identity:  # Only add activation if it's not Identity
                    layers.append(activation_fn())
                layers.append(nn.Dropout(dropout_rate))
                
            layers.append(nn.Linear(hidden_dim, output_dim))
            if add_ll_batch_norm:
                    layers.append(nn.BatchNorm1d(output_dim))
            
            network = nn.Sequential(*layers)
            self.networks.append(network)

    def forward(self, x):
        outputs = []
        for network in self.networks:
            outputs.append(network(x))
        return outputs

    def he_initializer(self):
        for network in self.networks:
            for layer in network:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.constant_(layer.bias, 0)  # Biases can be initialized to zero

    def reset_weights(self):
        for network in self.networks:
            for layer in network:
                if isinstance(layer, nn.Linear):
                    nn.init.constant_(layer.weight, 0.1)
                    nn.init.constant_(layer.bias, 0.0)

                   


# 2. plotting and summary utils


def plot_v_values(v_dict, num_replications, train_size):

    # Plotting all categories of V values
    plt.figure(figsize=(12, 6))
    for category, values in v_dict.items():
        plt.plot(range(1, num_replications + 1), values, 'o-', label=f'{category} Value function')
    plt.xlabel('Replications (Total: {})'.format(num_replications))
    plt.ylabel('Value function')
    plt.title('Value functions for {} Test Replications (Training Size: {})'.format(num_replications, train_size))
    plt.grid(True)
    plt.legend()
    plt.show()

def abbreviate_config(config):
    abbreviations = {
        "activation_function": "AF",
        "batch_size": "BS",
        "learning_rate": "LR",
        "num_layers": "NL"
    }
    abbreviated_config = {abbreviations[k]: v for k, v in config.items()}
    return str(abbreviated_config)
    
def plot_value_functions(results, folder):
    data = {
        "Configuration": [],
        "Value Function": []
    }

    for config_key, performance in results.items():
        config_dict = json.loads(config_key)
        abbreviated_config = abbreviate_config(config_dict)
        data["Configuration"].append(abbreviated_config)
        data["Value Function"].append(performance["Method's Value fn."])

    df = pd.DataFrame(data)
    
    # Sort the DataFrame by 'Value Function' in descending order
    df = df.sort_values(by="Value Function", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(df["Configuration"], df["Value Function"], color='skyblue')
    plt.xlabel("Value Function")
    plt.title("Value Function of Each Method")
    plt.yticks(rotation=0)  # Rotate configuration labels to vertical
    plt.tight_layout()
    plt.savefig(f'{folder}/value_function_plot.png')
    plt.close()
    

def plot_epoch_frequency(epoch_num_model_lst, n_epoch, run_name, folder='data'):
    """
    Plots a bar diagram showing the frequency of each epoch number where the model was saved.

    Args:
        epoch_num_model_lst (list of int): List containing the epoch numbers where models were saved.
        n_epoch (int): Total number of epochs for reference in the title.
    """
    # Count the occurrences of each number in the list
    frequency_counts = Counter(epoch_num_model_lst)

    # Separate the keys and values for plotting
    keys = sorted(frequency_counts.keys())
    values = [frequency_counts[key] for key in keys]

    # Create a bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(keys, values, color='skyblue')

    # Add title and labels
    plt.title(f'Bar Diagram of Epoch Numbers: n_epoch={n_epoch}')
    plt.xlabel('Epoch Number')
    plt.ylabel('Frequency')

    # Show the plot
    plt.grid(True)

    # Save the plot
    plot_filename = os.path.join(folder, f"{run_name}.png")
    plt.savefig(plot_filename)
    print(f"plot_epoch_frequency Plot saved as: {plot_filename}")
    plt.close()  # Close the plot to free up memory



def plot_simulation_surLoss_losses_in_grid(selected_indices, losses_dict, train_size, run_name, folder, cols=3):
    # Ensure the directory exists
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Calculate the number of rows needed based on the number of selected indices and desired number of columns
    rows = len(selected_indices) // cols + (len(selected_indices) % cols > 0)

    # Create a figure and a set of subplots
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))  # Adjust figure size as needed
    fig.suptitle(f'Training and Validation Loss for Selected Simulations @ train_size = {train_size}')

    # Flatten the axes array for easy indexing, in case of a single row or column
    axes = axes.flatten()
    
    for i, idx in enumerate(selected_indices):
        train_loss, val_loss = losses_dict[idx]

        # Plot on the ith subplot
        axes[i].plot(train_loss, label='Training')
        axes[i].plot(val_loss, label='Validation')
        axes[i].set_title(f'Simulation {idx}')
        axes[i].set_xlabel('Epochs')
        axes[i].set_ylabel('Loss')
        axes[i].legend()

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout to make room for the subtitle

    # Save the plot
    # Create the directory if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        
    plot_filename = os.path.join(folder, f"{run_name}_directSearch.png")
    plt.savefig(plot_filename)
    print(f"TrainVval Plot for Direct search saved as: {plot_filename}")
    plt.close(fig)  # Close the plot to free up memory


def plot_simulation_Qlearning_losses_in_grid(selected_indices, losses_dict, train_size, run_name, folder, cols=3):

    all_losses = {
        'train_losses_stage1': {},
        'train_losses_stage2': {},
        'val_losses_stage1': {},
        'val_losses_stage2': {}
    }

    # Iterate over each simulation and extract losses
    for simulation, losses in losses_dict.items():
        train_losses_stage1, train_losses_stage2, val_losses_stage1, val_losses_stage2 = losses

        all_losses['train_losses_stage1'][simulation] = train_losses_stage1
        all_losses['train_losses_stage2'][simulation] = train_losses_stage2
        all_losses['val_losses_stage1'][simulation] = val_losses_stage1
        all_losses['val_losses_stage2'][simulation] = val_losses_stage2

    
    rows = len(selected_indices) // cols + (len(selected_indices) % cols > 0)
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    fig.suptitle(f'Training and Validation Loss for Selected Simulations @ train_size = {train_size}')

    axes = axes.flatten()

    for i, idx in enumerate(selected_indices):
        # Check if the replication index exists in the losses for each type
        if idx in all_losses['train_losses_stage1']:
            axes[i].plot(all_losses['train_losses_stage1'][idx], label='Training Stage 1', linestyle='--')
            axes[i].plot(all_losses['val_losses_stage1'][idx], label='Validation Stage 1', linestyle='-.')
            axes[i].plot(all_losses['train_losses_stage2'][idx], label='Training Stage 2', linestyle='--')
            axes[i].plot(all_losses['val_losses_stage2'][idx], label='Validation Stage 2', linestyle='-.')
            axes[i].set_title(f'Simulation {idx}')
            axes[i].set_xlabel('Epochs')
            axes[i].set_ylabel('Loss')
            axes[i].legend()
        else:
            axes[i].set_title(f'Simulation {idx} - Data not available')
            axes[i].axis('off')

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Save the plot
    # Create the directory if it doesn't exist
    os.makedirs(folder, exist_ok=True)
    plot_filename = os.path.join(folder, f"{run_name}_deepQlearning.png")
    plt.savefig(plot_filename)
    print(f"TrainVval Plot for Deep Q Learning saved as: {plot_filename}")
    plt.close(fig)  # Close the plot to free up memory


# def extract_value_functions_separate(V_replications):

#     # Process predictive values
#     pred_data = V_replications.get('V_replications_M1_pred', defaultdict(list))

#     # Process behavioral values (assuming these are shared across models)
#     behavioral_data = V_replications.get('V_replications_M1_behavioral', [])

#     # Create DataFrames for each method
#     VF_df_DQL = pd.DataFrame({
#         "Method's Value fn.": pred_data.get('DQL', [None] * len(behavioral_data)),
#     })

#     VF_df_DS = pd.DataFrame({
#         "Method's Value fn.": pred_data.get('DS', [None] * len(behavioral_data)),
#     })


#     VF_df_Tao = pd.DataFrame({
#         "Method's Value fn.": pred_data.get('Tao', [None] * len(behavioral_data)),
#     })   

#     VF_df_Beh = pd.DataFrame({
#         "Method's Value fn.": behavioral_data,
#     })       
    
#     return VF_df_DQL, VF_df_DS, VF_df_Tao, VF_df_Beh



def extract_value_functions_separate(V_replications):

    # Process predictive values
    pred_data = V_replications.get('V_replications_M1_pred', defaultdict(list))

    # Process behavioral values 
    behavioral_data = V_replications.get('V_replications_M1_behavioral', [])


    # Helper function to ensure all tensors are converted to CPU and then to numpy
    def to_numpy(tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.cpu().numpy()  # Move to CPU and convert to NumPy
        return tensor  # If it's not a tensor, return as is

    # Create DataFrames for each method
    VF_df_DQL = pd.DataFrame({
        "Method's Value fn.": [to_numpy(val) for val in pred_data.get('DQL', [None] * len(behavioral_data))],
    })

    VF_df_DS = pd.DataFrame({
        "Method's Value fn.": [to_numpy(val) for val in pred_data.get('DS', [None] * len(behavioral_data))],
    })

    VF_df_Tao = pd.DataFrame({
        "Method's Value fn.": [to_numpy(val) for val in pred_data.get('Tao', [None] * len(behavioral_data))],
    })   

    VF_df_Beh = pd.DataFrame({
        "Method's Value fn.": [to_numpy(val) for val in behavioral_data],
    })       


    return VF_df_DQL, VF_df_DS, VF_df_Tao, VF_df_Beh




# 3. Loss function and surrogate opt utils

def compute_phi(x, option):
    if option == 1:
        return 1 + torch.tanh(5*x)
    elif option == 2:         
        return 1 + x / (1 + torch.abs(x))
    elif option == 3:
        return 1 + x / torch.sqrt(1 + x ** 2)
    elif option == 4:        
        return 1 + 2 * torch.atan(torch.pi * x / 2) / torch.pi
    elif option == 5:
        return torch.where(x >= 0, torch.tensor(1.0), torch.tensor(0.0))
    else:
        warnings.warn("Invalid phi option: %s", option)
        raise ValueError("Invalid phi option")


def gamma_function_old_vec(a, b, A, option):
    a = a.to(device)
    b = b.to(device)

    phi_a = compute_phi(a, option)
    phi_b = compute_phi(b, option)
    phi_b_minus_a = compute_phi(b - a, option)
    phi_a_minus_b = compute_phi(a - b, option)
    phi_neg_a = compute_phi(-a, option)
    phi_neg_b = compute_phi(-b, option)

    gamma = torch.where(A == 1, phi_a * phi_b,
                        torch.where(A == 2, phi_b_minus_a * phi_neg_a,
                                    torch.where(A == 3, phi_a_minus_b * phi_neg_b,
                                                torch.tensor(0.0).to(device))))
    return gamma


def compute_gamma(a, b, option):
    # Assume a and b are already tensors, check if they need to be sent to a specific device and ensure they have gradients if required
    a = a.detach().requires_grad_(True)
    b = b.detach().requires_grad_(True)

    # asymmetric
    if option == 1:
        result = ((torch.exp(a + b) - 1) / ((1 + torch.exp(a)) * (1 + torch.exp(b))) ) +  ( 1 / (1 + torch.exp(a) + torch.exp(b)))
    # symmetric
    elif option == 2:
        result = (torch.exp(a + b) * ((a * (torch.exp(b) - 1))**2 + (torch.exp(a) - 1) * (-torch.exp(a) + (torch.exp(b) - 1) * (torch.exp(a) - torch.exp(b) + b)))) / ((torch.exp(a) - 1)**2 * (torch.exp(b) - 1)**2 * (torch.exp(a) - torch.exp(b)))
    else:
        result = None
    return result


def gamma_function_new_vec(a, b, A, option):
    # a, b, and A are torch tensors and move them to the specified device
    a = torch.tensor(a, dtype=torch.float32, requires_grad=True).to(device)
    b = torch.tensor(b, dtype=torch.float32, requires_grad=True).to(device)

    # a = torch.tensor(a, dtype=torch.float32).to(device)
    # b = torch.tensor(b, dtype=torch.float32).to(device)
    A = torch.tensor(A, dtype=torch.int32).to(device)

    # Apply compute_gamma_vectorized across the entire tensors based on A
    result_1 = compute_gamma(a, b, option)
    result_2 = compute_gamma(b - a, -a, option)
    result_3 = compute_gamma(a - b, -b, option)

    gamma = torch.where(A == 1, result_1,
                        torch.where(A == 2, result_2,
                                    torch.where(A == 3, result_3,
                                                torch.tensor(0.0).to(device) )))

    return gamma


def main_loss_gamma(stage1_outputs, stage2_outputs, A1, A2, Ci, option, surrogate_num):

    if surrogate_num == 1:
        # # surrogate 1
        gamma_stage1 = gamma_function_old_vec(stage1_outputs[:, 0], stage1_outputs[:, 1], A1.int(), option)
        gamma_stage2 = gamma_function_old_vec(stage2_outputs[:, 0], stage2_outputs[:, 1], A2.int(), option)
    else:
        # surrogate 2 - contains symmetric and non symmetic cases
        gamma_stage1 = gamma_function_new_vec(stage1_outputs[:, 0], stage1_outputs[:, 1], A1.int(), option)
        gamma_stage2 = gamma_function_new_vec(stage2_outputs[:, 0], stage2_outputs[:, 1], A2.int(), option)

    loss = -torch.mean(Ci * gamma_stage1 * gamma_stage2)

    return loss



def process_batches(model1, model2, data, params, optimizer, is_train=True):
    batch_size = params['batch_size']
    total_loss = 0
    num_batches = (data['input1'].shape[0] + batch_size - 1) // batch_size 
    # print(" data['input1'].shape[0], batch_size, num_batches: -------->>>>>>>>>>>>>> ", data['input1'].shape, batch_size, num_batches)

    # print("num_batches DS: =============> ", num_batches)

    if is_train:
        model1.train()
        model2.train()
    else:
        model1.eval()
        model2.eval()

    for batch_idx in batches(data['input1'].shape[0], batch_size):
        batch_data = {k: v[batch_idx].to(params['device']) for k, v in data.items()}

        with torch.set_grad_enabled(is_train):
            outputs_stage1 = model1(batch_data['input1'])
            outputs_stage2 = model2(batch_data['input2'])

            outputs_stage1 = torch.stack(outputs_stage1, dim=1).squeeze()
            outputs_stage2 = torch.stack(outputs_stage2, dim=1).squeeze()

            loss = main_loss_gamma(outputs_stage1, outputs_stage2, batch_data['A1'], batch_data['A2'], 
                                   batch_data['Ci'], option=params['option_sur'], surrogate_num=params['surrogate_num'])
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                # Gradient Clipping (to prevent exploding gradients)
                if params['gradient_clipping']:
                    torch.nn.utils.clip_grad_norm_(model1.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(model2.parameters(), max_norm=1.0)
                
                optimizer.step()

        total_loss += loss.item()
    avg_loss = total_loss / num_batches
    return avg_loss


def initialize_and_prepare_model(stage, params):
    model = initialize_nn(params, stage).to(params['device'])
    
    # Check for the initializer type in params and apply accordingly
    if params['initializer'] == 'he':
        model.he_initializer()  # He initialization (aka Kaiming initialization)
    else:
        model.reset_weights()  # Custom reset weights to a specific constant eg. 0.1
    
    return model




def initialize_optimizer_and_scheduler(nn_stage1, nn_stage2, params):
    # Combine parameters from both models
    combined_params = list(nn_stage1.parameters()) + list(nn_stage2.parameters())

    # Select optimizer based on params
    if params['optimizer_type'] == 'adam':
        optimizer = optim.Adam(combined_params, lr=params['optimizer_lr'])
    elif params['optimizer_type'] == 'rmsprop':
        optimizer = optim.RMSprop(combined_params, lr=params['optimizer_lr'], weight_decay=params['optimizer_weight_decay'])
    else:
        warnings.warn("No valid optimizer type found in params['optimizer_type'], defaulting to Adam.")
        optimizer = optim.Adam(combined_params, lr=params['optimizer_lr'])  # Default to Adam if none specified

    # Initialize scheduler only if use_scheduler is True
    scheduler = None
    if params.get('use_scheduler', False):  # Defaults to False if 'use_scheduler' is not in params
        if params['scheduler_type'] == 'reducelronplateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.01, patience=10)
        elif params['scheduler_type'] == 'steplr':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params['scheduler_step_size'], gamma=params['scheduler_gamma'])
        elif params['scheduler_type'] == 'cosineannealing':
            T_max = (params['sample_size'] // params['batch_size']) * params['n_epoch']         # need to use the updated sample size 
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=0.0001)
        else:
            warnings.warn("No valid scheduler type found in params['scheduler_type'], defaulting to StepLR.")
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params['scheduler_step_size'], gamma=params['scheduler_gamma'])  # Default to StepLR if none specified

    return optimizer, scheduler


def update_scheduler(scheduler, params, val_loss=None):

    if scheduler is None:
        warnings.warn("Scheduler is not initialized but update_scheduler was called.")
        return
    
    # Check the type of scheduler and step accordingly
    if params['scheduler_type'] == 'reducelronplateau':
        # ReduceLROnPlateau expects a metric, usually the validation loss, to step
        if val_loss is not None:
            scheduler.step(val_loss)
        else:
            warnings.warn("Validation loss required for ReduceLROnPlateau but not provided.")
    else:
        # Other schedulers like StepLR or CosineAnnealingLR do not use the validation loss
        scheduler.step()



# 3. Q learning utils

def process_batches_DQL(model, inputs, actions, targets, params, optimizer, is_train=True):
    batch_size = params['batch_size']
    total_loss = 0
    num_batches = (inputs.shape[0] + batch_size - 1) // batch_size

    # print("num_batches DQL: =============> ", num_batches)
    # print(" inputs.shape[0], batch_size, num_batches: -------->>>>>>>>>>>>>> ", inputs.shape, batch_size, num_batches)

    if is_train:
        model.train()
    else:
        model.eval()

    for batch_idx in batches(inputs.shape[0], batch_size):

        with torch.set_grad_enabled(is_train):
                        
            batch_idx = batch_idx.to(device)
            inputs_batch = torch.index_select(inputs, 0, batch_idx).to(device)
            actions_batch = torch.index_select(actions, 0, batch_idx).to(device)
            targets_batch = torch.index_select(targets, 0, batch_idx).to(device)
            combined_inputs = torch.cat((inputs_batch, actions_batch.unsqueeze(-1)), dim=1)
            # print("combined_inputs shape ================================*****************: ", combined_inputs.shape)
            outputs = model(combined_inputs)
            loss = F.mse_loss(torch.cat(outputs, dim=0).view(-1), targets_batch)
            
            if is_train:
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

        total_loss += loss.item()
    avg_loss = total_loss / num_batches
    return avg_loss





def train_and_validate(config_number, model, optimizer, scheduler, train_inputs, train_actions, train_targets, val_inputs, val_actions, val_targets, params, stage_number):

    batch_size, device, n_epoch, sample_size = params['batch_size'], params['device'], params['n_epoch'], params['sample_size']
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_model_params = None
    epoch_num_model = 0

    for epoch in range(n_epoch):

        # print(" train_inputs.shape, val_inputs.shape: -------->>>>>>>>>>>>>> ", train_inputs.shape, val_inputs.shape)
        
        train_loss = process_batches_DQL(model, train_inputs, train_actions, train_targets, params, optimizer, is_train=True)
        train_losses.append(train_loss)

        val_loss = process_batches_DQL(model, val_inputs, val_actions, val_targets, params, optimizer, is_train=False)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            epoch_num_model = epoch
            best_val_loss = val_loss
            best_model_params = model.state_dict()

        # Update the scheduler with the current epoch's validation loss
        update_scheduler(scheduler, params, val_loss)

    # Define file paths for saving models
    model_dir = f"models/{params['job_id']}"
    # Check if the directory exists, if not, create it
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)     
        print(f"Directory '{model_dir}' created successfully.")
        
    # Save the best model parameters after all epochs
    if best_model_params is not None:
        model_path = os.path.join(model_dir, f'best_model_stage_Q_{stage_number}_{sample_size}_config_number_{config_number}.pt')
        torch.save(best_model_params, model_path)
        
    return train_losses, val_losses, epoch_num_model


def initialize_model_and_optimizer(params, stage):
    nn = initialize_nn(params, stage).to(device)
    
        
    # Select optimizer based on params
    if params['optimizer_type'] == 'adam':
        optimizer = optim.Adam(nn.parameters(), lr=params['optimizer_lr'])
    elif params['optimizer_type'] == 'rmsprop':
        optimizer = optim.RMSprop(nn.parameters(), lr=params['optimizer_lr'], weight_decay=params['optimizer_weight_decay'])
    else:
        warnings.warn("No valid optimizer type found in params['optimizer_type'], defaulting to Adam.")
        optimizer = optim.Adam(nn.parameters(), lr=params['optimizer_lr'])  # Default to Adam if none specified

    
    # Initialize scheduler only if use_scheduler is True
    scheduler = None
    if params.get('use_scheduler', False):  # Defaults to False if 'use_scheduler' is not in params
        if params['scheduler_type'] == 'reducelronplateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.01, patience=10)
        elif params['scheduler_type'] == 'steplr':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params['scheduler_step_size'], gamma=params['scheduler_gamma'])
        elif params['scheduler_type'] == 'cosineannealing':
            T_max = (params['sample_size'] // params['batch_size']) * params['n_epoch']
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=0.0001)
        else:
            warnings.warn("No valid scheduler type found in params['scheduler_type'], defaulting to StepLR.")
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params['scheduler_step_size'], 
                                                  gamma=params['scheduler_gamma'])  # Default to StepLR if none specified    
    
    return nn, optimizer, scheduler




def evaluate_model_on_actions(model, inputs, action_t):
    actions_list = [1, 2, 3]
    outputs_list = []
    for action_value in actions_list:
        action_tensor = torch.full_like(action_t, action_value).unsqueeze(-1)
        combined_inputs = torch.cat((inputs, action_tensor), dim=1).to(device)
        with torch.no_grad():
            outputs = model(combined_inputs)
        outputs_list.append(outputs[0])

    max_outputs, _ = torch.max(torch.cat(outputs_list, dim=1), dim=1)
    return max_outputs










# 5. Eval fn utils

def compute_test_outputs(nn, test_input, A_tensor, params, is_stage1=True):
    with torch.no_grad():
        if params['f_model'] == "surr_opt":
            # Perform the forward pass
            test_outputs_i = nn(test_input)

            # Directly stack the required outputs and perform computations in a single step
            test_outputs = torch.stack(test_outputs_i[:2], dim=1).squeeze()

            # Compute treatment assignments directly without intermediate variables
            test_outputs = torch.stack([
                torch.zeros_like(test_outputs[:, 0]),
                -test_outputs[:, 0],
                -test_outputs[:, 1]
            ], dim=1)
        else:
            # Modify input for each action and perform a forward pass
            input_tests = [
                torch.cat((test_input, torch.full_like(A_tensor, i).unsqueeze(-1)), dim=1).to(params['device'])
                for i in range(1, 4)  # Assuming there are 3 actions
            ]

            # Forward pass for each modified input and stack the results
            test_outputs = torch.stack([
                nn(input_stage)[0] for input_stage in input_tests
            ], dim=1)

    # Determine the optimal action based on the computed outputs
    optimal_actions = torch.argmax(test_outputs, dim=1) + 1
    return optimal_actions.squeeze().to(params['device'])
    



def initialize_and_load_model(stage, sample_size, params, config_number):
    # Initialize the neural network model
    nn_model = initialize_nn(params, stage).to(params['device'])
    
    # Define the directory and file name for the model
    model_dir = f"models/{params['job_id']}"
    if params['f_model']=="surr_opt":
        model_filename = f'best_model_stage_surr_{stage}_{sample_size}_config_number_{config_number}.pt'
    else:
        model_filename = f'best_model_stage_Q_{stage}_{sample_size}_config_number_{config_number}.pt'
        
    model_path = os.path.join(model_dir, model_filename)
    
    # Check if the model file exists before attempting to load
    if not os.path.exists(model_path):
        warnings.warn(f"No model file found at {model_path}. Please check the file path and model directory.")
        return None  # or handle the error as needed
    
    # Load the model's state dictionary from the file
    nn_model.load_state_dict(torch.load(model_path, map_location=params['device']))
    
    # Set the model to evaluation mode
    nn_model.eval()
    
    return nn_model



# utils value function estimator

def train_and_validate_W_estimator(config_number, model, optimizer, scheduler, train_inputs, train_actions, train_targets, val_inputs, val_actions, val_targets, batch_size, device, n_epoch, stage_number, sample_size, params, resNum):
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_model_params = None
    epoch_num_model = 0

    for epoch in range(n_epoch):
        
        train_loss = process_batches_DQL(model, train_inputs, train_actions, train_targets, params, optimizer, is_train=True)
        train_losses.append(train_loss)

        val_loss = process_batches_DQL(model, val_inputs, val_actions, val_targets, params, optimizer, is_train=False)
        val_losses.append(val_loss)
        
        if val_loss < best_val_loss:
            epoch_num_model = epoch
            best_val_loss = val_loss
            best_model_params = model.state_dict()

        # Update the scheduler with the current epoch's validation loss
        update_scheduler(scheduler, params, val_loss)


    # Define file paths for saving models
    model_dir = f"models/{params['job_id']}"

    
    # Check if the directory exists, if not, create it
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Directory '{model_dir}' created successfully.")
        
    # Save the best model parameters after all epochs
    if best_model_params is not None:
        model_path = os.path.join(model_dir, f"best_model_stage_Q_{stage_number}_{sample_size}_W_estimator_{params['f_model']}_config_number_{config_number}_result_{resNum}.pt")
        torch.save(best_model_params, model_path)

    return train_losses, val_losses, epoch_num_model



def valFn_estimate(Qhat1_H1d1, Qhat2_H2d2, Qhat1_H1A1, Qhat2_H2A2, A1_tensor, A2_tensor, A1, A2, Y1_tensor, Y2_tensor , P_A1_given_H1_tensor, P_A2_given_H2_tensor):
  
    # # IPW estimator
    # indicator1 = ((A1_tensor == A1)/P_A1_given_H1_tensor)
    # indicator2 = ((A2_tensor == A2)/P_A2_given_H2_tensor)
    # term = (Y1_tensor + Y2_tensor) * indicator1 * indicator2
    # return torch.mean(term).item()  
  
    # # term I got
    # term_1 = (Y1_tensor - Qhat1_H1A1.squeeze(1)) *((A1_tensor == A1)/P_A1_given_H1_tensor)
    # term_2 = (Y2_tensor - Qhat2_H2A2.squeeze(1) ) * ((A2_tensor == A2)/P_A2_given_H2_tensor)
    # return torch.mean(Qhat1_H1d1.squeeze(1)  + term_1 + term_2 + Qhat2_H2d2.squeeze(1)).item()
  
    # # 1st term on board (incorrect)
    # term_1 = (Y1_tensor - Qhat1_H1A1.squeeze(1) + Qhat2_H2d2.squeeze(1) ) *((A1_tensor == A1)/P_A1_given_H1_tensor)
    # term_2 = (Y2_tensor- Qhat2_H2A2.squeeze(1) ) * ((A2_tensor == A2)/P_A2_given_H2_tensor)
    # return torch.mean(Qhat1_H1d1.squeeze(1)  + term_1 + term_2).item()   
  
    # corrected doubly robust IPW estimator term by prof. 
    indicator1 = ((A1_tensor == A1)/P_A1_given_H1_tensor)
    indicator2 = ((A2_tensor == A2)/P_A2_given_H2_tensor)
    term_1 = (Y1_tensor - Qhat1_H1A1.squeeze(1) + Qhat2_H2d2.squeeze(1) ) * indicator1
    term_2 = (Y2_tensor - Qhat2_H2A2.squeeze(1) ) * indicator1 * indicator2
    return torch.mean(Qhat1_H1d1.squeeze(1) ).item() + torch.mean(term_1 + term_2).item() 


# def train_and_evaluate(train_data, val_data, test_data, params, config_number, resNum):
        
#     # Extracting elements
    
#     train_tensors, A1_train, A2_train, _, _ = train_data
#     val_tensors, A1_val, A2_val = val_data
#     test_tensors, A1_test, A2_test, P_A1_given_H1_tensor_test, P_A2_given_H2_tensor_test = test_data
    
    
#     train_input_stage1, train_input_stage2, train_Y1, train_Y2, train_A1, train_A2 = train_tensors
#     val_input_stage1, val_input_stage2, val_Y1, val_Y2, val_A1, val_A2 = val_tensors
#     test_input_stage1, test_input_stage2, test_Y1, test_Y2, test_A1, test_A2 = test_tensors


#     # Duplicate the params dictionary
#     param_W = params.copy()

#     # Update specific values in param_W 
#     param_W.update({
#         'num_networks': 1,
#         'activation_function': 'elu', #'elu', 'relu', 'sigmoid', 'tanh', 'leakyrelu', 'none' # comment this if need to parallelize over eval
#     })
        
#     if params["f_model"]!="DQlearning":
#         param_W.update({
#               'input_dim_stage1': params['input_dim_stage1'] + 1, # (H_1, A_1)
#               'input_dim_stage2': params['input_dim_stage2'] + 1, # (H_2, A_2)
#           })
    
#     nn_stage2, optimizer_2, scheduler_2 = initialize_model_and_optimizer(param_W, 2)
#     train_losses_stage2, val_losses_stage2, epoch_num_model_2 = train_and_validate_W_estimator(config_number, nn_stage2, optimizer_2, scheduler_2,
#                                                                                                train_input_stage2, train_A2, train_Y2,
#                                                                                                val_input_stage2, val_A2, val_Y2, 
#                                                                                                params['batch_size'], device, params['n_epoch'], 2,
#                                                                                                params['sample_size'], params, resNum)
    
    
#     model_dir = f"models/{params['job_id']}"
#     model_filename = f"best_model_stage_Q_{2}_{params['sample_size']}_W_estimator_{params['f_model']}_config_number_{config_number}_result_{resNum}.pt"
#     model_path = os.path.join(model_dir, model_filename)
#     nn_stage2.load_state_dict(torch.load(model_path, map_location=params['device']))
#     nn_stage2.eval()
    
#     combined_inputs2 = torch.cat((train_input_stage2, A2_train.unsqueeze(-1)), dim=1)
#     test_tr_outputs_stage2 = nn_stage2(combined_inputs2)[0]  
#     train_Y1_hat = test_tr_outputs_stage2.squeeze(1) + train_Y1 # pseudo outcome


#     combined_inputs2val = torch.cat((val_input_stage2, A2_val.unsqueeze(-1)), dim=1)
#     test_val_outputs_stage2 = nn_stage2(combined_inputs2val)[0]  
#     val_Y1_hat = test_val_outputs_stage2.squeeze() + val_Y1 # pseudo outcome


#     nn_stage1, optimizer_1, scheduler_1 = initialize_model_and_optimizer(param_W, 1)
#     train_losses_stage1, val_losses_stage1, epoch_num_model_1 = train_and_validate_W_estimator(config_number, nn_stage1, optimizer_1, scheduler_1, 
#                                                                                                train_input_stage1, train_A1, train_Y1_hat, 
#                                                                                                val_input_stage1, val_A1, val_Y1_hat, 
#                                                                                                params['batch_size'], device, 
#                                                                                                params['n_epoch'], 1, 
#                                                                                                params['sample_size'], params, resNum)    
#     model_dir = f"models/{params['job_id']}"
#     model_filename = f"best_model_stage_Q_{1}_{params['sample_size']}_W_estimator_{params['f_model']}_config_number_{config_number}_result_{resNum}.pt"
#     model_path = os.path.join(model_dir, model_filename)
#     nn_stage1.load_state_dict(torch.load(model_path, map_location=params['device']))
#     nn_stage1.eval()



#     combined_inputs2 = torch.cat((test_input_stage2, A2_test.unsqueeze(-1)), dim=1)
#     Qhat2_H2d2 = nn_stage2(combined_inputs2)[0]  

#     combined_inputs1 = torch.cat((test_input_stage1, A1_test.unsqueeze(-1)), dim=1)
#     Qhat1_H1d1 = nn_stage1(combined_inputs1)[0]  


#     combined_inputs2 = torch.cat((test_input_stage2, test_A2.unsqueeze(-1)), dim=1)
#     Qhat2_H2A2 = nn_stage2(combined_inputs2)[0]  

#     combined_inputs1 = torch.cat((test_input_stage1, test_A1.unsqueeze(-1)), dim=1)
#     Qhat1_H1A1 = nn_stage1(combined_inputs1)[0] 
    

#     V_replications_M1_pred = valFn_estimate(Qhat1_H1d1, Qhat2_H2d2, 
#                                             Qhat1_H1A1, Qhat2_H2A2, 
#                                             test_A1, test_A2, 
#                                             A1_test, A2_test,
#                                             test_Y1, test_Y2, 
#                                             P_A1_given_H1_tensor_test, P_A2_given_H2_tensor_test)

#     return V_replications_M1_pred 



def train_and_evaluate(train_data, val_data, test_data, params, config_number, resNum):
        
    # Extracting elements
    
    train_tensors, A1_train, A2_train, _, _ = train_data
    val_tensors, A1_val, A2_val = val_data
    test_tensors, A1_test, A2_test, P_A1_given_H1_tensor_test, P_A2_given_H2_tensor_test = test_data
    
    
    train_input_stage1, train_input_stage2, train_Y1, train_Y2, train_A1, train_A2 = train_tensors
    val_input_stage1, val_input_stage2, val_Y1, val_Y2, val_A1, val_A2 = val_tensors
    test_input_stage1, test_input_stage2, test_Y1, test_Y2, test_A1, test_A2 = test_tensors
    
    nn_stage2, optimizer_2, scheduler_2 = initialize_model_and_optimizer(params, 2)
    train_losses_stage2, val_losses_stage2, epoch_num_model_2 = train_and_validate_W_estimator(config_number, nn_stage2, optimizer_2, scheduler_2,
                                                                                               train_input_stage2, train_A2, train_Y2,
                                                                                               val_input_stage2, val_A2, val_Y2, 
                                                                                               params['batch_size'], device, params['n_epoch'], 2,
                                                                                               params['sample_size'], params, resNum)
    
    
    model_dir = f"models/{params['job_id']}"
    model_filename = f"best_model_stage_Q_{2}_{params['sample_size']}_W_estimator_{params['f_model']}_config_number_{config_number}_result_{resNum}.pt"
    model_path = os.path.join(model_dir, model_filename)
    nn_stage2.load_state_dict(torch.load(model_path, map_location=params['device']))
    nn_stage2.eval()
    
    combined_inputs2 = torch.cat((train_input_stage2, A2_train.unsqueeze(-1)), dim=1)
    test_tr_outputs_stage2 = nn_stage2(combined_inputs2)[0]  
    train_Y1_hat = test_tr_outputs_stage2.squeeze(1) + train_Y1 # pseudo outcome


    combined_inputs2val = torch.cat((val_input_stage2, A2_val.unsqueeze(-1)), dim=1)
    test_val_outputs_stage2 = nn_stage2(combined_inputs2val)[0]  
    val_Y1_hat = test_val_outputs_stage2.squeeze() + val_Y1 # pseudo outcome


    nn_stage1, optimizer_1, scheduler_1 = initialize_model_and_optimizer(params, 1)
    train_losses_stage1, val_losses_stage1, epoch_num_model_1 = train_and_validate_W_estimator(config_number, nn_stage1, optimizer_1, scheduler_1, 
                                                                                               train_input_stage1, train_A1, train_Y1_hat, 
                                                                                               val_input_stage1, val_A1, val_Y1_hat, 
                                                                                               params['batch_size'], device, 
                                                                                               params['n_epoch'], 1, 
                                                                                               params['sample_size'], params, resNum)    
    model_dir = f"models/{params['job_id']}"
    model_filename = f"best_model_stage_Q_{1}_{params['sample_size']}_W_estimator_{params['f_model']}_config_number_{config_number}_result_{resNum}.pt"
    model_path = os.path.join(model_dir, model_filename)
    nn_stage1.load_state_dict(torch.load(model_path, map_location=params['device']))
    nn_stage1.eval()



    combined_inputs2 = torch.cat((test_input_stage2, A2_test.unsqueeze(-1)), dim=1)
    Qhat2_H2d2 = nn_stage2(combined_inputs2)[0]  

    combined_inputs1 = torch.cat((test_input_stage1, A1_test.unsqueeze(-1)), dim=1)
    Qhat1_H1d1 = nn_stage1(combined_inputs1)[0]  


    combined_inputs2 = torch.cat((test_input_stage2, test_A2.unsqueeze(-1)), dim=1)
    Qhat2_H2A2 = nn_stage2(combined_inputs2)[0]  

    combined_inputs1 = torch.cat((test_input_stage1, test_A1.unsqueeze(-1)), dim=1)
    Qhat1_H1A1 = nn_stage1(combined_inputs1)[0] 
    

    V_replications_M1_pred = valFn_estimate(Qhat1_H1d1, Qhat2_H2d2, 
                                            Qhat1_H1A1, Qhat2_H2A2, 
                                            test_A1, test_A2, 
                                            A1_test, A2_test,
                                            test_Y1, test_Y2, 
                                            P_A1_given_H1_tensor_test, P_A2_given_H2_tensor_test)

    return V_replications_M1_pred 



def split_data(train_tens, A1, A2, P_A1_given_H1_tensor, P_A2_given_H2_tensor, params):

    # print("A1.shape[0]:------------------------>>>>>>>> ", A1.shape[0]) 
    train_val_size = int(0.5 *  A1.shape[0]) #int(0.5 *  params['sample_size'])
    validation_ratio = 0.20  # 20% of the train_val_size for validation
    
    val_size = int(train_val_size * validation_ratio)
    train_size = train_val_size - val_size  # Remaining part for training

    # Split tensors into training, validation, and testing
    train_tensors = [tensor[:train_size] for tensor in train_tens]
    val_tensors = [tensor[train_size:train_val_size] for tensor in train_tens]
    test_tensors = [tensor[train_val_size:] for tensor in train_tens]
    
    # Splitting A1 and A2 tensors
    A1_train, A1_val, A1_test = A1[:train_size], A1[train_size:train_val_size], A1[train_val_size:]
    A2_train, A2_val, A2_test = A2[:train_size], A2[train_size:train_val_size], A2[train_val_size:]
    
    p_A2_g_H2_train, p_A1_g_H1_test = P_A1_given_H1_tensor[:train_size], P_A1_given_H1_tensor[train_val_size:]
    p_A2_g_H2_train, p_A2_g_H2_test = P_A2_given_H2_tensor[:train_size], P_A2_given_H2_tensor[train_val_size:]
    
    
    # Create tuples for training, validation, and test sets
    train_data = (train_tensors, A1_train, A2_train, p_A2_g_H2_train, p_A2_g_H2_train)
    val_data = (val_tensors, A1_val, A2_val)
    test_data = (test_tensors, A1_test, A2_test, p_A1_g_H1_test, p_A2_g_H2_test)
    
    
    return train_data, val_data, test_data
    
def calculate_policy_values_W_estimator(train_tens, params, A1, A2, P_A1_given_H1_tensor, P_A2_given_H2_tensor, config_number):
    # First, split the data
    train_data, val_data, test_data = split_data(train_tens, A1, A2, P_A1_given_H1_tensor, P_A2_given_H2_tensor, params)

    # Train and evaluate with the initial split
    result1 = train_and_evaluate(train_data, val_data, test_data, params, config_number, resNum = 1)


    # Swap training/validation with testing, then test becomes train_val
    result2 = train_and_evaluate(test_data, val_data, train_data, params, config_number, resNum = 2)
    
    print("calculate_policy_values_W_estimator: ", result1, result2)
    
    return (result1+result2)/2



def evaluate_method(method_name, params, config_number, df, test_input_stage1, A1_tensor_test, test_input_stage2, A2_tensor_test, train_tensors, P_A1_g_H1, P_A2_g_H2, tmp):
    # Initialize and load models for the method 
    nn_stage1 = initialize_and_load_model(1, params['sample_size'], params, config_number)
    nn_stage2 = initialize_and_load_model(2, params['sample_size'], params, config_number)

    # Calculate test outputs for all networks in stage 1
    A1 = compute_test_outputs(nn=nn_stage1, 
                              test_input=test_input_stage1, 
                              A_tensor=A1_tensor_test, 
                              params=params, 
                              is_stage1=True)

    # Calculate test outputs for all networks in stage 2
    A2 = compute_test_outputs(nn=nn_stage2, 
                              test_input=test_input_stage2, 
                              A_tensor=A2_tensor_test, 
                              params=params, 
                              is_stage1=False)
    
    # Print first 20 predictions using current policy 
    print_predictions_in_box(method_name, A1, A2)   

    # Append to DataFrame
    new_row = {
        'Behavioral_A1': A1_tensor_test.cpu().numpy().tolist(),
        'Behavioral_A2': A2_tensor_test.cpu().numpy().tolist(),
        'Predicted_A1': A1.cpu().numpy().tolist(),
        'Predicted_A2': A2.cpu().numpy().tolist()
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Calculate policy values using the DR estimator

    # Duplicate the params dictionary
    param_W = params.copy()

    # Update specific values in param_W  if testing is fixed 
    param_W.update({
        'num_networks': 1,
        'num_layers':  tmp[0], #initial_config['num_layers'],
        'hidden_dim_stage1': tmp[1], #initial_config['hidden_dim_stage1'],
        'hidden_dim_stage2': tmp[2], #initial_config['hidden_dim_stage2']
        'activation_function': tmp[3], #initial_config['activation_function'], #'elu', 'relu', 'sigmoid', 'tanh', 'leakyrelu', 'none' 
    }) 
        
    if params["f_model"]!="DQlearning":
        param_W.update({
              'input_dim_stage1': params['input_dim_stage1'] + 1, # (H_1, A_1)
              'input_dim_stage2': params['input_dim_stage2'] + 1, # (H_2, A_2)
          })
        
        
    V_replications_M1_pred = calculate_policy_values_W_estimator(train_tensors, param_W, A1, A2, P_A1_g_H1, P_A2_g_H2, config_number)


    # V_replications_M1_pred = calculate_policy_values_W_estimator(train_tensors, params, A1, A2, P_A1_g_H1, P_A2_g_H2, config_number)


    # print(f"{method_name} estimator: ")

    return df, V_replications_M1_pred, param_W


# Helper function to handle both tensors and floats
def get_value(value):
    if isinstance(value, torch.Tensor):  # Check if it's a PyTorch tensor
        return value.item()
    return value  # If it's already a float or None, return it as is




def print_predictions_in_box(method_name, A1, A2):
    # Create a formatted string for the method name
    box_title = f" Print first 20 predictions using {method_name}'s current policy "

    # Calculate the length of the box
    box_length = len(box_title) + 4  # Add padding for borders

    # Print the box with the title
    print("-" * box_length)
    print(f"| {box_title} |")
    print("-" * box_length)

    # If A1 or A2 are lists of tensors, concatenate them
    if isinstance(A1, list):
        A1 = torch.cat(A1, dim=0)
    if isinstance(A2, list):
        A2 = torch.cat(A2, dim=0)

    # Print the top 20 values of A1 and A2
    print("First 20 A1 values:", A1[:20])
    print("First 20 A2 values:", A2[:20])
    
    # Calculate and print the counts of unique values in A1 and A2
    A1_unique, A1_counts = torch.unique(A1, return_counts=True)
    A2_unique, A2_counts = torch.unique(A2, return_counts=True)

    print("\nA1 unique values and their counts:")
    for value, count in zip(A1_unique, A1_counts):
        print(f"Value: {value}, Count: {count}")

    print("\nA2 unique values and their counts:")
    for value, count in zip(A2_unique, A2_counts):
        print(f"Value: {value}, Count: {count}")

    # Print the bottom border of the box
    print("-" * box_length)



# Generate Data
def load_and_preprocess_data(params, replication_seed, config_seed, run='train'):

    # Set seed for this configuration and replication
    seed_value = config_seed * 100 + replication_seed  # Ensures unique seed for each config and replication
    torch.manual_seed(seed_value) 
    np.random.seed(seed_value)

    # cutting off data points for faster testing
    df = pd.read_csv('final_data.csv') #.iloc[:3000, ]  #.iloc[:params["sample_size"], ] 
    print("df ==================> : ", df.shape, "Total data points: ",  df.shape[0]/2)

    # Shuffle
    # sample the rows creating a random order
    groups = [df for _, df in df.groupby('m:icustayid')]
    np.random.shuffle(groups)  # Shuffles the list of groups in place
    # Concatenate the shuffled groups back into a single DataFrame
    df = pd.concat(groups).reset_index(drop=True)

    # IV fluid is the treatment
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
        # filter based on indices in test
        test_patient_ids = torch.tensor(test_patient_ids)
        Ci = Ci[test_patient_ids]
        O1_filtered = O1_filtered[:, test_patient_ids]
        O2_filtered = O2_filtered[:, test_patient_ids]
        Y1_filtered = Y1_filtered[test_patient_ids]
        Y2_filtered = Y2_filtered[test_patient_ids]
        A1_filtered = A1_filtered[test_patient_ids]
        A2_filtered = A2_filtered[test_patient_ids]

        # calculate input stages
        input_stage1 = O1_filtered.t()         
        params['input_dim_stage1'] = input_stage1.shape[1] #  (H_1)  
        input_stage2 = torch.cat([O1_filtered.t(), A1_filtered.unsqueeze(1), Y1_filtered.unsqueeze(1), O2_filtered.t()], dim=1) 
        params['input_dim_stage2'] = input_stage2.shape[1] # (H_2)

        P_A1_given_H1_tensor_filtered = P_A1_given_H1_tensor_filtered[test_patient_ids]
        P_A2_given_H2_tensor_filtered = P_A2_given_H2_tensor_filtered[test_patient_ids]
        # ensure proper data types
        input_stage1 = input_stage1.float()
        input_stage2 = input_stage2.float()
        Ci = Ci.float()
        Y1_filtered = Y1_filtered.float()
        Y2_filtered = Y2_filtered.float()
        A1_filtered = A1_filtered.float()
        A2_filtered = A2_filtered.float()

        print("="*90)
        print("pi_10: ", probs1['pi_10'].mean().item(), "pi_11: ", probs1['pi_11'].mean().item(), "pi_12: ", probs1['pi_12'].mean().item())
        print("pi_20: ", probs2['pi_20'].mean().item(), "pi_21: ", probs2['pi_21'].mean().item(), "pi_22: ", probs2['pi_22'].mean().item())
        print("="*90)

        print()

        print("="*90)
        print("Y1_beh mean: ", torch.mean(Y1) )
        print("Y2_beh mean: ", torch.mean(Y2) )         
        print("Y1_beh+Y2_beh mean: ", torch.mean(Y1+Y2) )

        print("="*90)
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

    # Activate NumPy to R conversion
    numpy2ri.activate()

    # Suppress R console warnings and output
    ro.r['options'](warn=-1)  # Suppress R warnings
    # ro.r['sink']("/dev/null")  # Suppress R console messages

    # Load the R script that contains the required function
    ro.r('source("ACWL_tao.R")')

    # Call the R function with the parameters
    results = ro.globalenv['test_ACWL'](S1, S2, A1.cpu().numpy(), A2.cpu().numpy(), Y1.cpu().numpy(), Y2.cpu().numpy(), 
                                        config_number, params_ds['job_id'])

    # Extract the decisions and convert to PyTorch tensors on the specified device
    A1_Tao = torch.tensor(np.array(results.rx2('g1.a1')), dtype=torch.float32).to(params_ds['device'])
    A2_Tao = torch.tensor(np.array(results.rx2('g2.a1')), dtype=torch.float32).to(params_ds['device'])

    return A1_Tao, A2_Tao

def print_method_name_in_rectangle(method_name):
    # Format the method name with "method" suffix
    text = f" {method_name}'s method "
    
    # Determine the length of the text and create borders accordingly
    border_length = len(text) + 4  # Adding 4 for padding and borders
    
    # Print top border of the rectangle
    print("=" * border_length)
    
    # Print the text centered inside the rectangle
    print(f"| {text} |")
    
    # Print bottom border of the rectangle
    print("=" * border_length)

# def eval_DTR(V_replications, num_replications, nn_stage1_DQL, nn_stage2_DQL, nn_stage1_DS, nn_stage2_DS, df_DQL, df_DS, df_Tao, params_dql, params_ds, config_number):
def eval_DTR(V_replications, num_replications, df_DQL, df_DS, df_Tao, params_dql, params_ds, tmp, config_number):

    # Generate and preprocess data for evaluation
    processed_result = load_and_preprocess_data(params_ds, replication_seed=num_replications+1234, config_seed=config_number, run='test')
    test_input_stage1, test_input_stage2, test_O2, Y1_tensor, Y2_tensor, A1_tensor_test, A2_tensor_test, P_A1_g_H1, P_A2_g_H2  = processed_result
    train_tensors = [test_input_stage1, test_input_stage2, Y1_tensor, Y2_tensor, A1_tensor_test, A2_tensor_test]

    # Append policy values for DS
    V_replications["V_replications_M1_behavioral"].append(torch.mean(Y1_tensor + Y2_tensor).cpu())  

    # # Value function behavioral
    # message = f'\nY1 beh mean: {torch.mean(Y1_tensor)}, Y2 beh mean: {torch.mean(Y2_tensor)}, Y1_beh+Y2_beh mean: {torch.mean(Y1_tensor + Y2_tensor)} '
    # print(message)

    param_W_DQL = None
    param_W_DS = None

    #######################################
    # Evaluation phase using Tao's method #
    #######################################
    if params_ds.get('run_adaptive_contrast_tao', True):
        start_time = time.time()  # Start time recording
        A1_Tao, A2_Tao = evaluate_tao(test_input_stage1, test_O2, A1_tensor_test, A2_tensor_test, Y1_tensor, Y2_tensor, params_ds, config_number)
        end_time = time.time()  # End time recording

        # Print the top 20 values of A1 and A2
        print_predictions_in_box("Tao", A1_Tao, A2_Tao)

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


        # Duplicate the params dictionary
        param_W = params_ds.copy()

        # Update specific values in param_W  if testing is fixed 
        param_W.update({
            'num_networks': 1,
            'num_layers':  tmp[0], #initial_config['num_layers'],
            'hidden_dim_stage1': tmp[1], #initial_config['hidden_dim_stage1'],
            'hidden_dim_stage2': tmp[2], #initial_config['hidden_dim_stage2']
            'activation_function': tmp[3], #initial_config['activation_function'], #'elu', 'relu', 'sigmoid', 'tanh', 'leakyrelu', 'none' 
            'input_dim_stage1': params_ds['input_dim_stage1'] + 1, # (H_1, A_1)
            'input_dim_stage2': params_ds['input_dim_stage2'] + 1, # (H_2, A_2)
        })

        print()
        print_method_name_in_rectangle("Tao")
        print("="*60)   

        start_time1 = time.time()  # Start time recording
        V_rep_Tao = calculate_policy_values_W_estimator(train_tensors, param_W, A1_Tao, A2_Tao, P_A1_g_H1, P_A2_g_H2, config_number)
        end_time1 = time.time()  # End time recording
        # Append value function for Tao
        V_replications["V_replications_M1_pred"]["Tao"].append(V_rep_Tao)  

        message = f'Y1_tao+Y2_tao mean: {V_rep_Tao} \n'
        print(message) 
        print(f"Total time taken to run evaluate_method (Tao): { end_time - start_time + end_time1 - start_time1} seconds")
        print("="*60)
        print()

    #######################################
    # Evaluation phase using DQL's method #
    #######################################
    if params_ds.get('run_DQlearning', True):

        print()
        print_method_name_in_rectangle("Deep Q-Learning")
        print("="*60)
        
        start_time = time.time()  # Start time recording
        df_DQL, V_rep_DQL, param_W_DQL = evaluate_method('DQL', params_dql, config_number, df_DQL, test_input_stage1, A1_tensor_test, test_input_stage2, 
                                            A2_tensor_test, train_tensors, P_A1_g_H1, P_A2_g_H2, tmp)
        end_time = time.time()  # End time recording
        # Append value function for DQL
        V_replications["V_replications_M1_pred"]["DQL"].append(V_rep_DQL)     

        message = f'Y1_DQL+Y2_DQL mean: {V_rep_DQL} \n'
        print(message)

        print(f"Total time taken to run evaluate_method ('DQL'): { end_time - start_time} seconds")
        print("="*60)
        print()
    ########################################
    #  Evaluation phase using DS's method  #
    ########################################
    if params_ds.get('run_surr_opt', True):

        print()         
        print_method_name_in_rectangle("Direct Search")
        print("="*60)

        start_time = time.time()  # Start time recording
        df_DS, V_rep_DS, param_W_DS = evaluate_method('DS', params_ds, config_number, df_DS, test_input_stage1, A1_tensor_test, test_input_stage2, 
                                        A2_tensor_test, train_tensors, P_A1_g_H1, P_A2_g_H2, tmp)
        end_time = time.time()  # End time recording        
        
        # Append value function for DS
        V_replications["V_replications_M1_pred"]["DS"].append(V_rep_DS)

        message = f'Y1_DS+Y2_DS mean: {V_rep_DS} \n'
        print(message)         
        print(f"Total time taken to run evaluate_method ('DS'): { end_time - start_time} seconds")
        print("="*60)
        print()
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

    # Activate NumPy to R conversion
    numpy2ri.activate()

    # Suppress R console warnings and output
    ro.r['options'](warn=-1)  # Suppress R warnings
    # ro.r['sink']("/dev/null")  # Suppress R console messages

    # Load the R script containing the function
    ro.r('source("ACWL_tao.R")')

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


    print("\n\n")
    print("x"*90)
    for replication in tqdm(range(params['num_replications']), desc="Replications_M1"):

        print(f"\nReplication # -------------->>>>>  {replication+1}")

        # config_dict['replications'].append(replication+1)

        # Generate and preprocess data for training
        tuple_train, tuple_val, adapC_tao_Data = load_and_preprocess_data(params, replication_seed=replication, config_seed=config_number, run='train')

        # Estimate treatment regime : model --> surr_opt
        print("x"*90)
        print("\n\n")
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
            print()
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
        print("x"*90)
        print("\n\n")
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

    # Initialize empty lists to store the value functions across all configurations
    all_performances_DQL = []
    all_performances_DS = []
    all_performances_Tao = []
    all_performances_Beh = []

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
            performances_Beh = pd.concat([performances_Beh, performance_Beh], axis=0)

            # Process and store DQL performance
            dql_values = [get_value(value) for value in performances_DQL['Method\'s Value fn.']]
            all_performances_DQL.append(dql_values)

            # Process and store DS performance
            ds_values = [get_value(value) for value in performances_DS['Method\'s Value fn.']]
            all_performances_DS.append(ds_values)

            # Process and store Tao performance
            tao_values = [get_value(value) for value in performances_Tao['Method\'s Value fn.']]
            all_performances_Tao.append(tao_values)

            # Process and store Behavioral performance
            beh_values = [get_value(value) for value in performances_Beh['Method\'s Value fn.']]
            all_performances_Beh.append(beh_values)


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
                    'Behavioral': {"Method's Value fn.": performance_Beh_mean, 
                           "Method's Value fn. SD": performance_Beh_std,
                           },
                        
                    'DQL': {"Method's Value fn.": performance_DQL_mean, 
                            "Method's Value fn. SD": performance_DQL_std, 
                            },
                    'Tao': {"Method's Value fn.": performance_Tao_mean, 
                           "Method's Value fn. SD": performance_Tao_std,
                           }, 
                    'DS': {"Method's Value fn.": performance_DS_mean, 
                           "Method's Value fn. SD": performance_DS_std,
                           }
                }
            else:
                # Update existing entries with new means
                results[config_key]['Behavioral'].update({
                    "Method's Value fn.": performance_Beh_mean, 
                    "Method's Value fn. SD": performance_Beh_std,  
                })                  
                results[config_key]['DQL'].update({
                    "Method's Value fn.": performance_DQL_mean,                                 
                    "Method's Value fn. SD": performance_DQL_std, 
                })
                results[config_key]['Tao'].update({
                    "Method's Value fn.": performance_Tao_mean, 
                    "Method's Value fn. SD": performance_Tao_std,
                })
                results[config_key]['DS'].update({
                    "Method's Value fn.": performance_DS_mean,
                    "Method's Value fn. SD": performance_DS_std,
                })

            print("Performances for configuration: %s", config_key)
            print("performance_DQL_mean: %s", performance_DQL_mean)
            print("performance_DS_mean: %s", performance_DS_mean)
            print("performance_Tao_mean: %s", performance_Tao_mean)
            print("\n\n")
        

        
    folder = f"data/{config['job_id']}"
    save_simulation_data(all_performances_Beh, all_performances_DQL, all_performances_DS,  all_performances_Tao, all_dfs_DQL, all_dfs_DS, all_losses_dicts, all_epoch_num_lists, results, all_configurations, folder)
    load_and_process_data(config, folder)


        
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
    # these are the parameters usd for not-fixed config 

    # param_grid = {}

    param_grid = {
        'activation_function': ['elu'], # elu, relu, sigmoid, tanh, leakyrelu, none
        'num_layers': [4], # 2,4
        'optimizer_lr': [0.07], # 0.1, 0.01, 0.07, 0.001
        # 'n_epoch':[60]
    }


    # param_grid = {
    #     'activation_function': ['elu'], # elu, relu, sigmoid, tanh, leakyrelu, none
    #     'batch_size': [200],#,150],#,800,1000], # 50
    #     'num_layers': [4], # 1,2,3,4,5,6,7
    # }



    # param_grid = {
    #     'num_layers': [5, 12], # 2,4
    #     'n_epoch':[60, 150]
    # }

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


    
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    
    # Record the start time
    start_time = time.time()
    start_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))

    print()
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


    
