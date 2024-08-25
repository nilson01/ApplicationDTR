import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
import pickle
import yaml
import matplotlib.pyplot as plt
from collections import Counter
import torch.nn.functional as F
from collections import defaultdict

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.exceptions import ConvergenceWarning
import warnings



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


def save_simulation_data(all_dfs_DQL, all_dfs_DS, all_losses_dicts, all_epoch_num_lists, results, folder):
    # Check if the folder exists, if not, create it
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Define paths for saving files
    df_path_DQL = os.path.join(folder, 'simulation_data_DQL.pkl')
    df_path_DS = os.path.join(folder, 'simulation_data_DS.pkl')
    losses_path = os.path.join(folder, 'losses_dicts.pkl')
    epochs_path = os.path.join(folder, 'epoch_num_lists.pkl')
    results_path = os.path.join(folder, 'simulation_results.pkl')

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
    
    # Extract and process unique values for both DQL and DS
    columns_to_process = {
        'Predicted': ['Predicted_A1', 'Predicted_A2'],
    }

    unique_values_DQL = extract_unique_treatment_values(global_df_DQL, columns_to_process, name = "DQL")
    unique_values_DS = extract_unique_treatment_values(global_df_DS, columns_to_process, name = "DS")
    
    train_size = int(params['training_validation_prop'] * params['sample_size'])

    # Process and plot results from all simulations
    for i, method_losses_dicts in enumerate(all_losses_dicts):
        run_name = f"run_trainVval_{i}"
        selected_indices = [i for i in range(params['num_replications'])]  

        # Pass the appropriate sub-dictionary for DQL
        plot_simulation_Qlearning_losses_in_grid(selected_indices, method_losses_dicts['DQL'], train_size, run_name, folder)

        # Pass the appropriate sub-dictionary for DS
        plot_simulation_surLoss_losses_in_grid(selected_indices, method_losses_dicts['DS'], train_size, run_name, folder)

    # Print results for each configuration
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


    print("s_p: ===============>>>>>>>>>>>>>>>>>>>>> ", s_p, "\n\n")
    print("Minimum of -> Max values over each rows: ", np.min(np.max(s_p, axis=1)) , "\n\n\n")


    if stage == 1:
        col_names = ['pi_10', 'pi_11', 'pi_12']
    else:
        col_names = ['pi_20', 'pi_21', 'pi_22']
        
    #probs_df = pd.DataFrame(s_p, columns=col_names)
    #probs_df = {name: s_p[:, idx] for idx, name in enumerate(col_names)}
    probs_dict = {name: torch.tensor(s_p[:, idx], dtype=torch.float32) for idx, name in enumerate(col_names)}

    return probs_dict


# Neural networks utils

# def initialize_nn(params, stage):
#     nn = NNClass(
#         params[f'input_dim_stage{stage}'],
#         params[f'hidden_dim_stage{stage}'],
#         params[f'output_dim_stage{stage}'],
#         params['num_networks'],
#         dropout_rate=params['dropout_rate'],
#         activation_fn_name = params['activation_function'],
#     ).to(params['device'])
#     return nn

def initialize_nn(params, stage):

    nn = NNClass(
        input_dim=params[f'input_dim_stage{stage}'],
        hidden_dim=params[f'hidden_dim_stage{stage}'],
        output_dim=params[f'output_dim_stage{stage}'],
        num_networks=params['num_networks'],
        dropout_rate=params['dropout_rate'],
        activation_fn_name=params['activation_function'],
        num_hidden_layers=params['num_layers'] - 1  # num_layers is the number of hidden layers
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

        
class NNClass(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_networks, dropout_rate, activation_fn_name, num_hidden_layers):
        super(NNClass, self).__init__()
        self.networks = nn.ModuleList()

        # Map the string name to the actual activation function class
        if activation_fn_name.lower() == 'elu':
            activation_fn = nn.ELU
        elif activation_fn_name.lower() == 'relu':
            activation_fn = nn.ReLU
        else:
            raise ValueError(f"Unsupported activation function: {activation_fn_name}")

        for _ in range(num_networks):
            layers = []
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation_fn())
            layers.append(nn.Dropout(dropout_rate))
            
            for _ in range(num_hidden_layers):  # Adjusting the hidden layers count
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(activation_fn())
                layers.append(nn.Dropout(dropout_rate))
                
            layers.append(nn.Linear(hidden_dim, output_dim))
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

                    
        

# class NNClass(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, num_networks, dropout_rate, activation_fn_name):
#         super(NNClass, self).__init__()
#         self.networks = nn.ModuleList()
        
#         # Map the string name to the actual activation function class
#         if activation_fn_name.lower() == 'elu':
#             activation_fn = nn.ELU
#         elif activation_fn_name.lower() == 'relu':
#             activation_fn = nn.ReLU
#         else:
#             raise ValueError(f"Unsupported activation function: {activation_fn_name}")

#         for _ in range(num_networks):
#             network = nn.Sequential(
#                 nn.Linear(input_dim, hidden_dim),
#                 activation_fn(),  # Instantiate the activation function
#                 nn.Dropout(dropout_rate),
#                 nn.Linear(hidden_dim, output_dim),
#                 nn.BatchNorm1d(output_dim),
#             )
#             self.networks.append(network)
            
#     def forward(self, x):
#         outputs = []
#         for network in self.networks:
#             outputs.append(network(x))
#         return outputs

#     def he_initializer(self):
#         for network in self.networks:
#             for layer in network:
#                 if isinstance(layer, nn.Linear):
#                     nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
#                     nn.init.constant_(layer.bias, 0)  # Biases can be initialized to zero

#     def reset_weights(self):
#         for network in self.networks:
#             for layer in network:
#                 if isinstance(layer, nn.Linear):
#                     nn.init.constant_(layer.weight, 0.1)
#                     nn.init.constant_(layer.bias, 0.0)




# class NNClass(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, num_networks, dropout_rate):
#         super(NNClass, self).__init__()
#         self.networks = nn.ModuleList()
#         for _ in range(num_networks):
#             network = nn.Sequential(
#                 nn.Linear(input_dim, hidden_dim),
#                 nn.ELU(alpha=0.4),
#                 nn.Dropout(dropout_rate),
#                 nn.Linear(hidden_dim, output_dim),
#                 nn.BatchNorm1d(output_dim),
#             )
#             self.networks.append(network)

#     def forward(self, x):
#         outputs = []
#         for network in self.networks:
#             outputs.append(network(x))
#         return outputs

#     def he_initializer(self):
#         for network in self.networks:
#             for layer in network:
#                 if isinstance(layer, nn.Linear):
#                     nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
#                     nn.init.constant_(layer.bias, 0)  # Biases can be initialized to zero

#     def reset_weights(self):
#         for network in self.networks:
#             for layer in network:
#                 if isinstance(layer, nn.Linear):
#                     nn.init.constant_(layer.weight, 0.1)
#                     nn.init.constant_(layer.bias, 0.0)




# class NNClass(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, num_networks, dropout_rate):
#         super(NNClass, self).__init__()
#         self.networks = nn.ModuleList()
#         for _ in range(num_networks):
#             network = nn.Sequential(
#                 nn.Linear(input_dim, hidden_dim),
#                 nn.BatchNorm1d(hidden_dim),
#                 nn.ReLU(),
#                 nn.Dropout(dropout_rate),
#                 nn.Linear(hidden_dim, output_dim),
#                 nn.BatchNorm1d(output_dim),
#             )
#             self.networks.append(network)

#     def forward(self, x):
#         outputs = []
#         for network in self.networks:
#             outputs.append(network(x))
#         return outputs

#     def he_initializer(self):
#         for network in self.networks:
#             for layer in network:
#                 if isinstance(layer, nn.Linear):
#                     nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
#                     nn.init.constant_(layer.bias, 0)  # Biases can be initialized to zero

#     def reset_weights(self):
#         for network in self.networks:
#             for layer in network:
#                 if isinstance(layer, nn.Linear):
#                     nn.init.constant_(layer.weight, 0.1)
#                     nn.init.constant_(layer.bias, 0.0)



# class NNClass(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, num_networks, dropout_rate):
#         super(NNClass, self).__init__()
#         self.networks = nn.ModuleList()
#         for _ in range(num_networks):
#             network = nn.Sequential(
#                 nn.Linear(input_dim, hidden_dim),
#                 nn.BatchNorm1d(hidden_dim),
#                 nn.ReLU(),
#                 nn.Linear(hidden_dim, hidden_dim),
#                 nn.Dropout(dropout_rate),
#                 nn.Linear(hidden_dim, hidden_dim),
#                 nn.ReLU(),
#                 nn.Linear(hidden_dim, output_dim),
#                 nn.BatchNorm1d(output_dim),
#             )
#             self.networks.append(network)

#     def forward(self, x):
#         outputs = []
#         for network in self.networks:
#             outputs.append(network(x))
#         return outputs

#     def he_initializer(self):
#         for network in self.networks:
#             for layer in network:
#                 if isinstance(layer, nn.Linear):
#                     nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
#                     nn.init.constant_(layer.bias, 0)  # Biases can be initialized to zero

#     def reset_weights(self):
#         for network in self.networks:
#             for layer in network:
#                 if isinstance(layer, nn.Linear):
#                     nn.init.constant_(layer.weight, 0.1)
#                     nn.init.constant_(layer.bias, 0.0)














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


def extract_value_functions_separate(V_replications):

    # Process predictive values
    pred_data = V_replications.get('V_replications_M1_pred', defaultdict(list))

    # Process behavioral values (assuming these are shared across models)
    behavioral_data = V_replications.get('V_replications_M1_behavioral', [])

    # Create DataFrames for each method
    VF_df_DQL = pd.DataFrame({
        "Method's Value fn.": pred_data.get('DQL', []), #data_DQL["Method's Value fn."],
        "Behavioral Value fn.": behavioral_data
    })

    VF_df_DS = pd.DataFrame({
        "Method's Value fn.": pred_data.get('DS', []), #data_DS["Method's Value fn."],
        "Behavioral Value fn.": behavioral_data
    })


    VF_df_Tao = pd.DataFrame({
        "Method's Value fn.": pred_data.get('Tao', []), #data_Tao["Method's Value fn."],
        "Behavioral Value fn.": behavioral_data
    })    
    return VF_df_DQL, VF_df_DS, VF_df_Tao



# 3. Loss function and surrogate opt utils

def compute_phi(x, option):
    if option == 1:
        return 1 + torch.tanh(5*x)
    elif option == 2:
        return 1 + 2 * torch.atan(torch.pi * x / 2) / torch.pi
    elif option == 3:
        return 1 + x / torch.sqrt(1 + x ** 2)
    elif option == 4:
        return 1 + x / (1 + torch.abs(x))
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
        os.makedirs(model_dir)     
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


def train_and_evaluate(train_data, val_data, test_data, params, config_number, resNum):
        
    # Extracting elements
    
    train_tensors, A1_train, A2_train, _, _ = train_data
    val_tensors, A1_val, A2_val = val_data
    test_tensors, A1_test, A2_test, P_A1_given_H1_tensor_test, P_A2_given_H2_tensor_test = test_data
    
    
    train_input_stage1, train_input_stage2, train_Y1, train_Y2, train_A1, train_A2 = train_tensors
    val_input_stage1, val_input_stage2, val_Y1, val_Y2, val_A1, val_A2 = val_tensors
    test_input_stage1, test_input_stage2, test_Y1, test_Y2, test_A1, test_A2 = test_tensors


    # Duplicate the params dictionary
    param_W = params.copy()

    # Update specific values in param_W
    param_W.update({
      'num_networks': 1,
    })
        
    if params["f_model"]!="DQlearning":
        param_W.update({
              'input_dim_stage1': params['input_dim_stage1'] + 1, # (H_1, A_1)
              'input_dim_stage2': params['input_dim_stage2'] + 1, # (H_2, A_2)
          })
    
    nn_stage2, optimizer_2, scheduler_2 = initialize_model_and_optimizer(param_W, 2)
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
    

    nn_stage1, optimizer_1, scheduler_1 = initialize_model_and_optimizer(param_W, 1)
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








def evaluate_method(method_name, params, config_number, df, test_input_stage1, A1_tensor_test, test_input_stage2, A2_tensor_test, train_tensors, P_A1_g_H1, P_A2_g_H2):
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

    # Append to DataFrame
    new_row = {
        'Behavioral_A1': A1_tensor_test.cpu().numpy().tolist(),
        'Behavioral_A2': A2_tensor_test.cpu().numpy().tolist(),
        'Predicted_A1': A1.cpu().numpy().tolist(),
        'Predicted_A2': A2.cpu().numpy().tolist()
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Calculate policy values using the DR estimator
    V_replications_M1_pred = calculate_policy_values_W_estimator(train_tensors, params, A1, A2, P_A1_g_H1, P_A2_g_H2, config_number)

    # print(f"{method_name} estimator: ")

    return df, V_replications_M1_pred