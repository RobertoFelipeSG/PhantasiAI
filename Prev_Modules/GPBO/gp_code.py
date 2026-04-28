import argparse
import numpy as np
import torch
import pandas as pd
import warnings
import paho.mqtt.client as mqtt  # Importation du client MQTT
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement, NoisyExpectedImprovement, UpperConfidenceBound
from botorch.utils.transforms import normalize, standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from time import time

import os

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)

# Fonction pour lire les profils des sujets
def read_subject_profiles(file_name):
    data = pd.read_csv(file_name, sep=';')
    subject_profiles = []
    parameter_names = [col for col in data.columns if col not in ['Sujet', 'Interaction_Scale', 'Interaction_Sign']]
    n_dim = len(parameter_names)
    n_val_list = []
    for idx, row in data.iterrows():
        subject = {}
        n_val_subject = None
        for param in parameter_names:
            values = np.array(list(map(float, row[param].split())), dtype=np.float32)
            if n_val_subject is None:
                n_val_subject = len(values)
            else:
                if len(values) != n_val_subject:
                    raise ValueError(f"Le nombre de valeurs pour le paramètre {param} du sujet {row['Sujet']} n'est pas cohérent.")
            subject[param] = values
        n_val_list.append(n_val_subject)
        subject['interaction_scale'] = float(row['Interaction_Scale'])
        subject['interaction_sign'] = int(row['Interaction_Sign'])
        subject_profiles.append(subject)
    if len(set(n_val_list)) != 1:
        raise ValueError("Tous les sujets doivent avoir le même nombre de valeurs de paramètres (n_val).")
    n_val = n_val_list[0]
    return subject_profiles, parameter_names, n_dim, n_val

# Générer les réponses des sujets
def generate_subject_responses(n_suj, n_val, n_dim, parameter_names, subject_profiles):
    resp_nd = np.zeros((n_suj,) + (n_val,)*n_dim, dtype=np.float32)
    resp_1d = np.zeros((n_suj, n_dim, n_val), dtype=np.float32)

    for idx, subject in enumerate(subject_profiles):
        interaction = np.random.uniform(0.01, 0.1, size=n_val).astype(np.float32)

        for dim_idx, param in enumerate(parameter_names):
            resp_1d[idx, dim_idx, :] = subject[param]

        grids = np.meshgrid(*[resp_1d[idx, dim_idx, :] for dim_idx in range(n_dim)], indexing='ij')
        product = np.ones((n_val,)*n_dim)
        for grid in grids:
            product *= grid
        resp_nd[idx] = product

        for indices in np.ndindex((n_val,)*n_dim):
            i = indices[0]
            resp_nd[idx][indices] += subject['interaction_sign'] * interaction[i] * (subject['interaction_scale'] * (i + 1))

    min_vals = resp_nd.min(axis=tuple(range(1, n_dim+1)), keepdims=True)
    max_vals = resp_nd.max(axis=tuple(range(1, n_dim+1)), keepdims=True)
    resp_nd = (resp_nd - min_vals) / (max_vals - min_vals)

    return resp_1d, resp_nd

# Grille des paramètres
def get_parameter_grid(n_val, n_dim):
    val_p = np.arange(n_val, dtype=np.float32)
    grids = np.meshgrid(*[val_p]*n_dim, indexing='ij')
    X_test = np.stack(grids, axis=-1).reshape(-1, n_dim)
    return X_test

# Obtenir la réponse avec du bruit
def get_response(resp_nd, subject_idx, X_test, idx, noise_level):
    indices = tuple(int(X_test[idx, dim]) for dim in range(X_test.shape[1]))
    response = resp_nd[(subject_idx,) + indices]
    noise = np.random.normal(0, response * noise_level)
    return response + noise

# Fonction principale d'optimisation bayésienne
def run_bayesian_optimization(resp_nd, resp_1d, parameter_names, X_test_norm, X_test, bounds_discrete,
                              n_suj, n_iters, n_rnd, kappa, AF_name, noise_level, client):
    n_dim = X_test.shape[1]
    P_test = torch.zeros((n_suj, n_iters, n_dim + 1), dtype=torch.float32)
    rand_idx = np.random.permutation(len(X_test))[:n_rnd]
    for s in range(n_suj):
        print(f"Sujet {s+1}/{n_suj}")
        x_train = None
        y_train = None
        gp = None
        for i in range(n_iters):
            if i < n_rnd:
                next_idx = rand_idx[i]
            else:
                with torch.no_grad():
                    if AF_name == "UBC":
                        AF = UpperConfidenceBound(gp, beta=kappa**2, maximize=True)
                    elif AF_name == 'EI':
                        mu_sample_opt = torch.max(y_train)
                        AF = ExpectedImprovement(gp, best_f=mu_sample_opt, maximize=True)
                    elif AF_name == 'NEI':
                        AF = NoisyExpectedImprovement(gp, x_train, num_fantasies=20, maximize=True)
                    acq_values = AF(X_test_norm.unsqueeze(1)).detach().cpu().numpy()
                    next_idx = np.argmax(acq_values)
            x_next = X_test[next_idx]
            resp = get_response(resp_nd, s, X_test, next_idx, noise_level)
            P_test[s, i, :-1] = torch.from_numpy(x_next)
            P_test[s, i, -1] = torch.tensor(resp, dtype=torch.float32)
            x_train = P_test[s, :i+1, :-1]
            y_train = P_test[s, :i+1, -1].unsqueeze(-1)
            x_train_norm = normalize(x_train, bounds_discrete)
            y_train_std = standardize(y_train)
            if AF_name == "NEI":
                train_Yvar = torch.full_like(y_train_std, 0.15)
                gp = SingleTaskGP(x_train_norm, y_train_std, train_Yvar=train_Yvar)
            else:
                gp = SingleTaskGP(x_train_norm, y_train_std)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)

        best_idx = torch.argmax(P_test[s, :, -1])
        best_params_idx = P_test[s, best_idx, :-1].numpy().astype(int)
        best_response = P_test[s, best_idx, -1].item()

        best_params_values = {}
        for dim_idx, param in enumerate(parameter_names):
            best_params_values[param] = resp_1d[s, dim_idx, best_params_idx[dim_idx]]
        print(f"Meilleurs paramètres trouvés pour le sujet {s+1}:")
        for param, value in best_params_values.items():
            print(f"  {param} = {value}")
            ###
            f = open('stim.txt', 'a')
            f.write(str(value)+'\n')
            f.close()
            ###
        #if os.path.exists('hist_params.npy'):
        #    try:
        #        existing_array = np.load('hist_params.npy')
        #        new_array = np.vstack((existing_array, best_params_values))
        #    except ValueError:
        #        # Handle the case where the existing array is empty
        #        new_array = np.array([best_params_values])
        #else:
        #    new_array = np.array([best_params_values])
        #np.save('hist_params', new_array)
            ###
        print(f"Meilleure réponse : {best_response}")
        
        
        
        # Envoyer les paramètres optimaux via MQTT
        client.publish('GPBO/frequence', str(best_params_values.get('Freq', '')))
        client.publish('GPBO/amplitude', str(best_params_values.get('Amp', '')))
        client.publish('GPBO/dutycycle', str(best_params_values.get('DutyCycle', '')))
        print(f"Paramètres optimaux envoyés via MQTT : {best_params_values}")
    return P_test

if __name__ == "__main__":
    # Utilisation d'argparse pour récupérer les arguments de ligne de commande
    parser = argparse.ArgumentParser(description='Exécution de GPBO avec des paramètres personnalisés.')
    parser.add_argument('--n_iters', type=int, default=20, help='Nombre d\'itérations pour GPBO')
    parser.add_argument('--n_rnd', type=int, default=1, help='Nombre de points aléatoires pour GPBO')
    parser.add_argument('--kappa', type=float, default=3.0, help='Valeur de kappa pour GPBO')
    parser.add_argument('--AF_name', type=str, default="NEI", choices=["NEI", "EI", "UBC"], help='Nom de la fonction d\'acquisition')
    parser.add_argument('--noise_level', type=float, default=0.05, help='Niveau de bruit à ajouter aux réponses')

    args = parser.parse_args()

    # Paramètres globaux pour GPBO
    n_iters = args.n_iters
    n_rnd = args.n_rnd
    kappa = args.kappa
    AF_name = args.AF_name
    noise_level = args.noise_level

    # Configuration du client MQTT
    broker_address = "localhost"
    client = mqtt.Client("GPBO_Client")
    client.connect(broker_address, 1883, 60)
    client.loop_start()

    # Exécution directe de l'optimisation
    subject_profiles, parameter_names, n_dim, n_val = read_subject_profiles('./subject_3dim.txt')
    n_suj = len(subject_profiles)

    print(f"Nombre de sujets : {n_suj}")
    print(f"Nombre de paramètres (n_dim) : {n_dim}")
    print(f"Nombre de valeurs par paramètre (n_val) : {n_val}")
    print(f"Noms des paramètres : {parameter_names}")

    # Générer les réponses des sujets
    print("Génération des réponses des sujets...")
    resp_1d, resp_nd = generate_subject_responses(n_suj, n_val, n_dim, parameter_names, subject_profiles)
    print("Réponses générées.")

    X_test = get_parameter_grid(n_val, n_dim)
    X_test_norm = torch.from_numpy(X_test).float()

    l_bounds = [0.0]*n_dim
    u_bounds = [n_val - 1.0]*n_dim
    bounds_discrete = torch.tensor([l_bounds, u_bounds], dtype=torch.float32)
    X_test_norm = normalize(X_test_norm, bounds_discrete)

    print("Démarrage de l'optimisation bayésienne...")
    start_time = time()
    run_bayesian_optimization(resp_nd, resp_1d, parameter_names, X_test_norm, X_test, bounds_discrete, n_suj, n_iters, n_rnd, kappa, AF_name, noise_level, client)
    print(f"Optimisation terminée, it took {time() - start_time:.2f} seconds.")
    client.loop_stop()
    client.disconnect()
