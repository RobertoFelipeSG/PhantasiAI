import numpy as np
import torch
import pandas as pd
import warnings
from pathlib import Path
from time import time

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement, NoisyExpectedImprovement, UpperConfidenceBound
from botorch.utils.transforms import normalize, standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

from config.connection_manager import logging
from config.config_manager import load_config

CONFIG = load_config()

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)

class GPBOptimizer:
    def __init__(self, file_path, mqtt_client):
        self.file_path = file_path
        self.mqtt_client = mqtt_client
        self.base_path = Path(__file__).parent
        
        # set configuration parameters
        self.iters = CONFIG.get("iterations")
        self.num_rand = CONFIG.get("num_rand")
        self.kappa = CONFIG.get("kappa")
        self.AF_name = CONFIG.get("AF_name")
        self.noise_level = CONFIG.get("noise_level")

        # Load classification data
        self.subject_profiles, self.parameter_names, self.n_dim, self.n_val = self._read_subject_profiles(self.file_path)
        self.n_sub = len(self.subject_profiles)

        # Initialize simulation based on data 
        self.resp_1d, self.resp_nd = self._generate_subject_responses(self.n_sub, self.n_val, self.n_dim, self.parameter_names, self.subject_profiles)
        logging.info(f"Generated subject responses for {self.n_sub} subjects.")

        # Precalculate grid
        self.X_test = self._get_parameter_grid(self.n_val, self.n_dim)
        self.X_test_norm = torch.from_numpy(self.X_test).float()

        # Get bounds
        l_bounds = [0.0]*self.n_dim
        u_bounds = [self.n_val - 1.0]*self.n_dim
        self.bounds_discrete = torch.tensor([l_bounds, u_bounds], dtype=torch.float32)
        self.X_test_norm = normalize(self.X_test_norm, self.bounds_discrete)

    def _read_subject_profiles(self, file_name):
        data = pd.read_csv(file_name, sep=';') # Rows split by semicolon ;
        
        subject_profiles = []
        parameter_names = [col for col in data.columns if col not in ['sujet', 'Interaction_Scale', 'Interaction_Sign']]
        n_dim = len(parameter_names)
        n_val_list = []
        
        for idx, row in data.iterrows():
            subject = {}
            n_val_subject = None
            
            for param in parameter_names:
                raw_str = str(row[param])
                split_values = raw_str.split(',') if ',' in raw_str else raw_str.split()
                values = np.array(list(map(float, split_values)), dtype=np.float32)
                
                if n_val_subject is None:
                    n_val_subject = len(values)
                elif len(values) != n_val_subject:
                    logging.error(f"Mismatch in value counts for {param} of subject {row['sujet']}")
                    raise ValueError(f"Mismatch in value counts for {param} of subject {row['sujet']}.")
                subject[param] = values
            
            n_val_list.append(n_val_subject)
            subject['interaction_scale'] = float(row.get('Interaction_Scale', 0.5)) # default to 0.5 if file missing row
            subject['interaction_sign'] = int(row.get('Interaction_Sign', 1)) # default to 1 if file missing row
            subject_profiles.append(subject)
        
        if not n_val_list:
            logging.error("File is empty or no subjects found")
            raise ValueError("File is empty or no subjects found")
        
        if len(set(n_val_list)) != 1:
            logging.error("All subjects must have the same number of parameter values.")
            raise ValueError("All subjects must have the same number of parameter values.")
        
        n_val = n_val_list[0]
        return subject_profiles, parameter_names, n_dim, n_val
    
    def _generate_subject_responses(self, n_suj, n_val, n_dim, parameter_names, subject_profiles):
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
    
    def _get_parameter_grid(self, n_val, n_dim):
        val_p = np.arange(n_val, dtype=np.float32)
        grids = np.meshgrid(*[val_p]*n_dim, indexing='ij')
        X_test = np.stack(grids, axis=-1).reshape(-1, n_dim)
        
        return X_test

    def _get_response(self, resp_nd, subject_idx, X_test, idx):
        indices = tuple(int(X_test[idx, dim]) for dim in range(X_test.shape[1]))
        response = resp_nd[(subject_idx,) + indices]
        noise = np.random.normal(0, response * self.noise_level)
        
        return response + noise
    
    def _run_bayesian_optimization(self):
        n_dim = self.X_test.shape[1]
        P_test = torch.zeros((self.n_sub, self.iters, n_dim + 1), dtype=torch.float32)
        rand_idx = np.random.permutation(len(self.X_test))[:self.num_rand]
        
        for s in range(self.n_sub):
            print(f"Sujet {s+1}/{self.n_sub}")
            x_train = None
            y_train = None
            gp = None
            
            for i in range(self.iters):
                
                if i < self.num_rand:
                    next_idx = rand_idx[i]
                else:
                    with torch.no_grad():
                        if self.AF_name == "UBC":
                            AF = UpperConfidenceBound(gp, beta=self.kappa**2, maximize=True)
                        elif self.AF_name == 'EI':
                            mu_sample_opt = torch.max(y_train)
                            AF = ExpectedImprovement(gp, best_f=mu_sample_opt, maximize=True)
                        elif self.AF_name == 'NEI':
                            AF = NoisyExpectedImprovement(gp, x_train, num_fantasies=20, maximize=True)
                        acq_values = AF(self.X_test_norm.unsqueeze(1)).detach().cpu().numpy()
                        next_idx = np.argmax(acq_values)
                
                x_next = self.X_test[next_idx]
                resp = self._get_response(self.resp_nd, s, self.X_test, next_idx)
                P_test[s, i, :-1] = torch.from_numpy(x_next)
                P_test[s, i, -1] = torch.tensor(resp, dtype=torch.float32)
                x_train = P_test[s, :i+1, :-1]
                y_train = P_test[s, :i+1, -1].unsqueeze(-1)
                x_train_norm = normalize(x_train, self.bounds_discrete)
                y_train_std = standardize(y_train)
                if self.AF_name == "NEI":
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
            for dim_idx, param in enumerate(self.parameter_names):
                idx_val = best_params_idx[dim_idx]
                best_params_values[param] = self.resp_1d[s, dim_idx, idx_val]
            
            logging.info(f"Optimized response for subject {s+1}: {best_response}")
            logging.info(f"Optimized parameters for subject {s+1}: {best_params_values}")

            # Write to .txt file
            output_file = self.base_path / "stim.txt"
            try:
                with open(output_file, 'w') as f:
                    for param, value in best_params_values.items():
                        f.write(str(value)+'\n')
            except IOError as e:
                logging.error(f"Failed to write to stim.txt: {e}")

            # Publish optimized parameters via MQTT
            self.mqtt_client.publish('GPBO/min-amplitude', str(best_params_values.get('min_amplitude', '')))
            self.mqtt_client.publish('GPBO/amplitude', str(best_params_values.get('amplitude', '')))
            self.mqtt_client.publish('GPBO/mean-frequency', str(best_params_values.get('mean_frequency', '')))
            self.mqtt_client.publish('GPBO/median-frequency', str(best_params_values.get('median_frequency', '')))
            logging.info(f"Parameters published via MQTT")


    def run(self):
        logging.info("Starting Bayesian optimization...")
        start_time = time()
        
        self._run_bayesian_optimization()
        logging.info(f"Optimization completed. Duration: {time() - start_time:.2f} seconds.")