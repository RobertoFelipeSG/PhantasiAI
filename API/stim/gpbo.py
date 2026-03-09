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
LOG_FILE = Path(__file__).parent.parent / "test_timings.txt"

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)

class GPBOptimizer:
    def __init__(self, mqtt_client, client_topic):
        self.file_path = None
        self.mqtt_client = mqtt_client
        self.topic = client_topic
        self.base_path = Path(__file__).parent
        self.output_file = self.base_path / "stim.txt"
        
        # set configuration parameters
        self.iters = CONFIG.get("iterations")
        self.num_rand = CONFIG.get("num_rand")
        self.kappa = CONFIG.get("kappa")
        self.AF_name = CONFIG.get("AF_name")
        self.noise_level = CONFIG.get("noise_level")

    def initialize_simulation(self, file_path):
        self.file_path = file_path
        
        # Load classification data
        # get the param values, names, and number for each subject (min/max amp, mean/median freq)
        self.subject_profiles, self.parameter_names, self.n_dim, self.n_val = self._read_subject_profiles(self.file_path)
        self.n_sub = len(self.subject_profiles)

        # Initialize simulation based on data 
        # generates synthetic subject response data
        # stores in 1D grid for param values and ND total grid (4D here) 
        self.resp_1d, self.resp_nd = self._generate_subject_responses(self.n_sub, self.n_val, self.n_dim, self.parameter_names, self.subject_profiles)
        logging.info(f"[GPBO] Generated subject responses for {self.n_sub} subjects.")

        # Precalculate grid
        # calculates a testing grid of all combinations...
        self.X_test = self._get_parameter_grid(self.n_val, self.n_dim)
        self.X_test_norm = torch.from_numpy(self.X_test).float()

        # Get bounds
        # ...and then normalizes based on the bounds 
        l_bounds = [0.0]*self.n_dim
        u_bounds = [self.n_val - 1.0]*self.n_dim
        self.bounds_discrete = torch.tensor([l_bounds, u_bounds], dtype=torch.float32)
        self.X_test_norm = normalize(self.X_test_norm, self.bounds_discrete)

    def _read_subject_profiles(self, file_name):
        '''
        This reads the file and defines the search space of all possible parameter options

        Returns:
        subject_profiles: list of subject dictionaries
            - subject[param]: each param and their values
            - subject[interaction scale]: 0.5
            - subject[interaction sign]: 1
            - always going to be 0.5 and 1 because the rows don't exist in our features.txt
        parameter_names: list of params (min/max amplitude, mean/median frequency)
        n_dim: number of params
        n_val: number of values for each param
        '''
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
                    logging.error(f"[GPBO] Mismatch in value counts for {param} of subject {row['sujet']}")
                    raise ValueError(f"Mismatch in value counts for {param} of subject {row['sujet']}.")
                subject[param] = values
            
            n_val_list.append(n_val_subject)
            subject['interaction_scale'] = float(row.get('Interaction_Scale', 0.5)) # default to 0.5 if file missing row
            subject['interaction_sign'] = int(row.get('Interaction_Sign', 1)) # default to 1 if file missing row
            subject_profiles.append(subject)
        
        if not n_val_list:
            logging.error("[GPBO] File is empty or no subjects found")
            raise ValueError("File is empty or no subjects found")
        
        if len(set(n_val_list)) != 1:
            logging.error("[GPBO] All subjects must have the same number of parameter values.")
            raise ValueError("All subjects must have the same number of parameter values.")
        
        n_val = n_val_list[0]
        return subject_profiles, parameter_names, n_dim, n_val
    
    def _generate_subject_responses(self, n_suj, n_val, n_dim, parameter_names, subject_profiles):
        '''
        This generates synthetic response data for each subject (that will be used as our GT)

        Returns:
        - resp_1d: 1-dimensional array of all param values
        - resp_nd: n-dimensional array from 1D param values
            -> a grid of all possible combinations of min_amp x max_amp x mean_freq x max_freq
            with their generated synthetic response 
            -> response = normalized product of each param value + interaction term
        '''
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
            resp_nd[idx] = product # response at each combination is just the product of each param value

            for indices in np.ndindex((n_val,)*n_dim):
                i = indices[0]
                resp_nd[idx][indices] += subject['interaction_sign'] * interaction[i] * (subject['interaction_scale'] * (i + 1))
                # interaction term is added to each response

        min_vals = resp_nd.min(axis=tuple(range(1, n_dim+1)), keepdims=True)
        max_vals = resp_nd.max(axis=tuple(range(1, n_dim+1)), keepdims=True)
        resp_nd = (resp_nd - min_vals) / (max_vals - min_vals)
        # normalize each response

        return resp_1d, resp_nd
    
    def _get_parameter_grid(self, n_val, n_dim):
        ''' 
        This generates a parameter grid to be used for testing

        Returns:
        - X_test: array of all parameter combinations as indices
        '''
        val_p = np.arange(n_val, dtype=np.float32)
        grids = np.meshgrid(*[val_p]*n_dim, indexing='ij')
        X_test = np.stack(grids, axis=-1).reshape(-1, n_dim)
        
        return X_test

    def _get_response(self, resp_nd, subject_idx, X_test, idx):
        ''' 
        This adds some noise and gets the response value for each combination

        Returns:
        - response + noise: there response value from resp_nd (ground truth) and noise 
        '''
        indices = tuple(int(X_test[idx, dim]) for dim in range(X_test.shape[1]))
        response = resp_nd[(subject_idx,) + indices]
        noise = np.random.normal(0, response * self.noise_level)
        
        return response + noise
    
    def _run_bayesian_optimization(self):
        # P_test stores all iterations for all subjects
        # +1 holds the response score
        n_dim = self.X_test.shape[1]
        P_test = torch.zeros((self.n_sub, self.iters, n_dim + 1), dtype=torch.float32)
        # these are the random initial indices to test
        rand_idx = np.random.permutation(len(self.X_test))[:self.num_rand]
        
        # runs the optimization loop for every subject 
        for s in range(self.n_sub):
            print(f"Sujet {s+1}/{self.n_sub}")
            # initialize the training data
            x_train = None
            y_train = None
            gp = None

            # for every subject it runs the loop for _ iterations
            for i in range(self.iters):
                
                # for the first _ steps the algo picks random indices to test
                if i < self.num_rand:
                    next_idx = rand_idx[i]
                # once it has gone through _ random selections then it picks via an AF 
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
                
                x_next = self.X_test[next_idx] # get the parameter values we will test next
                resp = self._get_response(self.resp_nd, s, self.X_test, next_idx) # get the response of that index + some noise (this is our "real" response)
                P_test[s, i, :-1] = torch.from_numpy(x_next)
                P_test[s, i, -1] = torch.tensor(resp, dtype=torch.float32) # store the response for this iteration
                # update the training data with our tested params and the associated response
                x_train = P_test[s, :i+1, :-1]
                y_train = P_test[s, :i+1, -1].unsqueeze(-1)
                x_train_norm = normalize(x_train, self.bounds_discrete)
                y_train_std = standardize(y_train)
                
                if self.AF_name == "NEI":
                    train_Yvar = torch.full_like(y_train_std, 0.15)
                    gp = SingleTaskGP(x_train_norm, y_train_std, train_Yvar=train_Yvar)
                else:
                    gp = SingleTaskGP(x_train_norm, y_train_std)
                # re-train the GP with the data collected so far
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                fit_gpytorch_mll(mll)

            best_idx = torch.argmax(P_test[s, :, -1]) # gets index of the maximum score from P_test 
            best_params_idx = P_test[s, best_idx, :-1].numpy().astype(int) # gets index of the params of the maximum score
            best_response = P_test[s, best_idx, -1].item() # gets the maximum score

            best_params_values = {} # gets the best params
            for dim_idx, param in enumerate(self.parameter_names):
                idx_val = best_params_idx[dim_idx]
                best_params_values[param] = self.resp_1d[s, dim_idx, idx_val]
            
            logging.info(f"[GPBO] Optimized response for subject {s+1}: {best_response}")
            logging.info(f"[GPBO] Optimized parameters for subject {s+1}: {best_params_values}")

            # Write to .txt file
            try:
                with open(self.output_file, 'w') as f:
                    for param, value in best_params_values.items():
                        f.write(str(value)+'\n')
            except IOError as e:
                logging.error(f"[GPBO] Failed to write to stim.txt: {e}")

            # Publish optimized parameters via MQTT
            self.mqtt_client.publish(f"{self.topic}/GPBO/min-amplitude", str(best_params_values.get('min_amplitude', '')))
            self.mqtt_client.publish(f"{self.topic}/GPBO/amplitude", str(best_params_values.get('amplitude', '')))
            self.mqtt_client.publish(f"{self.topic}/GPBO/mean-frequency", str(best_params_values.get('mean_frequency', '')))
            self.mqtt_client.publish(f"{self.topic}/GPBO/frequency", str(best_params_values.get('median_frequency', '')))
            logging.info(f"[GPBO] Parameters published via MQTT")


    def run(self):
        logging.info("[GPBO] Starting Bayesian optimization...")
        start_time = time()
        
        self._run_bayesian_optimization()
        
        message = f"[GPBO] Optimization completed. Duration: {time() - start_time:.2f} seconds."
        logging.info(message)
        
        try:
            with open(LOG_FILE, "a") as f:
                f.write(f"{message}\n")
        except OSError as e:
            logging.error(f"Could not write to timing file: {e}")