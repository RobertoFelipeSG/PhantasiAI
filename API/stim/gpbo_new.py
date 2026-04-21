import numpy as np
import torch
import pandas as pd
import warnings
import time
import os
import csv
import threading
from pathlib import Path

import matplotlib
matplotlib.use('Agg') # ensures matplotlib works in an non-interactive backend
import matplotlib.pyplot as plt

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement, NoisyExpectedImprovement, UpperConfidenceBound
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.utils.transforms import normalize, standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

from config.connection_manager import logging
from config.config_manager import load_config

CONFIG = load_config()

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)

class GPBOOptimizer:
    def __init__(self, stimulator, dorsi_flag, n_iters, n_reps, recordings_directory, profiler, mqtt_client, client_topic, on_complete=None, on_stim_fail=None):
        self.file_path = None
        self.plot_lock = threading.Lock()
        self.stimulator = stimulator
        self.profiler = profiler
        self.dorsi_flag = dorsi_flag
        self.mqtt_client = mqtt_client
        self.recordings_directory = recordings_directory
        self.topic = client_topic
        self.on_complete = on_complete
        self.on_stim_fail = on_stim_fail
        
        # get ground truth/calibration data 
        parent_base_path = Path(__file__).parent.parent
        self.calb_dir = os.path.join(str(parent_base_path), "calibrate")
        os.makedirs(self.calb_dir, exist_ok=True)
        self.gt_path = os.path.join(str(self.calb_dir), "ground_truth.csv")
        
        self.n_optimizations = 0 # tracks how many optimization runs have been completed
        
        # set configuration parameters
        self.n_iters = n_iters
        self.n_reps = n_reps
        self.num_rand = CONFIG.get("num_rand")
        self.kappa = CONFIG.get("kappa")
        self.AF_name = CONFIG.get("AF_name")
        self.noise_level = CONFIG.get("noise_level")
        self.parameter_names = CONFIG.get("parameters")
        self.feature_names = CONFIG.get("features")

        # initialize overall optimizations directory
        self.opt_dir = os.path.join(str(self.recordings_directory), "optimizations")
        os.makedirs(self.opt_dir, exist_ok=True)

        # initialize state variables for optimization run
        self._initialize_optimization()

        # TO-DO: inject GP prior
        # we would probably do this at the start of each repetition?

    def _initialize_optimization(self):
        # state variables for one entire optimization run (10 repetitions)
        self.curr_iter = 0 # tracks how many iterations completed 
        self.curr_rep = 0 # tracks how many repetitions completed 
        self.opt_complete = False
        self.data_saved = False
        self.visuals_complete = False
        self.model_evals_complete = False
        self.history = {} # stores data, gp, and converged params of final iteration for each rep
        
        # state variables for current repetition, one iteration
        self.selected_params = None # last set of selected parameters 
        self.stim_success = False
        self.tested_params = torch.empty(0, 2)
        # initialize with default prior = default posterior mean and uncertainty maps 
        self.x_train = torch.empty(0, 2)
        self.y_train = torch.empty(0, 1) # single outcome
        self.gp_history = {} # stores posterior mean and variance for each iteration in one rep

        # create optimization folder with maps + stim log folders
        self.opt_log = [] # dict to store stimulation data 
        self._initialize_directory()

        # initialize the search grid (bc it's gonna be the same everytime)
        self.dutycycles = np.array(CONFIG.get("dutycycles"))
        self.frequencies = np.array(CONFIG.get("frequencies"))
        self.X_test, self.X_test_norm, self.bounds_discrete = self._initialize_input_grid()
        self._visualize_initial_maps()
    
    def _initialize_directory(self):  
        # directory to store visual heatmaps of posterior mean and standard deviations
        self.maps_dir = os.path.join(self.opt_dir, f"maps_{self.n_optimizations+1}")
        os.makedirs(self.maps_dir, exist_ok=True)

        # CSV to store stimulation params + outputs
        self.stim_data = os.path.join(self.opt_dir, f"stimulations_{self.n_optimizations+1}.csv")
    
    def _initialize_input_grid(self):
        '''
        Creates a grid of all possible stimulation parameter combinations of dutycycle x frequency 
        This is ran once (when the optimizer is initialized)
        Returns the X input grid and the upper and lower bounds
        '''
        grid_x, grid_y = np.meshgrid(self.dutycycles, self.frequencies, indexing='ij')
        X_test = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)
        X_test_norm = torch.from_numpy(X_test).float()

        l_bounds = [self.dutycycles.min(), self.frequencies.min()]
        u_bounds = [self.dutycycles.max(), self.frequencies.max()]
        bounds_discrete = torch.tensor([l_bounds, u_bounds], dtype=torch.float32)
        X_test_norm = normalize(X_test_norm, bounds_discrete)
        
        return X_test, X_test_norm, bounds_discrete
    
    def _visualize_initial_maps(self):
        '''
        Creates the initial Estimated Mean Map and Uncertainty Plot
        '''
        with self.plot_lock:
            grid_shape = (len(self.dutycycles), len(self.frequencies))
            estimated_map_0 = np.zeros(grid_shape).T # transpose
            uncertainty_0 = np.ones(grid_shape).T # transpose
    
            fig, axes = plt.subplots(1, 2, figsize=(10,4))
            extent = [(self.dutycycles.min() - 0.1), (self.dutycycles.max() + 0.1), 
                      (self.frequencies.min() - 5), (self.frequencies.max() + 5)]
        
            # plot estimated map
            im0 = axes[0].imshow(estimated_map_0, origin='lower', extent=extent, 
                                 aspect='auto', cmap='Reds', vmin=-1, vmax=1)
            axes[0].set_title("Estimated Map (Prior)")
            axes[0].set_xlabel("Duty Cycle")
            axes[0].set_ylabel("Frequency")
        
            # plot uncertainty
            im1 = axes[1].imshow(uncertainty_0, origin='lower', extent=extent, 
                                 aspect='auto', cmap='Reds', vmin=0, vmax=1)
            axes[1].set_title("Uncertainty (Prior)")
            axes[1].set_xlabel("Duty Cycle")
        
            plt.colorbar(im0, ax=axes[0])
            plt.colorbar(im1, ax=axes[1])
            
            filename = f"grids_prior.png"
            output_path = os.path.join(self.maps_dir, filename)
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()

        logging.info(f"[GPBO] Prior visualization saved.")
        
    def _visualize_maps(self, repetition, tested_params, converged_coord, 
                        estimated_map, uncertainty):
        '''
        Creates the Estimated Mean Map and Uncertainty plot for each iteration
        '''
        with self.plot_lock:
            # reshape 1D arrays back to 2D (dutycycle x frequency) and transpose
            grid_shape = (len(self.dutycycles), len(self.frequencies))
            estimated_map_2d = estimated_map.reshape(grid_shape).T
            uncertainty_2d = uncertainty.reshape(grid_shape).T
    
            # plotting
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            extent = [(self.dutycycles.min() - 0.1), (self.dutycycles.max() + 0.1), 
                      (self.frequencies.min() - 5), (self.frequencies.max() + 5)]
            
            # plot estimated mean maps
            plt1 = axes[0].imshow(estimated_map_2d, origin='lower', extent=extent, aspect='auto', cmap='Reds')
            axes[0].set_title(f"Estimated Map: Rep {repetition}")
            axes[0].set_xlabel("Duty Cycle")
            axes[0].set_ylabel("Frequency")
    
            # overlay tested points
            if len(tested_params) > 0:
                # previously tested points in black
                axes[0].scatter(tested_params[:, 0].numpy(), 
                                tested_params[:, 1].numpy(), 
                                c='black', s=60, zorder=4)
            if converged_coord is not None:
                axes[0].scatter(converged_coord[0], converged_coord[1], 
                            c='gold', marker='*', s=100, zorder=5)
    
            # plot uncertainty
            plt2 = axes[1].imshow(uncertainty_2d, origin='lower', extent=extent, aspect='auto', cmap='Reds')
            axes[1].set_title(f"Uncertainty: Rep {repetition}")
            axes[1].set_xlabel("Duty Cycle")
            
            plt.colorbar(plt1, ax=axes[0], fraction=0.046, pad=0.04)
            plt.colorbar(plt2, ax=axes[1], fraction=0.046, pad=0.04)
            
            filename = f"grids_rep_{repetition}.png"
            output_path = os.path.join(self.maps_dir, filename)
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
    
            #logging.info(f"[GPBO] Repetition {repetition} visualization saved.")

    def _visualize_all_maps(self):
        '''
        Runs after all repetitions are completed
        Generates an estimated mean and uncertainty map for each repetition (40 iterations)
        '''
        start_time = time.time()
        
        for rep_idx, rep_data in self.history.items():
            posteriors = rep_data['posteriors']

            # get final gp from current repetition
            final_iter = max(posteriors.keys())
            final_mean = posteriors[final_iter]['mean']
            final_var = posteriors[final_iter]['variance']
            
            # plot estimated mean and uncertainty map 
            self._visualize_maps(rep_idx, rep_data['tested_params'], rep_data['converged_coord'],
                                 final_mean, final_var) 

        logging.info(f"[GPBO] Visualizations complete. Duration: {time.time() - start_time:.2f} seconds.")
    
    def _visualize_evals(self, rep_idx, p_explore_list, p_exploit_list):
        ''' 
        Visualizes the evolution of the exploration and exploitation metrics for one repetition
        '''
        iters = list(range(1, self.n_iters + 1))
        
        with self.plot_lock:
        
            plt.figure(figsize=(8, 5))
            plt.plot(iters, p_explore_list, label='Exploration', marker='o', linestyle='--')
            plt.plot(iters, p_exploit_list, label='Exploitation', marker='x', linestyle='-')
            
            plt.title(f"Performance Evolution: Repetition {rep_idx}")
            plt.xlabel("Iteration")
            plt.ylabel("Efficacy Ratio")
            plt.ylim(0, 1.1)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Save the figure in the maps directory
            filename = f"performance_{rep_idx}.png"
            output_path = os.path.join(self.maps_dir, filename)
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
        
    def _perform_model_eval(self, backup=False):
        '''
        Calculates the exploration and exploitation metrics for each query to evaluate the model
        This is ran after all repetitions are complete
        '''
        start_time = time.time()
        
        # load ground truth data and get optimum
        if not os.path.exists(self.gt_path):
            logging.warning("[GPBO] Evaluation skipped: ground_truth.csv not found")
            return
        gt_df = pd.read_csv(self.gt_path)
        
        if len(gt_df) != (len(self.dutycycles) * len(self.frequencies)):
            logging.warning("[GPBO] Ground truth table does not match stimulation parameter options")
            return
        gt_max_val = gt_df['max_amplitude'].max()

        all_eval_data = []

        # iter through history to get posterior means and tested parameters for each repetition
        for rep_idx, rep_data in self.history.items():
            posteriors = rep_data['posteriors']
            tested_params = rep_data['tested_params']
            
            p_explore_list = []
            p_exploit_list = []

            # iter through the posterior mean in ascending order
            for iter_idx in sorted(posteriors.keys()):
                
                # calculate exploration metric: efficacy of predicted optimum
                pred_means = posteriors[iter_idx]['mean']
                pred_opt_idx = np.argmax(pred_means)
                pred_opt_params = self.X_test[pred_opt_idx]

                pred_max_val = gt_df[
                    np.isclose(gt_df['dutycycle'], pred_opt_params[0]) & 
                    np.isclose(gt_df['frequency'], pred_opt_params[1])
                ]['max_amplitude'].values[0]
                p_explore = pred_max_val / gt_max_val
                p_explore_list.append(p_explore)

                # calculate exploitation metric: efficacy of queried parameters
                # note: iter_idx starts at 1; tested_params starts at 0
                if (iter_idx - 1) < len(tested_params):
                    queried_params = tested_params[iter_idx - 1]
                    queried_val = gt_df[
                        np.isclose(gt_df['dutycycle'], float(queried_params[0])) & 
                        np.isclose(gt_df['frequency'], float(queried_params[1]))
                    ]['max_amplitude'].values[0]
                    p_exploit = queried_val / gt_max_val
                else:
                    p_exploit = None # avoids index error
                p_exploit_list.append(p_exploit)
                    
                all_eval_data.append({
                    'rep': rep_idx,
                    'iter': iter_idx,
                    'p_explore': p_explore,
                    'p_exploit': p_exploit
                })

            # create plots to visualize performance metrics over iterations
            self._visualize_evals(rep_idx, p_explore_list, p_exploit_list)

        # save evaluations to CSV
        eval_df = pd.DataFrame(all_eval_data)
        if backup: eval_path = os.path.join(self.opt_dir, f"model_evals_backup.csv")
        else: eval_path = os.path.join(self.opt_dir, f"model_evals_{self.n_optimizations}.csv")
        eval_df.to_csv(eval_path, index=False)

        logging.info(f"[GPBO] Evaluations complete. Duration: {time.time() - start_time:.2f} seconds.")
                    
    def _get_response(self):
        '''
        Calculates the mean value of each feature across the N analysis trials from features.txt
        This is ran at every iteration to get the "output"
        Returns the mean of each of the EMG response features

        CURRENT SETUP: 
        - features = max amplitude
        - n trials = 1
        '''
        data = pd.read_csv(self.file_path, sep=';') # Rows split by semicolon ;

        # validate features.txt
        if data.empty:
            logging.error("[GPBO] File is empty or no subjects found")
            raise ValueError("File is empty or no subjects found")
        
        resp_features = [col for col in self.feature_names if col in data.columns]
        feature_values = data.iloc[0] # all values are stored in the first row
        raw_data = {}
            
        for feat in self.feature_names:
            raw_str = str(feature_values[feat])
            split_vals = raw_str.split(',') if ',' in raw_str else raw_str.split()
            float_vals = [float(v) for v in split_vals if v.strip()] # ignore empty strings
            if not float_vals:
                logging.warning(f"Feature {feat} resulted in empty list. Setting to 0.0")
                raw_data[feat] = [0.0]
            else:
                raw_data[feat] = float_vals
 
        try:
            # get RMS for each selected feature
            if len(self.feature_names) > 1:
            # TO-DO: pre-scalarize: combine all features into one response value
                feats = [
                    np.sqrt(np.mean((raw_data.get('max_amplitude', [0.0])**2))),
                    np.sqrt(np.mean((raw_data.get('mean_frequency', [0.0])**2))),
                    np.sqrt(np.mean((raw_data.get('median_frequency', [0.0])**2)))
                ]
                weights = np.array([1.0, 1.0, 1.0]) # Adjust weights as needed
                resp = np.dot(feats, weights)
            else: # RIGHT NOW THIS IS WHAT WE ARE DOING 
                val = np.array(raw_data.get('max_amplitude', [0.0]))
                resp = np.sqrt(np.mean(val**2))
            
        except Exception as e:
            logging.error(f"Error calculating means: {e}") 
        
        return torch.tensor([resp], dtype=torch.float32)

    def _run_optimization(self, file_path, curr_trial):
        start_opt_time = time.time() # start timer for current optimization iteration
        self.dorsi_flag.clear() # reset dorsiflexion flag to trigger stimulation
        self.file_path = file_path
        
        # OPTIMIZATION CHECK: if all repetitions+iterations complete, start new optimization process
        if self.opt_complete:
            
            if self.on_complete:
                logging.info("[GPBO] Triggering auto-stop from main.py...")
                self.on_complete()
                return
            
            else:
                logging.info("[GPBO] Starting new optimization run...")
                self._initialize_optimization() # reset ALL state variables for an optimization run
        
        # ITERATION CHECK: if this is not the first iteration and stimulation ran successfully, record input->output pair 
        if (self.selected_params is not None) and self.stim_success:
            last_response = self._get_response() # get EMG response (output)
            
            self.curr_iter += 1 # increment iteration
            self.stim_success = False # reset success state
            # logging.info(f"[GPBO] Rep:{self.curr_rep + 1}, Iter:{self.curr_iter} | Response for {self.selected_params}: {last_response.item():.2f}")

            # add to dataset
            self.x_train = torch.cat([self.x_train, torch.tensor([self.selected_params]).float()])
            self.y_train = torch.cat([self.y_train, last_response.unsqueeze(0)])
            self.tested_params = torch.cat([self.tested_params,
                                            torch.tensor([self.selected_params]).float()])

            # add to data log
            iter_data = {
                'rep': self.curr_rep + 1, 
                'iter': self.curr_iter,
                'duration': self.curr_opt_time,
            }
            for i, p_name in enumerate(self.parameter_names):
                iter_data[p_name] = float(self.selected_params[i])
            for i, f_name in enumerate(self.feature_names):
                iter_data[f"RMS_{f_name}"] = float(last_response[i])

            self.opt_log.append(iter_data)

        # REPETITION CHECK: if n iterations done, repetition is complete
        if self.curr_iter >= self.n_iters:
            
            # fit model on final input->output pair
            x_train_norm = normalize(self.x_train, self.bounds_discrete)
            y_train_std = standardize(self.y_train)
            gp = SingleTaskGP(x_train_norm, y_train_std)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)

            gp.eval()
            gp.likelihood.eval()
        
            # save posterior mean and variance for current iteration
            with torch.no_grad():
                posterior = gp.posterior(self.X_test_norm.unsqueeze(1))
                posterior_mean = posterior.mean.squeeze()
                
                self.gp_history[self.curr_iter]  = {
                    'mean': posterior_mean.cpu().numpy(),
                    'variance': posterior.variance.squeeze().cpu().numpy()
                }
                
                converged_idx = posterior_mean.argmax().item()
                converged_coord = self.X_test[converged_idx]

            self.curr_rep += 1
            logging.info(f"[GPBO] Rep {self.curr_rep}; Converged on coordinates: {converged_coord}")

            # save history to overall dictionary
            self.history[self.curr_rep] = {
                'x_train': self.x_train.clone(),
                'y_train': self.y_train.clone(),
                'state_dict': gp.state_dict(),
                'tested_params': self.tested_params.clone(),
                'converged_coord': converged_coord,
                'posteriors': self.gp_history
            }

            # reset state repetition variables 
            self.selected_params = None
            self.curr_iter = 0
            self.x_train = torch.empty(0, 2)
            self.y_train = torch.empty(0, 1)
            self.tested_params = torch.empty(0, 2)
            self.gp_history = {}

            # check if all repetitions complete
            if self.curr_rep >= self.n_reps:
                self.opt_complete = True
                self.n_optimizations += 1

                # add to CSV with stimulation data
                df = pd.DataFrame(self.opt_log)
                df.to_csv(self.stim_data, index=False)
                self.data_saved = True
                
                # visualize all maps
                try: 
                    self._visualize_all_maps()
                    self.visuals_complete = True
                except Exception as e:
                    logging.error(f"[GPBO] Visualization error: {e}")

                # evaluate model: perform exploration/exploitation evaluation
                try:
                    self._perform_model_eval()
                    self.model_evals_complete = True
                except Exception as e:
                    logging.error(f"[GPBO] Model evaluation error: {e}")
                
                logging.info("[GPBO] All repetitions completed and maps generated!")
                return
        
        # SELECT NEXT POINT TO TEST
        # PHASE 1: Broad search (random sampling for n iterations if no GP prior)
        if self.curr_iter < self.num_rand:
            # logging.info(f"[GPBO] Random sampling for first iteration")
            next_idx = np.random.randint(len(self.X_test))
            self.selected_params = self.X_test[next_idx]

        # PHASE 2: Model guided search (update posterior mean and uncertainty maps)
        else:
            # fit GP to the EMG response outputs collected so far
            x_train_norm = normalize(self.x_train, self.bounds_discrete)
            y_train_std = standardize(self.y_train)
            
            # create single task GP
            if self.AF_name == "NEI":
                train_Yvar = torch.full_like(y_train_std, 0.15)
                gp = SingleTaskGP(x_train_norm, y_train_std, train_Yvar=train_Yvar)
            else:
                gp = SingleTaskGP(x_train_norm, y_train_std)
            
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)

            gp.eval()
            gp.likelihood.eval()

            # save posterior mean and variance for current iteration
            with torch.no_grad():
                posterior = gp.posterior(self.X_test_norm.unsqueeze(1))
                self.gp_history[self.curr_iter]  = {
                    'mean': posterior.mean.squeeze().cpu().numpy(),
                    'variance': posterior.variance.squeeze().cpu().numpy()
            }

            # define acquistion function
            if self.AF_name == "UCB":
                AF = UpperConfidenceBound(gp, beta=self.kappa**2, maximize=True)
            elif self.AF_name == 'EI':
                best_f = torch.max(y_train_std)
                AF = ExpectedImprovement(gp, best_f=best_f, maximize=True)
            elif self.AF_name == 'NEI':
                AF = NoisyExpectedImprovement(gp, x_train_norm, num_fantasies=20, maximize=True)
            
            # get best params based on AF
            with torch.no_grad():
                acq_vals = AF(self.X_test_norm.unsqueeze(1))
                next_idx = acq_vals.argmax().item()

            self.selected_params = self.X_test[next_idx]
        
        # log and publish chosen params
        best_params = {
            self.parameter_names[0]: float(self.selected_params[0]),
            self.parameter_names[1]: float(self.selected_params[1])
        }

        # store time duration for current optimization process
        self.curr_opt_time = time.time() - start_opt_time
        self.profiler.log_metric(curr_trial, "opt_iter", self.curr_opt_time)

        dutycycle = best_params.get('dutycycle')
        frequency = best_params.get('frequency')
        logging.info(f"[GPBO] Optimization done: FREQUENCY: {frequency}; DUTYCYCLE: {dutycycle}, waiting for event marker to start stimulation...")
        self.dorsi_flag.wait() # wait until event marker flag is raised to signal start of dorsiflexion

        self.dorsi_flag.clear() # clear the flag as soon as dorsiflexion begins
        try:
            self.stimulator.run(best_params, curr_trial)
            self.stim_success = True
            self.profiler.mark_process_complete(curr_trial)
        except (OSError, FileNotFoundError) as e:
            logging.error(f"[GPBO] Hardware Stim Error: Cannot access GPIO chip. {e}")
            if self.on_stim_fail: self.on_stim_fail()
        except KeyError as e:
            logging.error(f"[GPBO] Parameter Stim Error: Missing key in best_params: {e}")
            if self.on_stim_fail: self.on_stim_fail()
        except Exception as e:
            logging.error(f"[GPBO] Unexpected error during stimulation: {type(e).__name__}: {e}")
            if self.on_stim_fail: self.on_stim_fail()

    def run(self, file_path, curr_trial):
        #logging.info("[GPBO] Starting Bayesian optimization...")
        self._run_optimization(file_path, curr_trial)

    def handle_stop(self):
        ''' 
        Backup to save optimization data for current optimization run 
        (in case of sudden stop) 
        '''
        
        # add to CSV with stimulation data
        if not self.data_saved:
            df = pd.DataFrame(self.opt_log)
            filepath = os.path.join(self.opt_dir, f"stimulations_backup.csv")
            df.to_csv(filepath, index=False)

        # visualize all maps
        if not self.visuals_complete:
            try: 
                self._visualize_all_maps()
            except Exception as e:
                logging.error(f"[GPBO] Visualization error: {e}")

        # evaluate model: perform exploration/exploitation evaluation
        if not self.model_evals_complete:
            try:
                self._perform_model_eval(backup=True)
            except Exception as e:
                logging.error(f"[GPBO] Model evaluation error: {e}")
    