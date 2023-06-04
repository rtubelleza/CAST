from spinalsim import * # For BSG rate-based model
from brian2 import * # For Brian2 spiking model

""" Contains classes for simulations using various software/packages/codes, for
rate and spiking based neuron models. 

    Changes 27/4/2022:
    - Removed dependency for matrix. Only need edgelist. self.bmap, N neurons
      can be extracted from just the edgelist. Main reason for doing this is
      converting edgelist to matrix is time consuming / computationlly expensive
      especially when done in pandas/np. Defining synaptic connections here does
      not require a connection matrix.
    
    Changes 10/5/2023:
    - Modified spike times for simpler implementation. Also includes option for 
      irregularly spaced spikes.
"""

class BSG():
    """ Note that majority of the code here is from BSG. Cite appropriately. """
    def __init__(self, conn_symmatrix, neuron_master):
        self.conn_symmatrix = conn_symmatrix # Connectivity Matrix, must be ei. # To as rows, From as cols
        self.neuron_master = neuron_master # Neuron master accompanied by conn_symmatrix.
        self.eigenspectrum = None # Tuple [eigenvalue, eigenvector]
        self.start_neuron_rix = None
        self.neuron_params = None # 
        self.stimulus_params = None
        self.R = None # firing rate across time steps, for all neurons
    
    def set_start_neuron(self, bodyId):
        # Define a start neuron as input to the system
        # Assuming we have a dataframe; square matrix for connectivity

        # For multiple starts
        #if bodyId
        # Get list of bodyIds in pre
        if type(bodyId) == list:
            ix = []
            for ids in bodyId:
                ix.append(list(self.conn_symmatrix).index(ids))
            self.start_neuron_rix = ix

        elif type(bodyId) == int:
            self.start_neuron_rix = list(self.conn_symmatrix.index).index(bodyId)
        
    def simulate(self, **model_kwargs):
        """ Code from fig2.py, BSG """

        # Default parameters
        self.neuron_params = {
            'tau_V' : 50., # combined neuronal and synaptic timescale (ms), determines frequency of updating rate
            'seed' : 5, # random seed for noise generator
            'stim_func' : stim_varying_individual, #defines type of input I_e | stim_varying_individual: stimulate one neuon
            'gain_func' : gain_const_common, # defines type of gain
            'I_e':40., # external (synaptic) input; [0, 0, 20, 20, 0, 0]
            'noise_ampl' : 1., # standard deviation of neuronal noise term
            'f_V_func':tanh_f_V, # firing rate function 
            'threshold':20, # threshold V_th (mV); Level at which slope of firing rate is max. Relates to baseline firing rate
            'gain': 1., # default gain
            'fmax' : 50, # maximal firing, relative to threshold (1/s)
            'V_init':0., # initial value for V
            't_steps' : 10000, # simulation time (ms)
            }
            
        self.stimulus_params = {
            'stim_start': 2000,
            'stim_stop': 9000
        }

        # Change any neuron params
        for key, value in model_kwargs.items():
            if key in self.neuron_params.keys():
                self.neuron_params[key] = value
            elif key in self.stimulus_params.keys():
                self.stimulus_params[key] = value

        # Set up input vector. Stimulus control over N neurons and t time.
        stim_start = self.stimulus_params['stim_start']
        stim_stop = self.stimulus_params['stim_stop']
        I_e = self.neuron_params['threshold'] # 20
        I_e_vec = np.zeros(self.neuron_params['t_steps']) #[0, 0, 0, 0] 4 time steps
        I_e_vec[stim_start:stim_stop] = self.neuron_params['I_e'] #[0, 20, 20, 0] If start=1, stop =2
       
        # Give the stimulus train to specified neuron(s)
        N = self.conn_symmatrix.shape[0]
        I_e_vec_master = np.zeros((N, self.neuron_params['t_steps'])) #rows, tstep cols
        # Set the neuron or neurons to stimulate
        if type(self.start_neuron_rix) == int:
            I_e_vec_master[self.start_neuron_rix] = I_e_vec 
        elif type(self.start_neuron_rix) == list:
            for ids in self.start_neuron_rix:
                I_e_vec_master[ids] = I_e_vec

        # I_e_vec should be a matrix for a given neuron, i.e. for one neuron to be stimulated
        # If we know neuron 2 is our start neuron, and its index is 1, then should result in below
        # I_e[i, t]; [[0, 0, 0, 0], -- Neuron 1
                    # [0, 20, 20, 20]] -- Neuron 2
        self.neuron_params['I_e'] = I_e_vec_master 

        # Connectivity Matrix W, specify number of neurons, and exc/inh fracs.
        # Should be of the shape;
        #         from
        #       [[..],
        # to    [..]]
        # As Numpy Arrays

        # Simulate Network Activity
        self.R = simulate_network(W=self.conn_symmatrix, **self.neuron_params)
        return self.R
    
    def show_activity(self, which='all'):
        # Firingrate plot, sample all neurons
        import numpy as np
        import matplotlib.pyplot as plt

        t_steps = self.neuron_params['t_steps']
        stim_start_s = self.stimulus_params['stim_start']/1000
        stim_stop_s = self.stimulus_params['stim_stop']/1000

        tvec_s = np.linspace(0,t_steps/1000,t_steps)
        plt.figure(figsize=(10,10))
        plt.plot(tvec_s,self.R[::].T,'.5') 
        plt.axvline(x=stim_start_s, c='red') # Start
        plt.axvline(x=stim_stop_s, c='red') # Stop
        plt.xlim([0,tvec_s[-1]])
        plt.ylabel('Rate')
        plt.xlabel('Time Elapsed (s)')
        plt.show()
    
    def show_pca(self):
        import matplotlib.pyplot as plt
        stim_start = self.stimulus_params['stim_start']
        stim_stop = self.stimulus_params['stim_stop']
        _ ,proj_PC, _ = calc_PCA(self.R[:,stim_start:stim_stop]) #PCs, proj_PC, var_exp

        # PCA plot
        plt.figure(figsize=(10,10))
        plt.plot(proj_PC[0],-proj_PC[1],c='k')
        plt.axis('equal')
        plt.xlabel('PC1')
        plt.ylabel('PC2',labelpad=-25)
        plt.show()

class Br2Simulation():
    def __init__(self, edgelist, neuron_model, STD=False):
        self.edgelist = edgelist # Must be signed already (processed in conndf, by make_ei or retrieved by get_signed_edgelist)
        self.params = dict()
        self.init_default_params()
        self.neuron_model = neuron_model
        self.STD = STD
        self.NGroup = None
        self.SGroup = None
        self.stim = None
        self.syn_eqs = None
        self.syn_pre = None
    
    def get_neuronIds(self):
        pre = self.edgelist["bodyId_pre"].unique()
        post = self.edgelist["bodyId_post"].unique()
        return list(set(pre) | set(post)) # Union

    def get_neuron_count(self):
        return len(self.get_neuronIds())
    
    def get_bmap(self):
        neuronIds = sorted(self.get_neuronIds())
        return dict(zip(neuronIds, list(range(0, self.get_neuron_count()))))

    def init_default_params(self):
        ALL_PARAMS = [
            'g_L',     #leak conductance
            'e_L',     #leak/reversal potential
            'C_m',     #membrane capacitance
            'V_th',     #threshold potential
            'V_R',     #resting/reset potential
            'ref_tau', #absolute refractory period

            'e_E',     #conductance-model
            'exc_tau', #conductance-model
            'e_I',     #conductance-model
            'inh_tau', #conductance-model

            'E_Ia',    #SFA-model
            'tau_Ia',   #SFA-model
            'SFA',      #SFA-model

            'psc_q',   #Unit/magnitude depends on the model: quantifies a single post-synapse event. Scaled by w
            'syn_tau', #synaptic decay. for current-based
            'inh_w'    #Weighting on inhibitory connections. inh_w = 1 means same scale as exc connections.
        ]

        for p in ALL_PARAMS:
            self.params[p] = 0

    def set_params(self, **kwargs):
        ### Set general parameters for various models    
        for k,v in kwargs.items():
            self.params[k] = v

    def define_main_eqs(self, neuron_model, params, custom_eq=None):
        ## Which synaptic model to use. All are LIF neuron models.
        # Passed to NeuronGroup(model = ..)
        match neuron_model:
            case "current":
                _eqs = '''
                    dv/dt = (-g_L*(v - e_L) + I_syn)/C_m : volt (unless refractory)
                    dI_syn/dt = -I_syn/syn_tau : amp 
                    '''
                
            case "conductance":
                _eqs = '''
                    dv/dt = (g_L*(e_L - v) + g_E*(e_E - v) + g_I*(e_I - v))/C_m : volt (unless refractory)
                    dg_E/dt = -g_E/exc_tau : siemens
                    dg_I/dt = -g_I/inh_tau : siemens
                    '''

            case "conductance_SFA":
                _eqs = '''
                    dv/dt = (g_L*(e_L - v) + g_E*(e_E - v) + g_I*(e_I - v) - g_Ia*(E_Ia - v) + I0)/C_m : volt (unless refractory)
                    dg_E/dt = -g_E/exc_tau : siemens
                    dg_I/dt = -g_I/inh_tau : siemens
                    dg_Ia/dt = -g_Ia/tau_Ia : siemens
                    I0 : amp
                    '''
            case "custom":
                _eqs = custom_eq

            case _:
                raise ValueError('Not a valid neuron_model. Valid: "current", "conductance", "conductance_SFA"')
        
        return Equations(
            _eqs,
            g_L = params['g_L'],
            e_L = params['e_L'],
            C_m = params['C_m'],
            V_th = params['V_th'],
            V_R = params['V_R'],
            ref_tau = params['ref_tau'],
            e_E = params['e_E'],
            exc_tau = params['exc_tau'],
            e_I = params['e_I'],
            inh_tau = params['inh_tau'],
            E_Ia = params['E_Ia'],
            tau_Ia = params['tau_Ia'],
            SFA = params['SFA'],
            psc_q = params['psc_q'],
            syn_tau = params['syn_tau'],
            inh_w = params['inh_w']
            )

    def define_syn_eqs(self, neuron_model, STD):
        # Usually to set w to the correect unit, but also functionality for implementing STD
        # Passed to Synapses(model=..)
        syn_eqs = None

        if neuron_model == "current":
            syn_eqs = Equations(''' 
                w : amp 
            ''')

        else: # Then conductance based
            if not STD: 
                syn_eqs = Equations(''' 
                    w : siemens 
                ''')
            else: # STD
                syn_eqs = Equations(''' 
                    du_S/dt = -Omega_f * u_S : 1 (event-driven)
                    dx_S/dt = Omega_d * (1 - x_S) : 1 (event-driven)
                    w: siemens
                ''')
        return syn_eqs
    
    def define_threshold(self):
        # Modifiable in future if needed. By Default if V > V_th; threshold for spike
        return 'v > V_th'

    def define_reset(self, neuron_model):
        # Reset parameter depends on synapse model used, just SFA for now.
        if neuron_model == 'conductance_SFA':
            return 'v = V_R; g_Ia -= SFA'
        else:
            return 'v = V_R'
    
    def define_syn_pre(self, neuron_model, STD):
        syn_pre = None

        if neuron_model == "current":
            syn_pre = 'I_syn += w'

        else: # Then conductance based
            if not STD: 
                syn_pre = ['g_E += w', 'g_I += w']
            else:
                synapses_action = '''
                    u_S += U_0 * (1 - u_S)
                    r_S = u_S * x_S
                    x_S -= r_S
                '''
                syn_pre = [
                    synapses_action + 'g_E += w*r_S',
                    synapses_action + 'g_I += w*r_S'
                ]

        return syn_pre
    
    def build_neurons(self, custom_eq=None):
        NG = NeuronGroup(
            N=self.get_neuron_count(),
            model=self.define_main_eqs(self.neuron_model, self.params, custom_eq),
            threshold='v > V_th', #self.define_threshold(), #v > V_th
            reset=self.define_reset(self.neuron_model),
            refractory=self.params['ref_tau'],
            method='euler'
        )
        return NG

    def set_ngroup(self, NG):
        self.NGroup = NG

    def _cnx_helper(self, op):
        # op = '>' or '<'
        bmap = self.get_bmap()
        edges = self.edgelist.query(f'signed_weight {op} 0')[['bodyId_pre', 'bodyId_post', 'signed_weight']]
        pres = [bmap.get(x) for x in edges['bodyId_pre'].to_numpy().tolist()]
        posts = [bmap.get(x) for x in edges['bodyId_post'].to_numpy().tolist()]
        weights = edges['signed_weight'].to_numpy().tolist()
        return pres, posts, weights

    def build_synapses(self, NG, delay):
        # Builds synapse instances, and connects according to matrix
        syn_pre = self.define_syn_pre(self.neuron_model, self.STD)
        self.syn_pre = syn_pre
        syn_eqs = self.define_syn_eqs(self.neuron_model, self.STD)
        self.syn_eqs = syn_eqs

        if len(syn_pre) == 2: # Lazy way to check list
            exc_syn = Synapses(
                NG,
                NG,
                syn_eqs,
                on_pre = syn_pre[0],
                delay=delay
            )
            
            inh_syn = Synapses(
                NG,
                NG,
                syn_eqs,
                on_pre = syn_pre[1],
                delay=delay
            )

        else:
            exc_syn = Synapses(
                NG,
                NG,
                syn_eqs,
                on_pre = syn_pre,
                delay=delay
            )
            
            inh_syn = Synapses(
                NG,
                NG,
                syn_eqs,
                on_pre = syn_pre,
                delay=delay
            )

        exc_pres, exc_posts, exc_weights = self._cnx_helper(op='>')
        exc_syn.connect(i=exc_pres, j=exc_posts)
        exc_syn.w[:] = exc_weights*self.params['psc_q']

        inh_pres, inh_posts, inh_weights = self._cnx_helper(op='<')
        inh_syn.connect(i=inh_pres, j=inh_posts)
        inh_syn.w[:] = inh_weights*self.params['psc_q']*self.params['inh_w']
        return exc_syn, inh_syn
    
    def set_sgroup(self, e_syn, i_syn):
        self.SGroup = [e_syn, i_syn]

    def build_stimulus(
            self, 
            start_ms, 
            stop_ms, 
            n_spikes, 
            NG,
            irregular_spikes=False
            ):
        indices = np.array([0 for _ in range(n_spikes)])        
        duration = stop_ms - start_ms
        # Calculate the time interval between spikes
        interval = duration / n_spikes
        # Generate an array of regularly spaced time points
        times = np.round(np.arange(start_ms, start_ms+duration, interval)[:n_spikes]) # Truncate last spike if not fitting the stim window.

        if irregular_spikes:
            print("Making spikes irregular...")
            # Calculate the number of time steps
            n_steps = len(times)
            # Calculate the maximum amount by which each time step can be shifted
            max_shift = 200  # set this to the maximum amount of random shift you want
            # Calculate the minimum interval between adjacent time steps
            min_interval = 1
            # Initialize an array to hold the randomly spaced time steps
            rand_stim_times = np.zeros(n_steps)
            # Set the first time step to the first stimulation time
            rand_stim_times[0] = times[0]
            # Loop through the remaining time steps and randomly shift them while maintaining the ascending order
            for i in range(1, n_steps):
                # Calculate the maximum and minimum possible shifts for this time step
                max_possible_shift = min(max_shift, times[i] - rand_stim_times[i-1] - min_interval)
                min_possible_shift = max(0, times[i] - rand_stim_times[i-1] - max_shift)
                # Randomly select a shift amount within the possible range
                shift = np.random.uniform(min_possible_shift, max_possible_shift)
                # Update the time step with the shifted value
                rand_stim_times[i] = rand_stim_times[i-1] + shift + min_interval
            times = np.round(rand_stim_times)
            
        # Call it a channelrhodopsin for fun
        cs_chrimson = SpikeGeneratorGroup(
            N = 1,
            indices=indices,
            times=times*ms
        )
        
        # Connect 'artificial' channelrhodopsin to a select body_Id. for now only supports one
        stim_syn = Synapses(cs_chrimson, NG, model=self.syn_eqs, on_pre=self.syn_pre)
        return cs_chrimson, stim_syn # spike generator group and synapse

    def set_stim(self, stim_group):
        self.stim = stim_group