# import packages
import sys
sys.path.append('./VDAM2022')
import os
from data_processing import ConnDF
from analysis import AnalyseDF
from simulation import Br2Simulation
from neuprint import Client
from brian2 import *
import numpy as np
import pandas as pd
from copy import deepcopy
import random

def simulation(data_edgelist, params, roi_name, out_dir, irregular_spikes):
    print("Building simulation...")
    # Simulator Params
    defaultclock.dt = params['deltaT']

    # Simulator wrapper
    simulator = Br2Simulation(
        edgelist = data_edgelist,
        neuron_model = 'current',
        STD = False
    )
    simulator.set_params(**params)

    # Build neuron objects
    neurons = simulator.build_neurons()
    neurons.v = params['V_R'] # Set initial voltages to resting

    # Build synapse objects
    exc_synapses, inh_synapses = simulator.build_synapses(NG=neurons, delay=params['syn_delay'])

    # Build stimuli
    stimulus_generator, stimulus_syn = simulator.build_stimulus(
        start_ms=params['start_ms'],
        stop_ms=params['stop_ms'],
        n_spikes=params['n_spikes'],
        NG=neurons,
        irregular_spikes=irregular_spikes
        )
    target_bid = params['target_bid']
    bmap = simulator.get_bmap()
    if type(target_bid) == list:
        target_ix = [bmap.get(x) for x in target_bid]
    else:
        target_ix = target_bid
    stimulus_syn.connect(i=0, j=target_ix)
    stimulus_syn.w = params['psc_q']*params['stim_weight']

    # Build recording devices
    volt_mon = StateMonitor(neurons, 'v', record=target_ix) # Record voltage of target neuron for now
    spike_mon = SpikeMonitor(neurons, record=True)
    spike_mon_stim = SpikeMonitor(stimulus_generator, record=True)

    # Locals?
    allVars = dict()
    allVars.update(params)
    allVars.update(locals())

    # Network
    net = Network(
        neurons, 
        exc_synapses, 
        inh_synapses, 
        stimulus_generator, 
        stimulus_syn, 
        volt_mon, 
        spike_mon, 
        spike_mon_stim
        )
    
    print(f"Running network: weight={params['psc_q']}, presyndly={params['syn_delay']}, membcap={params['C_m']}")
    print(f"Stimulating: {params['target_bid']} with {params['n_spikes']} spikes over {params['stop_ms'] - params['start_ms']}ms")
    net.run(params['runtime'], namespace=allVars)

    # After running, save results
    # Make it after the weight
    weight = params['psc_q']/pA
    dly = params['syn_delay']/ms
    cm = params['C_m']/pF
    inverted_bmap = {v:k for k,v in bmap.items()} # Indices to bodyid

    # Save to out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    final_save = os.path.join(out_dir, f"{roi_name}_{params['target_bid']}_W{weight}pA_delay{dly}ms_cap{cm}pF_{params['n_spikes']}spikes")

    np.savez(
        final_save, 
        target_volt={'v': volt_mon.v[:], 't':volt_mon.t[:]},
        all_spikes={'i': [inverted_bmap.get(x) for x in spike_mon.i[:]], 't':spike_mon.t[:]}, # Body Ids instead of indices
        stim_spikes={'i': spike_mon_stim.i[:], 't':spike_mon_stim.t[:]},
        params=params
        )

def get_data_edgelist(client, roi, normalise, threshold, randomise=False):
    if roi is None:
        roi_name = "Entire VNC"
    else:
        roi_name = roi

    print(f"Getting {roi_name} network with: Normalisation = {normalise}, Thresholding = {threshold}...")
    if randomise:
        print(f"Randomising network...")
       
    data = ConnDF(client=client)
    data.extract_full(file_path='default')  
    if (roi is None) and (normalise is None) and (threshold is None):
        data.filter(roi=roi, normalise=normalise, threshold=threshold)
        _, edgelist = data.get_signed_edgelist(which='filtered', make_ei=True, inh_weighting=1, exc_weighting=1)
    else:
         _, edgelist = data.get_signed_edgelist(which='full', make_ei=True, inh_weighting=1, exc_weighting=1)

    if randomise:
        edgelist = data.randomise(edgelist)

    return edgelist

def sample_random_neurons(data, count, neuron_class=None):
    # Data = conndf, neuron_class = class of neuron, count = how many to sample
    df = data.neuron_master[data.neuron_master['bodyId'].isin(data.conn_symmatrix.index)]
    if neuron_class is None:
        return random.sample(list(df.bodyId), count)
    else:
        return random.sample(list(df[df["class"] == neuron_class].bodyId), count)

def main(
        norm_method,
        threshold,
        randomise,
        w_space, 
        delay_space, 
        capacitance_space, 
        stim_space, ko_space, 
        no_DN_feedback_space, 
        sim_time_space, 
        out_dir,
        irregular_spikes,
        roi
        ):
    # Database retrieval
    RT_AUTH_KEY = None # Removed for privacy purposes
    VNC_c = Client('neuprint-test.janelia.org', dataset='vnc', token=RT_AUTH_KEY)
    # Params from https://github.com/nawrotlab/DrosophilaOlfactorySparseCoding/blob/main/experiment_larva.py
    # Iterate through many different weights, synaptic delays and capacitances
    print("\n")

    # Defaults
    params = dict(
        # Neuron Params
        g_L = 5*nS, # KC 0.5, PN 2.5, main 5
        e_L = -60*mV, # Leak potential global
        C_m = 30*pF, ## 30, 100, 200 || SEARCHED
        V_th = -35*mV, # -30, -35 main
        V_R = -60*mV, # -60 main, -55 KC, -59 pn/ln
        ref_tau = 2*ms, # main 2ms
        # Synapse params
        inh_w = 1,
        syn_tau = 7*ms,
        psc_q = 10*pA, # weights in nS -> vary from 3, 30, 50, 100 || SEARCHED
        syn_delay = 4*ms, # || SEARCHED
        # Stimulation
        start_ms=2000,
        stop_ms=16000,
        n_spikes=1600,
        target_bid=13809,
        stim_weight=100, # Relative weight = 1, then for abs weight maybe 100
        # Sim
        deltaT = 0.1*ms,
        runtime=20000*ms
        )
    
    # Norm method
    if norm_method is None:
        norm_name = 'abs'
    else:
        norm_name = 'rel'
    
    if roi is None:
        roi_name = "VNC"
    else:
        roi_name = roi

    # Retrieve once
    data_edgelist = get_data_edgelist(client=VNC_c, roi=roi, normalise=norm_method, threshold=threshold, randomise=randomise)

    print("Setting up parameter search space...")
    # Parameter Value Loops
    for w in w_space:
        for d in delay_space:
            for c in capacitance_space:
                for s in stim_space:
                    for t in sim_time_space:
                        # Abs/Rel Weights
                        if norm_method is None:
                            w_d = w/1000 # Absolute wieghts -> 3.0, to 10.0 range
                            s_d = 100
                        else:
                            w_d = w
                            s_d = 1
                        params['psc_q'] = w_d
                        params['syn_delay'] = d
                        params['C_m'] = c
                        params['stim_weight'] = s_d
                        params['target_bid'] = s
                        params['start_ms'] = t[0]
                        params['stop_ms'] = t[1]
                        params['n_spikes'] = t[2]
                        params['runtime'] = t[3]
                        
                        # Network Manipulation Loops/Control
                        for f in no_DN_feedback_space:
                            # Rmfb for Descending/Stimulated Neurons;
                            if f: # True,
                                print(f"Removing connections going to neurons: {f}")
                                nfb_data_edgelist = deepcopy(data_edgelist)
                                nfb_data_edgelist = nfb_data_edgelist[~((nfb_data_edgelist['bodyId_post'].isin(s)))]
                                sim_name = f"{roi_name}_{norm_name}_nothresh_nofb"
                            else:
                                nfb_data_edgelist = None
                                sim_name = f"{roi_name}_{norm_name}_nothresh"

                            # Virtual neuron KO/ablation. Use r_data if you have
                            for remove in ko_space: # Enter KO space loop
                                if remove: 
                                    if nfb_data_edgelist is None: # If not rbfb, use normal
                                        rm_data_edgelist = deepcopy(data_edgelist)
                                    else:
                                        rm_data_edgelist = deepcopy(nfb_data_edgelist)
                                    print(f"Removing {remove} from the network...")
                                    rm_data_edgelist = rm_data_edgelist[~((rm_data_edgelist['bodyId_pre'].isin(remove) | rm_data_edgelist['bodyId_post'].isin(remove)))]
                                    simulation(rm_data_edgelist, params, sim_name+f"_rm{remove}", out_dir, irregular_spikes)
                                else:
                                    if nfb_data_edgelist is not None: # But if RBFB, use nfb_data
                                        simulation(nfb_data_edgelist, params, sim_name, out_dir, irregular_spikes)
                                    else: # Otherwise, if no network manipulation, use extracted data
                                        simulation(data_edgelist, params, sim_name, out_dir, irregular_spikes)
        
if __name__ == '__main__':
    # # Loop 1: 400Hz Generator, 2s stim MDNs, MF KOs
    # w_space = [x*pA for x in range(3000, 3501, 500)] 
    # delay_space = [x*ms for x in [0]] 
    # capacitance_space = [x*pF for x in [100]] # vals in paper above
    # stim_space = [[13809, 14419, 13438, 14523]] # MDNs
    # ko_space = [[], [10994, 11493], [10475, 10508], [10335, 10440], [14084, 13323, 13293, 13574], [10169, 10133]] # MF knockouts
    # sim_time_space = [[2000, 4000, x, 10000*ms] for x in range(100, 800, 100)] #start,stop,n_spikes,runtime in ms; 800/2s -> 400Hz?
    # no_DN_feedback_space = [True, False] # Remove feedback 
    # out_dir = "50thresh_VNC_VFB_allMDN_MFKO_2st10ss_results"
    # main(w_space, delay_space, capacitance_space, stim_space, ko_space, no_DN_feedback_space, sim_time_space, out_dir)
    # print(f"Finished loop 1: {out_dir} \n")

    # # Loop 2: 400Hz Generator, 10s stim MDNs,
    # w_space = [x*pA for x in range(3000, 3501, 500)] 
    # delay_space = [x*ms for x in [0]] 
    # capacitance_space = [x*pF for x in [100]] # vals in paper above
    # stim_space = [[13809, 14419, 13438, 14523]] # MDNs
    # ko_space = [[]] # MF knockouts
    # sim_time_space = [[2000, 12000, x, 18000*ms] for x in range(500, 4000, 500)] #start,stop,n_spikes,runtime in ms; 4000/10s -> 400Hz?
    # no_DN_feedback_space = [True, False] # Remove feedback 
    # out_dir = "50thresh_VNC_VFB_allMDN_10st18ss_results"
    # main(w_space, delay_space, capacitance_space, stim_space, ko_space, no_DN_feedback_space, sim_time_space, out_dir)
    # print(f"Finished loop 2: {out_dir} \n")

    # # Loop 3: 400Hz Generator, 2s stim MF0Xs
    # w_space = [x*pA for x in range(3000, 3501, 500)] 
    # delay_space = [x*ms for x in [0]] 
    # capacitance_space = [x*pF for x in [100]] # vals in paper above
    # stim_space = [[10994, 11493], [10475, 10508], [10335, 10440], [14084, 13323, 13293, 13574], [10169, 10133]] # MDNs
    # ko_space = [[]] 
    # sim_time_space = [[2000, 4000, x, 10000*ms] for x in range(100, 800, 100)] #start,stop,n_spikes,runtime in ms; 4000/10s -> 400Hz?
    # no_DN_feedback_space = [True, False] # Remove feedback 
    # out_dir = "50thresh_VNC_VFB_MFX_2st10ss_results"
    # main(w_space, delay_space, capacitance_space, stim_space, ko_space, no_DN_feedback_space, sim_time_space, out_dir)
    # print(f"Finished loop 3: {out_dir} \n")

    # Loop X: DNs
    # w_space = [x*pA for x in range(3000, 3501, 500)] 
    # delay_space = [x*ms for x in [0.0]] 
    # capacitance_space = [x*pF for x in [100]] 
    # stim_space = [[12926, 12955], [10118, 10126], [18107, 17155]] 
    # ko_space = [[]] # MF knockouts
    # sim_time_space = [[2000, 4000, x, 10000*ms] for x in range(100, 801, 100)] #start,stop,n_spikes,runtime in ms; 800/2s -> 400Hz?
    # no_DN_feedback_space = [False] # Remove feedback 
    # out_dir = "50thresh_VNC_W3_traj_updatednet_results"
    # main(w_space, delay_space, capacitance_space, stim_space, ko_space, no_DN_feedback_space, sim_time_space, out_dir)
    # print(f"Finished loop X: {out_dir} \n")

    # Loop Y: Randomised network
    # w_space = [x*pA for x in [3000]] 
    # delay_space = [x*ms for x in [0.0]] 
    # capacitance_space = [x*pF for x in [100]] 
    # stim_space = [[13809, 14419, 13438, 14523]] 
    # ko_space = [[]] 
    # sim_time_space = [[2000, 4000, 200, 10000*ms], [2000, 12000, 1000, 18000*ms]] # 200spikes -> 50Hz. for 2s and 10s
    # no_DN_feedback_space = [True, False] # With, without
    # out_dir = "VNC_abs_MDN_randomised_50Hz"
    # norm_method = None
    # threshold = None
    # randomise = True
    # irregular_spikes = False
    # roi = None
    # main(norm_method, threshold, randomise, w_space, delay_space, capacitance_space, stim_space, ko_space, no_DN_feedback_space, sim_time_space, out_dir, irregular_spikes, roi)
    # print(f"Finished loop: {out_dir} \n")      

    #  # Loop Y: Delay space for long 10s stim
    # w_space = [x*pA for x in [3000]] 
    # delay_space = [x*ms for x in [0.0, 0.1, 0.5, 1.0, 5.0]] 
    # capacitance_space = [x*pF for x in [100]] 
    # stim_space = [[13809, 14419, 13438, 14523]] 
    # ko_space = [[]] 
    # sim_time_space = [[2000, 12000, 1000, 18000*ms]] # 200spikes -> 50Hz. for 2s and 10s
    # no_DN_feedback_space = [True, False] # With and without feedback
    # out_dir = "VNC_abs_MDN_varydelay_50Hz"
    # norm_method = None
    # threshold = None
    # randomise = False
    # irregular_spikes = False
    # roi = None
    # main(norm_method, threshold, randomise, w_space, delay_space, capacitance_space, stim_space, ko_space, no_DN_feedback_space, sim_time_space, out_dir, irregular_spikes, roi)
    # print(f"Finished loop: {out_dir} \n")

    # # Loop Y: Irregular spiking
    # w_space = [x*pA for x in [3000]] 
    # delay_space = [x*ms for x in [0.0]] 
    # capacitance_space = [x*pF for x in [100]] 
    # stim_space = [[13809, 14419, 13438, 14523]] 
    # ko_space = [[]] 
    # sim_time_space = [[2000, 4000, 200, 10000*ms], [2000, 12000, 1000, 18000*ms]] # 200spikes -> 50Hz. for 2s and 10s
    # no_DN_feedback_space = [True, False] # With and without feedback
    # out_dir = "VNC_abs_MDN_irregular_50Hz"
    # norm_method = None
    # threshold = None
    # randomise = False
    # irregular_spikes = True
    # roi = None
    # main(norm_method, threshold, randomise, w_space, delay_space, capacitance_space, stim_space, ko_space, no_DN_feedback_space, sim_time_space, out_dir, irregular_spikes, roi)
    # print(f"Finished loop: {out_dir} \n")                                                            
                                                                                    
    # # Loop Z: MDN + MfKO longer
    # w_space = [x*pA for x in [3000]] 
    # delay_space = [x*ms for x in [0.0]] 
    # capacitance_space = [x*pF for x in [100]] 
    # stim_space = [[13809, 14419, 13438, 14523]] 
    # ko_space = [[], [10994, 11493], [10475, 10508], [10335, 10440], [14084, 13323, 13293, 13574], [10169, 10133]] # MF knockouts
    # sim_time_space = [[2000, 12000, 1000, 18000*ms]] # 10s
    # no_DN_feedback_space = [True, False] # With and without feedback
    # out_dir = "VNC_abs_MDN_MFKO_10s_50Hz"
    # norm_method = None
    # threshold = None
    # randomise = False
    # irregular_spikes = False
    # main(norm_method, threshold, randomise, w_space, delay_space, capacitance_space, stim_space, ko_space, no_DN_feedback_space, sim_time_space, out_dir, irregular_spikes, roi)
    # print(f"Finished loop: {out_dir} \n")                                                                               

    # # Loop Z: DNs
    # w_space = [x*pA for x in [3000]] 
    # delay_space = [x*ms for x in [0.0]] 
    # capacitance_space = [x*pF for x in [100]] 
    # stim_space = [[12926, 12955], [10118, 10126], [18107, 17155]] 
    # ko_space = [[], [10994, 11493], [10475, 10508], [10335, 10440], [14084, 13323, 13293, 13574], [10169, 10133]] # MF knockouts
    # sim_time_space = [[2000, 4000, 200, 10000*ms], [2000, 12000, 1000, 18000*ms]] # 10s
    # no_DN_feedback_space = [True, False] # With and without feedback
    # out_dir = "VNC_abs_oDNs_MFKO_varyHz"
    # norm_method = None
    # threshold = None
    # randomise = False
    # irregular_spikes = False
    # roi = None
    # main(norm_method, threshold, randomise, w_space, delay_space, capacitance_space, stim_space, ko_space, no_DN_feedback_space, sim_time_space, out_dir, irregular_spikes, roi)
    # print(f"Finished loop: {out_dir} \n")   

    # # Loop Z: DNs irregular
    # w_space = [x*pA for x in [3000]] 
    # delay_space = [x*ms for x in [0.0]] 
    # capacitance_space = [x*pF for x in [100]] 
    # stim_space = [[12926, 12955], [10118, 10126], [18107, 17155]] 
    # ko_space = [[], [10994, 11493], [10475, 10508], [10335, 10440], [14084, 13323, 13293, 13574], [10169, 10133]] # MF knockouts
    # sim_time_space = [[2000, 4000, 200, 10000*ms], [2000, 12000, 1000, 18000*ms]] # 10s
    # no_DN_feedback_space = [True, False] # With and without feedback
    # out_dir = "VNC_abs_oDNs_MFKO_irregular_varyHz"
    # norm_method = None
    # threshold = None
    # randomise = False
    # irregular_spikes = True
    # roi = None
    # main(norm_method, threshold, randomise, w_space, delay_space, capacitance_space, stim_space, ko_space, no_DN_feedback_space, sim_time_space, out_dir, irregular_spikes, roi)
    # print(f"Finished loop: {out_dir} \n")   

    # # Loop Z: MFs
    # w_space = [x*pA for x in [3000]] 
    # delay_space = [x*ms for x in [0.0]] 
    # capacitance_space = [x*pF for x in [100]] 
    # stim_space = [[10994, 11493], [10475, 10508], [10335, 10440], [14084, 13323, 13293, 13574], [10169, 10133]] 
    # ko_space = [[]] # MF knockouts
    # sim_time_space = [[2000, 4000, 200, 10000*ms], [2000, 12000, 1000, 18000*ms]] # 10s
    # no_DN_feedback_space = [True, False] # With and without feedback
    # out_dir = "VNC_abs_MFs_varyHz"
    # norm_method = None
    # threshold = None
    # randomise = False
    # irregular_spikes = False
    # roi = None
    # main(norm_method, threshold, randomise, w_space, delay_space, capacitance_space, stim_space, ko_space, no_DN_feedback_space, sim_time_space, out_dir, irregular_spikes, roi)
    # print(f"Finished loop: {out_dir} \n")   

    # # Loop Z: T3R Subset
    # w_space = [x*pA for x in [3000]] 
    # delay_space = [x*ms for x in [0.0]] 
    # capacitance_space = [x*pF for x in [100]] 
    # stim_space = [[13809, 14419, 13438, 14523]] 
    # ko_space = [[]] # MF knockouts
    # sim_time_space = [[2000, 4000, 200, 10000*ms], [2000, 4000, 400, 10000*ms], [2000, 12000, 1000, 18000*ms], [2000, 12000, 2000, 18000*ms]] # 10s
    # no_DN_feedback_space = [True, False] # With and without feedback
    # out_dir = "T3R_abs_MDN"
    # norm_method = None
    # threshold = None
    # randomise = False
    # irregular_spikes = False
    # roi = "LegNp(T3)(R)"
    # main(norm_method, threshold, randomise, w_space, delay_space, capacitance_space, stim_space, ko_space, no_DN_feedback_space, sim_time_space, out_dir, irregular_spikes, roi)
    # print(f"Finished loop: {out_dir} \n")   

    # # Loop Z: T3R Subset
    # w_space = [x*pA for x in [3000]] 
    # delay_space = [x*ms for x in [0.0]] 
    # capacitance_space = [x*pF for x in [100]] 
    # stim_space = [[13809, 14419, 13438, 14523]] 
    # ko_space = [[]] # MF knockouts
    # sim_time_space = [[2000, 4000, 200, 10000*ms], [2000, 4000, 400, 10000*ms], [2000, 12000, 1000, 18000*ms], [2000, 12000, 2000, 18000*ms]] # 10s
    # no_DN_feedback_space = [True, False] # With and without feedback
    # out_dir = "T3R_abs_MDN_randomised"
    # norm_method = None
    # threshold = None
    # randomise = True
    # irregular_spikes = False
    # roi = "LegNp(T3)(R)"
    # main(norm_method, threshold, randomise, w_space, delay_space, capacitance_space, stim_space, ko_space, no_DN_feedback_space, sim_time_space, out_dir, irregular_spikes, roi)
    # print(f"Finished loop: {out_dir} \n")   




    ### Run after
    # Loop O: VNC, irregular bursting, but low. so over 10s, do maybe like 200?, 
    w_space = [x*pA for x in [3000]] 
    delay_space = [x*ms for x in [0.0]] 
    capacitance_space = [x*pF for x in [100]] 
    stim_space = [[13809, 14419, 13438, 14523]] 
    ko_space = [[]] 
    sim_time_space = [[2000, 12000, x, 18000*ms] for x in [200, 300, 400, 500, 600, 700, 800]] # 200spikes -> 50Hz. for 2s and 10s
    no_DN_feedback_space = [True, False] # With and without feedback
    out_dir = "VNC_abs_MDN_irregular_LOWHz"
    norm_method = None
    threshold = None
    randomise = False
    irregular_spikes = True
    roi = None
    main(norm_method, threshold, randomise, w_space, delay_space, capacitance_space, stim_space, ko_space, no_DN_feedback_space, sim_time_space, out_dir, irregular_spikes, roi)
    print(f"Finished loop: {out_dir} \n")  