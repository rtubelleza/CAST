import os
import sys
import pickle
import random as rd
import numpy as np
import pandas as pd
import scipy as sc
# Network modules
import networkx as nx
import igraph as ig
# Data science packages
from umap.umap_ import UMAP
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
# Neuprint
import neuprint
import navis
import navis.interfaces.neuprint as neu
# Static and interactive plotting
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import gcf
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import ipywidgets as widgets
from ipywidgets import interact_manual

""" Contains main analysis classes. """

""" 27/3/2023 Changes
    - AnalyseDF.client now just points to the client of conndf, never need to
      point to a different client instance.
    - Resolved pandas SettingWithCopyWarning for future proofing by implementing
      pandas .assign method. 
    - Added cosine similarity calculations and heatmap plotting for seeing
      similarity profiles between premotor neurons, and between motor neurons.

    28/3/2023 Changes
    - Modified to add a self.transpose class state. Checks if we are clustering
    on PMNs or MNs. If true, clsuter on MNs. Methods commonly call this to 
    modify their clustering/processing/plotting appropriately. 

    29/3/2023 Changes
    - Added motor neuron typing. Appends neuron type to bodyId of motor neuron.

    30/3/2023 Changes
    - Updated plotly scatter for UMAP styling. Increased opacity, plot dpi/size,
    and download format for plot to svg.

    11/5/2023 Changes
    - Added PCA modular functionality to SpikeAnalysis, with automated traj plotter.
    - Added spike summary dataframes, getting neuron count and spike count by neuron
      attribute (i.e. mainRoi)

    18/5/2023 Changes/Additions
    - Added template for SummaryDF, for general summary statistics and graph properties.
    - Added draw_morphology. Instead of drawing subset network of chosen PMN/MN cluster,
      visualises all neurons in that cluster.
    
    24/5/2023 Changes
    - Renamed AnalyseDF to ClusterDF

    20/06/2023 Changes
    - Accomodates nomenclature changes to neuron classes; mainly decapitalisation.
"""

class ClusterDF():
    """ Analysis Class """
    def __init__(self, conndf):
        self.conndf = conndf 
        self.client = self.conndf.client
        self.premotor_extraction_report = None # premotor_extraction
        self.preMN_postMN_edgelist = None # premotor_extraction. thresholded connections
        self.preMN_postMN_matrix = None # premotor_extraction
        self.transpose = False # If clustering on PMN or MN side
        self.latest_umap_embedding = None # umap
        self.latest_umap_embedding_params = None
        self.latest_clusters = None
        self.latest_clustermap = None # draw_clustermap
        self.latest_HDBSCAN_tree = None # HDBSCAN
        self.latest_HDBSCAN_prob = None # HDBSCAN
        # For interactive plottting
        self._kept_cluster = None
        self._kept_cluster_col = None
        self._current_embedding = None
        self._prev_embedding = None
        self._prev_embedding_d = None # can be dervied from prev_embedding
        self._chosen_cluster = None
        self._palette = px.colors.qualitative.Plotly
        self._val_range = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] # Default 10 clusters, so default option 0 to 9
        self._stack_after = False
        self.latest_plotly_fig = None
        self.latest_mpl_fig = None

    def premotor_extraction(self, side='RHS', neuromere='T3', threshold=0.01, print_summary=True):
        """ Extract premotor neuron and motor neuron connections to a pool of motor neurons in a
            specified neuropil and side. """
        ## Helper function from neuprint.utils to convert table to matrix
        
        # MN Pool
        mn_pool = self.conndf.neuron_master[self.conndf.neuron_master['class'] == 'motor neuron']
        selected_mn_pool = mn_pool.copy().query(f"somaSide == '{side}' & somaNeuromere == '{neuromere}'")
        # Connections to MN pool
        PMN_MN_cnx = self.conndf.conn_master[self.conndf.conn_master['bodyId_post'].isin(selected_mn_pool.bodyId)]
        # Relative strength inputs. Divide weights by total incoming inputs of motor neuron.
        selected_mn_total_inputs = dict(zip(selected_mn_pool['bodyId'], selected_mn_pool['upstream']))
        PMN_MN_cnx = PMN_MN_cnx.assign(weight_relative = PMN_MN_cnx['weight']/PMN_MN_cnx['bodyId_post'].map(selected_mn_total_inputs))
        # from here, have an edggelist of pre and post syn cell, and weight and rel_weight

        # Extract premotor neurons by an arbitrary threshold
        # Threshold of 0.01 will include only connections, where a premotor 
        # neuron makes up 1% or more of a motorneuron total input
        thresholded_cnx = PMN_MN_cnx.query(f'weight_relative >= {threshold}')
        self.preMN_postMN_edgelist = thresholded_cnx
        self.preMN_postMN_matrix = self.conndf.neup_connection_table_to_matrix(
            conn_df=thresholded_cnx, 
            weight_col = 'weight_relative'
            )
        # Non-symmetric connectivity matrix. Rows = Premotor, Columns = Motor Neurons.

        # Create summary report
        n_pmn = thresholded_cnx.bodyId_pre.nunique()
        n_mn = len(selected_mn_pool)
        n_cnx = len(PMN_MN_cnx)
        n_thresholded_cnx = len(thresholded_cnx)
        summary_message = f"From {neuromere} {side} motor neurons, extracted {n_pmn} using a threshold of {threshold}."
        summary_report = {
            'N_PMN': n_pmn, 
            'N_MN': n_mn,
            'MN_neuromere': neuromere,
            'MN_side': side,
            'extraction_threshold': threshold,
            'N_cnx': n_cnx, 
            'N_t_cnx': n_thresholded_cnx, 
            'Message':summary_message}
        self.premotor_extraction_report = summary_report
        if print_summary:
            print(summary_message)

    def create_aesthetics(self):
        """ Instantiates and stores global and aesthetics for consistent plotting. """
        side = self.premotor_extraction_report['MN_side']
        neuromere = self.premotor_extraction_report['MN_neuromere']

        def automate_aesthetics(self, sns_palette, by):
            _n = self.conndf.neuron_master[by].nunique()
            _strs = self.conndf.neuron_master[by].unique()
            _pal = sns.color_palette(sns_palette, _n+1)
            _cval = dict(zip(map(str, _strs), _pal))
            return _cval

        # Transmitter palette: Blue=Acetylcholine, coral=Gaba, Green=glutamate, Grey=undef
        # Hard coded for preference
        transmitter_strings = ['Acetylcholine', 'Gaba', 'Glutamate', 'NaN']
        transmitter_palette = sns.color_palette(['dodgerblue', 'lightcoral', 'green', 'grey'])

        # Categorical palettes
        transmitter_cval = dict(zip(map(str, transmitter_strings), transmitter_palette)) # Used in the end
        class_cval = automate_aesthetics(self, sns_palette='dark', by='class')
        exitnerve_cval = automate_aesthetics(self, sns_palette='colorblind', by='exitNerve')

        # Continuous palettes
        pct_output_cval = sns.color_palette("light:g", as_cmap=True)

        return {
            'transmitter': transmitter_cval, 
            'class': class_cval, 
            'exitnerve': exitnerve_cval,
            f'pct_{neuromere}{side}': pct_output_cval}
    
    def create_meta_information(self, matrix):
        """ Create dataframe containing metainformation on neurons. Used for displaying as stakced row or column
        color bars in clustermap. """
        # Prior info
        side = self.premotor_extraction_report['MN_side']
        neuromere = self.premotor_extraction_report['MN_neuromere']

        # Maybe also keep a record of the order.
        pmn_list = list(matrix.index)
        mn_list = list(matrix.columns)

        def get_neuron_row(bid):
            return self.conndf.neuron_master.query(f"bodyId == {bid}")
        
        def get_neuron_class(row):
            return row['class'].iloc[0]

        def get_best_transmitter(row):
            _nt_map = {
                'ntGlutamateProb':'Glutamate', 
                'ntGabaProb':'Gaba', 
                'ntAcetylcholineProb':'Acetylcholine',
                'nan': 'NaN'}
            _max_predicted_nt = row[['ntGlutamateProb', 'ntGabaProb', 'ntAcetylcholineProb']].idxmax(axis=1).iloc[0]
            return _nt_map[str(_max_predicted_nt)]

        def get_exitnerve(row):
            return row['exitNerve'].iloc[0]

        def get_sideneuropil_output(self, bid, side, neuromere, row):
            _total_output = row['downstream'].iloc[0]
            _sideneuropil_output = self.conndf.conn_master.query(f"bodyId_pre == {bid}").query(f"roi == 'LegNp({neuromere})({side[0]})'").weight.sum()
            pct = _sideneuropil_output/_total_output
            return pct

        # Start with PMNs
        # Meta information to populate
        _classes = []
        _transmitter = []
        _sideneuropil_pct = []
        _exitnerve = []
        for pmn_id in pmn_list:
            current = get_neuron_row(pmn_id)
            _classes.append(get_neuron_class(current))
            _transmitter.append(get_best_transmitter(current))
            _sideneuropil_pct.append(get_sideneuropil_output(self, pmn_id, side, neuromere, current))
        pmn_meta = pd.concat([
            pd.Series(_classes), 
            pd.Series(_transmitter), 
            pd.Series(_sideneuropil_pct)], axis=1)
        pmn_meta = pmn_meta.set_axis(matrix.index)
        pmn_meta = pmn_meta.set_axis(['class', 'transmitter', f'pct_{neuromere}{side}'], axis=1)

        # End with MNs, reset lists
        _transmitter = []
        _exitnerve = []
        for mn_id in mn_list:
            current = get_neuron_row(mn_id)
            _transmitter.append(get_best_transmitter(current))
            _exitnerve.append(get_exitnerve(current))
        mn_meta = pd.concat([
            pd.Series(_transmitter), 
            pd.Series(_exitnerve)], axis=1)
        mn_meta = mn_meta.set_axis(matrix.columns)
        mn_meta = mn_meta.set_axis(['transmitter', 'exitnerve'], axis=1)

        return pmn_meta, mn_meta
            
    def draw_clustermap(
            self, 
            matrix=None, 
            MN_typing=False,
            xticklabels=True,
            yticklabels=True,
            row_cluster=False,
            col_cluster=False,
            cmap='viridis',
            **kwargs):
        """ kwargs passed to clustermap parameters. Performs Euclidean ward clustering on both row and col.
        Can specify if want to do olfactory paper-like clustering.  """
        # Prep.
        if matrix is None:
            matrix = self.preMN_postMN_matrix
        # Incoming matrix is transposed already, temporarily untranspose for this step
        if self.transpose:
            matrix = matrix.T

        side = self.premotor_extraction_report['MN_side']
        neuromere = self.premotor_extraction_report['MN_neuromere']
        pmn_meta, mn_meta = self.create_meta_information(matrix)
        
        # Convert string values to colour mappings
        aesthetics = self.create_aesthetics()
        pmn_meta_cv = pmn_meta.copy()
        for col in pmn_meta_cv.columns:
            pmn_meta_cv[col] = pmn_meta_cv[col].map(aesthetics[col])

        mn_meta_cv = mn_meta.copy()
        for col in mn_meta_cv.columns:
            mn_meta_cv[col] = mn_meta_cv[col].map(aesthetics[col])

        row_cls, col_cls = pmn_meta_cv, mn_meta_cv

        # Intermediary transpose step
        if self.transpose:
            matrix = matrix.T
            row_cls, col_cls = col_cls, row_cls
        
        # Append neuron type to motor neuron bodyids
        if MN_typing:
            if self.transpose: # Transposed. so mns are indices
                for mn_id in matrix.index:
                    mn_type = self.conndf.neuron_master.query(f"bodyId == {mn_id}")['type'].values[0]
                    matrix.rename(index={mn_id:f"{mn_id}_{mn_type}"}, inplace=True)

            else:
                for mn_id in matrix.columns:
                    mn_type = self.conndf.neuron_master.query(f"bodyId == {mn_id}")['type'].values[0]
                    matrix.rename(columns={mn_id:f"{mn_id}_{mn_type}"}, inplace=True)

        # Actual plotting
        cmg = sns.clustermap(
            matrix,
            method='ward', # Overriden 
            cmap=cmap,
            metric='euclidean', # Overriden
            row_colors=row_cls,
            col_colors=col_cls,
            yticklabels=yticklabels, #Show all ticks
            xticklabels=xticklabels,
            row_cluster=row_cluster,
            col_cluster=col_cluster,
            **kwargs
            )
        cmg.ax_heatmap.set_yticklabels(cmg.ax_heatmap.get_ymajorticklabels(), fontsize=8)
        #self.latest_clustermap = cmg
        # Main Colorbar 
        main_cbar = cmg.ax_cbar
        main_cbar.set_position([0.9, 0.9, 0.05, 0.1]) #left, bottom, width, height

        # Class Legend
        neurclass_objs = []
        for label in pmn_meta['class'].unique():
            x = cmg.ax_row_dendrogram.bar(0, 0, color=aesthetics['class'][label], label=label, linewidth=0)
            neurclass_objs.append(x)
        neurclass_legend = plt.legend(
            neurclass_objs, 
            pmn_meta['class'].unique(), 
            loc="center", 
            title='Neuron classes', 
            bbox_to_anchor=(.13, 0.87), 
            bbox_transform=gcf().transFigure,
            fontsize='xx-small'
            )
        
        # Transmitter Legend
        transmitter_objs = []
        for label in pmn_meta['transmitter'].unique():
            x = cmg.ax_row_dendrogram.bar(0, 0, color=aesthetics['transmitter'][label], label=label, linewidth=0)
            transmitter_objs.append(x)
        transmitter_legend = plt.legend(
            transmitter_objs, 
            pmn_meta['transmitter'].unique(), 
            loc="center", 
            title='Neurotransmitter', 
            bbox_to_anchor=(.13, 0.8), 
            bbox_transform=gcf().transFigure,
            fontsize='xx-small'
            )
        plt.gca().add_artist(neurclass_legend)

        # ExitNerve Legend
        exitnerve_objs = []
        for label in mn_meta['exitnerve'].unique():
            x = cmg.ax_row_dendrogram.bar(0, 0, color=aesthetics['exitnerve'][label], label=label, linewidth=0)
            exitnerve_objs.append(x)
        exitnerve_legend = plt.legend(
            exitnerve_objs, 
            mn_meta['exitnerve'].unique(), 
            loc="center", 
            title='Exit Nerve', 
            bbox_to_anchor=(0.95, 0.85), 
            bbox_transform=gcf().transFigure,
            fontsize='xx-small'
            )
        plt.gca().add_artist(transmitter_legend)

        # PCT ColorBar
        cbar_cax = cmg.fig.add_axes([0.20, 0.9, 0.01, 0.07]) #left, bottom, width. height
        #pct_cbar_ax = fig.add_axes([0.7, 0.75, 0.01, 0.05], sharex = plt.gca())
        pct_cbar = plt.colorbar(
            mpl.cm.ScalarMappable(
                norm=plt.Normalize(
                    min(pmn_meta[f'pct_{neuromere}{side}']), max(pmn_meta[f'pct_{neuromere}{side}'])),
                cmap=aesthetics[f'pct_{neuromere}{side}']
                ),
            cax=cbar_cax
            )
        pct_cbar.outline.set_visible(False)
        plt.show()

    def get_vnc_roi_meshes(self, offline=False):
        if offline:
            if 'ACCEPTED_ROIS.pkl' in os.listdir():
                with open('ACCEPTED_ROIS.pkl', 'rb') as f:
                    self.ACCEPTED_ROIS = pickle.load(f)
            else:
                raise ValueError("No ROIS downloaded yet.")
            
            if 'VNC_roi_meshes.pkl' in os.listdir():
                with open('VNC_roi_meshes.pkl', 'rb') as f:
                    self.VNC_roi_meshes = pickle.load(f)
            else:
                raise ValueError("No ROI meshes downloaded yet.")
            
        else:
            self.ACCEPTED_ROIS = neuprint.fetch_all_rois()
            with open('ACCEPTED_ROIS.pkl', 'wb') as f:  # open a text file
                pickle.dump(self.ACCEPTED_ROIS, f) # serialize the list

            self.VNC_roi_meshes = [neu.fetch_roi(x) for x in self.ACCEPTED_ROIS]
            with open('VNC_roi_meshes.pkl', 'wb') as f:  # open a text file
                pickle.dump(self.VNC_roi_meshes, f) # serialize the list    

        return self.VNC_roi_meshes

    def get_neuron_skeletons(self, bodyId_list):
        return neu.fetch_skeletons(bodyId_list, missing_swc='raise', heal=True, with_synapses=False)
    
    def get_combined_meshes(self, bodyId_list):
        neuron_skeletons = self.get_neuron_skeletons(bodyId_list)
        VNC_roi_meshes = self.get_vnc_roi_meshes()
        return [neuron_skeletons, VNC_roi_meshes]

    def draw_morphologies(self, bodyId_list=None, matrix=None):
        if matrix is None and bodyId_list is not None:
            bodyId_list = bodyId_list
        elif matrix is not None and bodyId_list is None:
            rows = list(matrix.index)
            cols = list(matrix.columns)
            bodyId_list = list(set(rows + cols))
        else:
            raise ValueError()
        
        combined_meshes = self.get_combined_meshes(bodyId_list)

        def automate_aesthetics(self, sns_palette, by):
            _n = self.conndf.neuron_master[by].nunique()
            _strs = self.conndf.neuron_master[by].unique()
            _pal = sns.color_palette(sns_palette, _n+1)
            _cval = dict(zip(map(str, _strs), _pal))
            return _cval
        
        def shift_colour(rgb, max_shift):
            r, g, b = rgb
            r_shift = rd.uniform(-max_shift, max_shift)
            g_shift = rd.uniform(-max_shift, max_shift)
            b_shift = rd.uniform(-max_shift, max_shift)

            new_r = min(max(r + r_shift, 0), 1)
            new_g = min(max(g + g_shift, 0), 1)
            new_b = min(max(b + b_shift, 0), 1)

            return new_r, new_g, new_b
    
        class_cval = automate_aesthetics(self, sns_palette='dark', by='class')
        
        n_df = self.conndf.neuron_master.query(f"bodyId == {bodyId_list}")[["bodyId", "class"]]
        # Add class-dep cval, 
        n_df["class_cval"] = n_df["class"].map(class_cval)
        n_df["class_cval"] = n_df["class_cval"].apply(lambda x: shift_colour(rgb=x, max_shift=0.3))

        neuron_alpha = 0.9
        mesh_alpha = 0.1
        cmapping = []
        # Colour neurons by class
        for ix, _ in enumerate(combined_meshes[0]):
            bid = combined_meshes[0][ix].id
            cval = n_df.query(f"bodyId == {bid}")['class_cval'].item()
            cmapping.append(tuple([x for x in cval] + [neuron_alpha]))

        # If want to decrease alpha of ROIs,
        alpha = 0.1
        col = (0.85, 0.85, 0.85, alpha)
        for ix, _ in enumerate(combined_meshes[1]):
            cmapping.append(col)

        fig = navis.plot3d(
            x=combined_meshes,
            inline=False,
            width=500,
            height=800,
            color=cmapping,
            legend_group={13809:'MDN (13809)',14419:'MDN (14419)',13438:'MDN (13438)',14523:'MDN (14523)',11493:'LBL40 (11493)', 10994:'LBL40 (10994)', 13293:'LUL130 (13293)', 13574:'LUL130 (13574)', 14084:'LUL130 (14084)', 13323:'LUL130 (13323)'},
            backend='plotly'
        )
        fig.show(renderer='notebook')

    def draw_network(self, matrix=None, **kwargs):
        """ Kwargs passed to NetworkX plotter. """
        if matrix is None:
            matrix = self.preMN_postMN_matrix
            master_edgelist = self.preMN_postMN_edgelist
        else:
            master_edgelist = self.conndf.neuron_master
            
        if self.transpose:
            matrix = matrix.T

        
        edgelist = master_edgelist.query(
            f"bodyId_pre in {list(matrix.index)}").query(
                f"bodyId_post in {list(matrix.columns)}")

        gr = nx.from_pandas_edgelist(
            edgelist,
            source='bodyId_pre',
            target='bodyId_post',
            edge_attr='weight_relative',
            create_using=nx.DiGraph
            )
        
        attrs = {}
        neurons = self.conndf.neuron_master.to_dict('index')
        for kam in neurons.keys():
            b_id = neurons[kam]['bodyId']
            del neurons[kam]['bodyId']
            attrs[b_id] = neurons[kam]

        nx.set_node_attributes(gr, attrs)

        # Reassign according to order of clustermap. 
        # MNs layer 1, first in dictionary iterables
        for v,data in gr.nodes(data=True):
            # Transmitter Assignment
            idx = np.argmax([data['ntGlutamateProb'], data['ntGabaProb'], data['ntAcetylcholineProb']]) # 0 = glut, 1 = gaba, 2 = acetylchol
            gr.nodes[v]['transmitter'] = ['Glutamate', 'Gaba', 'Acetylcholine'][idx] 
            # Class Assignment
            if self.transpose:
                if data['class'] == 'motor neuron':
                    gr.nodes[v]['layer'] = 2
                else:
                    gr.nodes[v]['layer'] = 1
            else:
                if data['class'] == 'motor neuron':
                    gr.nodes[v]['layer'] = 1
                else:
                    gr.nodes[v]['layer'] = 2
            # Need to implement 3 layers for motor neurons that are also 'premotor'?
            ##### FOR NOW; remove MN from PMN

        plt.figure(figsize=(10,5))
        aesthetics = self.create_aesthetics()
        node_color = [aesthetics['class'][data] for v, data in gr.nodes(data='class')]
        edge_weights = [30*gr[u][v]['weight_relative'] for u,v in gr.edges()]
        edge_arrows = [x/3 + 5 for x in edge_weights]

        transmitter_cval = aesthetics['transmitter'].copy()
        transmitter_cval['Acetylcholine'] = transmitter_cval['Glutamate']
        transmitter_cval['Glutamate'] = transmitter_cval['Gaba']
        edge_colours = [transmitter_cval[data] for v,data in gr.nodes(data='transmitter')]

        pos = nx.multipartite_layout(gr, subset_key='layer', align='horizontal', scale=3)

        if self.transpose:
            MN_order = list(matrix.index)[::-1]#[list(matrix.columns)[i] for i in self.latest_clustermap.dendrogram_col.reordered_ind]
            PMN_order = list(matrix.columns) #[list(matrix.index)[i] for
        else:
            MN_order = list(matrix.columns)#[list(matrix.columns)[i] for i in self.latest_clustermap.dendrogram_col.reordered_ind]
            PMN_order = list(matrix.index)[::-1] #[list(matrix.index)[i] for i in self.latest_clustermap.dendrogram_row.reordered_ind[::-1]]
            
        for n in MN_order:
            if n in PMN_order:
                PMN_order.remove(n)

        new_coord = dict()
        ix = 0
        for _, coord in dict(list(pos.items())).items():
            new_coord[(MN_order + PMN_order)[ix]] = coord
            ix += 1

        nx.draw(
            gr, 
            new_coord, 
            with_labels=False, 
            node_color=node_color,
            font_size=0, # 15; 5
            node_size=50, # 15; 600
            width=edge_weights,
            edge_color=edge_colours,
            arrowsize=edge_arrows
            )
        plt.show()

    def umap(
            self, 
            n_neighbors=10, 
            min_dist=0.0, 
            n_components=2,
            metric='euclidean', 
            matrix=None, 
            random_state=0):
        """ Passed to UMAP() to construct reducer instance. Returns embedding """
        if matrix is None:
            matrix = self.preMN_postMN_matrix
        if self.transpose:
            matrix = matrix.T

        reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric, random_state=random_state)
        embedding = reducer.fit_transform(matrix)
        embedding = pd.DataFrame(embedding, index=matrix.index)
        self.latest_umap_embedding = embedding
        self.latest_umap_embedding_params = {
            'n_neighbors':n_neighbors,
            'min_dist':min_dist,
            'n_components':n_components,
            'metric':metric
            }
        
        return embedding
    
    def HDBSCAN(self, matrix=None, **kwargs):
        # read on soft-clustering https://hdbscan.readthedocs.io/en/latest/soft_clustering_explanation.html
        if matrix is None:
            matrix = self.preMN_postMN_matrix
        if self.transpose:
            matrix = matrix.T

        clusterer = hdbscan.HDBSCAN(**kwargs).fit(matrix)
        self.latest_HDBSCAN_tree = clusterer.condensed_tree_
        self.latest_HDBSCAN_prob = clusterer.probabilities_
        return clusterer.labels_

    def kmeans(self, n_clusters=10, n_init=10, random_state=0, input_data=None, **kwargs):
        """ Passed to KMeans() instance. Returns clusters. """
        if input_data is None:
            if self.latest_umap_embedding is not None:
                input_data = self.latest_umap_embedding
            else:
                self.umap()
                input_data = self.latest_umap_embedding

        km = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state, **kwargs)
        return km.fit_predict(input_data.to_numpy())

    def subset_by_cluster(self, k, annot_df=None):
        """ Must have an annotated matrix with clusters. """
        if annot_df is None:
            if self.latest_umap_embedding is not None:
                annot_df = self.latest_umap_embedding
            else:
                self.kmeans()
                annot_df = self.latest_umap_embedding
        main_matrix = self.preMN_postMN_matrix ## here
        if self.transpose:
            main_matrix = main_matrix.T

        return main_matrix.loc[
            list(annot_df.query(f"km_cluster == {k}").index),
            main_matrix.loc[list(annot_df.query(f"km_cluster == {k}").index)].any()]
  
        # Interactive UMAP/KMeans computation then Plotting. First Step
    def show_cluster_scatter(
            self, transpose, n_neighbors, metric, n_clusters, load_predef_clust=False):
        self.transpose = transpose
        embedding = self.umap(n_neighbors=n_neighbors, metric=metric, matrix=None)
        clusters = self.kmeans(n_clusters=n_clusters, input_data=embedding)
        embedding['km_cluster'] = clusters
        embedding['km_cluster'] = embedding['km_cluster']
        self._current_embedding = embedding.copy() # Store

        # Replace with preivous 'kept' clusters
        if load_predef_clust == True and self._prev_embedding_d is not None:
            embedding['km_cluster'].update(pd.Series(self._prev_embedding_d)) # Replace bodyid cluster with previous
            if self._kept_cluster_col in self._palette:
                self._palette.remove(self._kept_cluster_col)
        # Marker. Keep previous colour of chosen cluster, make it colour unavailable for current embedding
        marker_styles = []
        for label in embedding['km_cluster']:
            if label == self._kept_cluster: # skipped if false, since will be None, or -X
                marker_styles.append(
                    {'symbol': 'circle', 
                    'size':4, 'color':self._kept_cluster_col, 
                    'line': {'width': 3, 'color': 'black'}} # highlight it too
                    )
            else: # Current embedding colours
                marker_styles.append(
                    {'symbol': 'circle', 
                    'size':10, 'color':self._palette[int(label)%len(self._palette)], # Cycle through palette
                    'line': {'width': 0.7, 'color': 'black'}}
                    )
                
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=embedding[0],
            y=embedding[1],
            customdata=np.stack( (list(embedding.index), list(embedding['km_cluster'])), axis=-1),
            mode='markers',
            marker=dict(
                symbol=[marker['symbol'] for marker in marker_styles],
                size=[marker['size'] for marker in marker_styles],
                color=[marker['color'] for marker in marker_styles],
                line=dict(
                    width=[marker['line']['width'] for marker in marker_styles],
                    color=[marker['line']['color'] for marker in marker_styles]
                    )
            ),
            hovertemplate='<br>bodyId: %{customdata[0]} <br>cluster: %{customdata[1]}<extra></extra>'
        ))
        fig.update_traces(marker=dict(opacity=0.9))
        fig.update_layout(
            title=f"UMAP embedding (Neighbours = {n_neighbors}), Kmeans (Clusters = {n_clusters}), on {'MNs' if self.transpose else 'PMNs'}",
            width=800, 
            height=800
        )
        fig.show(renderer='notebook')

        # Store current as prev (latest?)
        self._prev_embedding = embedding.copy()

        interact_manual(
            self.show_heat_network,
            cluster=widgets.BoundedIntText(value=0,min=0,disabled=False,continuous_update=False, description='Choose cluster to viz:', style = {'description_width': 'initial'}),
            with_previous=widgets.Dropdown(options=['Latest only', f'Include {self._kept_cluster}'], value='Latest only', description='Include manually kept clusters', style = {'description_width': 'initial'}, layout=widgets.Layout(visibility = ['hidden' if self._kept_cluster is None else 'visible'][0])),
            MN_typing=widgets.Dropdown(options=[True, False], value=False, description='If True, types MNs', style = {'description_width': 'initial'}),
            )
            
        interact_manual(
            self.keep_cluster,
            choose=widgets.Dropdown(options=self._val_range, value=0, disabled=False,description='Choose cluster to keep:', style = {'description_width': 'initial'}),
            )
    
    def show_heat_network(self, cluster, with_previous=False, MN_typing=False):   
        if with_previous: # Includes version of the embedding with kept cluster
            embedding = self._prev_embedding 
        else:
            embedding = self._current_embedding # Latest, without any overwrite
        matrix_subset = self.subset_by_cluster(cluster, annot_df=embedding)
        self.draw_clustermap(matrix=matrix_subset, MN_typing=MN_typing)
        
        # try morphology
        self.draw_morphologies(matrix=matrix_subset)

        # Replace network with morphology maps?
        #self.draw_network(matrix=matrix_subset)
    
    def keep_cluster(self, choose):
        # Limit range to previous embedding
        # Resets any kept clusters by referring to svar.current_embedding which is the embedding str8 after 
        # kmeans/umap
        self._val_range = self._current_embedding['km_cluster'].unique()
        if choose in self._val_range: # Valid options
            self._kept_cluster = choose + 0.1
            cluster_embed = self._current_embedding.query(f"km_cluster == {choose}")
            self._prev_embedding_d = dict(zip(cluster_embed.index, cluster_embed.km_cluster+0.1)) # dict to replace
            self._kept_cluster_col = self._palette[choose%len(self._palette)]

        if self._stack_after:
            self.start_premotor_interact()
    
    def start_premotor_interact(self, stack_after=False):
        self._stack_after = stack_after
        interact_manual(
            self.show_cluster_scatter, 
            transpose=widgets.Dropdown(options=[True, False], value=False, description='If True, clusters MNs', style = {'description_width': 'initial'}),
            n_neighbors=widgets.BoundedIntText(value=10,min=1,disabled=False,continuous_update=False, description='N Neighbours (UMAP)', style = {'description_width': 'initial'}),
            metric=widgets.Dropdown(options=['euclidean', 'cosine'], value='euclidean', description='Distance Metric (UMAP)', style = {'description_width': 'initial'}),
            n_clusters=widgets.BoundedIntText(value=10,min=1,disabled=False,continuous_update=False, description='N Clusters (Kmeans)', style = {'description_width': 'initial'}),
            load_predef_clust=widgets.ToggleButton(value=False, disabled=False,continuous_update=False, description=f'Load: cluster {self._kept_cluster}', style = {'description_width': 'initial'}, layout=widgets.Layout(visibility = ['hidden' if self._kept_cluster is None else 'visible'][0]))
            )
        
    def cosine_similarity(self, matrix, linkage='ward'):
        """ Given a M observation x N dims/vars matrix, returns M x M pairwise
        cosine similarity matrix. Also returns the row linkage for the 
        matrix, as calculated by @linkage param. Default ward linkage. 
        Rows determine the 'things' to perform pairwise. 

        Note: Does not care about directionality in this case.
        
        'Similar downstream profiles'
        
        """
        cosine_dist = sc.spatial.distance.pdist(matrix, metric='cosine')
        cosine_simi = 1 - sc.spatial.distance.squareform(cosine_dist)
        cosine_link = sc.cluster.hierarchy.linkage(cosine_simi, method='ward')
        return cosine_link, cosine_simi

    def draw_cosine(self, matrix=None, between='rows'):
        """ Given a matrix of M rows, and N columns, calculate the pairwise 
        cosine similarity between @betweens. If rows, then based on common
        columns. If columns, then based on common rows. Shows this information 
        as a square seaborn heatmap. 
        """

        # By default uses the current premn_postmn_matrix stored
        if matrix is None:
            mat = self.preMN_postMN_matrix
        else: # Supports any matrix, but will do the same steps of transposing
            mat = matrix
        
        if between == 'columns':
            mat = mat.T # Columns are now rows, as sc.pdist/cosine_sim, does row-wise comparisons
        
        # Perform cosine_similarity 
        link, simi = self.cosine_similarity(mat)

        # Draw heatmap
        labels = mat.index

        cg = sns.clustermap(
            data=simi, 
            cmap='viridis',
            row_linkage=link,
            col_linkage=link,
            yticklabels=True,
            xticklabels=False # same as yticklabels
            )
        cg.ax_heatmap.set_yticklabels(labels, fontsize=8)
        plt.show()

from brian2 import *
from scipy.ndimage import gaussian_filter1d

class SpikeAnalysis():
    """ Analyse spike data from simulations. """
    def __init__(self, all_spikes, params):
        self.spike_indices = all_spikes['i']
        self.spike_times = all_spikes['t']
        self.params = params
        self.id_bmap = None
        self.dense_spike_train = None
        self.rates = None
        self.to_dense_spike_train()

    # Preprocess Brian2 spike data
    def to_dense_spike_train(self):
        "Convert compressed spike data into sparse format, where the spike of each neuron is encoded as 1 in its own list."
        # [2ms, ..., ..., 5ms]
        # [13, ..., ..., 13]
        # -> train for neuron 13: [1, 0, 0, 5] 
        # Store as 2d matrix, rows = neuron, col = timepoint/samplepoint, values = 1 or 0 for spike or not

        # spikes = dict of indices and times
        id_list = list(dict.fromkeys(self.spike_indices)) # Ordered unique list; first = first spiked, should be the target neuron
        id_bmap = {bid: index for index, bid in enumerate(id_list)} # {13809:0, 16109:1, etc}
        self.id_bmap = id_bmap
        # Omitting 0-2000ms, taking 2000ms as the 'start'
        stim_start = self.params['start_ms']
        dt = self.params['deltaT']/ms

        spike_sample_times = self.spike_times/ms/dt - stim_start/dt # 20000 - 20000 -> 0, 100000 - 20000 -> 80000 time points
        sim_dur = int(self.params['runtime']/ms/dt - stim_start/dt)

        spike_matrix = np.zeros((len(id_list), sim_dur), dtype=int) # Declare int type as vals will be just 1s and 0s for spikes
        for ix, bodyid in enumerate(self.spike_indices):
            row = int(id_bmap[bodyid])
            col = int(spike_sample_times[ix] - 1) # if occurs at tpt = 250, insert at index 249
            spike_matrix[row, col] = 1
        self.dense_spike_train = spike_matrix
    
    def get_avg_rate(self, spike_train):
        "Convert a single spike train into avg firing rate using interspike intervals"
        # Spiking frequency/firing rate
        dt = self.params['deltaT']
        ISIs = np.where(spike_train == 1)[0]
        if ISIs.size == 0: ## Avoid dividing by 0
            return 0*Hz
        else:
            return 1 / np.mean(np.diff(ISIs*dt))
    
    def firing_rate(self, spike_train, window, deltaT):
        sampling_rate = 0.1
        window_size = int(window/sampling_rate)
        step = int(deltaT/sampling_rate)
        tvec = np.arange(0, len(spike_train)*0.1, step*0.1)

        rates = list()
        for i in range(0, len(spike_train) - window_size +1, step):
            window = spike_train[i:i+window_size]
            rate = np.sum(window) / (window_size * 0.0001)
            rates.append(rate)
        rates = np.array(rates)
        tvec = tvec[:len(rates)]
        return tvec, rates
    
    def get_smooth_rates(self, window=50, deltaT=1, sigma=2):
        dense_spikes = self.dense_spike_train
        rate_matrix = list()
        for i in range(dense_spikes.shape[0]):
            tvec,rates = self.firing_rate(dense_spikes[i, :], window, deltaT)
            rate_matrix.append(list(rates))
        rates_matrix = np.array(rate_matrix)
        filtered_rates_matrix = gaussian_filter1d(rates_matrix, sigma)
        return tvec, filtered_rates_matrix
    
    def get_jagged_rates(self, window=50, deltaT=1):
        # No gaussian convolution as above
        dense_spikes = self.dense_spike_train
        rate_matrix = list()
        for i in range(dense_spikes.shape[0]):
            tvec,rates = self.firing_rate(dense_spikes[i, :], window, deltaT)
            rate_matrix.append(list(rates))
        rates_matrix = np.array(rate_matrix)
        return tvec, rates_matrix
    
    def get_LR_list(self, sub_df):
        """ Get list of left and right neurons based on dominant ROI or somaSide. """
        LEFT_IXS = list()
        RIGHT_IXS = list()

        for ix, row in sub_df.iterrows():
            bid = row["bodyId"]
            main_roi = row["mainRoi"]
            if "(R)" in main_roi:
                RIGHT_IXS.append(self.id_bmap[bid])
            elif "(L)" in main_roi:
                LEFT_IXS.append(self.id_bmap[bid])
            else: # Then annotate by somaSide
                soma_side = row["somaSide"]
                if soma_side == "RHS":
                    RIGHT_IXS.append(self.id_bmap[bid])
                else: 
                    LEFT_IXS.append(self.id_bmap[bid])

        return LEFT_IXS, RIGHT_IXS

    def get_col_list(self, sub_df, col_name, col_query):
        """ Get list of neurons within the specified col_query list for col_name
            string. """
        cl = list(sub_df[sub_df[col_name].isin(col_query)]['bodyId'])
        col_ixs = [self.id_bmap[x] for x in cl]
        return col_ixs

    def get_inclusion_list(self, sub_df, neuron_class=None, neuropil=None, neuron_type=None):
        incl_ls = list()
        incl_ls_set = list()

        if neuron_class is not None: # Assuming neuron_class is a valid class string. LIST
            incl_ls_set.append(self.get_col_list(sub_df, 'class', neuron_class))
        # Label by neuropil
        if neuropil is not None: # LIST
            incl_ls_set.append(self.get_col_list(sub_df, 'mainRoi', neuropil))
        # Label by neuron type
        if neuron_type is not None:
            incl_ls_set.append(self.get_col_list(sub_df, 'type', neuron_type))

        for label in incl_ls_set:
            if label:
                if not incl_ls:
                    incl_ls += label
                else:
                    incl_ls = list(set(incl_ls) & set(label))
        return incl_ls

    def bid_to_string(self, sub_df, body_id):
        """ String formatted representation of the neuron, given its bodyId.
        Current format: bodyId:class:type:mainRoi"""
        nclass = sub_df.query(f'bodyId == {body_id}')['class'].item()
        ntype = sub_df.query(f'bodyId == {body_id}')['type'].item()
        nmroi = sub_df.query(f'bodyId == {body_id}')['mainRoi'].item()
        return f"{body_id}:{nclass}:{ntype}:{nmroi}"

    def show_ordered_raster(self, conndf=None, side=False, neuron_class=None, neuropil=None, neuron_type=None, custom_bids=None, figsize=None, pfr=False, cluster_colors=None):
        """ Raster plot of spikes, in ascending order from first spiker (bottom-most: should be the stimulated neuron)
        to last spiker (top-most). """
        if self.dense_spike_train is None:
            self.to_dense_spike_train()
        bmap = {v:k for k,v in self.id_bmap.items()}
        annotated_neurons = list()
        if figsize is None:
            figsize = (15,12)
        # If labelling
        if conndf is not None: 
            bid_ls = [{v:k for k,v in self.id_bmap.items()}[x] for x in range(self.dense_spike_train.shape[0])]
            neurons_here = conndf.neuron_master.query(f"bodyId in {bid_ls}")
            # Label by side: by mainRoi side, then by somaSide (if no L/R roi annotation)
            if side:
                LEFT_ixs, RIGHT_ixs = self.get_LR_list(neurons_here)
            # Label by class. If all none returns empty.
            incl_ls = self.get_inclusion_list(neurons_here, neuron_class, neuropil, neuron_type)

            fig, ax = plt.subplots(figsize=figsize)
            
            if cluster_colors is None:
                for i, neuron in enumerate(self.dense_spike_train):
                    stimes = np.where(neuron == 1)[0]*self.params['deltaT']/ms
                    bid = bmap[i]
                    if custom_bids is not None:
                        if bid not in custom_bids:
                            continue
                    if not incl_ls: # If not subsetting by neuropil or class
                        if i in RIGHT_ixs:
                            ax.scatter(stimes, np.full_like(stimes, i), color='r', s=0.5)
                        elif i in LEFT_ixs:
                            ax.scatter(stimes, np.full_like(stimes, i), color='b', s=0.5)
                    else: # Subsetting by neuropil or class
                        if i in incl_ls:
                            annotated_neurons.append(i)
                            if i in RIGHT_ixs:
                                ax.scatter(stimes, np.full_like(stimes, i), color='r', s=1)
                            elif i in LEFT_ixs:
                                ax.scatter(stimes, np.full_like(stimes, i), color='b', s=1)
                        else:
                            ax.scatter(stimes, np.full_like(stimes, i), color='k', s=0.01)
            else:
                for i, neuron in enumerate(self.dense_spike_train):
                    stimes = np.where(neuron == 1)[0]*self.params['deltaT']/ms
                    bid = bmap[i]
                    if custom_bids is not None:
                        if bid not in custom_bids:
                            continue
                    if not incl_ls: # If not subsetting by neuropil or class
                        if bid in cluster_colors.keys():
                            ax.scatter(stimes, np.full_like(stimes, i), color=cluster_colors[bid], s=5, edgecolors='black', lw=0.25)
                    else: # Subsetting by neuropil or class
                        if i in incl_ls:
                            annotated_neurons.append(i)
                            if bid in cluster_colors.keys():
                                ax.scatter(stimes, np.full_like(stimes, i), color=cluster_colors[bid], s=5, edgecolors='black', lw=0.25)
                        else:
                            ax.scatter(stimes, np.full_like(stimes, i), color='k', s=0.005)
        # If default plotting
        else:
            fig, ax = plt.subplots(figsize=figsize)
            for i, neuron in enumerate(self.dense_spike_train):
                stimes = np.where(neuron == 1)[0]*self.params['deltaT']/ms
                ax.scatter(stimes, np.full_like(stimes, i), color='k', s=0.5)
            #ax.set_yticklabels([])
            #ax.set_yticks([])
            #ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Neuron index")
            params = self.params
            ax.set_title(f"W{params['psc_q']}, dly{params['syn_delay']}, memcap{params['C_m']}")
            #self.latest_mpl_fig = fig, ax

        # Global plotting settings
        #ax.set_yticks(list(self.id_bmap.values())) # {v:k for k,v in spike_analyser.id_bmap.items()}[i]
        #ax.set_yticklabels(list(self.id_bmap.keys()))
        #ax.set_yticklabels([])
        #ax.set_yticks([])
        ax.set_xlabel("Time (ms)")
        params = self.params
        params = self.params
        stim_bids = params["target_bid"]
        stim_ids = [self.id_bmap[s] for s in stim_bids]
        stim_rates = [self.get_avg_rate(self.dense_spike_train[s,:]) for s in stim_ids]
        sim_params = f"W{params['psc_q']}, dly{params['syn_delay']}, memcap{params['C_m']}"
        fig_title = f"{sim_params} | {stim_bids} ~ {stim_rates}"
        ax.set_title(fig_title)

        # Population firing rate plot
        if pfr:
            tvec, population_rate = self.get_population_firing_rates(window=50, deltaT=1, sigma=8, smooth=True, exclude_stim=True)
            ax2 = ax.twinx()
            ax2.plot(tvec, population_rate, linewidth=2)
            ax2.set_ylabel('Mean population firing rate (Hz)')
        

        self.latest_mpl_fig = fig, ax
        plt.show() # Or return fig/ax
        savefig(f'{fig_title}.png')
        return [{v:k for k,v in self.id_bmap.items()}[x] for x in annotated_neurons]

    def show_ordered_raster_plotly(self, conndf=None, side=False, neuron_class=None, neuropil=None, neuron_type=None, include_all=False, custom_bids=None):
        """ Raster plot of spikes, in ascending order from first spiker (bottom-most: should be the stimulated neuron)
        to last spiker (top-most). Plotly backend"""
        if self.dense_spike_train is None:
            self.to_dense_spike_train()
        bmap = {v:k for k,v in self.id_bmap.items()}
        bid_ls = [bmap[x] for x in range(self.dense_spike_train.shape[0])]

        neurons_here = conndf.neuron_master.query(f"bodyId in {bid_ls}")
        # If labelling
        if side or neuron_class is not None or neuropil is not None: 
            # Label by side,             
            if side:
                LEFT_ixs, RIGHT_ixs = self.get_LR_list(neurons_here)
            # Label by class. If all none returns empty.
            incl_ls = self.get_inclusion_list(neurons_here, neuron_class, neuropil, neuron_type)

            data = []
            for i, neuron in enumerate(self.dense_spike_train):
                stimes = np.where(neuron == 1)[0]*self.params['deltaT']/ms
                bid = bmap[i]
                sformat = self.bid_to_string(neurons_here, bid)

                # Iterate only ids in custom_bids
                if custom_bids is not None:
                    if bid not in custom_bids:
                        continue

                if not incl_ls: # If not subsetting by neuropil or class
                    if i in RIGHT_ixs:
                        trace = go.Scatter(x=stimes, y=np.full_like(stimes, i), mode='markers', 
                                marker=dict(color='red', size=2), showlegend=False, text=sformat, hoverinfo='text')
                        data.append(trace)
                    elif i in LEFT_ixs:
                        trace = go.Scatter(x=stimes, y=np.full_like(stimes, i), mode='markers', 
                                marker=dict(color='blue', size=2), showlegend=False, text=sformat, hoverinfo='text')
                        data.append(trace)
                else: # Subsetting by neuropil or class
                    if i in incl_ls:
                        if i in RIGHT_ixs:
                            trace = go.Scatter(x=stimes, y=np.full_like(stimes, i), mode='markers', 
                                marker=dict(color='red', size=3), showlegend=False, text=sformat, hoverinfo='text')
                            data.append(trace)
                        elif i in LEFT_ixs:
                            trace = go.Scatter(x=stimes, y=np.full_like(stimes, i), mode='markers', 
                                marker=dict(color='blue', size=3), showlegend=False, text=sformat, hoverinfo='text')
                            data.append(trace)
                        else:
                            if include_all:
                                trace = go.Scatter(x=stimes, y=np.full_like(stimes, i), mode='markers', 
                                    marker=dict(color='black', size=1), showlegend=False, text=sformat, hoverinfo='text')
                                data.append(trace)

                    else:
                        trace = go.Scatter(x=stimes, y=np.full_like(stimes, i), mode='markers', 
                                marker=dict(color='black', size=0.5), showlegend=False, text=sformat, hoverinfo='text')
                        data.append(trace)

        # If default plotting
        else:
            data = []
            for i, neuron in enumerate(self.dense_spike_train):
                stimes = np.where(neuron == 1)[0]*self.params['deltaT']/ms
                bid = bmap[i]
                sformat = self.bid_to_string(neurons_here, bid)
                trace = go.Scatter(x=stimes, y=np.full_like(stimes, i), mode='markers', 
                                marker=dict(color='black', size=2), showlegend=False, text=sformat, hoverinfo='text')
                data.append(trace)

        # Global plotting settings
        params = self.params
        stim_bids = params["target_bid"]
        stim_ids = [self.id_bmap[s] for s in stim_bids]
        stim_rates = [self.get_avg_rate(self.dense_spike_train[s,:]) for s in stim_ids]
        sim_params = f"W{params['psc_q']}, dly{params['syn_delay']}, memcap{params['C_m']}"
        fig_title = f"{sim_params} | {stim_bids} ~ {stim_rates}"
        layout = go.Layout(
            width=1500, 
            height=800, 
            title=fig_title,
            xaxis_title="Time(ms)")
        fig = go.Figure(data=data, layout=layout)
        self.latest_plotly_fig = fig
        return fig

    def show_firing_rates(self, window=50, deltaT=1, sigma=8, conndf=None, side=False, neuron_class=None, neuropil=None, neuron_type=None, custom_bids=None):
        """ Raster plot of spikes, in ascending order from first spiker (bottom-most: should be the stimulated neuron)
        to last spiker (top-most). """
        if self.dense_spike_train is None:
            self.to_dense_spike_train()
        bmap = {v:k for k,v in self.id_bmap.items()}
        annotated_neurons = list()
        tvec, rates = self.get_smooth_rates(window, deltaT, sigma)
        #rates = rates[:, :self.params['stop_ms']]
        #tvec = tvec[self.params['start_ms']:self.params['stop_ms']]
        # If labelling
        if conndf is not None: 
            bid_ls = [{v:k for k,v in self.id_bmap.items()}[x] for x in range(self.dense_spike_train.shape[0])]
            neurons_here = conndf.neuron_master.query(f"bodyId in {bid_ls}")
            # Label by side: by mainRoi side, then by somaSide (if no L/R roi annotation)
            if side:
                LEFT_ixs, RIGHT_ixs = self.get_LR_list(neurons_here)
            # Label by class. If all none returns empty.
            incl_ls = self.get_inclusion_list(neurons_here, neuron_class, neuropil, neuron_type)

            fig, ax = plt.subplots(figsize=(18,8))
            plt.xlim([0, self.params['stop_ms']-self.params['start_ms']])
            for i in range(rates.shape[0]):
                bid = bmap[i]
                if custom_bids is not None:
                    if bid not in custom_bids:
                        continue
                if not incl_ls: # If not subsetting by neuropil or class
                    if i in RIGHT_ixs:
                        ax.plot(tvec, rates[i,:], color='r', linewidth=0.5)
                    elif i in LEFT_ixs:
                        ax.plot(tvec, rates[i,:], color='b', linewidth=0.5)
                else: # Subsetting by neuropil or class
                    if i in incl_ls:
                        annotated_neurons.append(i)
                        if i in RIGHT_ixs:
                            ax.plot(tvec, rates[i,:], color='r', linewidth=1)
                        elif i in LEFT_ixs:
                            ax.plot(tvec, rates[i,:], color='b', linewidth=1)
                    else:
                        ax.plot(tvec, rates[i,:], color='k', linewidth=0.01)

        # If default plotting
        else:
            fig, ax = plt.subplots(figsize=(15,12))
            for i in range(rates.shape[0]):
                ax.plot(tvec, rates[i,:], color='k', linewidth=0.5)
            
        # Global plotting settings
        ax.set_ylabel("Firing Rate (Hz)")
        ax.set_xlabel("Time (ms)")
        params = self.params
        ax.set_title(f"W{params['psc_q']}, dly{params['syn_delay']}, memcap{params['C_m']}")
        self.latest_mpl_fig = fig, ax
        return fig, ax

    def get_population_firing_rates(self, window=50, deltaT=1, sigma=8, smooth=True, exclude_stim=True):
        if smooth:
            tvec, rates = self.get_smooth_rates(window, deltaT, sigma)
        else:
            tvec, rates = self.get_smooth_rates(window, deltaT)

        if exclude_stim:
            params = self.params
            # Excludes the stimulated neurons
            stimulated_count = len(params["target_bid"])
            rates = rates[stimulated_count:,:]
        population_rate = rates.mean(axis=0)
        return tvec, population_rate
    
    def show_population_firing_rates(self, window=50, deltaT=1, sigma=8, smooth=True, exclude_stim=True, figsize=None):
        tvec, population_rate = self.get_population_firing_rates(window, deltaT, sigma, smooth, exclude_stim)
        if figsize is None:
            plt.figure(figsize=(18,8))
        else:
            plt.figure(figsize=figsize)
        plt.xlim([0, self.params['stop_ms']-self.params['start_ms']])
        plt.plot(tvec, population_rate)
        plt.ylabel('Mean population firing rate (Hz)')
        plt.xlabel('Time (ms)')

    def save_latest_plotly_fig(self, filename):
        if self.latest_plotly_fig is not None:
            fig = self.latest_plotly_fig
            if filename.split('.')[1] == 'png':
                self.latest_plotly_fig.write_image(filename)
            elif filename.split('.')[1] == 'html':
                self.latest_plotly_fig.write_html(filename)
            else:
                raise ValueError("Invalid file format")
            
    def get_pca(self, matrix=None, exclude_stim=True):
        if matrix is None: # If no matrix, get rate matrix by default
            _, matrix = self.get_smooth_rates()
        
        # params
        params = self.params
        # Excludes the stimulated neurons
        stimulated_count = len(params["target_bid"])
        if exclude_stim:
            matrix = matrix[stimulated_count:,:]
        
        # PCA
        pca = PCA(n_components=6)
        pca.fit_transform(matrix[:,:params["stop_ms"]-params["start_ms"]])

        return pca
    
    @staticmethod
    def get_pca_traj(pca_comp, PCX, PCY, theme='light'):
        #assert PCX < PCY
        
        p = pca_comp
        arrow_locations = p[:, :-1]
        arrow_directions = np.diff(p, axis=1)

        # Elapsed time
        sampling_rate = 1 #0.001
        elapsed_time = np.arange(p.shape[1] - 1) * sampling_rate

        # Plot by theme
        if theme == 'dark':
            plt.style.use('dark_background')
        else:
            plt.style.use('default')
        
        fig, ax = plt.subplots(figsize=(12,10))
        q = ax.quiver(
            arrow_locations[PCX-1, :],
            arrow_locations[PCY-1, :],
            arrow_directions[PCX-1, :],
            arrow_directions[PCY-1, :],
            elapsed_time,
            cmap='viridis',
            alpha=0.5,
            scale_units='xy',
            angles='xy',
            scale=1
            )
        ax.set_xlabel(f"PC{PCX} (arb. units)", fontsize=14)
        ax.set_ylabel(f"PC{PCY} (arb. units)", fontsize=14)
        ax.set
        ax.set_title("PCA Trajectory")
        cbar = fig.colorbar(q, ax=ax)
        cbar.set_label("Time (ms)")
        # Set tick locations and labels, for now statically for stim2s and stim10s
        if p.shape[1] == 2000:
            ticks = [0, 0.5, 1.0, 1.5, 2.0]
        else:
            ticks = [0, 2000, 4000, 6000, 8000, 10000]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([str(x) for x in ticks])
        cbar.ax.tick_params(rotation=90)

        if theme == 'dark':
            ax.set_facecolor('black')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
        
        return fig

    @staticmethod
    def get_pca_explained_variance(pca_var_ratio):
        pc_variance_df = pd.DataFrame({"Explained Variance Ratio": pca_var_ratio, "PC":list(range(pca_var_ratio.shape[0]))})
        fig, ax = plt.subplots()
        # Use seaborn for nicer
        sns.barplot(x="PC", y="Explained Variance Ratio", data=pc_variance_df, ax=ax)
        
        return fig, pc_variance_df
    
    def get_activity_summary(self, conndf, by='mainRoi', class_subset=None, mainRoi_subset=None):
        conndf.annotate_top_roi()
        df = conndf.neuron_master
        if class_subset is not None:
            df = df[df['class'] == class_subset]
        if mainRoi_subset is not None:
            df = df[df['mainRoi'] == mainRoi_subset]

        neuron_count_by = df.query(f"bodyId in {list(self.id_bmap.keys())}")[by].value_counts()
        spike_count_by = pd.DataFrame.from_dict(dict(zip(list(self.id_bmap.keys()), self.dense_spike_train.sum(axis=1))), orient='index', columns=['new_col'])
        spike_count_by = spike_count_by.reset_index()
        spike_count_by = spike_count_by.rename(columns={'index': 'bodyId', 'new_col':'spike_count'})
        spike_count_by = pd.merge(df, spike_count_by, on='bodyId', how='left')
        spike_count_by = spike_count_by.query(f"bodyId in {list(self.id_bmap.keys())}").groupby(by)['spike_count'].sum().sort_values(ascending=False).astype(int64)
        main_summary = pd.DataFrame.from_dict({'neuron_count': neuron_count_by.to_dict(), 'spike_count': spike_count_by.to_dict()})
        main_summary.index = neuron_count_by.keys()

        return main_summary
    
    def autocorrelation():
        pass

    def neural_tangling(self):
        # Q (t) = maxt' (abs(Rt - Rt')**2)/(abs(rt - rt))
        pass
    
class SummaryDF():
    def __init__(self, conndf, graph=None):
        self.conndf = conndf
        self.graph = graph
        self.client = self.conndf.client
    
    def get_degrees():
        pass

    def get_compartment_statistics():
        pass

    def MN_summary(self, nerves=['AbN1', 'PDMNp', 'MesoAN', 'LN', 'ProAN', 'VProN', 'DProN']):
        """ Summary table for motor neuron numbers per neuromere, side and value. """
        soma_df = self.conndf.neuron_master.copy()
        # Motor neurons only
        soma_df = soma_df[soma_df['class'] == 'motor neuron']
        # Multilevel index. By T and by Side. Then count motor neurons only.
        soma_df = soma_df.groupby(['somaNeuromere', 'somaSide'])['exitNerve'].value_counts()#apply(lambda x: x[x == 'Motor neuron'])
        # Exclude A neuromeres. and midline
        soma_df = soma_df.to_frame()
        soma_df = soma_df.query('somaNeuromere.str.startswith("T")').query('somaSide != "Midline"')
        soma_df = soma_df.rename(columns={'exitNerve':'Counts'})
        soma_df = soma_df.reset_index(level='exitNerve').pivot(columns='exitNerve', values='Counts')
        soma_df = soma_df.fillna(0)

        # Formatting based on kai table
        tdf = pd.DataFrame() # Empty dataframe
        for n in nerves:
            new_tdf = soma_df.filter(regex=f'{n}').agg(np.sum, axis=1).to_frame(name=f'{n}')
            if not tdf.empty:
                tdf = tdf.join(new_tdf, on=['somaNeuromere', 'somaSide'])
            else:
                tdf = new_tdf

        soma_df = tdf
        soma_df = soma_df.astype(int)
        soma_df = soma_df.replace(0,'')
        soma_df = soma_df.reindex(index = ['RHS','LHS'], level=1)
        return soma_df
    
    def count_summary(self):
        """ The number of neurons, and total connections pairs, and synapses. """
        neuron_count = self.conndf.neuron_master.bodyId.nunique()
        tot_conn = self.conndf.conn_master.groupby(['bodyId_pre', 'bodyId_post'])['weight'].sum().reset_index()
        pair_count = len(tot_conn)
        synapse_count = tot_conn['weight'].sum()
        return neuron_count, pair_count, synapse_count
    
    def class_summary(self):
        """ Summary of class counts, etc."""
        return self.conndf.neuron_master['class'].value_counts()


class GraphDF():
    """ Graph related analysis. Takes in graph data representation of the VNC network,
        or subsets of it. For now, supports only iGraph due to large network.
        Coupled to conndf for now, need to uncouple by making an aesthetics
        class. """
    def __init__(self, graph, profile, conndf):
        self.graph = graph
        self.conndf = conndf
        # iGraph and community vars
        self.profile = profile
        self.vertex_df = self.graph.get_vertex_dataframe()
        self.vertex_df['comm_attr'] = self.profile.membership #annotate nodes by assigned community
        self.unique_communities = self.vertex_df['comm_attr'].unique()
        self.node_list = None
        self.cmap = self.create_aesthetics()['class']

    def create_aesthetics(self):
        """ Instantiates and stores global and aesthetics for consistent plotting. """

        def automate_aesthetics(self, sns_palette, by):
            _n = self.conndf.neuron_master[by].nunique()
            _strs = self.conndf.neuron_master[by].unique()
            _pal = sns.color_palette(sns_palette, _n+1)
            _cval = dict(zip(map(str, _strs), _pal))
            return _cval

        # Transmitter palette: Blue=Acetylcholine, coral=Gaba, Green=glutamate, Grey=undef
        # Hard coded for preference
        transmitter_strings = ['Acetylcholine', 'Gaba', 'Glutamate', 'NaN']
        transmitter_palette = sns.color_palette(['dodgerblue', 'lightcoral', 'green', 'grey'])

        # Categorical palettes
        transmitter_cval = dict(zip(map(str, transmitter_strings), transmitter_palette)) # Used in the end
        class_cval = automate_aesthetics(self, sns_palette='dark', by='class')
        exitnerve_cval = automate_aesthetics(self, sns_palette='colorblind', by='exitNerve')

        return {
            'transmitter': transmitter_cval, 
            'class': class_cval, 
            'exitnerve': exitnerve_cval}

    def annotate_nodes(self, ctype='strength'):
        """ Annotates nodes by graph metrics such as strength (weighted degree)
        or betweenness centrality scores. Used for ranking nodes, and sampling 
        the top N nodes in method sample. Can annotate a given vertex_df by 
        multiple centrality types. 

        Args (currently supported): # should change this to pass methods
            'strength'
            'betweenness'
        """

        _df = None
        for comm in self.unique_communities:
            comm_nodes = list(self.vertex_df.query(f'comm_attr == {comm}').index)
            comm_subgraph = self.graph.subgraph(comm_nodes)

            if ctype == 'strength':
                wc_strength = comm_subgraph.strength(weights='weight')
                add_dict = {'wc_strength': wc_strength}

            elif ctype == 'betweenness':
                wc_betweenness = comm_subgraph.betweenness(weights='weight')
                add_dict = {'wc_betweenness': wc_betweenness}

            temp_df = pd.DataFrame(add_dict, index=comm_nodes)
            if _df is None:
                _df = temp_df
            else:
                _df = pd.concat([_df, temp_df])
        self.vertex_df = self.vertex_df.join(_df)
    
    def sample(self, n, ctype='strength'):
        """ Gets the top n nodes ranked by centrality type. Invokes node
        annotation method, annotate_nodes if not centrality type does not exist
        in the current vertex_df. Used for constructing a subgraph. """
        self.node_list = []

        if f'wc_{ctype}' not in self.vertex_df.columns:
            self.annotate_nodes(ctype=ctype)

        for comm in self.unique_communities:
            comm_nodes = list(
                self.vertex_df.query(
                    f'comm_attr == {comm}'
                    ).sort_values(by=f'wc_{ctype}', ascending=False).head(n).index
                )
            self.node_list += comm_nodes

    def get_manual_partition(self, n, ctype='strength'):
        # check if annotated by ctype
        if f'wc_{ctype}' not in self.vertex_df.columns:
            self.annotate_nodes(ctype=ctype)
        self.sample(n, ctype)

        manual_partition = ig.VertexClustering(
            self.graph.subgraph(sorted(self.node_list)),
            list(self.vertex_df.loc[sorted(self.node_list)]['comm_attr']))
        return manual_partition

    def plot_by_class(
        self,
        manual_partition=None,
        label_nodes=False,
        ctype='strength',
        layout='fruchterman_reingold',
        vertex_size=5,
        edge_width=0.1,
        edge_arrow_size=0.1,
        edge_size=0.1,
        **kwargs):
        """ iGraph plot of subgraph network with matplotlib backend. **kwargs
        passed to ig.plot function. MDS layout by default. Has reasonable 
        ig.plot default vertex and edge parameters. Calls annotate_nodes and
        sample class methods, combining functionality to produce plot.
        """
        
        # Manual partition with VertexClustering object for shading aesthetic
        if manual_partition is None:
            manual_partition = self.get_manual_partition(ctype)

        # Add neuron class colours as node attributes. Colour nodes by neuron class
        self.vertex_df['class_col'] = self.vertex_df['class'].map(self.cmap)
        node_cols = list(self.vertex_df.loc[sorted(self.node_list)]['class_col'])
        if label_nodes:
            labels = manual_partition.graph.vs['_nx_name']
        else:
            labels = None

        # Plot
        return ig.plot(
            manual_partition,
            mark_groups=True, # Shade by community
            layout=layout,
            vertex_size=vertex_size,
            vertex_color=node_cols,
            edge_width=edge_width,
            edge_arrow_size=edge_arrow_size,
            edge_size=edge_size,
            vertex_label=labels,
            hovermode='closest',
            **kwargs
            )