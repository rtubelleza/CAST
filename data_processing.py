# General processing modules
import numpy as np
import pandas as pd
import scipy
import ast
import random
# Network modules
import networkx as nx
import igraph as ig
# Neuprint python API tools
from neuprint import fetch_adjacencies, NeuronCriteria, \
    fetch_all_rois, fetch_neurons, Client
from neuprint.utils import connection_table_to_matrix

""" Includes main connectome data processing class, ConnDF as main output for 
downstream classes in other .py modules. Also includes graph sampling plotting 
class GraphSampler. """

""" 27/3/2023 Changes
    - Added extract one before and after functions to get pre or post syn
      neurons relative to a given pool. For getting 'layers' and 'immediately'
      surrounding neurons to a given neuron pool of interest.
    
    - Added connection_table_to_matrix from neuprint.utils as a helper method 
      neup_connection_table_to_matrix to suppress pandas FutureWarning. Simply 
      just calls the keywords for the arguments in .pivot usage. 

    - Externalised normalisation function within filter as a standalone function
      normalise.

    21/4/2023 Changes
    - Added a 'meta' function default_data to instantiate ConnDF from default 
      client; for quick query functionality with neuprint-python API.
    - Added an annotate method to annotate neurons in neuron_master by main ROI,
      which is the ROI they have the most connections in (synweight). Uses
      roiInfo column.

    27/4/2023 Changes
    - Modified _prepare_matrix into get_signed_edgelist. Needed as a 'public'
      method to directly extract signed edgelist, as Br2Simulation no longer
      requires an instance of ConnDF, nor the symmatrix (reducing Class
      coupling).

    10/5/2023 Changes
    - Added network randomisation by shuffling post-synaptic partners.

    20/07/2023 Changes
    - Removed default_data method for public release
    - Remove pre and post normalisation methods due to redundancy with 'relative' method.

"""

class ConnDF():
    """ Creates an instance of a connectome analysis class, ConnDF. 
    Args:
        client: neuprint.Client instance; Can be any connectome dataset from neuPrint. 
    """
    def __init__(self, client):
        self.client = client
        self.neuron_master = None #neuron_df
        self.conn_master = None #conn_df
        self.conn_filter = None
        self.conn_symmatrix = None
        self.conn_matrix = None

        self.graph = None #edgelist
        self.graph_attrs = None #attribute list

        self.extract_state = None #How dataframe is extracted
        self.df_source = None #For extract_full with no start source. Used for method layer()
        self.df_dict = None #{x: conn_df subset}
        self.conn_master = None #conn_df
        self.df_layer_dict = None #{[1-N]: layer}
        self.df_comm_dict = None
    
    ## At the moment, setters, no getters. Need to devise a Class coupling/cohesion plan.
    def set_conn_filter(self, df):
        self.conn_filter = df
    
    def set_conn_matrix(self, mat):
        self.conn_matrix = mat
    
    def set_conn_symmatrix(self, mat):
        self.conn_symmatrix = mat
    
    def annotate_top_roi(self):
        self.neuron_master['mainRoi'] = self.neuron_master['roiInfo'].map(self._get_main_roi)

    def _get_main_roi(self, d):
        biggest_weight = 0
        biggest_roi = ""
        for k,v in ast.literal_eval(d).items():
            if v['synweight'] > biggest_weight:
                biggest_weight = v['synweight']
                biggest_roi = k
        return biggest_roi
    
    def extract_full(self, file_path=None):
        DATASET = self.client.dataset.split(':')[0] #vnc, hem
        STORAGE_NAME = f'{DATASET}_traced_adjacencies' #vnc, hemi
        """
        Args:
            file_path(str): 
                Option 1: 'default' retrieves stored output of None
                Option 2: None, retrieves latest traced, non-cropped dfs from database.
                Option 3: Path to file containing .pkled object of neuron df, conn df
        """
        if file_path == 'default': # Extract from pre-existing csv files
            self.neuron_master = pd.read_csv(f'{STORAGE_NAME}/neurons.csv')
            self.conn_master = pd.read_csv(f'{STORAGE_NAME}/roi-connections.csv')

        elif file_path == None: # Extract newest adjacencies. Fetch all Traced, non-cropped neurons.
            criteria = NeuronCriteria(status='Traced', cropped=False, client=self.client)
            self.neuron_master, self.conn_master = fetch_adjacencies(
                criteria,
                criteria,
                include_nonprimary=False,
                export_dir=f'{STORAGE_NAME}',  # store in folder called VNC_traced-adjacencies
                batch_size=200, # Default 200
                client=self.client,
                properties=[
                    'instance', 'type', 'pre', 'post', 'downstream', 'upstream',
                    'size', 'status', 'cropped', 'statusLabel', 'somaRadius',
                    'somaLocation', 'roiInfo', 'ntGlutamateProb', 'predictedNtProb',
                    'somaSide', 'subclass', 'origin', 'user', 'ntAcetylcholineProb',
                    'class', 'group', 'predictedNt', 'subcluster', 'hemilineage',
                    'entryNerve', 'somaNeuromere', 'target', 'description', 'birthtime',
                    'prefix', 'ntGabaProb', 'rootPosition', 'namingUser', 'serial',
                    'ntUnknownProb', 'lastModifiedBy', 'tosomaPosition', 'position',
                    'synonyms', 'rootSide', 'exitNerve', 'inputRois', 'outputRois'
                    ]
                ) # Full metadata property list
                
        else: # Extract from pickled. outdated. try not to use.
            self.neuron_master = pd.read_pickle(file_path[0])
            self.conn_master = pd.read_pickle(file_path[1])

    def update_neurons(self):
        """ Updates self.neuron_master with all available properties. """
        self.neuron_master, _ = fetch_neurons(self.neuron_master)

    def extract_one_before(self, neurons):
        """ Gets the edgelist of all pre-connecting neurons to a given pool of 
        neurons. """
        return self.conn_master.query(f"bodyId_post in {neurons}")

    def extract_one_after(self, neurons):
        """ Gets the edgelist of all neurons that a given pool of neurons
        connects to. """
        return self.conn_master.query(f"bodyId_pre in {neurons}")

    def neup_connection_table_to_matrix(
        self,
        conn_df, 
        group_cols='bodyId', 
        weight_col='weight', 
        sort_by=None, 
        make_square=False):
            """ neuprint.utils method. Modified pivot to explicitly call the
            keywords to suppress the FutureWarning, current neuprint implementation 
            will probably be deprecated in future pandas versions, so call it here.
            """
            if isinstance(group_cols, str):
                group_cols = (f"{group_cols}_pre", f"{group_cols}_post")

            assert len(group_cols) == 2, \
                "Please provide two group_cols (e.g. 'bodyId_pre', 'bodyId_post')"

            assert group_cols[0] in conn_df, \
                f"Column missing: {group_cols[0]}"

            assert group_cols[1] in conn_df, \
                f"Column missing: {group_cols[1]}"

            assert weight_col in conn_df, \
                f"Column missing: {weight_col}"

            col_pre, col_post = group_cols
            dtype = conn_df[weight_col].dtype

            agg_weights_df = conn_df.groupby([col_pre, col_post], sort=False)[weight_col].sum().reset_index()
            matrix = agg_weights_df.pivot(index=col_pre, columns=col_post, values=weight_col)
            matrix = matrix.fillna(0).astype(dtype)

            if sort_by:
                if isinstance(sort_by, str):
                    sort_by = (f"{sort_by}_pre", f"{sort_by}_post")

                assert len(sort_by) == 2, \
                    "Please provide two sort_by column names (e.g. 'type_pre', 'type_post')"

                pre_order = conn_df.sort_values(sort_by[0])[col_pre].unique()
                post_order = conn_df.sort_values(sort_by[1])[col_post].unique()
                matrix = matrix.reindex(index=pre_order, columns=post_order)
            else:
                # No sort: Keep the order as close to the input order as possible.
                pre_order = conn_df[col_pre].unique()
                post_order = conn_df[col_post].unique()
                matrix = matrix.reindex(index=pre_order, columns=post_order)

            if make_square:
                matrix, _ = matrix.align(matrix.T).fillna(0.0).astype(matrix.dtype)
                matrix = matrix.rename_axis('bodyId_pre', axis=0).rename_axis('bodyId_post', axis=1)
                matrix = matrix.loc[sorted(matrix.index), sorted(matrix.columns)]

            return matrix
    
    # Network Graphs
    def df_as_graph(
            self, 
            which='full', 
            make_ei=False, 
            package='igraph', 
            parallel_edges=False):
        """ Uses current stored connectivity matrix and neuron df to construct
        graph objects in a specified package/module. 

        Args:
            package(str): 
                'networkx': NetworkX graph object. Only good for small networks
                'igraph': iGraph graph object. Generally better.

            which(str): Specify which conn_df to convert. Default: 'filtered'.
                'filtered' converts self.conn_filter (filtered to an ROI)
                'full' converts self.conn_master (the full VNC)

            make_ei(bool):
                True = Invoke make_ei to use a signed connection table as input.
                False = Use raw connection table.
        """
        # First, increase number of properties contained in neuron_master df.
        # Use fetch_neurons to retrieve ALL properties.
        if package == 'networkx':
            if which == 'full':
                df = self.conn_master
            elif which == 'filtered':
                df = self.conn_filter

            weight_col = 'weight'
            if make_ei:
                df = self.make_ei(conn_df=df)
                weight_col = 'signed_weight'

            # Set graph attributes using full neuron property list
            attrs = {}
            neurons = self.neuron_master.to_dict('index')
            for k in neurons.keys():
                b_id = neurons[k]['bodyId']
                del neurons[k]['bodyId']
                attrs[b_id] = neurons[k]
            self.graph_attrs = attrs

            # Construct nx.DiGraph
            if type(df) == pd.core.frame.DataFrame:
                if parallel_edges:
                    gtype = nx.MultiDiGraph
                    edge_key='roi'
                else:
                    gtype = nx.DiGraph
                    edge_key='bodyId_pre'

                graph = nx.from_pandas_edgelist(
                    df=df,
                    source='bodyId_post',
                    target='bodyId_pre',
                    edge_key=edge_key,
                    edge_attr=weight_col,
                    create_using=gtype
                )
                self.graph = graph
                nx.set_node_attributes(self.graph, self.graph_attrs)
            
        elif package == 'igraph':
            # # Convert df to tuple
            # if type(df) == pd.core.frame.DataFrame:
            #     graph = ig.Graph.TupleList(
            #         edges=df[['bodyId_pre', 'bodyId_post', weight_col]].values, 
            #         directed=True,
            #         edge_attrs=df['roi'] #pre, post, weight
            #     )
            #     self.graph = graph
            #     # iGraph Annotation
            #     for column in self.neuron_master:
            #         self.graph.vs[column] = self.neuron_master.loc[self.graph.vs['name'], column]
            
            # For weird some reason the above is slower than creating the graph 
            # as networkx, then converting it to igraph?????
            self.df_as_graph(which=which, make_ei=make_ei, package='networkx')
            self.graph = ig.Graph.from_networkx(self.graph)
    
    # Connectivity Dataframe Filtering, Annotation
    def filter(self, roi=None, threshold=None, normalise=None):
        ACCEPTED_ROIS = fetch_all_rois(client=self.client)
        """ Filter connection table to a specific roi(s) and/or to only include edges
        with weights > threshold (if specified).

        Args:
            roi(str): Include only connections within this roi.
            threshold(int): Include only connections with weights >= threshold. If normalise is not None,
                            this applies to normalised weights.
        
        """        
        if roi is not None:
            if type(roi) == str:
                if roi not in ACCEPTED_ROIS:
                    raise ValueError('Invalid ROI.')
                df =  self.conn_master.query(f'roi == "{roi}"')
            
            elif type(roi) == list: #ensure list of valid strings
                for r in roi:
                    if r not in ACCEPTED_ROIS:
                        raise ValueError('List contains an invalid ROI.')
                df =  self.conn_master.query(f'roi in {roi}')
        else:
            df = self.conn_master

        df = df.fillna(0)

        # Weird control structure, improve later maybe
        if threshold is not None:
            if normalise is not None: 
                df = self.normalise(df, by=normalise)
            df = df.query(f'weight >= {threshold}')
        else:
            if normalise is not None: 
                df = self.normalise(df, by=normalise)
        
        self.set_conn_filter(df)

    def normalise(self, df, by='relative'):
        weight_mapping = dict()
        if by == 'relative': # Pct output to postsynaptic neuron. See AnalyseDF(), premotor extraction example for PMN-MNs
            post_neurons = df['bodyId_post'].unique()
            pool = self.neuron_master[self.neuron_master['bodyId'].isin(post_neurons)]
            weight_mapping = dict(zip(pool['bodyId'], pool['upstream']))
            df = df.assign(weight = df['weight']/df['bodyId_post'].map(weight_mapping))
        elif by is None:
            return df
        else:
            raise ValueError
        
        return df

    def make_ei(
        self, 
        conn_df, 
        prob_threshold=0.5, 
        inh_weighting=1, 
        exc_weighting=1
        ):

        """ Makes connection tables weights positive or negative based on 
        neurotransmitter probability. 
        
        Current settings assume GABA and glutamate as inhibitory,
        cholinergic as excitatory.

        Args:
            neuron_df: Neuron master
            conn_df: conn_df to convert. 
            prob_threshold: Threshold to use for classifying neuron as exc/inh.
            exc_weighting: Scaling factor applied to excitatory weights.
            inh_weighting: Scaling factor applied to inhibitory weights.
        
        """
        neuron_df = self.neuron_master # VNC neuron list as reference
        signed_conn_df = conn_df.copy() # Use a copy

        weight_mapping = dict()
        for n in conn_df['bodyId_pre'].unique():
            neuron = neuron_df[neuron_df['bodyId'] == n]

            if neuron['ntUnknownProb'].values[0] > prob_threshold: # If neurotransmitter is probably unknown, take the highest val prob
                next_largest = neuron[['ntGlutamateProb', 'ntGabaProb', 'ntAcetylcholineProb']].idxmax(axis=1).values[0]
                if next_largest in ['ntGlutamateProb', 'ntGabaProb']:
                    weight_mapping[n] = -1*inh_weighting
                else:
                    weight_mapping[n] = 1*exc_weighting
            
            else: # Somewhat confident neurotransmitter probs, ntUnknownProb < threshold
                if neuron['ntAcetylcholineProb'].values[0] > prob_threshold: # Cholinerg = exc.
                    weight_mapping[n] = 1*exc_weighting
                
                else: # 
                    weight_mapping[n] = -1*inh_weighting # Make Glutamatergic and Gabaergic are negative

        signed_conn_df['weight_map'] = signed_conn_df['bodyId_pre'].apply(lambda x: weight_mapping[x])
        signed_conn_df['signed_weight'] = signed_conn_df['weight']*signed_conn_df['weight_map']
        return signed_conn_df

    def get_signed_edgelist(self, which, make_ei, inh_weighting, exc_weighting):
        """ Add signed weights to edgelist using neurotransmitter probabilities.
            Calculated in make_ei. Used for df_as_matrix and df_as_symmatrix. """
        if which == 'filtered':
            if self.conn_filter is None: # If None, no filtered df yet.
                raise AttributeError('No filtered conn df.')
            mat = self.conn_filter

        elif which == 'full':
            if self.conn_master is None:
                raise AttributeError('No conn df master. Import a VNC.')
            mat = self.conn_master
        
        weight_col = 'weight'
        if make_ei:
            weight_col = 'signed_weight'
            mat = self.make_ei(
                conn_df=mat, inh_weighting=inh_weighting, exc_weighting=exc_weighting
                )
        return weight_col, mat

    def randomise(self, edgelist):
        """ Randomises a network (edgelist) by simply shuffling the post-synaptic bodyId list. """        
        shuffled_df = edgelist.copy()
        posts = shuffled_df["bodyId_post"].to_list()
        shuffled_posts = random.sample(posts, len(posts))
        shuffled_df = shuffled_df.assign(bodyId_post = shuffled_posts)
        shuffled_df = shuffled_df.drop("roi", axis=1) # roi information is not preserved here with shuffling so should be removed
        return shuffled_df

    def df_as_matrix(
        self,
        which='filtered',
        make_ei=False,
        inh_weighting=1,
        exc_weighting=1):
        weight_col, mat = self.get_signed_edgelist(which, make_ei, inh_weighting, exc_weighting) #_ is weight_col String
        mat = connection_table_to_matrix(conn_df=mat, weight_col=weight_col)
        self.set_conn_matrix(mat)
        return mat
    
    def df_as_symmatrix(
        self, 
        which='filtered', 
        make_ei=False, 
        inh_weighting=1, 
        exc_weighting=1,
        as_csr=False):

        """ Convert connection table to symmetric/square adjacency matrix.
        Args:
            which(str): Specify which conn_df to convert. Default: 'filtered'.
                'filtered' converts self.conn_filter (filtered to an ROI)
                'full' converts self.conn_master (the full VNC)

            as_csr(bool): 
                True = Return as scipy CSR matrix
                False = Return as pandas DF

            make_ei(bool):
                True = Invoke make_ei to use a signed connection table as input.
                False = Use raw connection table.
        """
        weight_col, mat = self.get_signed_edgelist(which, make_ei, inh_weighting, exc_weighting)

        # Utils.py implementation for make_square
        _mat = mat.groupby(
            ['bodyId_pre', 'bodyId_post'], sort=False)[weight_col].sum().reset_index()
        _mat = _mat.pivot(index='bodyId_pre', columns='bodyId_post', values=weight_col)
        _mat = _mat.fillna(0).astype(mat[weight_col].dtype)
        _mat = _mat.align(_mat.T)[0].fillna(0) 
        _mat = _mat.rename_axis('bodyId_pre', axis=0).rename_axis('bodyId_post', axis=1)
        _mat = _mat.loc[sorted(_mat.index), sorted(_mat.columns)]

        if as_csr:
            _mat = scipy.sparse.csr_matrix(_mat.values)
        self.set_conn_symmatrix(_mat)
        return _mat