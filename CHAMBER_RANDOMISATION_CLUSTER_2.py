#!/usr/bin/env python
# coding: utf-8

# In[37]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[6]:


import ast
import sys
import math
import numpy as np #v 1.23.3
import networkx as nx
import argparse
import itertools
import pandas as pd
import os
import scipy
from numpy import roots
import random
import pickle
#os.chdir('/home/ll16598/Documents/ARCHITECTURAL_IMMUNITY/ANALYSIS')

import scipy
from numpy import roots
import random
import pickle

import sys

# Get command-line arguments

def paths (G, list1, list2, traversal_list):
    chamber_nodes=list1
    nest_nodes=list2
    all_short_paths=[]
    all_short_paths_spatial=[]
    all_tot_paths=[]
    all_spatial_paths=[]
    shortest_short_paths=[]
    shortest_short_paths_spatial=[]
    shortest_tot_paths=[]
    shortest_spatial_paths=[]
    all_network_cham_traversed=[]
    all_spatial_cham_traversed=[]
    all_path_ids=[]
    if len(chamber_nodes)>0:
        for chamber in chamber_nodes:
            shortest_paths=[]
            total_paths=[]
            spatial_paths=[]
            shortest_paths_spatial=[]
            p_shortest_paths=[]
            p_total_paths=[]
            p_spatial_paths=[]
            p_shortest_paths_spatial=[]
        
            net_cham_traversed_list=[]
            sptl_cham_traversed_list=[]
            path_ids=[]
            for nest in nest_nodes:
                try:
                    path = nx.shortest_path(G, source=chamber, target=nest)
                    path_ids.append(tuple([chamber, nest]))
                    network_cham_traversed=[]
                    for node in path:
                        if node in traversal_list and node!=chamber:
                            network_cham_traversed.append(node)
                    network_cham_traversed=len(network_cham_traversed)
                    net_cham_traversed_list.append(network_cham_traversed)
                    
                    #path_spatial is the shortest euclidean path
                    path_spatial = nx.dijkstra_path(G, source=nest, target=chamber, weight='weight')
                    
                    sptl_cham_traversed=[]
                    for node in path_spatial:
                        if node in traversal_list and node!=chamber:
                            sptl_cham_traversed.append(node)
                    sptl_cham_traversed=len(sptl_cham_traversed)
                    sptl_cham_traversed_list.append(sptl_cham_traversed)
                    
                    edges_spatial = [(path_spatial[i], path_spatial[i+1]) for i in range(len(path_spatial)-1)]
                    edge_weights_spatial = [G.get_edge_data(path_spatial[i], path_spatial[i+1])['weight'] for i in range(len(path_spatial) - 1)]
                    edge_num_spatial=len(edges_spatial)
                    path_dist_spatial=np.sum(edge_weights_spatial)
                    shortest_paths_spatial.append(edge_num_spatial)
                    spatial_paths.append(path_dist_spatial)#print("Shortest path between {} and {}: {}".format(chamber, nest, path))
                    edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
                    edge_weights = [G.get_edge_data(path[i], path[i+1])['weight'] for i in range(len(path) - 1)]
                    edge_num=len(edges)
                    path_dist=np.sum(edge_weights)
                    shortest_paths.append(edge_num)
                    total_paths.append(path_dist)
                    
                except nx.NetworkXNoPath:
                    print("No path found between {} and {}".format(chamber, nest))
            l=100
            for path in shortest_paths:
                if path<l:
                    l=path
            if l==100:
                l=np.nan
            p_shortest_paths.append(l)
            l=100
            for path in total_paths:
                if path<l:
                    l=path
            if l==100:
                l=np.nan
            p_total_paths.append(l)
            l=100
            for path in spatial_paths:
                if path<l:
                    l=path
            if l==100:
                l=np.nan
            p_spatial_paths.append(l)
            l=100
            for path in shortest_paths_spatial:
                if path<l:
                    l=path
            if l==100:
                l=np.nan
            p_shortest_paths_spatial.append(l)
            
            all_short_paths.extend(shortest_paths)
            all_tot_paths.extend(total_paths)
            all_spatial_paths.extend(spatial_paths)
            all_short_paths_spatial.extend(shortest_paths_spatial)
            shortest_short_paths.extend(p_shortest_paths)
            shortest_short_paths_spatial.extend(p_shortest_paths_spatial)
            shortest_tot_paths.extend(p_total_paths)
            shortest_spatial_paths.extend(p_shortest_paths_spatial)
        
            all_network_cham_traversed.extend(net_cham_traversed_list)
            all_spatial_cham_traversed.extend(sptl_cham_traversed_list)
            all_path_ids.extend(path_ids)

    else:
        all_short_paths=np.nan
        all_tot_paths=np.nan
        all_spatial_paths=np.nan
        all_short_paths_spatial=np.nan
        shortest_short_paths=np.nan
        shortest_short_paths_spatial=np.nan
        shortest_tot_paths=np.nan
        shortest_spatial_paths=np.nan
        
        all_network_cham_traversed=np.nan
        all_sptl_cham_traversed=np.nan
        all_path_ids=np.nan
    return all_short_paths, all_tot_paths, all_spatial_paths, all_short_paths_spatial, shortest_short_paths, shortest_short_paths_spatial, shortest_tot_paths, shortest_spatial_paths, all_network_cham_traversed, all_spatial_cham_traversed, all_path_ids
# In[2]:

def chamber_connection_proportion(G):
    """
    Compute the proportion of possible connections to other chambers 
    in the network which are actually present for each chamber node.
    
    Parameters:
    - G (networkx.Graph): The input graph.
    
    Returns:
    - dict: A dictionary with chamber nodes as keys and their connection proportions as values.
    """
    
    # Get node attributes
    attributes_OG = nx.get_node_attributes(G, 'TYPE')
    
    # Identify all chamber nodes
    chamber_nodes = [node for node, type_value in attributes_OG.items() if type_value == 'CHAM']
    
    # Calculate the proportion of connections for each chamber node
    proportions = {}
    total_chambers = len(chamber_nodes)
    
    for node in chamber_nodes:
        chamber_neighbors = sum(1 for neighbor in G.neighbors(node) if attributes_OG[neighbor] == 'CHAM')
        try:
            proportions[node] = chamber_neighbors / (total_chambers - 1)
        except ZeroDivisionError:
            proportions=np.nan
    
    return proportions



# In[6]:


#USR='LUKE'


# In[7]:


#os.getcwd()
list_paths_skl=[]
list_paths_vol=[]

USR='CLUSTER'
if USR=='CLUSTER':
    directory="/user/work/ll16598"

    TREATS = pd.read_csv(directory+'/COLONY_INFO.csv')
    nurse_k = 0
    forag_k = 0
    nurse_ne_k = 0
    forag_ne_k = 0
    density_exp = float(sys.argv[5])
    array_id = int(sys.argv[6])
    iteration = int(sys.argv[7])
    timestamp = int(sys.argv[8])
    array_id=array_id%4000
    array_id=array_id-1

    dir_G=directory+"/RAND_CHAM_LENGTH"
    dir_G_widths_o=directory+'/RAND_CHAM_WIDTH'
    dir_G_widths=directory+'/WIDTH'
    save_directory=directory+'/ABM_RESULTS_RAND_CHAM'
else:
    directory="/home/ll16598/Documents/ARCHITECTURAL_IMMUNITY/CURRENT_Gs/01-05-23_2/GRAPHS"

    TREATS = pd.read_csv('/home/ll16598/Documents/ARCHITECTURAL_IMMUNITY/COLONY_INFO.csv')
    nurse_k = 10
    forag_k = -10
    nurse_ne_k = -10
    forag_ne_k = 10
    density_exp = 1
    array_id = 1
    iteration = 1
    timestamp = 1

    dir_G=directory+"/RAND_CHAM_LENGTH"
    dir_G_widths_o=directory+'/RAND_CHAM_WIDTH'
    dir_G_widths=directory+'/WIDTHS'
    save_directory=directory+'/ABM_RESULTS_RAND_CHAM'
#TREATS = pd.read_csv('/media/cf19810/One Touch/CT_ANALYSIS/COLONY_INFO.csv')
analysis_df=TREATS
name_list=[]
for i in TREATS['name']:
    name_list.append(i)
nind=array_id%40
wed_seq = [96 - i*5 for i in range(20)]
wed_seq.reverse()
mon_seq = [99 - i*5 for i in range(20)]
mon_seq.reverse()
WED_MON_Gs=list(wed_seq+mon_seq)
name_list_2=[]
for k in WED_MON_Gs:
    name_list_2.append(name_list[k])
name=name_list_2[nind]


junction_capacity1=0.9333236#0.3374093
junction_capacity2=0.9022186#0.2759298
junction_num=2
#1.105
chamber_capacity1=1.568653
chamber_capacity2=1.454409
print('RUNNING ON:', name)
# In[8]:
def process_interactions(interactions_list):
    contacts_dict = {}
    for interaction in interactions_list:
        # Sort the tuple to treat (A, B) the same as (B, A)
        sorted_interaction = tuple(sorted(interaction))
        
        # Update the contacts count
        if sorted_interaction in contacts_dict:
            contacts_dict[sorted_interaction] += 1
        else:
            contacts_dict[sorted_interaction] = 1
            
    return [(edge, weight) for edge, weight in contacts_dict.items()]


# In[ ]:


def convert_edge_format(edges):
    new_edges = []
    for edge in edges:
        new_edges.append(tuple([edge[0][0], edge[0][1], edge[1]]))
    return new_edges
    

files_width = [ f.path for f in os.scandir(dir_G_widths_o)]
files = [ f.path for f in os.scandir(dir_G)]

matching_files = [file for file in files if os.path.basename(file).split("_")[0] == str(array_id)]
matching_files_width = [file for file in files_width if os.path.basename(file).split("_")[0] == str(array_id)]

G_list_w=[]
all_names=[]
l=0
files2 = [ f.path for f in os.scandir(dir_G_widths)]
while len(G_list_w)<100:
    if len(G_list_w)==100:
        break
    for file in range(0,len(files2)):
        day=os.path.basename(files2[file])
        filename = "_".join(day.split("_")[:2])  # Split by underscore, take first two parts, join with underscore
        if l==len(files2):
            break
        if filename == name_list[l]:
            G=nx.read_graphml(files2[file])
            Gs = sorted(nx.connected_components(G), key=len, reverse=True)
            Gmax = G.subgraph(Gs[0])
            G_list_w.append(Gmax)
            all_names.append(filename)
            l+=1
        else:
            l+=0
wed_seq = [96 - i*5 for i in range(20)]
wed_seq.reverse()
mon_seq = [99 - i*5 for i in range(20)]
mon_seq.reverse()
WED_MON_Gs=list(wed_seq+mon_seq)
G_list_w2=[]
for k in WED_MON_Gs:
    G_list_w2.append(G_list_w[k])
if nind in wed_seq:
    chamber_capacity=chamber_capacity1
    junction_capacity=junction_capacity1
else:
    chamber_capacity=chamber_capacity2
    junction_capacity=junction_capacity2
Gex=G_list_w2[nind]
Gsex = sorted(nx.connected_components(Gex), key=len, reverse=True)
Gmaxex = Gex.subgraph(Gsex[0])
Gexp=Gmaxex
# In[9]:


def calculate_distance (node1, node2):
    np_node1=np.array(node1)
    np_node2=np.array(node2)
    squared_dist = np.sum((np_node1-np_node2)**2, axis=0) #this calculates distance between two points
    dist = np.sqrt(squared_dist)
    return dist


# In[13]:


G_list=[]
all_names=[]
l=0
G=nx.read_graphml(matching_files[0])
Gs = sorted(nx.connected_components(G), key=len, reverse=True)
Gmax1 = G.subgraph(Gs[0])
G_list.append(Gmax1)

G_list_width=[]
all_names=[]
G=nx.read_graphml(matching_files_width[0])
Gs = sorted(nx.connected_components(G), key=len, reverse=True)
Gmax = G.subgraph(Gs[0])
G_list_width.append(Gmax)

def invert_edge_weights (G):
    G=G.copy()
    for u, v, data in G.edges(data=True):
        data['weight'] = 1/data['weight']
    return G
def divide_edge_weights(G1, G2):
    """
    Returns a graph where edge weights are obtained by dividing weights in G1 by weights in G2.
    Assumes G1 and G2 have the same structure.
    """
    G = G1.copy()
    for (u, v, data) in G1.edges(data=True):
        G[u][v]['weight'] = data['weight'] / G2[u][v]['weight']
    return G
Gnew=divide_edge_weights(Gmax1, Gmax)
Gnew_inv=invert_edge_weights(Gnew)
Gexp_inv=invert_edge_weights(Gexp)

attributes = nx.get_node_attributes(Gexp, 'TYPE')
ALL_JUNC_NUMS = [node for node, type_value in attributes.items() if type_value == 'JUNC']
ALL_CHAM_NUMS = [node for node, type_value in attributes.items() if type_value == 'CHAM']
if len(ALL_CHAM_NUMS)==0:
    print('NO CHAMBERS', array_id)
    sys.exit()
ALL_NE_NUMS = [node for node, type_value in attributes.items() if type_value == 'NEST EN']
END = [node for node, type_value in attributes.items() if type_value == 'END']
JEC=list(ALL_JUNC_NUMS+ALL_CHAM_NUMS+END)

closeness_centrality_spatial = nx.closeness_centrality(Gexp, distance='weight', wf_improved=True)
total_closeness = sum(closeness_centrality_spatial[node] for node in ALL_CHAM_NUMS if node in closeness_centrality_spatial)
count_nodes = sum(1 for node in ALL_CHAM_NUMS if node in closeness_centrality_spatial)
# Check to avoid division by zero
if count_nodes > 0:
    mean_closeness_centrality = total_closeness / count_nodes
else:
    mean_closeness_centrality=np.nan
flow_centrality = nx.current_flow_betweenness_centrality(Gexp_inv, weight='weight', normalized=False)
total_bc = sum(flow_centrality.values())
flow_centrality = {node: centrality/total_bc for node, centrality in flow_centrality.items()}
# Sum the centrality values of the specific nodes and count the number of nodes
total_centrality = sum(flow_centrality[node] for node in ALL_CHAM_NUMS if node in flow_centrality)
count_nodes = sum(1 for node in ALL_CHAM_NUMS if node in flow_centrality)
# Check to avoid division by zero
if count_nodes > 0:
    mean_flow_centrality = total_centrality / count_nodes
else:
    mean_flow_centrality=np.nan
    
#info_cen = nx.current_flow_closeness_centrality(G2, weight='weight')
#max_centrality = max(info_cen.values())
i#nfo_cen = {node: centrality / max_centrality for node, centrality in info_cen.items()}
    
nech_all_short_paths, nech_all_tot_paths, nech_all_sptl_paths, \
        nech_all_short_paths_spatial, nech_shortest_short_paths, \
        nech_shortest_short_paths_spatial, nech_shortest_tot_paths, nech_shortest_spatial_paths,\
        nech_all_network_cham_traversed, nech_all_spatial_cham_traversed, nech_path_ids=paths(Gexp, ALL_NE_NUMS, ALL_CHAM_NUMS, ALL_CHAM_NUMS)

mean_path=np.mean(nech_all_sptl_paths)
try:    
    chamber_connection_proportion_mean=np.mean(list(chamber_connection_proportion(Gexp).values()))
except AttributeError:
    chamber_connection_proportion_mean=np.nan

#now for new
attributes = nx.get_node_attributes(Gnew, 'TYPE')
ALL_JUNC_NUMS = [node for node, type_value in attributes.items() if type_value == 'JUNC']
ALL_CHAM_NUMS = [node for node, type_value in attributes.items() if type_value == 'CHAM']
ALL_NE_NUMS = [node for node, type_value in attributes.items() if type_value == 'NEST EN']
END = [node for node, type_value in attributes.items() if type_value == 'END']
JEC=list(ALL_JUNC_NUMS+ALL_CHAM_NUMS+END)

closeness_centrality_spatial = nx.closeness_centrality(Gnew, distance='weight', wf_improved=True)
total_closeness = sum(closeness_centrality_spatial[node] for node in ALL_CHAM_NUMS if node in closeness_centrality_spatial)
count_nodes = sum(1 for node in ALL_CHAM_NUMS if node in closeness_centrality_spatial)
# Check to avoid division by zero
if count_nodes > 0:
    mean_closeness_centrality_new = total_closeness / count_nodes
else:
    mean_closeness_centrality_new=np.nan
flow_centrality = nx.current_flow_betweenness_centrality_subset(Gnew_inv, \
                                                         sources=ALL_NE_NUMS,\
                                                         targets=JEC, weight='weight', normalized=False)
total_bc = sum(flow_centrality.values())
flow_centrality = {node: centrality/total_bc for node, centrality in flow_centrality.items()}
# Sum the centrality values of the specific nodes and count the number of nodes
total_centrality = sum(flow_centrality[node] for node in ALL_CHAM_NUMS if node in flow_centrality)
count_nodes = sum(1 for node in ALL_CHAM_NUMS if node in flow_centrality)
# Check to avoid division by zero
if count_nodes > 0:
    mean_flow_centrality_new = total_centrality / count_nodes
else:
    mean_flow_centrality_new=np.nan
    
nech_all_short_paths, nech_all_tot_paths, nech_all_sptl_paths, \
        nech_all_short_paths_spatial, nech_shortest_short_paths, \
        nech_shortest_short_paths_spatial, nech_shortest_tot_paths, nech_shortest_spatial_paths,\
        nech_all_network_cham_traversed, nech_all_spatial_cham_traversed, nech_path_ids=paths(Gnew, ALL_NE_NUMS, ALL_CHAM_NUMS, ALL_CHAM_NUMS)

mean_path_new=np.mean(nech_all_sptl_paths)
try:
    chamber_connection_proportion_mean_new=np.mean(list(chamber_connection_proportion(Gnew).values()))
except AttributeError:
    chamber_connection_proportion_mean_new=np.nan
# structure edges so: ((end1, end2), list_coG_ords, length, width, ants)

# In[16]:



# In[35]:


# In[43]:


#i1 is head index, i2 is tail
class Ant:
    def __init__(self, spores, caste, identity, edge, t, prev_node, next_node):
        #self.orientation=orientation
        self.spores=spores
        self.caste=caste
        self.identity=identity
        self.edge=edge#just basic edge
        self.t=t
        self.prev_node=prev_node
        self.next_node=next_node


# In[17]:


#head_index, tail_index, spores, caste, identity, edge, t
caste=['Nurse', 'Forager', 'Inoculated_Forager', 'Inoculated_Nurse']
orientation=np.nan
A=0#identity
edge=np.nan
prev_node=np.nan
next_node=np.nan
nurse=Ant(0, caste[0], A, edge, 0, prev_node, next_node)
forager=Ant(0, caste[1], A, edge, 0, prev_node, next_node)
inoculated=Ant(1, caste[2], A, edge, 0, prev_node, next_node)


# In[18]:


def create_agents(nurse_number, forager_number, inoculated_forager_number, inoculated_nurse_number):
    nurses=[]
    foragers=[]
    inoculated_foragers=[]
    inoculated_nurses=[]
    A=0
    edge=np.nan
    prev_node=np.nan
    next_node=np.nan
    for a in range(0, nurse_number):
        nurse=Ant(0, caste[0], A, edge, 0, prev_node, next_node)
        nurses.append(nurse)
        A+=1

    for a in range(0, forager_number):
        forager=Ant(0, caste[1], A, edge, 0, prev_node, next_node)
        foragers.append(forager)
        A+=1

    for a in range(0, inoculated_forager_number):
        inoculated=Ant(1, caste[2], A, edge, 0, prev_node, next_node)
        inoculated_foragers.append(inoculated)
        A+=1
        # no inoculated nurses
    for a in range(0, inoculated_nurse_number):
        inoculated=Ant(1, caste[2], A, edge, 0, prev_node, next_node)
        inoculated_nurses.append(inoculated)
        A+=1
        #we re
    return nurses, foragers, inoculated_foragers, inoculated_nurses


# In[ ]:


def generate_number(mean, std_dev, min_val, max_val):
    while True:
        num = np.random.normal(mean * max_val, std_dev * max_val)
        if min_val <= num <= max_val:
            return int(num)


# In[19]:

#Time to run the simulation
#wait_t=16 #Time for an ant to stop if it bumps into another ant

#mean = 0.4925927  # proportion nurses
#mean = 0.7 # proportion nurses 2
#std_dev = 0.1026826  # specify the standard deviation of the normal distribution
t_res=10
#Time to run the simulation
#wait_t=16 #Time for an ant to stop if it bumps into another ant

#mean = 0.4925927  # proportion nurses
#mean = 0.7 # proportion nurses 2
#std_dev = 0.1026826  # specify the standard deviation of the normal distribution
mean=0.432
std_dev=0.212
# generate the first number from a normal distribution with the specified mean and standard deviation
rand_num = random.uniform(0, 1)

nurse_number = generate_number(mean, std_dev, 0, 180)
nurse_number=180
forager_number = 180 - nurse_number
rand_num = random.uniform(0, 1)

# scale the random number to the desired range using the mean and standard deviation
num1 = mean + std_dev * math.sqrt(-2 * math.log(rand_num)) * math.cos(2 * math.pi * rand_num)

inoculated_nurse_number= generate_number(mean, std_dev, 0, 20)
inoculated_nurse_number= 0
inoculated_forager_number = 20 - inoculated_nurse_number

transmission_rate=(np.mean(list([0.00066,0.00072, 0.00069])))/2
#spores per time step to be transmitted. Mean of front, back and side contacts, from science
ant_movement_speed=0.2#cm/s half a body length per second
ants_cm=0.4
ant_length=0.4


#transmission_cap=10
B= np.mean(list([0.39,0.36, 0.23])) #(probability of contact 
#should we * by 2?
#(‘front contacts’:β=0.39; ‘front and back contacts’: β=0.36; ‘any overlap’: β=0.23).
epidemic_start=21600
dt=1#timestep
Tmax=64800
sigma=0.00024#transmission threshold
rando_stasis=False


# In[45]:


# In[20]:


XMID= 5.564687499999999
YMID= 5.579222039473684
INITIAL_NE=tuple([XMID,YMID,0])


# In[21]:
max_interact=2
max_out_prop=0.3

# In[ ]:
def get_free_ant_encounters(N1, s1, v_bar, Dt, DS):

  I11 = (1/2) * (N1**2) * s1*((math.sqrt(2) * v_bar * Dt) / DS)
  
  return I11
def transmit(n, lst):
    
    max_attempts = 200  # You can adjust this based on your needs
    attempts = 0

    usage_count = {item: 0 for item in lst}

    # Dictionary to keep track of how many times each item has been used
    usage_count = {item.identity: 0 for item in lst}

    # Set to keep track of pairs that have already been formed
    formed_pairs = set()

    pairs = []

    while len(pairs) < n and len(pairs)<len(lst) and attempts < max_attempts:

        # Shuffle the list
        random.shuffle(lst)
        attempts+=1
        if attempts>=max_attempts:
            print('attemptmax')
            return lst
        # Try forming a pair
        ant, ant2 = lst[0], lst[1]

        # Ensure we don't have duplicate items in a pair and form pairs with distinct items
        if ant != ant2:
            # Convert pair to a frozenset so order doesn't matter, e.g., (1, 2) is same as (2, 1)
            current_pair = frozenset([ant.identity, ant2.identity])
            #print(current_pair)
            # Check if pair hasn't been formed before and items haven't been used more than twice
            if current_pair not in formed_pairs and usage_count[ant.identity] < 2 and usage_count[ant2.identity] < 2:
                pairs.append((ant.identity, ant2.identity))
                formed_pairs.add(current_pair)
                usage_count[ant.identity] += 1
                usage_count[ant2.identity] += 1
                #this is where social net would be added
                if random.random()>B:
                    continue
                #print('transmitted')
                if ant.spores>ant2.spores and ant.spores>sigma:
                    load_difference=ant.spores-ant2.spores
                    if load_difference<=0:
                        continue
                    transmitted=transmission_rate*load_difference*dt
                    ant2.spores+=transmitted
                    ant.spores-=transmitted
                elif ant2.spores>ant.spores and ant2.spores>sigma:
                    load_difference=ant2.spores-ant.spores
                    if load_difference<=0:
                        continue
                    transmitted=transmission_rate*load_difference*dt
                    ant.spores+=transmitted
                    ant2.spores-=transmitted
    return lst

wed_seq = [96 - i*5 for i in range(20)]
wed_seq.reverse()
mon_seq = [99 - i*5 for i in range(20)]
mon_seq.reverse()
WED_MON_Gs=list(wed_seq+mon_seq)


# In[23]:


all_nest_nodes = []
all_chamber_nodes = []
all_initial_nes=[]
all_junction_nodes = []
all_end_nodes = []
all_edges=[]
all_nodes=[]
#for g in range(0,len(wed_seq)):
G=G_list[0]
G2=G_list_width[0]
#list_ant=[]
attributes = nx.get_node_attributes(G, 'TYPE')
nest_nodes = [node for node, type_value in attributes.items() if type_value == 'NEST EN']
min_dist=10
nest_nodes2=[]
initial_nes=[]
for ne in nest_nodes:
    x,y,z=ast.literal_eval(G.nodes[ne]['coord'])


    xy=tuple([x,y,0])
    ne2=tuple([ne,xy])
    nest_nodes2.append(ne2)
    dist=calculate_distance(xy,INITIAL_NE)
    if dist<min_dist:
        initial_ne=tuple([ne,xy])
        min_dist=dist
initial_nes.append(initial_ne)
nest_nodes2=list(set(nest_nodes2)-set(initial_nes))
chamber_nodes = [node for node, type_value in attributes.items() if type_value == 'CHAM']
junction_nodes = [node for node, type_value in attributes.items() if type_value == 'JUNC']
end_nodes = [node for node, type_value in attributes.items() if type_value == 'END']
if len(chamber_nodes)==0:
    chamber_nodes=np.nan


all_initial_nes.append(initial_nes)
all_nest_nodes.append(nest_nodes2)
all_chamber_nodes.append(chamber_nodes)
all_junction_nodes.append(junction_nodes)
all_end_nodes.append(end_nodes)
try:
    node_list = chamber_nodes + initial_nes+nest_nodes2+ end_nodes+ junction_nodes
except TypeError:
    node_list = initial_nes+nest_nodes2+ end_nodes+ junction_nodes

all_nodes.append(node_list)
edge_weights = []
for u, v, data in G.edges(data=True):
    length = data['weight']
    width = G2.get_edge_data(u, v)['weight']
    if u in nest_nodes:
        x,y,z=ast.literal_eval(G.nodes[u]['coord'])
        xy=tuple([x,y,0])
        u=tuple([u,xy])
    if v in nest_nodes:
        x,y,z=ast.literal_eval(G.nodes[v]['coord'])
        xy=tuple([x,y,0])
        v=tuple([v,xy])   
    list_coords=[]
    #This is subject to change depending on if cells or real coords are used
    l=0
    length_mm=round(length*10)
    for i in range(0, length_mm):
        list_coords.append(l)
        l+=1
    edge_weights.append((tuple([u, v]), length, width))
    print(length,width)
all_edges.append(edge_weights)



# In[ ]:


# DIR_PICK_EDGE=directory+'/EDGE_PICKLES_RAND_1'
# DIR_PICK_NODE=directory+'/NODE_PICKLES_RAND_1'

# DIR_PICK_GRAPHS=directory+'/OUTPUT_RAND'

# In[ ]:
mega_df=[]
mega_df2=[]
for iterr in range(0,5):

    import copy
    print('ITER', iterr)
    ALL_ITS_NODE_ANTS=[]
    ALL_ITS_EDGE_ANTS=[]
    max_outside=180*max_out_prop
    min_cham_time_list=[]
    ALL_NODE_ANTS=[]
    ALL_EDGE_ANTS=[]
    ALL_NODE_ANTS=[]
    ALL_EDGE_ANTS=[]

    ALL_INTERACTIONS=[]
    ALL_INTERACTIONS_ANT_SPACE=[]
    t_save1=[i for i in range(0,epidemic_start,100)]
    t_save2=[i for i in range(epidemic_start, Tmax+100,100)]
    t_save=t_save1+t_save2
    kk=array_id
    chamber_list=all_chamber_nodes[0]
    if chamber_list is np.nan:
        chamber_list=[]
    #     continue
    print('SIMULATION ON = ',array_id)
    min_cham_time=Tmax
    ne_list=all_nest_nodes[0]
    initial_ne_list=all_initial_nes[0]
    end_list=all_end_nodes[0]
    junction_list=all_junction_nodes[0]
    edge_list=all_edges[0]
    G=G_list[0]
    nurses, foragers, inoculated_foragers, inoculated_nurses=create_agents(nurse_number, forager_number, \
                                                                           inoculated_forager_number, inoculated_nurse_number)
    inoculateds=list(inoculated_foragers+ inoculated_nurses)        
    original_ants=list(nurses+foragers)
    G_NODE_ANTS=[]
    G_EDGE_ANTS=[]
    G_INTERACTIONS=[]
    G_INTERACTIONS_ANT_SPACE=[]
    g_transmission_chamber=[]
    g_transmission_tunnel=[]
    g_transmission_junction=[]
    #SET-UP THE SIMULATION BY ALLOWING ANTS TO SETTLE IN THE NEST AND RUN FOR A TIME
    if isinstance(chamber_list, float) and np.isnan(chamber_list):
        chamber_list = []
    elif not chamber_list:
        chamber_list = []

    node_list = chamber_list + ne_list + end_list + junction_list+initial_ne_list
    nech_list = chamber_list + ne_list+initial_ne_list
    inodes=ne_list+initial_ne_list
    edge_ants={edge: [] for edge in edge_list}
    node_ants = {node: [] for node in node_list} # create an empty list for each node
    d1_list_ne=[node for node in ne_list if len(list(G.neighbors(node[0])))==1]
    d1_list_ine=[node for node in initial_ne_list if len(list(G.neighbors(node[0])))==1]
    d1_list=end_list+d1_list_ne+d1_list_ine
    for ant in original_ants:
        chosen_ne = random.choice(node_list)
        ant.prev_node=chosen_ne
        if chosen_ne in inodes:
            ant.next_node=random.choice(list(G.neighbors(chosen_ne[0])))
        else:
            no=random.choice(list(G.neighbors(chosen_ne)))
            for nod in node_list:
                if nod[0]==no:
                        ant.next_node=nod
                        break
        node_ants[chosen_ne].append(ant) # add the ant to the list for the chosen node

        #print(ant.spores)
    T=0
    while T<Tmax:
        transmission_chamber=0
        transmission_tunnel=0
        transmission_junction=0

#                         print('chamber ants=', len(ants))
        if T==epidemic_start:
            max_outside=200*max_out_prop
            for ant in inoculateds:
                chosen_ne = random.choice(inodes)
                ant.prev_node=chosen_ne
                ant.next_node=random.choice(list(G.neighbors(chosen_ne[0])))
                node_ants[chosen_ne].append(ant) # add the ant to the list for the chosen node
        ##FIRST STEP ADVANCING ANTS FROM NODES
        for node in node_list:
            ants=node_ants[node]
            random.shuffle(ants)
            choices=[]
            for ant in ants:
                #print(ant.t)
                outside_num=len([ant for ne_node in node_list for \
                     ant in node_ants[ne_node]]) 

                if T % t_res ==0 and T>epidemic_start:
                    G_INTERACTIONS_ANT_SPACE.append(tuple([ant.identity, node, ant.caste,T]))
                leaving=False
                
                if node in ne_list  and outside_num>=max_outside and ant.t<T:
                        leaving=True
                elif node in initial_ne_list and  outside_num>=max_outside and ant.t<T:
                        leaving=True
                elif node in chamber_list and ant.t<T:
                    ants_occupying=len(node_ants[node])
                    pn=ants_occupying#/10 #pheromone

                    try:
                        if ant.caste=='Nurse'or ant.caste=='Inoculated_Nurse':
                            leaving_probability=0.5/((pn)+(nurse_k))
                            #print(ants_occupying, T, leaving_probability)
                        elif ant.caste=='Forager' or ant.caste=='Inoculated_Forager':
                            leaving_probability=0.5/((pn)+(forag_k))
                    except ZeroDivisionError:
                        leaving_probability=1
                    if random.random()<leaving_probability or leaving_probability<0:
                        leaving=True
                       # print(leaving_probability, rando)

                elif node in ne_list and ant.t<T:
                    ants_occupying=len(node_ants[node])
                    pn=ants_occupying#/10 #pheromone
                    try:
                        if ant.caste=='Nurse'or ant.caste=='Inoculated_Nurse':
                            leaving_probability=0.5/((pn)+(nurse_ne_k))
                        elif ant.caste=='Forager' or ant.caste=='Inoculated_Forager':
                            leaving_probability=0.5/((pn)+(forag_ne_k))
                    except ZeroDivisionError:
                        leaving_probability=1
                    if leaving_probability>random.random() or leaving_probability<0:
                        #print(leaving_probability, 'left_ne')
                        leaving=True
                elif node in initial_ne_list:
                    ants_occupying=len(node_ants[node])
                    pn=ants_occupying#/10 #pheromone
                    try:
                        if ant.caste=='Nurse' or ant.caste=='Inoculated_Nurse':
                            leaving_probability=0.5/((pn)+(nurse_ne_k))
                        elif ant.caste=='Forager' or ant.caste=='Inoculated_Forager':
                            leaving_probability=0.5/((pn)+(forag_ne_k))
                    except ZeroDivisionError:
                        leaving_probability=1
                    if leaving_probability>random.random() or leaving_probability<0:
                        #print(leaving_probability, 'left_ne')
                        leaving=True
                elif node in junction_list and ant.t<T:
                    leaving=True

                elif node in d1_list and ant.t<T:
                    leaving=True
                    #random_ne_choice
                    #if node in ne_list:
                    #    node=random.choice(ne_list)
                   #     ant_nodes=list([node])
                    for edge in edge_list:
                        if node==edge[0][1]:
                        
                            ant.prev_node=edge[0][1]
                            ant.next_node=edge[0][0]
                            ant.t+=edge[1]/ant_movement_speed
                            chosen_edge=edge
                            break
                        if node==edge[0][0]:
                        
                            ant.prev_node=edge[0][0]
                            ant.next_node=edge[0][1]
                            ant.t+=edge[1]/ant_movement_speed
                            chosen_edge=edge
                            break
                    edge_ants[chosen_edge].append(ant)
                    node_ants[node] = [a for a in node_ants[node] if a.identity != ant.identity]  # remove the ant by ID
                    continue
                    #IF ANT IS in a junction and not waiting
                if leaving==False:
                    #print('staying')
                    continue
                if leaving==True: 
                    choices=[]
                    ant_nodes=list([ant.next_node, ant.prev_node])
                    for edge in edge_list:
                        if edge[0][1] not in ant_nodes and edge[0][0] not in ant_nodes:
                            continue
                        #capacity=(edge[2]*10)+1
                        #25/07
                        capacity=(edge[2]*edge[1])/ants_cm
                        e_ants=edge_ants[edge]
                        if len(e_ants)>=capacity:
                            continue
                        else:
                            choices.append(edge)
                    if len(choices)==0:
                        continue
                    #stops going back if at junction
                    if len(choices)>1 and node not in inodes and node not in chamber_list:
                        choices2=[]
                        for edge in choices:
                            if edge[0][1]==ant.prev_node and edge[0][0]==ant.next_node:
                                continue
                            elif edge[0][0]==ant.prev_node and edge[0][1]==ant.next_node:
                                continue
                            else:
                                choices2.append(edge)
                        choices=choices2
                    chosen_edge=random.choice(choices)
                    ant.prev_node=node
                    if chosen_edge[0][1] == node:
                        ant.next_node = chosen_edge[0][0]
                    else:
                        ant.next_node = chosen_edge[0][1]
#                             if ant.next_node in chamber_nodes:
#                                 print('chamber_habitation')
                    time=(T+((edge[1])/ant_movement_speed))
                    #print(time)
                    #print(ant.t)
                    ant.t=time
                    #print(ant.t)

                    #print(ant.t)
                    edge_ants[chosen_edge].append(ant)
                    #print(chosen_edge, ant)
                    node_ants[node] = [a for a in node_ants[node] if a.identity != ant.identity]  # remove the ant by ID

        for edge in edge_list:
        
            ants=edge_ants[edge]
            outside_num=len([ant for ne_node in node_list for \
                     ant in node_ants[ne_node]]) 
            for ant in ants:
                if ant.t<T:
                    num_ants=len(node_ants[ant.next_node])
                    #check capacity of junction. using width of tunnel leaving from
                   # capacity=(edge[2]*10)+1
                    capacity_j=junction_num#/ants_cm
                    capacity_e=1#/ants_cm
                    n=ant.next_node
                    if  num_ants>=capacity_j and n in junction_list:
                        continue
                    elif  num_ants>=capacity_e and n in end_list:
                        continue
                    elif n in ne_list and outside_num>=max_outside or n in initial_ne_list and outside_num>=max_outside:
                        continue
                    elif n in ne_list:
                    
                        n1=n
                        n=random.choice(ne_list)
                        #same
                        ant.prev_node=n
                        ant.next_node=n
                        dist=calculate_distance(n1[1],n[1])
                        time=(T+(dist/ant_movement_speed))
                        ant.t=time

                    node_ants[n].append(ant)

           #         print(n, 'node entered')
                   #print('ant entering node', n)
#                         if n in chamber_list and T>1000:
#                             print(n, ant)
                    #print(node_ants)
                    edge_ants[edge] = [a for a in edge_ants[edge] if a.identity != ant.identity]  # remove the ant by ID

        #TRANSMISSION 
        #TRANSMISSION
           
        if T<epidemic_start and T in t_save:
            node_ants2=copy.deepcopy(node_ants)
            edge_ants2=copy.deepcopy(edge_ants)
            G_NODE_ANTS.append(node_ants2)
            G_EDGE_ANTS.append(edge_ants2)
        if T<epidemic_start:
        
            T+=dt  
            continue
        for node in node_list:
            if node in ne_list or node in initial_ne_list or node in end_list:
                continue

            elif node in chamber_list:
                capacity=chamber_capacity

            ants=node_ants[node]
            ants_in_node=len(ants)
            n_encounters=get_free_ant_encounters(ants_in_node, ant_length, ant_movement_speed,dt, capacity)
            if node in junction_list:
                n_encounters=1#=junction_capacity#change to 1
            if ants_in_node<=1:
                continue
            #print(T, 'transmission in node')
            ants=transmit(n_encounters, ants)
            node_ants[node]=ants




        #edge transmission            
        for edge in edge_list:
            capacity=edge[2]*edge[1]
            ants=edge_ants[edge]
            ants_in_edge=len(ants)
            if ants_in_edge<=1:
                continue
            n_encounters=get_free_ant_encounters(ants_in_edge, ant_length, ant_movement_speed, dt, capacity)

            if ants_in_node<=1:
                continue
            #print(T, 'transmission in edge')
            ants=transmit(n_encounters, ants)
            edge_ants[edge]=ants
            
        T+=dt  
       
        if T in t_save:
            node_ants2=copy.deepcopy(node_ants)
            edge_ants2=copy.deepcopy(edge_ants)
            G_NODE_ANTS.append(node_ants2)
            G_EDGE_ANTS.append(edge_ants2)



    ALL_NODE_ANTS.append(G_NODE_ANTS)
    ALL_EDGE_ANTS.append(G_EDGE_ANTS)





    #         all_transmission_junction.append(g_transmission_junction)
    #         all_transmission_chamber.append(g_transmission_chamber)
    #         all_transmission_tunnel.append(g_transmission_tunnel)

    #         all_transmission_junction.append(g_transmission_junction)
    #         all_transmission_chamber.append(g_transmission_chamber)
    #         all_transmission_tunnel.append(g_transmission_tunnel)
    #indent end here
   # paras=str([nurse_k,forag_k,nurse_ne_k,forag_ne_k,density_exp])
    strits=str(epidemic_start)
    strarr=str(Tmax)
    stramp=str(timestamp)
    strad=str(array_id)

    #print(OUT_NODE)
    #OUT_INT=DIR_PICK_INT+'/_IT_'+strits+'_'+strarr+'_'+paras+'_'+stramp+'.pickle'
    #with open(OUT_INT, "wb") as f:
    #        pickle.dump(ALL_INTERACTIONS, f)

    #OUT_INT2=DIR_PICK_INT2+'/_IT_'+strits+'_'+strarr+'_'+paras+'_'+stramp+'.pickle'
    #with open(OUT_INT2, "wb") as f:
    #        pickle.dump(ALL_INTERACTIONS_ANT_SPACE, f)       


    #     ALL_ITS_NODE_ANTS.append(ALL_NODE_ANTS)
    #     ALL_ITS_EDGE_ANTS.append(ALL_EDGE_ANTS)
    print('ITERATION COMPLETED = ', stramp)

    gg=[]
    noms=[]


    list_spores_nurse=[]
    list_spores_forager=[]
    list_spores_nurse_sum=[]
    list_spores_forager_sum=[]
    list_nurse_prev=[]
    list_forager_prev=[]
    list_inoculated_cham=[]
    list_t=[]
    list_r=[]
    list_condition=[]
    list_name=[]
    list_day=[]
    iterations=[]
    nurse_ks=[]
    forag_ks=[]
    nurse_ne_ks=[]
    forag_ne_ks=[]
    density_exps=[]
    list_nurse_cham=[]
    list_forage_en=[]
    first_dt_list=[]
    #resolution of every 400 timesteps and every 100 for the 1000 timesteps following inoculation
    t_res=10000 
    downsampling_factor2=1
    downsampling_factor=1

    t_seq1=[i for i in range(0, epidemic_start, 100)]
    print('WARNING: CHECK T INTERVAL IN T_seq=experiment')
    t_seq2=[i for i in range(epidemic_start, Tmax+500, 100)]
    t_seq=t_seq1+t_seq2
    for g in range(0, len(ALL_NODE_ANTS)):
       # print(g)
        attributes = nx.get_node_attributes(G, 'TYPE')
        chamber_nodes = [node for node, type_value in attributes.items() if type_value == 'CHAM']
        nest_nodes = [node for node, type_value in attributes.items() if type_value == 'NEST EN']

        junction_nodes = [node for node, type_value in attributes.items() if type_value == 'JUNC']
        end_nodes = [node for node, type_value in attributes.items() if type_value == 'END']
        node_dict = {'chamber': chamber_nodes, 'nest': nest_nodes, 'junction': junction_nodes, 'end': end_nodes}
        num_chamber=len(chamber_nodes)
        num_nest=len(nest_nodes)
        g_node=ALL_NODE_ANTS[0]
        g_edge=ALL_EDGE_ANTS[0]

        if "WED" in name:
            day='WED'
        else:
            day='MON'
        if 'P' in name:
            condition='PATHOGEN'
        else:
            condition='SHAM'

        #now iterate through timesteps
    #         for dt in range(0, len(g_node)):
    #             if ep_start <= dt < t_res:
    #                 step = downsampling_factor2
    #             else:
    #                 step = downsampling_factor
        step=downsampling_factor
        for d in range(0, len(g_node), step):

            time=t_seq[d]
            #print(dt)
            #print(time)
            t_node=g_node[d]
            t_edge=g_edge[d]
            # Iterating through each key in t_node
            total_forager_spores=[]
            total_nurse_spores=[]
            nurse_cham=0
            inoculated_cham=0
            forage_en=0
            forager_infected=0
            nurse_infected=0
            for key in t_node:
                ant_node = t_node[key]
                for ant in ant_node:
                    if ant.caste=='Forager':
                        total_forager_spores.append(ant.spores)
                        if ant.spores>0:
                            forager_infected+=1
                    if ant.caste=='Nurse':
                        total_nurse_spores.append(ant.spores)
                        if ant.spores>0:
                            nurse_infected+=1
                    if key in chamber_nodes and ant.caste=='Nurse':
                        nurse_cham+=1
                    elif key in chamber_nodes and ant.caste=='Forager':
                        forage_en+=1
                    if key in chamber_nodes and ant.caste=='Inoculated_Forager':
                        inoculated_cham+=1


            for key in t_edge:
                ant_edge = t_edge[key]
                for ant in ant_edge:
                    if ant.caste=='Forager':
                        total_forager_spores.append(ant.spores)
                        if ant.spores>0:
                            forager_infected+=1
                    if ant.caste=='Nurse':
                        total_nurse_spores.append(ant.spores)
                        if ant.spores>0:
                            nurse_infected+=1

            list_t.append(time)
            list_day.append(day)
            list_condition.append(condition)
            list_name.append(name)
            nurse_ks.append(nurse_k)
            forag_ks.append(forag_k)
            list_r.append(array_id)
            nurse_ne_ks.append(nurse_ne_k)
            forag_ne_ks.append(forag_ne_k)
            list_spores_nurse.append(np.mean(total_nurse_spores))
            list_spores_forager.append(np.mean(total_forager_spores))
            list_spores_nurse_sum.append(np.sum(total_nurse_spores))
            list_spores_forager_sum.append(np.sum(total_forager_spores))
            try:
                nur_prev=nurse_infected/len(total_nurse_spores)
            except ZeroDivisionError:
                nur_prev=0
            list_nurse_prev.append(nur_prev)
            try:
                for_prev=forager_infected/len(total_forager_spores)
            except ZeroDivisionError:
                for_prev=0
            list_forager_prev.append(for_prev)
            if len(chamber_nodes)>0:
                try:
                    nurse_cham=nurse_cham/len(total_nurse_spores)
                except ZeroDivisionError:
                    nurse_cham=np.nan
                try:
                    inoculated_cham=inoculated_cham/20
                except ZeroDivisionError:
                    inoculated_cham=np.nan
                try:
                    forage_en=forage_en/len(total_forager_spores)
                except ZeroDivisionError:
                    forage_en=np.nan
            else:
                nurse_cham=np.nan
                inoculated_cham=np.nan
                forage_en=np.nan
            list_nurse_cham.append(nurse_cham)
            list_forage_en.append(forage_en)
            list_inoculated_cham.append(inoculated_cham)
            if dt in range(epidemic_start, t_res):
                step=downsampling_factor2
            else:
                step = downsampling_factor



    print('LENGTH: ',len(list_t))
    data = {
        'dt': list_t,
        'condition': list_condition,
        'name': list_name,
        'day': list_day,
        'spores_nurse': list_spores_nurse,
        'spores_forager': list_spores_forager,
        'spores_nurse_sum': list_spores_nurse_sum,
        'spores_forager_sum': list_spores_forager_sum,
        'nurse_ks': nurse_ks,
        'forag_ks': forag_ks,
        'list_r':list_r,
        'nurse_ne_ks': nurse_ne_ks,
        'forag_ne_ks': forag_ne_ks,
        'list_forager_prev':list_forager_prev,
        'list_nurse_prev':list_nurse_prev,
        'list_nurse_cham':list_nurse_cham,
        'list_forage_en':list_forage_en,
        'list_inoculated_cham':list_inoculated_cham
    }
    df = pd.DataFrame(data)
    mega_df.append(df)
    df=df[df['dt']>epidemic_start]
    av_nurse=np.mean(df['spores_nurse'])
    av_forage=np.mean(df['spores_forager'])
    sum_nurse=np.sum(df['spores_nurse'])
    sum_forage=np.sum(df['spores_forager'])
    condition=list_condition[0]
    data2={
        'av_nurse':av_nurse,
        'av_forage':av_forage,
        'sum_nurse':sum_nurse,
        'sum_forage':sum_forage,
        'final_nurse':list_spores_nurse[len(df['spores_nurse'])-1],
        'final_forage':list_spores_forager[len(df['spores_forager'])-1],
        'condition':condition,
        'name':name, 
        'array_id':array_id,
        'mean_closeness_centrality':mean_closeness_centrality,
        'mean_flow_centrality':mean_flow_centrality,
        'mean_path':mean_path,
        'chamber_connection_proportion_mean':chamber_connection_proportion_mean,
        'mean_closeness_centrality_new':mean_closeness_centrality_new,
        'mean_flow_centrality_new':mean_flow_centrality_new,
        'mean_path_new':mean_path_new,
        'chamber_connection_proportion_mean_new':chamber_connection_proportion_mean_new}
    

#params='_'+str(nurse_k)+'_'+str(forag_k)+'_'+str(density_exp)
    df = pd.DataFrame(data2, index=[0])
    mega_df2.append(df)
mega_df=pd.concat(mega_df)
mega_df = mega_df.reset_index(drop=True)
mega_df2=pd.concat(mega_df2)
mega_df2 = mega_df2.reset_index(drop=True)
mega_df.to_csv(save_directory+'/_FULL_'+strad+'_'+strits+'_'+name+'_'+stramp+'.csv',index=False)
mega_df2.to_csv(save_directory+'/_SUMMARY_'+strad+'_'+strits+'_'+name+'_'+stramp+'.csv',index=False)



# In[ ]:




