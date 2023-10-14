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


# In[2]:





# In[6]:


#USR='LUKE'


# In[7]:


#os.getcwd()
list_paths_skl=[]
list_paths_vol=[]


directory="/user/work/ll16598"

TREATS = pd.read_csv(directory+'/COLONY_INFO.csv')
nurse_k = -201
forag_k =  -201
nurse_ne_k =  -201
forag_ne_k =  -201
density_exp =  -201
array_id = int(sys.argv[6])
iteration = int(sys.argv[7])
timestamp = int(sys.argv[8])

dir_G=directory+"/BASE"
dir_G_widths_o=directory+'/WIDTH_O'

#TREATS = pd.read_csv('/media/cf19810/One Touch/CT_ANALYSIS/COLONY_INFO.csv')
analysis_df=TREATS
name_list=[]
for i in TREATS['name']:
    name_list.append(i)


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
while l<len(files) and len(G_list)<100:

    for file in range(0,len(files)):
        day=os.path.basename(files[file])
        filename = "_".join(day.split("_")[:2])  # Split by underscore, take first two parts, join with underscore
        #print(file)
        if l==len(files):
            break
        if filename == name_list[l]:
            G=nx.read_graphml(files[file])
            Gs = sorted(nx.connected_components(G), key=len, reverse=True)
            Gmax = G.subgraph(Gs[0])
            G_list.append(Gmax)
            print(filename)
            all_names.append(filename)
            l+=1
        else:
            l+=0
            
G_list_width=[]
all_names=[]
l=0
while l<len(files) and len(G_list_width)<100:
    for file in range(0,len(files)):
        day=os.path.basename(files_width[file])
        filename = "_".join(day.split("_")[:2])  # Split by underscore, take first two parts, join with underscore
        #print(file)
        if l==len(files):
            break
        if filename == name_list[l]:
            G=nx.read_graphml(files_width[file])
            Gs = sorted(nx.connected_components(G), key=len, reverse=True)
            Gmax = G.subgraph(Gs[0])
            G_list_width.append(Gmax)
            print(filename)
            all_names.append(filename)
            l+=1
        else:
            l+=0


# Need to make sure t is in right unit (mm_. add a chamber retention time

# structure edges so: ((end1, end2), list_coords, length, width, ants)

# In[16]:


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
forager_number = 180 - nurse_number
rand_num = random.uniform(0, 1)

# scale the random number to the desired range using the mean and standard deviation
num1 = mean + std_dev * math.sqrt(-2 * math.log(rand_num)) * math.cos(2 * math.pi * rand_num)

inoculated_nurse_number= generate_number(mean, std_dev, 0, 20)
inoculated_nurse_number= 0
inoculated_forager_number = 20 - inoculated_nurse_number

transmission_rate=(np.mean(list([0.00066,0.00072, 0.00069])))/2
#spores per time step to be transmitted. Mean of front, back and side contacts, from science
ant_movement_speed=2#cm/s half a body length per second
ants_cm=0.7#ant length
#exponents for how much time foragers and nurses want to spend in chambers or nest_entrances,
#dependent on ant density. Could also sample from different normal distributions,
#which may certainly be more appropriate in foraging area

random_entrance=True
junction_capacity=0.2933291039638014
junction_num=2
#1.105
chamber_capacity=1.5026958128065395
foraging_capacity=95.03#main forage
foraging_capacity2=63.62#petri
density_aversion=0.1
#py=2#not sure what this is @the amount of pheromone that reduces the probabiliy of leaving by half"
sigmoidal=False #if There should be a sigmoid of occupancy in chambers and junctions, 
#with inflexion at the capacity
density_dependence=1 #this impacts the influence of density on transmission, my multiplying it
#can safely set at 0
#transmission_cap=10
B= np.mean(list([0.39,0.36, 0.23])) #(probability of contact 
#should we * by 2?
#(‘front contacts’:β=0.39; ‘front and back contacts’: β=0.36; ‘any overlap’: β=0.23).
epidemic_start=21600
dt=1#timestep
Tmax=64800
sigma=0.00024#transmission threshold
rando_stasis=False


# In[20]:


XMID= 5.564687499999999
YMID= 5.579222039473684
INITIAL_NE=tuple([XMID,YMID,0])


# In[21]:


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
for g in range(0,len(G_list)):
#for g in range(0,len(wed_seq)):
    G=G_list[g]
    G2=G_list_width[g]
    #list_ant=[]
    attributes = nx.get_node_attributes(G, 'TYPE')
    nest_nodes = [node for node, type_value in attributes.items() if type_value == 'NEST EN']
    min_dist=10
    initial_nes=[]
    for ne in nest_nodes:
        x,y,z=ast.literal_eval(G.nodes[ne]['coord'])
        xy=tuple([x,y,0])
        dist=calculate_distance(xy,INITIAL_NE)
        if dist<min_dist:
            initial_ne=ne
            min_dist=dist
    initial_nes.append(initial_ne)
    nest_nodes=list(set(nest_nodes)-set(initial_nes))
    chamber_nodes = [node for node, type_value in attributes.items() if type_value == 'CHAM']
    junction_nodes = [node for node, type_value in attributes.items() if type_value == 'JUNC']
    end_nodes = [node for node, type_value in attributes.items() if type_value == 'END']
    if len(chamber_nodes)==0:
        chamber_nodes=np.nan


#     nest_nodes = [(node, list_ant) for node in nest_nodes]
#     chamber_nodes = [(node, list_ant) for node in chamber_nodes]
#     junction_nodes = [(node, list_ant) for node in junction_nodes]
#     end_nodes = [(node, list_ant) for node in end_nodes]

    all_initial_nes.append(initial_nes)
    all_nest_nodes.append(nest_nodes)
    all_junction_nodes.append(chamber_nodes)
    all_junction_nodes.append(junction_nodes)
    all_end_nodes.append(end_nodes)
    try:
        node_list = chamber_nodes + initial_nes+nest_nodes+ end_nodes+ junction_nodes
    except TypeError:
        node_list = initial_nes+nest_nodes+ end_nodes+ junction_nodes

    all_nodes.append(node_list)
    edge_weights = []
    for u, v, data in G.edges(data=True):
        length = data['weight']
        width = G2.get_edge_data(u, v)['weight']
        
        
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


DIR_PICK_EDGE=directory+'/EDGE_PICKLES_NOCHAM'
DIR_PICK_NODE=directory+'/NODE_PICKLES_NOCHAM'

# In[ ]:


import copy
ALL_ITS_NODE_ANTS=[]
ALL_ITS_EDGE_ANTS=[]

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
for g in range(0,len(WED_MON_Gs)):
    kk=WED_MON_Gs[g]
    chamber_list=all_chamber_nodes[kk]
    if chamber_list is np.nan:
    	chamber_list=[]
   #     continue
    print('SIMULATION ON = ',name_list[kk])
    min_cham_time=Tmax
    ne_list=all_nest_nodes[kk]
    initial_ne_list=all_initial_nes[kk]
    end_list=all_end_nodes[kk]
    junction_list=all_junction_nodes[kk]
    edge_list=all_edges[kk]
    G=G_list[kk]
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
    d1_list=[node for node in node_list if len(list(G.neighbors(node)))==1]
    for ant in original_ants:
        chosen_ne = random.choice(node_list)
        ant.prev_node=chosen_ne
        ant.next_node=random.choice(list(G.neighbors(chosen_ne)))
        node_ants[chosen_ne].append(ant) # add the ant to the list for the chosen node

        #print(ant.spores)
    T=0
    while T<Tmax:
        transmission_chamber=0
        transmission_tunnel=0
        transmission_junction=0
        for node in chamber_list:
            ants=node_ants[node]
            for ant in ants:
                if ant.caste=='Inoculated_Forager' and T<min_cham_time:
                    min_cham_time=T

#             if T in list_t_check:
#                 forage_complete3=False
#                 ants=[ant for ne_node in node_list for \
#                     ant in node_ants[ne_node]] 
#                 eants=[ant for ne_node in edge_list for \
#                     ant in edge_ants[ne_node]] 
#                 antsj=[ant for ne_node in junction_list for \
#                     ant in node_ants[ne_node]] 
#                 print('edge_ants', len(eants))
#                 print('jun_ants', len(antsj))

#                 print('total', len(ants)+len(eants))
#                 print('time=', T)
#                 for node in node_list:

#                     if node in ne_list and forage_complete3==False:
#                         ants=[ant for ne_node in ne_list for \
#                                             ant in node_ants[ne_node]] 
#                         forage_complete3=True
#                         print('foraging=',len(ants))
#                     if node in chamber_list:
#                         print(node)
#                         ants=node_ants[node]
#                         print('chamber ants=', len(ants))
        if T==epidemic_start:
            for ant in inoculateds:
                chosen_ne = random.choice(inodes)
                ant.prev_node=chosen_ne
                ant.next_node=random.choice(list(G.neighbors(chosen_ne)))
                node_ants[chosen_ne].append(ant) # add the ant to the list for the chosen node
        ##FIRST STEP ADVANCING ANTS FROM NODES
        for node in node_list:
            ants=node_ants[node]
            random.shuffle(ants)
            choices=[]
            for ant in ants:
                #print(ant.t)
                if T % t_res ==0 and T>epidemic_start:
                    G_INTERACTIONS_ANT_SPACE.append(tuple([ant.identity, node, ant.caste,T]))
                leaving=False
                if node in chamber_list and ant.t<=T:
                    ants_occupying=len(node_ants[node])
                    pn=ants_occupying#/10 #pheromone

                    try:
                        if ant.caste=='Nurse'or ant.caste=='Inoculated_Nurse':
                            leaving_probability=0.5/((pn)+(nurse_k))
                            #print(ants_occupying, T, leaving_probability)
                        elif ant.caste=='Forager' or ant.caste=='Inoculated_Forager':
                            leaving_probability=0.5/((pn)+(forag_k))
                    except ZeroDivisionError:
                        leaving_probability=0
                    if random.random()<leaving_probability:
                        leaving=True
                       # print(leaving_probability, rando)

                elif node in ne_list and ant.t<=T:
                    ants_occupying=len(node_ants[node])
                    pn=ants_occupying#/10 #pheromone
                    try:
                        if ant.caste=='Nurse'or ant.caste=='Inoculated_Nurse':
                            leaving_probability=0.5/((pn)+(nurse_ne_k))
                        elif ant.caste=='Forager' or ant.caste=='Inoculated_Forager':
                            leaving_probability=0.5/((pn)+(forag_ne_k))
                    except ZeroDivisionError:
                        leaving_probability=0
                    if leaving_probability>random.random():
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
                        leaving_probability=0
                    if leaving_probability>random.random():
                        #print(leaving_probability, 'left_ne')
                        leaving=True
                elif node in junction_list and ant.t<=T:
                    leaving=True

                elif node in d1_list and ant.t<=T:
                    leaving=True
                    apn=ant.prev_node
                    ann=ant.next_node
                    ant.next_node=ant.prev_node
                    ant.prev_node=ann
                    ant_nodes=list([ant.next_node, ant.prev_node])
                    #random_ne_choice
                    if node in ne_list:
                        node=random.choice(ne_list)
                        ant_nodes=list([node])
                    for edge in edge_list:
                        if edge[0][1] in ant_nodes:
                            ant.t+=edge[1]
                            chosen_edge=edge
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
                    #stops going back unless at tunnel end
                    if len(choices)>1 and node not in inodes:
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
                    time=(T+round((edge[1])/ant_movement_speed))
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
            for ant in ants:
                if ant.t<=T:
                    num_ants=len(node_ants[ant.next_node])
                    #check capacity of junction. using width of tunnel leaving from
                   # capacity=(edge[2]*10)+1
                    capacity=junction_num#/ants_cm
                    n=ant.next_node
                    if  num_ants>=capacity and n in junction_list:
#                             if T>1000:
#                                 print(capacity, num_ants)
                        continue
                    #if n in ne_list:
                        #n=random.choice(ne_list)

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
            if node in ne_list or node in initial_ne_list:
                continue
            elif node in junction_list:
                capacity=junction_capacity
            elif node in chamber_list:
                capacity=chamber_capacity

            ants=node_ants[node]
            random.shuffle(ants)
            ants2=ants
            interacted_pairs=[]

            ants_in_node=len(ants)
            if ants_in_node<=1:
                continue
            density=ants_in_node/capacity

            for ant in ants:
                for ant2 in ants2:
                    load_difference=ant.spores-ant2.spores

                    if ant.identity!=ant2.identity  and\
                    random.random() > 1-((1-B)**(density**density_exp)):
                        G_INTERACTIONS.append(tuple([ant.identity, ant2.identity, \
                                                     ant.caste, ant2.caste, T]))
                                                     
                    else:
                        continue
                    if ant.spores<sigma or load_difference<=0:

                            continue
                            
                    else:
                        load_difference=ant.spores-ant2.spores
                        #transmitted=(load_difference/2)-((1-(2*transmission_rate))**dt)*(load_difference/2)
                        transmitted=transmission_rate*load_difference*dt
                        ant2.spores+=transmitted
                        ant.spores-=transmitted


        #edge transmission            
        for edge in edge_list:
            ants=edge_ants[edge]
            ants2=ants
            interacted_pairs=[]

            capacity=edge[1]*edge[2]
            ants_in_edge=len(ants)
            if ants_in_edge<=1:
                continue
            density=ants_in_edge/capacity
            #print(capacity, density)
            #            adjust=math.log(ants_in_contact)
            for ant in ants:
                for ant2 in ants2:
                    load_difference=ant.spores-ant2.spores


                    if ant.identity!=ant2.identity  and\
                    random.random() > 1-((1-B)**(density**density_exp)):
                        G_INTERACTIONS.append(tuple([ant.identity, ant2.identity, \
                                                     ant.caste, ant2.caste, T]))
                    else:
                        continue
                    if ant.spores<sigma or load_difference<=0:
                            continue
                            
                    else:

                        load_difference=ant.spores-ant2.spores
                        #transmitted=(load_difference/2)-((1-(2*transmission_rate))**dt)*(load_difference/2)
                        transmitted=transmission_rate*load_difference*dt
                        ant2.spores+=transmitted
                        ant.spores-=transmitted
#                             if ant.spores<0:
#                                 ant.spores=0
#                             if ant2.spores>100:
#                                 ant2.spores=100
    #                     print(ant.spores, ant2.spores)
#             if T==400:
#                 for key in node_ants.keys():
#                     for ant in node_ants[key]:
#                         print(ant.spores)
        g_transmission_junction.append(transmission_junction)
        g_transmission_chamber.append(transmission_chamber)
        g_transmission_tunnel.append(transmission_tunnel)
        T+=dt  
       
        if T in t_save:
            node_ants2=copy.deepcopy(node_ants)
            edge_ants2=copy.deepcopy(edge_ants)
            G_NODE_ANTS.append(node_ants2)
            G_EDGE_ANTS.append(edge_ants2)



    ALL_NODE_ANTS.append(G_NODE_ANTS)
    ALL_EDGE_ANTS.append(G_EDGE_ANTS)
    ALL_INTERACTIONS.append(G_INTERACTIONS)
    ALL_INTERACTIONS_ANT_SPACE.append(G_INTERACTIONS_ANT_SPACE)
    min_cham_time_list.append(min_cham_time)

    
#         all_transmission_junction.append(g_transmission_junction)
#         all_transmission_chamber.append(g_transmission_chamber)
#         all_transmission_tunnel.append(g_transmission_tunnel)

#         all_transmission_junction.append(g_transmission_junction)
#         all_transmission_chamber.append(g_transmission_chamber)
#         all_transmission_tunnel.append(g_transmission_tunnel)
#indent end here
paras=str([nurse_k,forag_k,nurse_ne_k,forag_ne_k,density_exp])
strits=str(epidemic_start)
strarr=str(Tmax)
stramp=str(timestamp)
OUT_NODE=DIR_PICK_NODE+'/_IT_'+strits+'_'+strarr+'_'+paras+'_'+stramp+'.pickle'
with open(OUT_NODE, "wb") as f:
        pickle.dump(ALL_NODE_ANTS, f)

         
OUT_EDGE=DIR_PICK_EDGE+'/_IT_'+strits+'_'+strarr+'_'+paras+'_'+stramp+'.pickle'
with open(OUT_EDGE, "wb") as f:
        pickle.dump(ALL_EDGE_ANTS, f)     
#     ALL_ITS_NODE_ANTS.append(ALL_NODE_ANTS)
#     ALL_ITS_EDGE_ANTS.append(ALL_EDGE_ANTS)
print('ITERATION COMPLETED = ', stramp)

