# ARCHITECTURAL_IMMUNITY
NOTE: Since this project is still a work in progress the file organisation is quite chaotic (if you are not me). If you would like to use any of these tools please do reach out and I would be happy to help.

This code is intended for the automated extraction and analysis of 3D spatial networks from 3D skeletons. Used in tandem with a 3D CT-reconstruction, 
converted into a volume thickness map, this allows the automatic assignment of qualities such as nest entrances, chambers, junctions, and tunnel-ends.

The network extraction should run perfectly without the volume and can produce greatly optimised graphs, compared to if you were to use each
branch of a skeleton as an edge.  Example useage of these tools is within the NETWORK_EXTRACTION file. I would recommend using all functions in this order. The parameters used in this file have undergone extensive testing and are optimised for
Lasius niger ant nests. Modifying the parameters can produce quite varying results. 


Here are the main scripts
- ThreeD_Net_Tools is a Python module for the extraction and analysis of 3D spatial networks, providing an extensive range of tools beyond which is conventionally available within NetworkX
- Within G_ANALYSIS, there is a pipeline for the network analysis of extracted spatial networks
- Within GRAPH_RANDOM, there is code for realistically randomising spatial networks. The randomisation maintains the same degree distribution, prevents overlapping edges and ensures that the new network is within the bounded convex hull of the old network.
- ABM_CLUSTER_NETWORK is an agent-based model for analysing disease transmission within a spatial network. Additionally, it is possible to extract the agent social networks from this, allowing an overview for how spatial structure can modify social structure

