# ARCHITECTURAL_IMMUNITY
This code is intended for the automated extraction and analysis of 3D spatial networks from 3D skeletons. Used in tandem with a 3D CT-reconstruction, 
converted into a volume thickness map, this allows the automatic assignment of qualities such as nest entrances, chambers, junctions, and tunnel-ends.

The network extraction should run perfectly without the volume and can produce greatly optimised graphs, compared to if you were to use each
branch of a skeleton as an edge. Without 'reducing' the network there are many small spurious edges and nodes whenever a tunnel bends.
The module ThreeD_Net_Tools provides tools for network reduction. Example useage of these tools is within the NETWORK_EXTRACTION file. 
I would recommend using all functions in this order. The parameters used in this file have undergone extensive testing and are optimised for
Lasius niger ant nests. Modifying the parameters can produce quite varying results. In particular, the prune and reduce parameters, which
control how long an endpoint edge can be and how long any segment can be, are very important. The extraction is very powerful if you have
networks collected over a number of days, since it can link node identities over time, which additionally can support accurate node quality 
assignment.



The G_ANALYSIS provides a number of tools for analysing 3d spatial networks. 
- We can reduce graphs to their 'effective' nodes
- find path lengths between nodes of different attributes, based upon edge number and dijkstra's shortest path
- Simulate an agent moving from a given source node to a target node, with edge choices taken at random

The parameters for assigning node qualities and for needs when analysing 3D spatial networks are likely very case specific.
Please do not hesitate to reach out for help or collaboration: luke.leckie@bristol.ac.uk
