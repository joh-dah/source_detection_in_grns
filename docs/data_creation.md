# Data Creation Documentation

The Data Creation consists of two Data Sources. The Data for the underlying Network Structur and the Data for the Signal Propagation.

## Network Structur

Our Networks are based on real world gene regulatory networks (GRN's) aquired from the GRAND Database. The Available Networks are stored in data/cellline_network. The Neworks are preprocess by hand. The respective code can be found in data_exploration/celline_networks. This will likely be adjusted later.

The desired Network can be specified via the parameter --network. E.g ```--network="A549_landmark_t2"

A GRN typically is a directed network, but to make it compatible with the si signal propagation we converted it to undirected.This will also likely be changed later.

## Signal Propagation

The SI propagation model is applied to the generated graph. In the SI model, each node in the graph can be in one of two states: susceptible (S) or infected (I). The infection propagates through the graph based on predefined rules, simulating the spread of an infectious disease.

We will replace this with a more sophisticated approach in the future.

### Data Split

The dataset is split into training and validation sets based on a predefined ratio.

### Data Saving

The dataset, including the graphs and infection status, is saved `data/training/raw` and `data/validation/raw`.
