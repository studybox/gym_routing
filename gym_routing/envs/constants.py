import numpy as np
import pandas as pd
data_nodes = pd.read_csv("/media/mSATA/UM/Simulation/test_ann_arbor_nodes.csv")
nn2idx = {}
for i in range(data_nodes['node'].shape[0]):
    nn2idx[data_nodes['node'][i]] = i
def create_map(graph):
    XBOUNDLEFT = np.Inf
    XBOUNDRIGHT = -np.Inf
    YBOUNDTOP = -np.Inf
    YBOUNDBOT = np.Inf
    datamap = {}
    for node in graph.nodes:
        nidx = nn2idx[node]
        datamap[node] = (data_nodes['x'][nidx], data_nodes['y'][nidx])
        if data_nodes['x'][nidx]-50 < XBOUNDLEFT:
            XBOUNDLEFT = data_nodes['x'][nidx]-50
        if data_nodes['x'][nidx]+50 > XBOUNDRIGHT:
            XBOUNDRIGHT = data_nodes['x'][nidx]+50
        if data_nodes['y'][nidx]-500 < YBOUNDBOT:
            YBOUNDBOT = data_nodes['y'][nidx]-500
        if data_nodes['y'][nidx]+500 > YBOUNDTOP:
            YBOUNDTOP = data_nodes['y'][nidx]+500
    world_widthx = XBOUNDRIGHT-XBOUNDLEFT
    world_widthy = YBOUNDTOP-YBOUNDBOT
    return datamap, world_widthx, world_widthy, XBOUNDLEFT, YBOUNDBOT
