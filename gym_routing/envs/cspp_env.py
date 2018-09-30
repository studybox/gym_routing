import math
import numpy as np
import networkx as nx
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from problem import Problem
MAX_ARRAY_LENGTH = 100
DATA = "/media/mSATA/UM/Upper routing simulation/SUMOdata/routingdataset.hdf5"
class CsppEnv(gym.Env):
    metadata = {'render.modes': []}
    def __init__(self):
        self.seed()
        self.problem = Problem(DATA, self.np_random) # load all the graph examples
        self.prunedistance = 1 # the maximum level to jump when pruning
        self.pointer = 1
        self.ppointer = 0
        self.nodelist = []#np.ndarray((MAX_ARRAY_LENGTH,), dtype=np.int32)
        self.edgelist = []
        self.levellist = []#np.ndarray((MAX_ARRAY_LENGTH,), dtype=np.int32)
        self.r1list = []#np.ndarray((MAX_ARRAY_LENGTH,), dtype=np.float32)
        self.costlist = []#np.ndarray((MAX_ARRAY_LENGTH,), dtype=np.float32)
        self.path = []#np.ndarray((MAX_ARRAY_LENGTH,), dtype=np.int32)
        self.numglobals = 2
        self.action_space = None
        #self.obervation_space = spaces.Box(low=0.0, high=2.0, shape=(self.problem.numnodes["graph1"], 2*self.problem.numnodes["graph1"]+4))#, dtype=np.float32)
        self.obervation_space = spaces.Box(low=0.0, high=200.0, shape=(self.problem.numnodes["graph1"]+self.problem.numedges["graph1"]*4+self.numglobals))#, dtype=np.float32)
        self.viewer = None
    def seed(self, seed= None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def _draw_obs(self, curnode):

        #nodestate[:, 1] = self.problem.instance.maxR1
        #TODO change nodestate cost layer
        #for index in range(self.problem.num_nodes):
        #    nodestate[index, 2] = self.problem.mincost[index]/self.problem.mincost[self.problem.dest]
        #for (index, value) in enumerate(self.nodelist):
        #    nindex = self.problem.graphs["graph1"].nodes[value]["Index"]
        #    if value != self.problem.instance.dest:
        #        nodestate[nindex, 0] = self.levellist[index]#/float(self.problem.maxlevel) #TODO change maxlevel
        #    else:
        #        nodestate[nindex, 0] += self.levellist[index]#/float(self.problem.maxlevel)
        #    nodestate[nindex, 1] = self.r1list[index]
        #    nodestate[nindex, 2] = self.costlist[index]
        #for p in self.path:
        #    pindex = self.problem.graphs["graph1"].nodes[p]["Index"]
        #    nodestate[pindex, 0] = 1.0


        obs = np.concatenate((self.nodestate, self.edgestate, self.u), axis=-1)
        reward = 0.0
        if len(self.edgelist) == 0:
            reward = -1.0
        if curnode == self.problem.instance.dest:
            reward = np.exp(np.abs(self.problem.instance.optimal-self.problem.instance.primalbound))
        #if self.levellist[index] == self.levellist[0:self.pointer][-1]:
        #    reward
        done = False
        if len(self.edgelist) == 0 or curnode == self.problem.instance.dest:
            done = True
        info = {}
        return obs, reward, done, info
    def _respond_to_action(self, action):
        #assert action in self.nodelist[0:self.pointer], "%r (%s) invalid"%(action, type(action))
        assert action in self.edgelist, "%r (%s) invalid"%(action, type(action))
        # get the last occurrence, this favors pulse over prune
        #index = np.where(self.nodelist == action)[0][-1]
        actindexlist = []
        for idx, act in enumerate(self.edgelist):
            if act == action:
                actindexlist.append(idx)
        index = actindexlist[-1]
        levelthres = self.levellist[index]
        #assert self.levellist[index] >= (self.levellist[0:self.pointer][-1]-self.prunedistance), "prune out of distance"
        #assert self.levellist[index] >= (self.levellist[-1]-self.prunedistance), "prune out of distance"
        #self.pointer = index + 1
        for idx in range(index, len(self.levellist)):
            if self.levellist[idx] > levelthres:
                self.nodelist = self.nodelist[0:idx]
                self.edgelist = self.edgelist[0:idx]
                self.levellist = self.levellist[0:idx]
                self.r1list = self.r1list[0:idx]
                self.costlist = self.costlist[0:idx]
                break
        self.pointer = len(self.nodelist)-1
        return (self.nodelist.pop(index),
                self.edgelist.pop(index),
                self.levellist.pop(index),
                self.r1list.pop(index),
                self.costlist.pop(index))
    def _pulse(self, curr1, curcost, curlevel, curnode):
        if curnode != self.problem.instance.dest:
            self._change_labels(curr1, curcost, curnode)
            #if self.problem.visited[curnode] == 0:
            assert self.problem.graphs["graph1"].node[curnode]['visited'] == 0 , "Current node is appears on the path, making path a loop"
            self.ppointer += 1
            #self.path[self.ppointer-1] = curnode
            self.path.append(curnode)
            self.problem.graphs["graph1"].node[curnode]['visited'] = 1
            for (u, v) in self.problem.graphs["graph1"].out_edges(curnode):
                newr1 = curr1 + self.problem.graphs["graph1"][u][v]['r1']
                newcost = curcost + self.problem.graphs["graph1"][u][v]['c']
                newlevel = curlevel + 1
                if (newr1 <= (self.problem.instance.maxR1-self.problem.instance.R1underbar[v])) and \
                    (not self._check_labels(newr1, newcost, v)) and \
                    (self.problem.graphs["graph1"].node[v]['visited'] == 0):
                    #(newcost <= (self.problem.instance.primalbound-self.problem.instance.Cunderbar[v])) and \
                    # pulse
                    self.pointer += 1
                    #self.nodelist[self.pointer-1] = v
                    self.nodelist.append(v)
                    self.edgelist.append(self.problem.graphs["graph1"][u][v]['Index'])
                    #self.levellist[self.pointer-1] = newlevel
                    self.levellist.append(newlevel)
                    #self.r1list[self.pointer-1] = newr1
                    self.r1list.append(newr1)
                    #self.costlist[self.pointer-1] = newcost
                    self.costlist.append(newcost)
                    assert(self.pointer == len(self.nodelist))
                elif (self.problem.graphs["graph1"].node[v]['visited'] == 1):
                    # update visited node on loop
                    self._change_labels(newr1, newcost, v)
        else:
            self.ppointer += 1
            #self.path[self.ppointer-1] = curnode
            self.path.append(curnode)
            if (curcost <= self.problem.instance.primalbound) and (curr1 <= self.problem.instance.maxR1):
                #self.problem.instance.finalpath = self.path[0:self.ppointer]
                self.problem.instance.finalpath = self.path

                assert(len(self.path)==self.ppointer)
                self.problem.instance.r1star = curr1
                self.problem.instance.primalbound = curcost
    def _change_labels(self, r1, c, v):
            if (c <= self.problem.graphs["graph1"].nodes[v]['c1']):
                self.problem.graphs["graph1"].nodes[v]['c1'] = c
                self.problem.graphs["graph1"].nodes[v]['r11'] = r1
            elif (r1 <= self.problem.graphs["graph1"].nodes[v]['r12']):
                self.problem.graphs["graph1"].nodes[v]['c2'] = c
                self.problem.graphs["graph1"].nodes[v]['r12'] = r1
            else:
                self.problem.graphs["graph1"].nodes[v]['c3'] = c
                self.problem.graphs["graph1"].nodes[v]['r13'] = r1
    def _check_labels(self, r1, c, v):
            if ((r1>self.problem.graphs["graph1"].nodes[v]['r11'] and \
                 c>self.problem.graphs["graph1"].nodes[v]['c1']) or \
                (r1>self.problem.graphs["graph1"].nodes[v]['r12'] and \
                 c>self.problem.graphs["graph1"].nodes[v]['c2']) or \
                (r1>self.problem.graphs["graph1"].nodes[v]['r13'] and \
                 c>self.problem.graphs["graph1"].nodes[v]['c3'])):
                return True
            return False
    def step(self, action):
        curnode, curedge, curlevel, curr1, curcost = self._respond_to_action(action)
        if self.ppointer >= curlevel:
            for i in range(curlevel-1, self.ppointer):
                self.problem.graphs["graph1"].node[self.path[i]]['visited'] = 0
                self.path.pop()
            self.ppointer = curlevel - 1
            assert(len(self.path) == self.ppointer)
        self._pulse(curr1, curcost, curlevel, curnode)
        self.edgestate3 = np.zeros((1, self.problem.numedges["graph1"]), dtype=np.float32)
        self.edgestate4 = np.zeros((1, self.problem.numedges["graph1"]), dtype=np.float32)
        for idx, edgeidx in enumerate(self.edgelist):
            self.edgestate4[0, edgeidx] = 0.1 * self.levellist[idx]
        for idx in range(len(self.path)-1):
            eidx = self.problem.graphs["graph1"][self.path[idx]][self.path[idx+1]]["Index"]
            self.edgestate3[0, eidx] = 1.0
        self.edgestate = np.concatenate((self.edgestate1, self.edgestate2, self.edgestate3, self.edgestate4), axis=-1)
        return self._draw_obs(curnode)
    def reset(self, **kargs):
        self.pointer = 1
        self.ppointer = 0
        self.nodelist = []
        self.edgelist = []
        self.levellist = []
        self.r1list = []
        self.costlist = []
        self.path = []
        self.problem.reset(**kargs)
        #populate edgestate
        self.edgestate1 = np.zeros((1, self.problem.numedges["graph1"]), dtype=np.float32)
        self.edgestate2 = np.zeros((1, self.problem.numedges["graph1"]), dtype=np.float32)
        self.edgestate3 = np.zeros((1, self.problem.numedges["graph1"]), dtype=np.float32)
        self.edgestate4 = np.zeros((1, self.problem.numedges["graph1"]), dtype=np.float32)


        self.nodestate1 = np.zeros((1, self.problem.numnodes["graph1"]), dtype=np.float32) #destination
        self.outdegrees = np.zeros((1, self.problem.numnodes["graph1"]), dtype=np.int32) #out_degree

        self.outedges = []
        self.outnodes = []
        self.edgev = []
        for node in self.problem.graphs["graph1"].nodes:
            neighbor_nodes = []
            neighbor_edges = []
            for (u, v) in self.problem.graphs["graph1"].out_edges(node):
                neighbor_edges.append(self.problem.graphs["graph1"][u][v]["Index"])
                neighbor_nodes.append(self.problem.graphs["graph1"].nodes[v]["Index"])
            self.outedges.append(neighbor_edges)
            self.outnodes.append(neighbor_nodes)
        for (u, v) in self.problem.graphs["graph1"].edges:
            self.edgev.append(self.problem.graphs["graph1"].nodes[v]["Index"])
        print len(self.outedges)
        assert(len(self.outedges) == self.problem.numnodes["graph1"])
        self.edgev = np.array(self.edgev, dtype=np.int32)



        self.u = np.zeros((1,2))
        self.u[0,0] = self.problem.instance.maxR1
        self.u[0,1] = self.problem.instance.tstep

        startidx = self.problem.graphs["graph1"].nodes[self.problem.instance.start]["Index"]
        destidx = self.problem.graphs["graph1"].nodes[self.problem.instance.dest]["Index"]
        self.nodestate1[0, destidx] = 2.0
        self.nodestate1[0, startidx] = 1.0

        for idx , node in enumerate(self.problem.graphs["graph1"].nodes):
            self.outdegrees[0, idx] = self.problem.graphs["graph1"].out_degree(node)

        #Reset the graph to initial point
        self.pointer = 1
        self.ppointer = 0
        curnode = self.problem.instance.start
        curlevel = 1
        curr1 = 0.0
        curcost = 0.0
        self.pointer -= 1
        self._pulse(curr1, curcost, curlevel, curnode)
        for idx, (u, v) in enumerate(self.problem.graphs["graph1"].edges):
            self.edgestate1[0, idx] = self.problem.graphs["graph1"][u][v]['r1']
            self.edgestate2[0, idx] = self.problem.graphs["graph1"][u][v]['c']
        for idx, edgeidx in enumerate(self.edgelist):
            self.edgestate4[0, edgeidx] = 0.1 * self.levellist[idx]
        self.edgestate = np.concatenate((self.edgestate1, self.edgestate2, self.edgestate3, self.edgestate4), axis=-1)
        self.nodestate = self.nodestate1
        return self._draw_obs(curnode)
    def render(self, mode='human'):

        if self.viewer is None:
            import render
            import constants
            datamap, xx, yy, xl, yb = constants.create_map(self.problem.graphs["graph1"])
            self.xx = xx
            self.yy = yy
            screen_width = 600
            screen_height = 800

            scalex = screen_width/xx
            scaley = screen_height/yy
            self.scalex = scalex
            self.scaley = scaley
            self.viewer = render.Viewer(screen_width, screen_height)
            self.network = {"p":{}, "r":{}}

            for edge in self.problem.graphs["graph1"].edges:

                road = render.Line(((datamap[edge[0]][0]-xl)*scalex, (datamap[edge[0]][1]-yb)*scaley),
                                   ((datamap[edge[1]][0]-xl)*scalex, (datamap[edge[1]][1]-yb)*scaley))
                self.network["r"][self.problem.graphs["graph1"][edge[0]][edge[1]]["Index"]] = road
                self.viewer.add_geom(road)
            for node in self.problem.graphs["graph1"].nodes:
                junction = render.make_circle(2.5,15,(datamap[node][0]-xl)*scalex,(datamap[node][1]-yb)*scaley, True)
                self.network["p"][node] = junction
                self.viewer.add_geom(junction)
        for node in self.problem.graphs["graph1"].nodes:
            self.network["p"][node].set_color(0.1, 0.1, 0.8)
        self.network["p"][self.problem.instance.start].set_color(.8,.6,.4)
        self.network["p"][self.problem.instance.dest].set_color(.5,.8,.5)
        for edge in self.problem.graphs["graph1"].edges:
            eidx = self.problem.graphs["graph1"][edge[0]][edge[1]]["Index"]

            if eidx in self.edgelist:
                self.network["r"][eidx].set_color(0.0,1.0,0.0)
                if (edge[1], edge[0]) in self.problem.graphs["graph1"].edges:
                    eidx2 = self.problem.graphs["graph1"][edge[1]][edge[0]]["Index"]
                    self.network["r"][eidx2].set_color(0.0,1.0,0.0)
            else:
                if (edge[1], edge[0]) in self.problem.graphs["graph1"].edges:
                    eidx2 = self.problem.graphs["graph1"][edge[1]][edge[0]]["Index"]
                if eidx2 not in self.edgelist:
                    self.network["r"][eidx].set_color(0.75,0.75,0.75)
        for idx in range(len(self.path)-1):
            eidx = self.problem.graphs["graph1"][self.path[idx]][self.path[idx+1]]["Index"]
            self.network["r"][eidx].set_color(1.0, 0.0, 0.0)
            if (self.path[idx+1], self.path[idx]) in self.problem.graphs["graph1"].edges:
                eidx2 = self.problem.graphs["graph1"][self.path[idx+1]][self.path[idx]]["Index"]
                self.network["r"][eidx2].set_color(1.0,0.0,0.0)
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
