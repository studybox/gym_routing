import math
import numpy as np
import networkx as nx
import gym
import h5py
from gym import error, spaces, utils
from gym.utils import seeding
MAX_ARRAY_LENGTH = 100
class ProblemInstance:
    def __init__(self, route, traffic):
        self.route = route
        self.traffic = traffic
    def init(self, graph):
        # initialize the graph with this problem instance


class Problem:
    def __init__(self, filename, val_frac=0.7):
        self.val_frac = val_frac
        print "validation fraciton: {}".format(self.val_frac)

        print "loading data..."
        self._load_data(filename)

    def _load_data(self, filename):
        f = h5py.File(filename, 'r')
        ri = f['graph1/route_info']
        ti = f['graph1/traffic_info']
        gs = f['graph1/graph_structure']
        self.graphs = {"graph1":nx.DiGraph()}
        for link in gs:
            self.graphs["graph1"].add_edge(link[0], link[1])
        self.numsteps = ri.shape[ri.attrs["stepaxis"]-1]
        self.numroutes = ri.shape[ri.attrs["routeaxis"]-1]
        self.numproblems = self.numsteps*self.numroutes

        self.numsteptrain = int(math.floor(self.val_frac*self.numsteps))
        self.numroutetrain = int(math.floor(self.val_frac*self.numroutes))

        self.numstepval = self.numsteps - self.numsteptrain
        self.numrouteval = self.numroutes - self.numroutetrain

        self.numtrain = self.numroutetrain * self.numsteptrain
        self.numval = self.numproblems - self.numtrain

        self.numstudy1 = self.numrouteval * self.numsteptrain #seen step unseen routes
        self.numstudy2 = self.numroutetrain * self.numstepval #unseen step seen routes
        self.numstudy3 = self.numrouteval * self.numstepval # both unseen
        # shuffle data
        stepp = np.random.permutation(self.numsteps)
        routep = np.random.permutation(self.numroutes)

        print "Shift and scale"
        self._shift_scale(ti)

        self.train = []
        self.val = []
        self.study1 = []
        self.study2 = []
        self.study3 = []
        for i, step in enumerate(stepp):
            for j, rid in enumerate(routep):
                if i < self.numsteptrain and j < self.numroutetrain:
                    self.train.append(ProblemInstance(route=ri[step,rid,:]),traffic=self.ti_scaled[:,step,:])
                else:
                    self.val.append(ProblemInstance(route=ri[step,rid,:]),traffic=self.ti_scaled[:,step,:])

        for step in stepp[:self.numsteptrain]:
            for rid in routep[self.numroutetrain:]:
                self.study1.append(ProblemInstance(route=ri[step,rid,:]),traffic=ti_scaled[:,step,:])

        for step in stepp[self.numsteptrain:]:
            for rid in routep[:self.numroutetrain]:
                self.study2.append(ProblemInstance(route=ri[step,rid,:]),traffic=ti_scaled[:,step,:])

        for step in stepp[self.numsteptrain:]:
            for rid in routep[self.numroutetrain:]:
                self.study3.append(ProblemInstance(route=ri[step,rid,:]),traffic=ti_scaled[:,step,:])


    def _shift_scale(self,ti):
        self.shiftfeatures = np.mean(ti, axis=(0,1))
        self.scalefeatures = np.std(ti, axis=(0,1))
        assert(self.shiftfeatures.shape[0]==ti.shape[ti.attrs["featureaxis"]])
        self.ti_scaled = (ti[:,:,:] - self.shiftfeatures)/self.scalefeatures
    def reset(self):
class CsppEnv(gym.Env):
    metadata = {'render.modes': []}

    def __init__(self):
        self.problempool = # load all the graph examples
        self.prunedistance = 1 # the maximum level to jump when pruning
        self.pointer = 1
        self.ppointer = 0
        self.nodelist = np.ndarray((MAX_ARRAY_LENGTH,), dtype=np.int32)
        self.levellist = np.ndarray((MAX_ARRAY_LENGTH,), dtype=np.int32)
        self.r1list = np.ndarray((MAX_ARRAY_LENGTH,), dtype=np.float32)
        self.costlist = np.ndarray((MAX_ARRAY_LENGTH,), dtype=np.float32)
        self.path = np.ndarray((MAX_ARRAY_LENGTH,), dtype=np.int32)
        self.action_space = None
        self.obervation_space = spaces.Box(low=0.0, high=2.0, shape=(self.problempool.num_nodes, 2*self.problempool.num_nodes+4), dtype=np.float32)
        self.viewer = None
    def _draw_obs(self, curnode):
        actionopts = np.zeros((self.problem.num_nodes,), dtype=np.float32)
        for (index, value) in enumerate(self.nodelist[0:self.pointer]):
            if self.levellist[index] >= self.levellist[0:self.pointer][-1]-self.prunedistance:
                # TODO check graph u,v
                actionopts[value] = 1
        nodestate = np.zeros((self.problem.num_nodes,3), dtype=np.float32)
        nodestate[self.problem.dest, 0] = 2.0
        nodestate[:, 1] = self.problem.maxR1
        #TODO change nodestate cost layer
        #for index in range(self.problem.num_nodes):
        #    nodestate[index, 2] = self.problem.mincost[index]/self.problem.mincost[self.problem.dest]
        for (index, value) in enumerate(self.nodelist[0:self.pointer]):
            if value != self.problem.dest:
                nodestate[value, 0] = self.levellist[index]/float(self.problempool.maxlevel)
            else:
                nodestate[value, 0] += self.levellist[index]/float(self.problempool.maxlevel)
            nodestate[value, 1] = self.r1list[index]
            nodestate[value, 2] = self.costlist[index]
        for p in self.path[0:self.ppointer]:
            nodestate[p, 0] = 1.0

        edgestate = self.edgestate
        obs = np.concatenate((nodestate, actionopts[:, np.newaxis], edgestate), axis=1)
        reward = 0.0
        if self.pointer == 0:
            reward = -1.0
        if curnode == self.dest:
            reward = 1.0 + np.exp(-np.abs(self.problem.optimal-self.problem.primalbound))
        #if self.levellist[index] == self.levellist[0:self.pointer][-1]:
        #    reward
        done = False
        if self.pointer == 0 or curnode == self.dest:
            done = True
        info = {}
        return obs, reward, done, info
    def _respond_to_action(self, action):
        assert action in self.nodelist[0:self.pointer], "%r (%s) invalid"%(action, type(action))
        # get the last occurrence, this favors pulse over prune
        index = np.where(self.nodelist[0:self.pointer] == action)[0][-1]
        assert self.levellist[index] >= (self.levellist[0:self.pointer][-1]-self.prunedistance), "prune out of distance"
        self.pointer = index + 1
        return (self.nodelist[self.pointer-1],
                self.levellist[self.pointer-1],
                self.r1list[self.pointer-1],
                self.costlist[self.pointer-1])
    def _pulse(self, curr1, curcost, curlevel, curnode):
        if curnode != self.problem.dest:
            self._change_labels(curr1, curcost, curnode)
            #if self.problem.visited[curnode] == 0:
            assert self.problem.graph.node[curnode]['visited'] == 0 , "Current node is appears on the path, making path a loop"
            self.ppointer += 1
            self.path[self.ppointer-1] = curnode
            self.problem.visited[curnode] = 1
            for (u, v) in self.problem.graph.out_edges(curnode):
                newr1 = curr1 + self.problem.graph[u][v][self.problem.r1]
                newcost = curcost + self.problem.graph[u][v][self.problem.cost]
                newlevel = curlevel + 1
                if (newr1 <= (self.problem.maxR1-self.problem.R1underbar[v])) and
                    (newcost <= (self.problem.primalbound-self.problem.Cunderbar[v])) and
                    (not _check_labels(newr1, newcost, v)) and
                    (self.problem.graph.node[curnode]['visited'] == 0):
                    # pulse
                    self.pointer += 1
                    self.nodelist[self.pointer-1] = v
                    self.levellist[self.pointer-1] = newlevel
                    self.r1list[self.pointer-1] = newr1
                    self.costlist[self.pointer-1] = newcost
                elif (self.problem.graph.node[curnode]['visited'] == 1):
                    # update visited node on loop
                    self._change_labels(newr1, newcost, v)
      else:
            self.ppointer += 1
            self.path[self.ppointer-1] = curnode
            if (curcost <= self.problem.primalbound) and (curr1 <= self.problem.maxR1):
                self.problem.finalpath = self.path[0:self.ppointer]
                self.problem.r1star = curr1
                self.problem.primalbound = curcost
    def _change_labels(self, r1, c, v):
            if (c <= self.problem.graph.nodes[v]['c1']):
                self.problem.graph.nodes[v]['c1'] = c
                self.problem.graph.nodes[v]['r11'] = r1
            elif (r1 <= self.problem.graph.nodes[v]['r12']):
                self.problem.graph.nodes[v]['c2'] = c
                self.problem.graph.nodes[v]['r12'] = r1
            else:
                self.problem.graph.nodes[v]['c3'] = c
                self.problem.graph.nodes[v]['r13'] = r1
    def _check_labels(self, r1, c, v):
            if ((r1>=self.problem.graph.nodes[v]['r11'] and
                 c>=self.problem.graph.nodes[v]['c1']) or
                (r1>=self.problem.graph.nodes[v]['r12'] and
                 c>=self.problem.graph.nodes[v]['c2']) or
                (r1>=self.problem.graph.nodes[v]['r13'] and
                 c>=self.problem.graph.nodes[v]['c3'])):
                return True
            return False
    def step(self, action):
        curnode, curlevel, curr1, curcost = self._respond_to_action(action)
        if self.ppointer >= curlevel:
            for i in range(curlevel-1, self.ppointer):
                self.problem.visited[self.path[i]] = 0
            self.ppointer = curlevel - 1
        self._pulse(curr1, curcost, curlevel, curnode):
        return self._draw_obs(curnode)
    def reset(self):
        self.problem = self.problempool.reset()
        #populate edgestate
        self.edgestate1 = np.zeros((self.problem.num_nodes, self.problem.num_nodes), dtype=np.float32)
        self.edgestate2 = np.zeros((self.problem.num_nodes, self.problem.num_nodes), dtype=np.float32)
        for (u, v) in self.problem.graph.edges:
            self.edgestate1[u, v] = self.problem.graph[u][v][self.problem.r1]
            self.edgestate2[u, v] = self.problem.graph[u][v][self.problem.cost]
        self.edgestate = np.concatenate((self.edgestate1, self.edgestate2), axis=-1)
        #Reset the graph to initial point
        self.pointer = 1
        self.ppointer = 0
        self.nodelist[self.pointer-1] = self.problem.start
        self.levellist[self.pointer-1] = 1
        self.r1list[self.pointer-1] = 0.0
        self.costlist[self.pointer-1] = 0.0
        curnode = self.nodelist[self.pointer-1]
        curlevel = self.levellist[self.pointer-1]
        curr1 = self.r1list[self.pointer-1]
        curcost = self.costlist[self.pointer-1]
        self.pointer -= 1
        self._pulse(curr1, curcost, curlevel, curnode)
        return self._draw_obs(curnode)
    def render(self, mode='human', close=False):
        if mode == 'human':
            screen_width = 600
            screen_height = 400
            world_width = self.problempool.xrange
            super(MyEnv, self).render(mode=mode)
        elif mode == 'rgb_array':
            super(MyEnv, self).render(mode=mode)
        else:
            super(MyEnv, self).render(mode=mode)
