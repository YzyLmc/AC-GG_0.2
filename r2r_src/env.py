''' Batched Room-to-Room navigation environment '''

import sys
sys.path.append('build')
import MatterSim
import csv
import numpy as np
import math
import base64
import utils
import json
import os
import random
import networkx as nx
from param import args

from utils import load_datasets, load_nav_graphs, Tokenizer

csv.field_size_limit(sys.maxsize)


class EnvBatch():
    ''' A simple wrapper for a batch of MatterSim environments, 
        using discretized viewpoints and pretrained features '''

    def __init__(self, feature_store=None, candidate_store=None, batch_size=100):
        """
        1. Load pretrained image feature
        2. Init the Simulator.
        :param feature_store: The name of file stored the feature.
        :param batch_size:  Used to create the simulator list.
        """
        if candidate_store:
            self.candidates = candidate_store

        if feature_store:
            if type(feature_store) is dict:     # A silly way to avoid multiple reading
                self.features = feature_store
                self.image_w = 640
                self.image_h = 480
                self.vfov = 60
                self.feature_size = next(iter(self.features.values())).shape[-1]
                if "detectfeat" in args.features:       # Detection feature will contain the angel feat.
                    self.feature_size -= 4
                print('The feature size is %d' % self.feature_size)
        else:
            print('Image features not provided')
            self.features = None
            self.image_w = 640
            self.image_h = 480
            self.vfov = 60
        self.featurized_scans = set([key.split("_")[0] for key in list(self.features.keys())])
        self.sims = []
        for i in range(batch_size):
            sim = MatterSim.Simulator()
            sim.setRenderingEnabled(False)
            sim.setDiscretizedViewingAngles(True)   # Set increment/decrement to 30 degree. (otherwise by radians)
            sim.setCameraResolution(self.image_w, self.image_h)
            sim.setCameraVFOV(math.radians(self.vfov))
            sim.init()
            self.sims.append(sim)

    def _make_id(self, scanId, viewpointId):
        return scanId + '_' + viewpointId   

    def newEpisodes(self, scanIds, viewpointIds, headings):
        for i, (scanId, viewpointId, heading) in enumerate(zip(scanIds, viewpointIds, headings)):
            # print("New episode %d" % i)
            # sys.stdout.flush()
            self.sims[i].newEpisode(scanId, viewpointId, heading, 0)
  
    def getStates(self):
        """
        Get list of states augmented with precomputed image features. rgb field will be empty.
        Agent's current view [0-35] (set only when viewing angles are discretized)
            [0-11] looking down, [12-23] looking at horizon, [24-35] looking up
        :return: [ ((30, 2048), sim_state) ] * batch_size
        """
        feature_states = []
        for i, sim in enumerate(self.sims):
            # print("Get State %d"%i)
            # sys.stdout.flush()
            state = sim.getState()

            long_id = self._make_id(state.scanId, state.location.viewpointId)
            if self.features:
                feature = self.features[long_id]     # Get feature for
                candidate = self.candidates[long_id]
                feature_states.append((feature, candidate, state))
            else:
                feature_states.append((None, None, state))
        return feature_states

    def makeActions(self, actions):
        ''' Take an action using the full state dependent action interface (with batched input). 
            Every action element should be an (index, heading, elevation) tuple. '''
        for i, (index, heading, elevation) in enumerate(actions):
            self.sims[i].makeAction(index, heading, elevation)

class R2RBatch():
    ''' Implements the Room to Room navigation task, using discretized viewpoints and pretrained features '''

    def __init__(self, feature_store, candidate_store, batch_size=100, seed=10, splits=['train'], tokenizer=None,
                 name=None):
        self.env = EnvBatch(feature_store=feature_store, candidate_store=candidate_store,
                            batch_size=batch_size)
        if feature_store:
            self.feature_size = self.env.feature_size
        self.data = []
        if tokenizer:
            self.tok = tokenizer
        scans = []
        for split in splits:
            for item in load_datasets([split]):
                # Split multiple instructions into separate entries
                for j,instr in enumerate(item['instructions']):
                    if item['scan'] not in self.env.featurized_scans:   # For fast training
                        continue
                    new_item = dict(item)
                    new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
                    new_item['instructions'] = instr
                    if tokenizer:
                        new_item['instr_encoding'] = tokenizer.encode_sentence(instr)
                    if not tokenizer or new_item['instr_encoding'] is not None:  # Filter the wrong data
                        self.data.append(new_item)
                        scans.append(item['scan'])
        if name is None:
            self.name = splits[0] if len(splits) > 0 else "FAKE"
        else:
            self.name = name

        self.scans = set(scans)
        self.splits = splits
        self.seed = seed
        random.seed(self.seed)
        random.shuffle(self.data)

        if args.filter != "":
            filter_name, percent = args.filter.split("_")
            percent = int(percent) / 100
            scan_list = list(self.scans)
            scan_list = sorted(scan_list)
            scan_num = len(scan_list)
            scan_num_in_use = int(scan_num * percent)
            scan_in_use = set(scan_list[:scan_num_in_use])
            data_in_use = [datum for datum in self.data if datum['scan'] in scan_in_use]
            data_num_in_use = len(data_in_use)
            if self.name == 'train':
                if filter_name == 'env':
                    print("With the top %d scans and %d data" % (scan_num_in_use, data_num_in_use))
                    print("With percent %0.4f and %0.4f" % (scan_num_in_use / len(self.scans), data_num_in_use / len(self.data)))
                    print(scan_in_use)
                    self.scans = scan_in_use
                    self.data = data_in_use
                    assert len(self.data) == data_num_in_use
                elif filter_name == 'data':
                    print("With the all %d scans and %d data" % (len(self.scans), data_num_in_use))
                    self.data = self.data[:data_num_in_use]
                    for datum in self.data[:5]:
                        print(datum['instr_id'])
                    assert len(self.data) == data_num_in_use
            # elif self.name == 'aug':
            #     if filter_name == 'env':
            #         print("With the top %d scans and %d data" % (scan_num_in_use, data_num_in_use))
            #         print("With percent %0.4f and %0.4f" % (scan_num_in_use / len(self.scans), data_num_in_use / len(self.data)))
            #         print(scan_in_use)
            #         self.scans = scan_in_use
            #         self.data = data_in_use
            #         assert len(self.data) == data_num_in_use
            #     elif filter_name == 'data':
            #         print("With the all %d scans and %d data" % (len(self.scans), len(self.data)))

        self.ix = 0
        self.batch_size = batch_size
        self._load_nav_graphs()

        self.angle_feature = utils.get_all_point_angle_feature()
        self.sim = utils.new_simulator()
        self.buffered_state_dict = {}

        # It means that the fake data is equals to data in the supervised setup
        self.fake_data = self.data
        print('R2RBatch loaded with %d instructions, using splits: %s' % (len(self.data), ",".join(splits)))

    def size(self):
        return len(self.data)

    def _load_nav_graphs(self):
        """
        load graph from self.scan,
        Store the graph {scan_id: graph} in self.graphs
        Store the shortest path {scan_id: {view_id_x: {view_id_y: [path]} } } in self.paths
        Store the distances in self.distances. (Structure see above)
        Load connectivity graph for each scan, useful for reasoning about shortest paths
        :return: None
        """
        print('Loading navigation graphs for %d scans' % len(self.scans))
        self.graphs = load_nav_graphs(self.scans)
        self.paths = {}
        for scan, G in self.graphs.items(): # compute all shortest paths
            self.paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.distances = {}
        for scan, G in self.graphs.items(): # compute all shortest paths
            self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _next_minibatch(self, tile_one=False, batch_size=None, **kwargs):
        """
        Store the minibach in 'self.batch'
        :param tile_one: Tile the one into batch_size
        :return: None
        """
        if batch_size is None:
            batch_size = self.batch_size
        if tile_one:
            batch = [self.data[self.ix]] * batch_size
            self.ix += 1
            if self.ix >= len(self.data):
                random.shuffle(self.data)
                self.ix -= len(self.data)
        else:
            batch = self.data[self.ix: self.ix+batch_size]
            if len(batch) < batch_size:
                random.shuffle(self.data)
                self.ix = batch_size - len(batch)
                batch += self.data[:self.ix]
            else:
                self.ix += batch_size
        self.batch = batch

    def reset_epoch(self, shuffle=False):
        ''' Reset the data index to beginning of epoch. Primarily for testing. 
            You must still call reset() for a new episode. '''
        if shuffle:
            random.shuffle(self.data)
        self.ix = 0

    def _shortest_path_action(self, state, goalViewpointId):
        ''' Determine next action on the shortest path to goal, for supervised training. '''
        if state.location.viewpointId == goalViewpointId:
            return goalViewpointId      # Just stop here
        path = self.paths[state.scanId][state.location.viewpointId][goalViewpointId]
        nextViewpointId = path[1]
        return nextViewpointId

    def make_candidate(self, feature, candidate, scanId, viewpointId, viewId):
        def _loc_distance(loc):
            return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)
        base_heading = (viewId % 12) * math.radians(30)
        adj_dict = {}
        id2feat = {c['viewpointId']: c['feature'] for c in candidate}
        long_id = "%s_%s" % (scanId, viewpointId)
        if long_id not in self.buffered_state_dict:
            for ix in range(36):
                if ix == 0:
                    self.sim.newEpisode(scanId, viewpointId, 0, math.radians(-30))
                elif ix % 12 == 0:
                    self.sim.makeAction(0, 1.0, 1.0)
                else:
                    self.sim.makeAction(0, 1.0, 0)

                state = self.sim.getState()
                assert state.viewIndex == ix

                # Heading and elevation for the viewpoint center
                heading = state.heading - base_heading
                elevation = state.elevation

                visual_feat = feature[ix]

                # get adjacent locations
                for j, loc in enumerate(state.navigableLocations[1:]):
                    # if a loc is visible from multiple view, use the closest
                    # view (in angular distance) as its representation
                    distance = _loc_distance(loc)

                    # Heading and elevation for for the loc
                    loc_heading = heading + loc.rel_heading
                    loc_elevation = elevation + loc.rel_elevation
                    angle_feat = utils.angle_feature(loc_heading, loc_elevation)
                    if (loc.viewpointId not in adj_dict or
                            distance < adj_dict[loc.viewpointId]['distance']):
                        #if True:
                            #visual_feat = id2feat[loc.viewpointId]
                        adj_dict[loc.viewpointId] = {
                            'heading': loc_heading,
                            'elevation': loc_elevation,
                            "normalized_heading": state.heading + loc.rel_heading,
                            'scanId':scanId,
                            'viewpointId': loc.viewpointId, # Next viewpoint id
                            'pointId': ix,
                            'distance': distance,
                            'idx': j + 1,
                            'feature': np.concatenate((visual_feat, angle_feat), -1)
                        }
            candidate = list(adj_dict.values())
            self.buffered_state_dict[long_id] = [
                {key: c[key]
                 for key in
                ['normalized_heading', 'elevation', 'scanId', 'viewpointId',
                 'pointId', 'idx']}
            for c in candidate]
            return candidate
        else:
            candidate = self.buffered_state_dict[long_id]
            candidate_new = []
            for c in candidate:
                c_new = c.copy()
                ix = c_new['pointId']
                normalized_heading = c_new['normalized_heading']
                visual_feat = feature[ix]
                loc_heading = normalized_heading - base_heading
                c_new['heading'] = loc_heading
                angle_feat = utils.angle_feature(c_new['heading'], c_new['elevation'])
                c_new['feature'] = np.concatenate((visual_feat, angle_feat), -1)
                c_new.pop('normalized_heading')
                candidate_new.append(c_new)
            return candidate_new

    def make_simple_candidate(self, candidate, viewId):
        base_heading = (viewId % 12) * math.radians(30)
        new_candidate = []
        for c in candidate:
            c_new = c.copy()
            heading = c['heading'] - base_heading
            c_new['heading'] = heading
            c_new['feature'] = np.concatenate((c['feature'],
                                               utils.angle_feature(heading, c['elevation']) ) )
            new_candidate.append(c_new)
        return new_candidate

    def _get_obs(self):
        obs = []
        for i, (feature, candidate, state) in enumerate(self.env.getStates()):
            # print("Get obs %d"%i)
            # sys.stdout.flush()
            item = self.batch[i]
            base_view_id = state.viewIndex

            # Full features
            candidate = self.make_candidate(feature, candidate, state.scanId, state.location.viewpointId, state.viewIndex)

            # By using this, only the heading is shifted, the angle_feature is added.
            # candidate = self.make_simple_candidate(candidate, base_view_id)

            # (visual_feature, angel_feature) for views
            feature = np.concatenate((feature, self.angle_feature[base_view_id]), -1)
            obs.append({
                'instr_id' : item['instr_id'],
                'scan' : state.scanId,
                'viewpoint' : state.location.viewpointId,
                'viewIndex' : state.viewIndex,
                'heading' : state.heading,
                'elevation' : state.elevation,
                'feature' : feature,
                'candidate': candidate,
                'navigableLocations' : state.navigableLocations,
                'instructions' : item['instructions'],
                'teacher' : self._shortest_path_action(state, item['path'][-1]),
                'path_id' : item['path_id']
            })
            if 'instr_encoding' in item:
                obs[-1]['instr_encoding'] = item['instr_encoding']
            # A2C reward. The negative distance between the state and the final state
            obs[-1]['distance'] = self.distances[state.scanId][state.location.viewpointId][item['path'][-1]]
        return obs

    def reset(self, batch=None, inject=False, **kwargs):
        ''' Load a new minibatch / episodes. '''
        if batch is None:       # Allow the user to explicitly define the batch
            self._next_minibatch(**kwargs)
        else:
            if inject:          # Inject the batch into the next minibatch
                self._next_minibatch(**kwargs)
                self.batch[:len(batch)] = batch
            else:               # Else set the batch to the current batch
                self.batch = batch
        scanIds = [item['scan'] for item in self.batch]
        viewpointIds = [item['path'][0] for item in self.batch]
        headings = [item['heading'] for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, headings)
        return self._get_obs()

    def step(self, actions):
        ''' Take action (same interface as makeActions) '''
        self.env.makeActions(actions)
        return self._get_obs()

    def get_statistics(self):
        stats = {}
        length = 0
        path = 0
        for datum in self.data:
            length += len(self.tok.split_sentence(datum['instructions']))
            path += self.distances[datum['scan']][datum['path'][0]][datum['path'][-1]]
        stats['length'] = length / len(self.data)
        stats['path'] = path / len(self.data)
        return stats


class SemiBatch(R2RBatch):
    def __init__(self, online=False, load=None, name=None, *args, **kwargs):
        if name is None:
            name = "FAKE"
        super().__init__(*args, name=name, **kwargs)
        if not online:
            if os.path.exists(load):
                self.fake_data = json.load(open(load, 'r'))
            else:
                self.fake_data = self.get_all_data()
                self.filter_out()
                json.dump(self.fake_data, open(load, 'w'), sort_keys=True, indent=4, separators=(',', ': '))
        self.fake_data = [datum for datum in self.fake_data if datum['scan'] in self.env.featurized_scans]

    def filter_out(self):
        """
        Remove the datum appeared in training / validation
        """
        make_id = lambda d: d['scan']+"_"+d['path'][0]+"_"+d['path'][-1]
        remove = set()
        for datum in self.data:     # The data needs to be removed.
            remove.add(make_id(datum))
        old_size = len(self.fake_data)
        self.fake_data = list(filter(lambda d:make_id(d) not in remove, self.fake_data))
        print("The training data %d is removed from the augmentation." % (len(self.fake_data) - old_size))

    def make_item(self, scan, src, trg, heading):
        return {
            'scan': scan,
            'heading': heading,
            'path': [src, trg],
            'path_id': "fake"+src+trg,
            'instr_encoding': [],
            'instr_id': "",
            'instructions': ""
        }

    def validate_path(self, scan, src, trg):
        length = len(self.paths[scan][src][trg]) - 1        # Because the len() include src and trg
        if 3 < length < 7:
            return True
        else:
            return False

    def get_all_data(self):
        data = []
        for scan in self.scans:
            g = self.graphs[scan]
            for src_viewpoint in g:
                for trg_viewpoint in g:         # (src, trg) is an ordered pair
                    if src_viewpoint == trg_viewpoint:
                        continue
                    if self.validate_path(scan, src_viewpoint, trg_viewpoint):
                        data.append(self.make_item(scan,
                                                   src_viewpoint,
                                                   trg_viewpoint,
                                                   random.random() * math.pi * 2)
                                    )
        print("Create %d path for splits" % (len(data)), self.splits)
        return data

    # Override the methods involving self.data
    def _next_minibatch(self, tile_one=False):
        if tile_one:
            batch = [self.fake_data[self.ix]] * self.batch_size
            self.ix += 1
            if self.ix >= len(self.fake_data):
                random.shuffle(self.fake_data)
                self.ix -= len(self.fake_data)
        else:
            batch = self.fake_data[self.ix: self.ix + self.batch_size]
            if len(batch) < self.batch_size:
                random.shuffle(self.fake_data)
                self.ix = self.batch_size - len(batch)
                batch += self.fake_data[:self.ix]
            else:
                self.ix += self.batch_size
        self.batch = batch

    def reset_epoch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.fake_data)
        self.ix = 0

    def size(self):
        return len(self.fake_data)

class ArbiterBatch(R2RBatch):
    def __init__(self, gt_env, gen_env, gt_batch, gen_batch, *args, **kwargs):
        """

        :param gt_env: ground truth environment
        :param gen_env:  generating environment
        :param gt_batch: ground truth batch size
        :param gen_batch: generating batch size
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.gt = gt_env
        self.gen = gen_env
        self.scans = self.gt.scans.union(self.gen.scans)    # Set the correct scans
        self._load_nav_graphs()                             # Load the navigation graph
        self.gt_batch = gt_batch
        # self.gt.batch_size = self.gt_batch
        self.gen_batch = gen_batch
        # self.gen.batch_size = self.gen_batch
        self.batch_size = self.gt_batch + self.gen_batch

    def size(self):
        return self.gt.size() + self.gen.size()

    def reset_epoch(self, shuffle=False):
        self.gt.reset_epoch(shuffle)
        self.gen.reset_epoch(shuffle)

    def _next_minibatch(self, **kwargs):
        self.gt._next_minibatch(batch_size=self.gt_batch, **kwargs)        # Get from two batch
        gt_data = self.gt.batch
        self.gen._next_minibatch(batch_size=self.gen_batch, **kwargs)
        gen_data = self.gen.batch
        gt_data.extend(gen_data)                  # Joint the two batches together.
        self.batch = gt_data

    def _get_obs(self):
        """
        Add the label to the obs
        :return: the obs with correct label
        """
        obs = super()._get_obs()
        for i, ob in enumerate(obs):
            ob['label'] = (i < self.gt_batch)
        return obs

    def get_answer(self):
        path2answer = {}
        for datum in self.gt.data:
            path2answer[datum['instr_id']] = True
        for datum in self.gen.data:
            path2answer[datum['instr_id']] = False
        return path2answer









