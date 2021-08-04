#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 12:43:36 2021

@author: ziyi
"""

import numpy as np
import json
import networkx as nx
import math
import torch

# CLS and DTW method copy from: https://github.com/aimagelab/perceive-transform-and-act/
class CLS(object):
    """ Coverage weighted by length score (CLS).
        Link to the original paper:
        https://arxiv.org/abs/1905.12255
    """
    def __init__(self, graph, weight='weight', threshold=3.0):
        """Initializes a CLS object.
        Args:
          graph: networkx graph for the environment.
          weight: networkx edge weight key (str).
          threshold: distance threshold $d_{th}$ (float).
        """
        self.graph = graph
        self.weight = weight
        self.threshold = threshold
        self.distance = dict(
            nx.all_pairs_dijkstra_path_length(
                self.graph, weight=self.weight))

    def __call__(self, prediction, reference):
        """Computes the CLS metric.
        Args:
          prediction: list of nodes (str), path predicted by agent.
          reference: list of nodes (str), the ground truth path.
        Returns:
          the CLS between the prediction and reference path (float).
        """

        def length(nodes):
            lens = []
            for edge in zip(nodes[:-1], nodes[1:]):
                try:
                    lens.append(self.graph.edges[edge].get(self.weight, 1.0))
                except KeyError:
                    pass
            return np.sum(lens)

        coverage = np.mean([
            np.exp(-np.min([  # pylint: disable=g-complex-comprehension
                self.distance[u][v] for v in prediction
            ]) / self.threshold) for u in reference
        ])
        expected = coverage * length(reference)
        score = expected / (expected + np.abs(expected - length(prediction)))
        return coverage * score


class DTW(object):
    """ Dynamic Time Warping (DTW) evaluation metrics. """

    def __init__(self, graph, weight='weight', threshold=3.0):
        """Initializes a DTW object.
        Args:
          graph: networkx graph for the environment.
          weight: networkx edge weight key (str).
          threshold: distance threshold $d_{th}$ (float).
        """
        self.graph = graph
        self.weight = weight
        self.threshold = threshold
        self.distance = dict(
            nx.all_pairs_dijkstra_path_length(self.graph, weight=self.weight))

    def __call__(self, prediction, reference, metric='sdtw'):
        """Computes DTW metrics.
        Args:
          prediction: list of nodes (str), path predicted by agent.
          reference: list of nodes (str), the ground truth path.
          metric: one of ['ndtw', 'sdtw', 'dtw'].
        Returns:
          the DTW between the prediction and reference path (float).
        """
        assert metric in ['ndtw', 'sdtw', 'dtw']

        dtw_matrix = np.inf * np.ones((len(prediction) + 1, len(reference) + 1))
        dtw_matrix[0][0] = 0
        for i in range(1, len(prediction)+1):
            for j in range(1, len(reference)+1):
                best_previous_cost = min(
                    dtw_matrix[i-1][j], dtw_matrix[i][j-1], dtw_matrix[i-1][j-1])
                cost = self.distance[prediction[i-1]][reference[j-1]]
                dtw_matrix[i][j] = cost + best_previous_cost
        dtw = dtw_matrix[len(prediction)][len(reference)]

        if metric == 'dtw':
            return dtw

        ndtw = np.exp(-dtw/(self.threshold * len(reference)))
        if metric == 'ndtw':
            return ndtw

        success = self.distance[prediction[-1]][reference[-1]] <= self.threshold
        return success * ndtw


def load_nav_graphs(scans):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3] - pose2['pose'][3])**2
                + (pose1['pose'][7] - pose2['pose'][7])**2
                + (pose1['pose'][11] - pose2['pose'][11])**2)**0.5

    graphs = {}
    for scan in scans:
        with open('connectivity/%s_connectivity.json' % scan) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i, item in enumerate(data):
                if item['included']:
                    for j, conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3],
                                                                    item['pose'][7], item['pose'][11]])
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(
                                item['image_id'], data[j]['image_id'], weight=distance(item, data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G
    return graphs


def _load_nav_graphs(scans):
    ''' Load connectivity graph for each scan, useful for reasoning about shortest paths '''
    print('Loading navigation graphs for %d scans' % len(scans))
    graphs = load_nav_graphs(scans)
    paths = {}
    for scan, G in graphs.items():  # compute all shortest paths
        paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
    distances = {}
    for scan, G in graphs.items():  # compute all shortest paths
        distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    return distances


if __name__ == '__main__':
    # Load json
    with open('tasks/R2R/data/R2R_train_aug.json') as f:
        data = json.load(f)

    # Load connectiviy graph
    scans = []
    for traj in data:
        if traj['scan'] not in scans:
            scans.append(traj['scan'])

    graphs = load_nav_graphs(scans)
    DTWs = {}
    for scan in scans:
        graph_i = graphs[scan]
        DTWs[scan] = DTW(graph_i)


    for i in range(len(data)):
        scan = data[i]['scan']
        path_gt = data[i]['path']
        viewpoint_st = path_gt[0]
        viewpoint_end = path_gt[-1]
        graph_i = graph[scan]
        all_path = nx.all_simple_paths(graph_i, source=viewpoint_st, target=viewpoint_end)
        for path in all_path:
            dtw_score = DTWs[scan](path, path_gt)
