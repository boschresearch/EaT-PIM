#!/usr/bin/python3

# Copyright (c) 2022 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# This source code is derived from the PyTorch implementation of the RotatE model 
#  "RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space"
#  (https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding)
# Copyright (c) 2019 Edward-Sun, licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import random
from frozendict import frozendict

from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, negative_sample_size, mode):
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(triples)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        positive_sample = self.triples[idx]

        head, relation, tail = positive_sample

        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation - 1)]
        subsampling_weight = torch.sqrt(1 /
                                        torch.cuda.FloatTensor([subsampling_weight], device=self.device)
                                        if self.device == 'cuda'
                                        else torch.FloatTensor([subsampling_weight], device=self.device)
                                        )

        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size * 2)
            if self.mode == 'head-batch':
                mask = np.in1d(
                    negative_sample,
                    self.true_head[(relation, tail)],
                    assume_unique=True,
                    invert=True
                )
            elif self.mode == 'tail-batch':
                mask = np.in1d(
                    negative_sample,
                    self.true_tail[(head, relation)],
                    assume_unique=True,
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]

        negative_sample = torch.cuda.LongTensor(negative_sample, device=self.device) if self.device == 'cuda' else torch.LongTensor(negative_sample, device=self.device)

        positive_sample = torch.cuda.LongTensor(positive_sample, device=self.device) if self.device == 'cuda' else torch.LongTensor(positive_sample, device=self.device)

        return positive_sample, negative_sample, subsampling_weight, self.mode

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode

    @staticmethod
    def count_frequency(triples, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation - 1) not in count:
                count[(tail, -relation - 1)] = start
            else:
                count[(tail, -relation - 1)] += 1
        return count

    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''

        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))

        return true_head, true_tail


class TestDataset(Dataset):
    def __init__(self, triples, all_true_triples, nentity, nrelation, mode):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]

        if self.mode == 'head-batch':
            tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.triple_set
                   else (-1, head) for rand_head in range(self.nentity)]
            tmp[head] = (0, head)
        elif self.mode == 'tail-batch':
            tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                   else (-1, tail) for rand_tail in range(self.nentity)]
            tmp[tail] = (0, tail)
        else:
            raise ValueError('negative batch mode %s not supported' % self.mode)

        tmp = torch.cuda.LongTensor(tmp, device=self.device) if self.device == 'cuda' else torch.LongTensor(tmp, device=self.device)
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]

        positive_sample = torch.cuda.LongTensor((head, relation, tail), device=self.device) if self.device == 'cuda' else torch.LongTensor((head, relation, tail), device=self.device)

        return positive_sample, negative_sample, filter_bias, self.mode

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode


def count_leaves(input_dict, leafs):
    lcount = 0
    for k, v_list in input_dict.items():
        for v in v_list:
            if isinstance(v, dict) or isinstance(v, frozendict):
                inner_lcount, leafs = count_leaves(v, leafs)
                lcount += inner_lcount
            else:
                lcount += 1
                leafs.add(v)
    return lcount, leafs

class TrainGraphDataset(Dataset):
    def __init__(self, graphs, nentity, nrelation, negative_sample_size, mode, ingredient_ids):
        self.len = len(graphs)
        self.graphs = graphs
        self.leaf_counts = {}
        self.leaf_nodes = {}
        for (gops, _) in graphs:
            leafs = set()
            leaf_count, leafs = count_leaves(gops, leafs)
            self.leaf_counts[gops] = leaf_count
            self.leaf_nodes[gops] = leafs
        # self.triple_set = set(triples)
        self.nentity = nentity
        self.entity_ids = {i for i in range(nentity)}
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(self.graphs)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.graphs)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.ingredient_ids = set(ingredient_ids)

    def get_leaf_counts(self, gops):
        return self.leaf_counts[gops]

    def get_leaf_nodes(self, gops):
        return self.leaf_nodes[gops]

    def convert_to_tensors(self, op_leaf_count, rand_options, input_dict):
        output_dict = dict()
        for k, v_list in input_dict.items():
            id_content = []
            for v in v_list:
                if isinstance(v, dict) or isinstance(v, frozendict):
                    inner_dict = self.convert_to_tensors(op_leaf_count, rand_options, v)
                    id_content.append(inner_dict)
                else:
                    v_list = np.random.choice(rand_options, size=op_leaf_count+1)
                    v_list[0] = v
                    v_tensor = torch.cuda.LongTensor(v_list,
                                                     device=self.device) if self.device == 'cuda' else torch.LongTensor(
                        v_list, device=self.device)
                    id_content.append(v_tensor)
            k_list = [k for _ in range(op_leaf_count + 1)]
            k_tensor = torch.cuda.LongTensor(k_list, device=self.device) if self.device == 'cuda' else torch.LongTensor(
                k_list, device=self.device)
            output_dict[k_tensor] = frozenset(id_content)
        return frozendict(output_dict)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        positive_sample = self.graphs[idx]

        graph_ops, tail = positive_sample

        # removed inverse relations
        subsampling_weight = self.count[graph_ops]
        subsampling_weight = torch.sqrt(1 /
                                        torch.cuda.FloatTensor([subsampling_weight], device=self.device)
                                        if self.device == 'cuda'
                                        else torch.FloatTensor([subsampling_weight], device=self.device)
                                        )
        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size * 2)
            if self.mode == 'head-batch':
                mask = np.in1d(
                    negative_sample,
                    self.true_head[tail],
                    assume_unique=True,
                    invert=True
                )
            elif self.mode == 'tail-batch' or self.mode == 'graph-tail-batch':
                mask = np.in1d(
                    negative_sample,
                    self.true_tail[graph_ops],
                    assume_unique=True,
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample_t = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_sample_t = torch.cuda.LongTensor(negative_sample_t, device=self.device) if self.device == 'cuda'\
            else torch.LongTensor(negative_sample_t, device=self.device)

        op_leaf_count = self.get_leaf_counts(graph_ops)
        op_leafs = self.get_leaf_nodes(graph_ops)
        rand_options = list(self.ingredient_ids-op_leafs)
        positive_sample_ops = self.convert_to_tensors(op_leaf_count, rand_options, graph_ops)
        tail_list = [tail for _ in range(op_leaf_count+1)]
        positive_sample_t = torch.cuda.LongTensor(tail_list, device=self.device) \
            if self.device == 'cuda' else torch.LongTensor(tail_list, device=self.device)

        return (positive_sample_ops, positive_sample_t), negative_sample_t, \
               subsampling_weight, self.mode

    @staticmethod
    def collate_fn(data):
        # positive_sample_ops = data[0][0]
        # positive_sample_t = torch.stack([_[2] for _ in data], dim=0)
        # negative_sample_h = torch.stack([_[3] for _ in data], dim=0)
        # negative_sample_r = torch.stack([_[4] for _ in data], dim=0)
        # negative_sample_t = torch.stack([_[5] for _ in data], dim=0)
        # subsample_weight = torch.cat([_[6] for _ in data], dim=0)
        # mode = data[0][7]
        # return
        positive_sample = data[0][0]
        negative_sample_t = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample_t, \
               filter_bias, mode

    @staticmethod
    def count_frequency(graphs, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for graph_ops, graph_tail in graphs:
            if graph_ops not in count:
                count[graph_ops] = start
            else:
                count[graph_ops] += 1

            # if (tail, -relation-1) not in count:
            #     count[(tail, -relation-1)] = start
            # else:
            #     count[(tail, -relation-1)] += 1
        return count

    @staticmethod
    def get_true_head_and_tail(graphs):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''

        true_head = {}
        true_tail = {}

        for graph_ops, graph_tail in graphs:
            if graph_ops not in true_tail:
                true_tail[graph_ops] = []
            true_tail[graph_ops].append(graph_tail)
            if graph_tail not in true_head:
                true_head[graph_tail] = []
            true_head[graph_tail].append(graph_ops)

        for th in true_head:
            true_head[th] = np.array(list(set(true_head[th])))
        for tt in true_tail:
            true_tail[tt] = np.array(list(set(true_tail[tt])))

        return true_head, true_tail


class TestGraphDataset(Dataset):
    def __init__(self, graphs, all_true_graphs, nentity, nrelation, mode):
        self.len = len(graphs)
        self.graphs = graphs
        self.all_graphs = set(all_true_graphs)
        # self.triple_set = set(triples)
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        positive_sample = self.graphs[idx]

        graph_ops, tail = positive_sample

        negative_sample_list = []
        negative_sample_size = 0

        if self.mode == 'head-batch':
            print("this isn't implemented at all yet")
            tmp = [(0, rand_head) if (graph_ops, tail) not in self.all_graphs
                   else (-1, graph_ops) for rand_head in range(self.nentity)]
            tmp[graph_ops] = (0, graph_ops)
        elif self.mode == 'tail-batch' or self.mode == 'graph-tail-batch':
            tmp = [(0, rand_tail) if (graph_ops, rand_tail) not in self.all_graphs
                   else (-1, tail) for rand_tail in range(self.nentity)]
            tmp[tail] = (0, tail)
        else:
            raise ValueError('negative batch mode %s not supported' % self.mode)

        tmp = torch.cuda.LongTensor(tmp, device=self.device) if self.device == 'cuda' else torch.LongTensor(tmp, device=self.device)
        filter_bias = tmp[:, 0].float()

        negative_sample = tmp[:, 1]

        negative_sample_t = negative_sample

        def convert_to_tensors(input_dict):
            output_dict = dict()
            for k, v_list in input_dict.items():
                id_content = []
                for v in v_list:
                    if isinstance(v, dict) or isinstance(v, frozendict):
                        id_content.append(convert_to_tensors(v))
                    else:
                        v_tensor = torch.cuda.LongTensor([v], device=self.device) if self.device == 'cuda' else torch.LongTensor([v], device=self.device)
                        id_content.append(v_tensor)
                k_tensor = torch.cuda.LongTensor([k], device=self.device) if self.device == 'cuda' else torch.LongTensor([k], device=self.device)
                output_dict[k_tensor] = frozenset(id_content)
            return frozendict(output_dict)

        positive_sample_ops = convert_to_tensors(graph_ops)
        positive_sample_t = torch.cuda.LongTensor([tail], device=self.device) if self.device == 'cuda' else torch.LongTensor([tail], device=self.device)

        return (positive_sample_ops, positive_sample_t), negative_sample_t, \
               filter_bias, self.mode

    @staticmethod
    def collate_fn(data):
        positive_sample = data[0][0]
        negative_sample_t = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample_t, \
               filter_bias, mode

class PathlengthDictIterator(object):
    def __init__(self, dict_data):
        self.step = 0
        self.iterator_dict = {k:self.one_shot_iterator(dict_data[k]) for k in dict_data.keys()}
        self.iter_order = list(dict_data.keys())
        random.shuffle(self.iter_order)
        self.current_iter_index = 0
        self.current_path_iterator = self.iterator_dict[self.iter_order[self.current_iter_index]]

    def __next__(self):

        data = next(self.current_path_iterator)
        # iterate through all the data with pathlength of the current size, then move on to the next
        if data is None:
            self.current_iter_index += 1
            # all path lengths have been used in training - reset, and shuffle the order
            if self.current_iter_index == len(self.iter_order):
                self.current_iter_index = 0
                random.shuffle(self.iter_order)
            self.current_path_iterator = self.iterator_dict[self.iter_order[self.current_iter_index]]
            data = next(self.current_path_iterator)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data
            yield None

    
class PathTrainGraphDataset(Dataset):
    def __init__(self, graphs, nentity, nrelation, negative_sample_size, mode, ingredient_ids):
        self.len = len(graphs)
        self.graphs = graphs
        self.leaf_counts = {}
        self.leaf_nodes = {}
        for (gops, _) in graphs:
            leafs = set()
            leaf_count, leafs = count_leaves(gops, leafs)
            self.leaf_counts[gops] = leaf_count
            self.leaf_nodes[gops] = leafs
        # self.triple_set = set(triples)
        self.nentity = nentity
        self.entity_ids = {i for i in range(nentity)}
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(self.graphs)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.graphs)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.ingredient_ids = set(ingredient_ids)
        self.fg_head_rels = dict()

    def get_leaf_counts(self, gops):
        return self.leaf_counts[gops]

    def get_leaf_nodes(self, gops):
        return self.leaf_nodes[gops]

    def get_head_paths(self, input_dict):
        output_lists = []
        for k, v_list in input_dict.items():
            path_content = []
            for v in v_list:
                if isinstance(v, dict) or isinstance(v, frozendict):
                    inner_content = self.get_head_paths(v)
                    path_content.extend(inner_content)
                else:
                    path_content.extend([[v]])

            for p in path_content:
                p.append(k)
            output_lists.extend(path_content)
        return output_lists

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        positive_sample = self.graphs[idx]

        graph_ops, tail = positive_sample

        # removed inverse relations
        subsampling_weight = self.count[graph_ops]
        subsampling_weight = torch.sqrt(1 /
                                        torch.cuda.FloatTensor([subsampling_weight], device=self.device)
                                        if self.device == 'cuda'
                                        else torch.FloatTensor([subsampling_weight], device=self.device)
                                        )
        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size * 2)
            if self.mode == 'head-batch':
                mask = np.in1d(
                    negative_sample,
                    self.true_head[tail],
                    assume_unique=True,
                    invert=True
                )
            elif self.mode == 'tail-batch' or self.mode == 'graph-tail-batch':
                mask = np.in1d(
                    negative_sample,
                    self.true_tail[graph_ops],
                    assume_unique=True,
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample_t = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        negative_sample_t = torch.cuda.LongTensor(negative_sample_t, device=self.device) if self.device == 'cuda'\
            else torch.LongTensor(negative_sample_t, device=self.device)

        op_leaf_count = self.get_leaf_counts(graph_ops)
        op_leafs = self.get_leaf_nodes(graph_ops)
        rand_options = list(self.ingredient_ids-op_leafs)
        negative_sample_h = np.random.choice(rand_options, size=op_leaf_count)
        negative_sample_h = torch.cuda.LongTensor(negative_sample_h,
                                         device=self.device) if self.device == 'cuda' else torch.LongTensor(
            negative_sample_h, device=self.device)

        if graph_ops in self.fg_head_rels.keys():
            head_rels = self.fg_head_rels[graph_ops]
        else:
            head_rels = self.get_head_paths(graph_ops)
            self.fg_head_rels[graph_ops] = head_rels

        positive_sample_h = torch.cuda.LongTensor([hr[0] for hr in head_rels],
                                         device=self.device) if self.device == 'cuda' else torch.LongTensor(
            [hr[0] for hr in head_rels], device=self.device)
        positive_sample_r = tuple(torch.cuda.LongTensor(hr[1:],
                                         device=self.device) if self.device == 'cuda' else torch.LongTensor(
            [hr[0] for hr in head_rels], device=self.device) for hr in head_rels)

        tail_list = [tail for _ in range(op_leaf_count)]
        positive_sample_t = torch.cuda.LongTensor(tail_list, device=self.device) \
            if self.device == 'cuda' else torch.LongTensor(tail_list, device=self.device)
        return ((positive_sample_h, negative_sample_h), positive_sample_r, positive_sample_t), negative_sample_t, \
               subsampling_weight, self.mode

    @staticmethod
    def collate_fn(data):
        # positive_sample_ops = data[0][0]
        # positive_sample_t = torch.stack([_[2] for _ in data], dim=0)
        # negative_sample_h = torch.stack([_[3] for _ in data], dim=0)
        # negative_sample_r = torch.stack([_[4] for _ in data], dim=0)
        # negative_sample_t = torch.stack([_[5] for _ in data], dim=0)
        # subsample_weight = torch.cat([_[6] for _ in data], dim=0)
        # mode = data[0][7]
        # return
        positive_sample = data[0][0]
        negative_sample_t = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample_t, \
               filter_bias, mode

    @staticmethod
    def count_frequency(graphs, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for graph_ops, graph_tail in graphs:
            if graph_ops not in count:
                count[graph_ops] = start
            else:
                count[graph_ops] += 1

            # if (tail, -relation-1) not in count:
            #     count[(tail, -relation-1)] = start
            # else:
            #     count[(tail, -relation-1)] += 1
        return count

    @staticmethod
    def get_true_head_and_tail(graphs):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''

        true_head = {}
        true_tail = {}

        for graph_ops, graph_tail in graphs:
            if graph_ops not in true_tail:
                true_tail[graph_ops] = []
            true_tail[graph_ops].append(graph_tail)
            if graph_tail not in true_head:
                true_head[graph_tail] = []
            true_head[graph_tail].append(graph_ops)

        for th in true_head:
            true_head[th] = np.array(list(set(true_head[th])))
        for tt in true_tail:
            true_tail[tt] = np.array(list(set(true_tail[tt])))

        return true_head, true_tail



class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data



class OneShotIterator(object):
    def __init__(self, dataloader_tail):
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0

    def __next__(self):
        self.step += 1
        data = next(self.iterator_tail)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data
