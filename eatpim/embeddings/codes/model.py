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
# Copyright (c) 2019 Edward Sun, licensed under the MIT license,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score
from frozendict import frozendict

from torch.utils.data import DataLoader

from dataloader import TestDataset, PathlengthDictIterator
from dataloader import TestGraphDataset, OneShotIterator
import time

class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma,
                 double_entity_embedding=False, double_relation_embedding=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.gamma = nn.Parameter(
            torch.cuda.FloatTensor([gamma], device=self.device)
            if self.device == 'cuda' else torch.FloatTensor([gamma], device=self.device),
            requires_grad=False
        )
        
        self.embedding_range = nn.Parameter(
            torch.cuda.FloatTensor([(self.gamma.item() + self.epsilon) / hidden_dim], device=self.device)
            if self.device == 'cuda'
            else torch.FloatTensor([(self.gamma.item() + self.epsilon) / hidden_dim], device=self.device),
            requires_grad=False
        )
        
        self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim*2 if double_relation_embedding else hidden_dim

        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim, device=self.device))
        nn.init.uniform_(
            tensor=self.entity_embedding, 
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim, device=self.device))
        nn.init.uniform_(
            tensor=self.relation_embedding, 
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.agg_w = nn.Sequential(
            nn.Linear(self.entity_dim, self.entity_dim),
            nn.ReLU(),
            nn.Linear(self.entity_dim, self.entity_dim),
            nn.ReLU()
        )

        if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))
        
        #Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE', 'PathTransE']:
            raise ValueError('model %s not supported' % model_name)
            
        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')
        
    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements 
        in their triple ((head, relation) or (relation, tail)).
        '''

        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
        elif mode == 'graph-tail-batch':
            if self.model_name == 'PathTransE':
                (head_part, relation_part, tail_part), neg_sample_t = sample
                batch_size, negative_sample_size = 1, neg_sample_t.size(1)

                true_head_tens = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=head_part[0]
                ).unsqueeze(1)
                neg_head_tens = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=head_part[1]
                ).unsqueeze(1)

                relation_tens = self.PathTransECalcOperation_Relation(relation_part)

                head = ((true_head_tens, neg_head_tens), relation_tens)
                true_tail = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=tail_part.view(-1)
                ).unsqueeze(1)#.view(batch_size, negative_sample_size, -1)

                neg_tail = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=neg_sample_t.view(-1)
                )
                #.view(head_tens.size(0), negative_sample_size, -1)
            else:
                (sample_h, sample_t), neg_sample_t = sample
                batch_size, negative_sample_size = 1, neg_sample_t.size(1)

                head = sample_h

                true_tail = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=sample_t.view(-1)
                ).unsqueeze(1)

                neg_tail = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=neg_sample_t.view(-1)
                ).view(batch_size, negative_sample_size, -1)

        elif mode == 'graph-single':
            print("!?!?! you should not be here")
            sample_h, sample_t = sample
            batch_size, negative_sample_size = sample_t.size(0), 1

            head = sample_h
            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample_t.view(-1)
            ).unsqueeze(1)
            
        else:
            raise ValueError('mode %s not supported' % mode)
            
        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE,
            'PathTransE': self.TransE,
        }

        graph_model_func = {
            'RotatE': self.GOpRotatE,
            'TransE': self.GOpTransE,
            'DistMult': self.GOpDistMult,
            #'PathRotatE': self.PathRotatE,
            'PathTransE': self.PathTransE,
        }

        graph_tail_model_func = {
            'TransE': self.GOpTranseScores,
            'DistMult': self.GOpDistMultScores,
            'RotatE': self.GOpRotateScores,
            #'PathRotatE': self.PathRotatEScores,
            'PathTransE': self.PathTransEScores,
        }

        if mode == 'graph-tail-batch':
            pos_score, neg_tail_score, neg_head_score = graph_tail_model_func[self.model_name](head, true_tail, neg_tail)
            return pos_score, neg_tail_score, neg_head_score
        elif mode == 'graph-single':
            score = graph_model_func[self.model_name](head, tail)
        elif self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)
        
        return score

    def do_agg(self, heads):
        return self.agg_w(heads)

    def GOpRotateCalcOperation(self, ops):
        if isinstance(ops, frozendict):
            # we expect only 1 key/val pair in this dict. the key is the relation type, the val is
            # a list of entities or other operations that are performed
            for k,v in ops.items():
                relation_id = k
                entity_list = v
        else:
            # otherwise, ops is a tensor representing the ID of an entity
            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=ops
            ).unsqueeze(1)
            return head

        pi = 3.14159265358979323846

        relation = torch.index_select(
                        self.relation_embedding,
                        dim=0,
                        index=relation_id
                    ).unsqueeze(1)
        phase_relation = relation/(self.embedding_range.detach()/pi)

        entity_list = [self.GOpRotateCalcOperation(ent) for ent in entity_list]
        # aggregate using mean
        head_content = torch.stack(entity_list, dim=0).mean(dim=0)

        re_head, im_head = torch.chunk(head_content, 2, dim=2)
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation

        stacked_score = torch.cat([re_score, im_score], dim=2)

        return stacked_score

    def GOpRotatE(self, ops, tail):
        # this implementation is based on RotatE
        pi = 3.14159265358979323846

        re_tail, im_tail = torch.chunk(tail, 2, dim=2)
        scores = self.GOpRotateCalcOperation(ops)
        re_score, im_score = torch.chunk(scores, 2, dim=2)

        re_score = re_score - re_tail
        im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = self.gamma.detach() - score.sum(dim = 2)

        return score

    def GOpTranseCalcOperation(self, ops):
        if isinstance(ops, frozendict):
            # we expect only 1 key/val pair in this dict. the key is the relation type, the val is
            # a list of entities or other operations that are performed
            for k,v in ops.items():
                relation_id = k
                entity_list = v
        else:
            # otherwise, ops is a tensor representing the ID of an entity
            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=ops
            ).unsqueeze(1)
            return head

        relation = torch.index_select(
                        self.relation_embedding,
                        dim=0,
                        index=relation_id
                    ).unsqueeze(1)

        entity_list = [self.GOpTranseCalcOperation(ent) for ent in entity_list]
        # the aggregation strategy here is to just use mean
        if len(entity_list) == 1:
            head_content = entity_list[0]
        else:
            head_content = torch.stack(entity_list, dim=0).mean(dim=0)
        # head_content = self.do_agg(head_content)

        score = (head_content + relation)

        return score

    def GOpTransE(self, ops, tail):

        score = self.GOpTranseCalcOperation(ops) - tail

        score = self.gamma.detach() - torch.norm(score, p=1, dim=2)

        return score

    def GOpTranseScores(self, head, true_tail, neg_tail):
        head_calc = self.GOpTranseCalcOperation(head)

        pos_score = head_calc[0].view(1,1,-1)-true_tail[0].view(1,1,-1)
        pos_score = self.gamma.detach() - torch.norm(pos_score, p=1, dim=2)

        neg_head_score = head_calc[1:]-true_tail[1:]
        neg_head_score = self.gamma.detach() - torch.norm(neg_head_score, p=1, dim=2)

        neg_tail_score = head_calc[0]-neg_tail
        neg_tail_score = self.gamma.detach() - torch.norm(neg_tail_score, p=1, dim=2)

        return pos_score, neg_head_score, neg_tail_score

    def GOpDistMultScores(self, head, true_tail, neg_tail):
        head_calc = self.GOpDistMultCalcOperation(head)

        pos_score = head_calc[0].view(1,1,-1)*true_tail[0].view(1,1,-1)
        pos_score = pos_score.sum(dim = 2)

        neg_head_score = head_calc[1:]*true_tail[1:]
        neg_head_score = neg_head_score.sum(dim=2)

        neg_tail_score = head_calc[0]*neg_tail
        neg_tail_score = neg_tail_score.sum(dim=2)

        return pos_score, neg_head_score, neg_tail_score

    def GOpRotateScores(self, head, true_tail, neg_tail):
        pi = 3.14159265358979323846
        head_calc = self.GOpRotateCalcOperation(head)

        #####################################
        #####################################
        re_trueh_tail, im_trueh_tail = torch.chunk(true_tail[0].view(1,1,-1), 2, dim=2)
        re_pos_score, im_pos_score = torch.chunk(head_calc[0].view(1,1,-1), 2, dim=2)

        re_pos_score = re_pos_score - re_trueh_tail
        im_pos_score = im_pos_score - im_trueh_tail

        pos_score = torch.stack([re_pos_score, im_pos_score], dim = 0)
        pos_score = pos_score.norm(dim = 0)

        pos_score = self.gamma.detach() - pos_score.sum(dim = 2)

        #####################################
        re_truen_tail, im_truen_tail = torch.chunk(true_tail[1:], 2, dim=2)
        re_negh_score, im_negh_score = torch.chunk(head_calc[1:], 2, dim=2)

        re_negh_score = re_negh_score - re_truen_tail
        im_negh_score = im_negh_score - im_truen_tail

        negh_score = torch.stack([re_negh_score, im_negh_score], dim = 0)
        negh_score = negh_score.norm(dim = 0)

        negh_score = self.gamma.detach() - negh_score.sum(dim = 2)

        #####################################
        re_truet_tail, im_truet_tail = torch.chunk(neg_tail, 2, dim=2)
        re_negt_score, im_negt_score = torch.chunk(head_calc[0].view(1,1,-1), 2, dim=2)

        re_negt_score = re_negt_score - re_truet_tail
        im_negt_score = im_negt_score - im_truet_tail

        negt_score = torch.stack([re_negt_score, im_negt_score], dim = 0)
        negt_score = negt_score.norm(dim = 0)

        negt_score = self.gamma.detach() - negt_score.sum(dim = 2)
        #####################################

        return pos_score, negh_score, negt_score

    def PathTransEScores(self, head_relation, true_tail, neg_tail):

        head, relations = head_relation
        true_head, neg_head = head

        pos_score = true_head+relations-true_tail
        pos_score = self.gamma.detach() - torch.norm(pos_score, p=1, dim=2)

        neg_head_score = neg_head+relations-true_tail
        neg_head_score = self.gamma.detach() - torch.norm(neg_head_score, p=1, dim=2)

        neg_tail_score = true_head+relations-neg_tail
        neg_tail_score = self.gamma.detach() - torch.norm(neg_tail_score, p=1, dim=2)

        return pos_score, neg_head_score, neg_tail_score

    def PathTransECalcOperation_Relation(self, ops):
        relation_parts = [
            torch.index_select(
                self.entity_embedding,
                dim=0,
                index=op).sum(dim=0)
            for op in ops
        ]

        relation = torch.stack(relation_parts, dim=0).unsqueeze(1)

        return relation

    def PathTransE(self, head, relation, tail):

        score = head + relation - tail

        score = self.gamma.detach() - torch.norm(score, p=1, dim=2)

        return score

    def TransE(self, head, relation, tail, mode):
        # head = self.do_agg(head)
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.detach() - torch.norm(score, p=1, dim=2)
        return score

    def GOpDistMultCalcOperation(self, ops):
        if isinstance(ops, frozendict):
            # we expect only 1 key/val pair in this dict. the key is the relation type, the val is
            # a list of entities or other operations that are performed
            for k, v in ops.items():
                relation_id = k
                entity_list = v
        else:
            # otherwise, ops is a tensor representing the ID of an entity
            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=ops
            ).unsqueeze(1)
            return head

        relation = torch.index_select(
            self.relation_embedding,
            dim=0,
            index=relation_id
        ).unsqueeze(1)

        entity_list = [self.GOpDistMultCalcOperation(ent) for ent in entity_list]
        # the aggregation strategy here is to just use mean
        if len(entity_list) == 1:
            head_content = entity_list[0]
        else:
            head_content = torch.stack(entity_list, dim=0).mean(dim=0)

        score = (head_content * relation)

        return score

    def GOpDistMult(self, ops, tail):

        score = self.GOpDistMultCalcOperation(ops) * tail
        score = score.sum(dim = 2)

        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim = 2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim = 2)
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846
        
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation/(self.embedding_range.detach()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = self.gamma.detach() - score.sum(dim = 2)
        return score

    def pRotatE(self, head, relation, tail, mode):
        pi = 3.14159262358979323846
        
        #Make phases of entities and relations uniformly distributed in [-pi, pi]

        phase_head = head/(self.embedding_range.detach()/pi)
        phase_relation = relation/(self.embedding_range.detach()/pi)
        phase_tail = tail/(self.embedding_range.detach()/pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)            
        score = torch.abs(score)

        score = self.gamma.detach() - score.sum(dim = 2) * self.modulus
        return score
    
    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if mode == 'tail-batch' or mode == 'head-batch':
            positive_score = model(positive_sample, mode='single')
            negative_score = model((positive_sample, negative_sample), mode=mode)
            if args.negative_adversarial_sampling:
                # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
                negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                                  * F.logsigmoid(-negative_score)).sum(dim=1)
            else:
                negative_score = F.logsigmoid(-negative_score).mean(dim=1)
        elif mode == 'graph-tail-batch':
            positive_score, negative_head_score, negative_tail_score =\
                model((positive_sample, negative_sample), mode=mode)
            if args.negative_adversarial_sampling:
                #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
                negative_head_score = (F.softmax(negative_head_score * args.adversarial_temperature, dim = 1).detach()
                                  * F.logsigmoid(-negative_head_score)).sum(dim = 1)
                negative_tail_score = (F.softmax(negative_tail_score * args.adversarial_temperature, dim = 1).detach()
                                  * F.logsigmoid(-negative_tail_score)).sum(dim = 1)
            else:
                negative_head_score = F.logsigmoid(-negative_head_score).mean(dim = 1)
                negative_tail_score = F.logsigmoid(-negative_tail_score).mean(dim = 1)


        positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            if mode == 'tail-batch' or mode == 'head-batch':
                negative_sample_loss = - negative_score.mean()
            else:
                negative_sample_loss = - (negative_head_score.mean()+negative_tail_score.mean())
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            if mode == 'tail-batch' or mode == 'head-batch':
                negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()
            else:
                negative_head_loss = - (subsampling_weight * negative_head_score).sum()/subsampling_weight.sum()
                negative_tail_loss = - (subsampling_weight * negative_tail_score).sum()/subsampling_weight.sum()
                negative_sample_loss = negative_head_loss+negative_tail_loss

        loss = (positive_sample_loss + negative_sample_loss)/2

        if args.regularization != 0.0:
            #Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                model.entity_embedding.norm(p = 3)**3 + 
                model.relation_embedding.norm(p = 3).norm(p = 3)**3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.detach()}
        else:
            regularization_log = {}
        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.detach(),
            'negative_sample_loss': negative_sample_loss.detach(),
            'loss': loss.detach()
        }

        return log

    @staticmethod
    def predict(model, input_triple, mode, vocab_size, args):
        model.eval()
        with torch.no_grad():
            triple_torch = torch.from_numpy(np.array([input_triple]))
            negative_torch = torch.from_numpy(np.array([[i for i in range(vocab_size)]]))

            score = model((triple_torch, negative_torch), mode)

            argsort = torch.argsort(score, dim=1, descending=True)

            if mode == 'head-batch':
                positive_arg = triple_torch[:, 0]
            elif mode == 'tail-batch':
                positive_arg = triple_torch[:, 2]
            else:
                raise ValueError('mode %s not supported' % mode)

            return argsort[0, :].tolist()

    
    @staticmethod
    def test_step(model, test_graphs, all_true_graphs, test_triples, all_true_triples, args):
        '''
        Evaluate the model on test or valid datasets
        '''
        
        model.eval()
        #Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
        #Prepare dataloader for evaluation
        # TODO BIG TODO
        triple_test_dataloader_head = DataLoader(
            TestDataset(
                test_triples,
                all_true_triples,
                args.nentity,
                args.nrelation,
                'head-batch'
            ),
            batch_size=args.test_batch_size,
            num_workers=0,#max(1, args.cpu_num//2),
            collate_fn=TestDataset.collate_fn
        )

        triple_test_dataloader_tail = DataLoader(
            TestDataset(
                test_triples,
                all_true_triples,
                args.nentity,
                args.nrelation,
                'tail-batch'
            ),
            batch_size=args.test_batch_size,
            num_workers=0,#max(1, args.cpu_num//2),
            collate_fn=TestDataset.collate_fn
        )

        graph_test_dataloader_tail = DataLoader(
            TestGraphDataset(test_graphs,
                             all_true_graphs,
                             args.nentity,
                             args.nrelation,
                        'graph-tail-batch'),
            batch_size=1,#args.batch_size,
            shuffle=True,
            num_workers=0,#max(1, args.cpu_num//2),
            collate_fn=TestGraphDataset.collate_fn)

        # TODO BIG TODO
        triple_test_dataset_list = [triple_test_dataloader_head, triple_test_dataloader_tail]
        graph_test_dataset_list = [graph_test_dataloader_tail]

        logs = []

        step = 0
        total_steps = len(test_graphs)

        with torch.no_grad():
            for datasets in [graph_test_dataset_list]:#[triple_test_dataset_list]:#, triple_test_dataset_list]:
                for test_dataset in datasets:
                    for positive_sample,\
                        negative_sample,\
                        filter_bias, mode in test_dataset:

                        batch_size = 1

                        # this scoring is only valid for the graph-tail-batch evaluation. for other kinds of eval
                        # remove the first two vars and change this to just score = model...
                        _, _, score = model((positive_sample, negative_sample), mode)

                        score += filter_bias

                        #Explicitly sort all the entities to ensure that there is no test exposure bias
                        argsort = torch.argsort(score, dim = 1, descending=True)

                        if mode == 'head-batch':
                            positive_arg = positive_sample[:, 0]
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2]
                        elif mode == 'graph-tail-batch':
                            positive_arg = positive_sample[1]
                        else:
                            raise ValueError('mode %s not supported' % mode)

                        for i in range(batch_size):
                            #Notice that argsort is not ranking
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                            assert ranking.size(0) == 1

                            #ranking + 1 is the true ranking used in evaluation metrics
                            ranking = 1 + ranking.detach()
                            logs.append({
                                'MRR': 1.0/ranking,
                                'MR': float(ranking),
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                                'HITS@100': 1.0 if ranking <= 100 else 0.0,
                            })

                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                        step += 1

        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs])/len(logs)

        return metrics
