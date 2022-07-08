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

import argparse
import json
import logging
import os

import numpy as np
import torch

from torch.utils.data import DataLoader

from model import KGEModel

from dataloader import TrainDataset, TrainGraphDataset, PathTrainGraphDataset
from dataloader import OneShotIterator, BidirectionalOneShotIterator
from eatpim.utils import path
from frozendict import frozendict

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')
    
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--do_example', action='store_true', help='show an example ingredient prediction')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')

    parser.add_argument('--regions', type=int, nargs='+', default=None, 
                        help='Region Id for Countries S1/S2/S3 datasets, DO NOT MANUALLY SET')
    
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--model', default='TransE', type=str)
    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')
    
    parser.add_argument('-n', '--negative_sample_size', default=2048, type=int)
    parser.add_argument('-d', '--hidden_dim', default=200, type=int)
    parser.add_argument('-g', '--gamma', default=24.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('--train_triples_every_n', default=100, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true', 
                        help='Otherwise use subsampling weighting like in word2vec')
    
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=1, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)
    
    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')
    
    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')
    
    return parser.parse_args(args)

def override_config(args):
    '''
    Override model and data configuration
    '''
    
    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)
    

    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']
    
def save_model(model, optimizer, save_variable_list, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )
    
    entity_embedding = model.entity_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'entity_embedding'), 
        entity_embedding
    )
    
    relation_embedding = model.relation_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'relation_embedding'), 
        relation_embedding
    )

    agg_w = model.agg_w
    torch.save(agg_w, os.path.join(args.save_path, 'agg_weight'))

def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples

def read_graphs(file_path, entity2id, relation2id):
    '''
    Read graphs and map them into ids.
    '''
    def content_to_ids(input_dict):
        output_dict = dict()
        for k, v_list in input_dict.items():
            id_content = []
            for v in v_list:
                if isinstance(v, dict):
                    id_content.append(content_to_ids(v))
                else:
                    id_content.append(entity2id[str(v)])
            output_dict[relation2id[str(k)]] = frozenset(id_content)
        return frozendict(output_dict)

    graphs = []
    with open(file_path) as fin:
        for line in fin:
            graph_dict = json.loads(line)
            # there should only be one item in the first depth of this dict
            # the key is the output recipe node, the value is the dictionary representation of the flowgraph
            for k,v in graph_dict.items():
                graphs.append((content_to_ids(v), entity2id[str(k)]))
    return graphs


def read_triple_pathlengths(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = {}
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            # we expect h to be a list of ingredients, and r to also be a list.
            # r is a path, so order matters
            pathlen = len(eval(r))
            if pathlen not in triples:
                triples[pathlen] = []
            triples[pathlen].append((tuple(entity2id[str(h_content)] for h_content in eval(h)),
                                     tuple(relation2id[str(r_content)] for r_content in eval(r)),
                                     entity2id[t]))
    return triples

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))
        
        
def main(args):
    if (not args.do_train) and (not args.do_valid) and (not args.do_test) and (not args.do_example):
        raise ValueError('one of train/val/test mode must be choosed.')

    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')

    args.save_path = os.path.join(path.DATA_DIR, args.data_path, args.save_path)#(path.DATA_DIR / args.data_path / args.save_path).resolve()

    if args.init_checkpoint:
        args.init_checkpoint = os.path.join(path.DATA_DIR, args.data_path, args.init_checkpoint)
        override_config(args)
    elif args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be choosed.')

    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    # Write logs to checkpoint and console
    set_logger(args)

    with open((path.DATA_DIR / args.data_path / "ingredient_list.json").resolve()) as f:
        ingredient_list = json.load(f)

    data_dir = (path.DATA_DIR / args.data_path / "eatpim_triple_data").resolve()

    with open((data_dir / 'entities.dict').resolve()) as fin:
        id2entity = dict()
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)
            id2entity[int(eid)] = entity

    ingredient_ids = [entity2id[ing] for ing in ingredient_list if ing in entity2id.keys()]

    with open((data_dir / 'relations.dict').resolve()) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)

    nentity = len(entity2id)
    nrelation = len(relation2id)
    
    args.nentity = nentity
    args.nrelation = nrelation
    
    logging.info('Model: %s' % args.model)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)
    
    train_graphs = read_graphs((data_dir / 'train.txt').resolve(), entity2id, relation2id)
    logging.info('#train: %d' % len(train_graphs))
    valid_graphs = read_graphs((data_dir / 'valid.txt').resolve(), entity2id, relation2id)
    logging.info('#valid: %d' % len(valid_graphs))
    test_graphs = read_graphs((data_dir / 'test.txt').resolve(), entity2id, relation2id)
    logging.info('#test: %d' % len(test_graphs))

    train_triples = read_triple((data_dir / 'trip_train.txt').resolve(), entity2id, relation2id)
    logging.info('#train triples: %d' % len(train_triples))
    valid_triples = read_triple((data_dir / 'trip_valid.txt').resolve(), entity2id, relation2id)
    logging.info('#valid triples: %d' % len(valid_triples))
    test_triples = read_triple((data_dir / 'trip_test.txt').resolve(), entity2id, relation2id)
    logging.info('#test triples: %d' % len(test_triples))
    
    #All true triples
    # all_true_triples = []#dict()
    # dicts = [train_graphs, valid_graphs, test_graphs]
    # for d in dicts:
    #     for k, v in d.items():
    #         # curr_content = all_true_triples.get(k, [])
    #         # curr_content.extend(v)
    #         # all_true_triples[k] = curr_content
    #         all_true_triples.extend(v)
    all_true_graphs = train_graphs+valid_graphs+test_graphs
    all_true_triples = train_triples+valid_triples+test_triples

    
    kge_model = KGEModel(
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding
    )
    
    logging.info('Model Parameter Configuration:')
    for name, param in kge_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    if args.cuda:
        kge_model = kge_model.cuda()
        # # set to half precision (float16). CPU version of torch doesn't handle half.
        # kge_model = kge_model.half()

    if args.do_train:
        # Set training dataloader iterator
        triple_train_dataloader_head = DataLoader(
            TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'head-batch'),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,#max(1, args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn
        )
        triple_train_dataloader_tail = DataLoader(
            TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'tail-batch'),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,#max(1, args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn
        )
        
        graph_train_dataloader_tail = DataLoader(
            TrainGraphDataset(train_graphs, nentity, nrelation,
                              args.negative_sample_size*8,
                              'graph-tail-batch',
                              ingredient_ids=ingredient_ids), # tail-batch
            batch_size=1,#args.batch_size,
            shuffle=True, 
            num_workers=0,#max(1, args.cpu_num//2),
            collate_fn=TrainGraphDataset.collate_fn
        )

        # logging.info("Training path-based TransE experiment...")
        # graph_train_dataloader_tail = DataLoader(
        #     PathTrainGraphDataset(train_graphs, nentity, nrelation,
        #                       args.negative_sample_size*8,
        #                       'graph-tail-batch',
        #                       ingredient_ids=ingredient_ids), # tail-batch
        #     batch_size=1,#args.batch_size,
        #     shuffle=True,
        #     num_workers=0,#max(1, args.cpu_num//2),
        #     collate_fn=PathTrainGraphDataset.collate_fn
        # )

        # TODO BIG TODO
        graph_train_iterator = OneShotIterator(graph_train_dataloader_tail)
        triple_train_iterator = BidirectionalOneShotIterator(triple_train_dataloader_head, triple_train_dataloader_tail)

        # Set training configuration
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, kge_model.parameters()), 
            lr=current_learning_rate,
            # eps=1e-4  # this parameter is important to prevent nans from occurring during training with half precision
        )
        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_steps // 2

    if args.init_checkpoint:
        # Restore model from checkpoint directory
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        init_step = checkpoint['step']
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.model)
        init_step = 0
    
    step = init_step
    
    logging.info('Start Training...')
    logging.info('init_step = %d' % init_step)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)
    logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
    if args.negative_adversarial_sampling:
        logging.info('adversarial_temperature = %f' % args.adversarial_temperature)
    train_triples_every_n = args.train_triples_every_n

    # Set valid dataloader as it would be evaluated during training
    if args.do_train:
        logging.info('learning_rate = %f' % current_learning_rate)

        training_logs = []

        #Training Loop
        for step in range(init_step, args.max_steps):

            if step % train_triples_every_n == 0 and step > 0:
                # every 100 steps, train the triples in the graph
                # triples give data about relations among entities, as well as recipes and their
                # nodes (recipe, hasNode, ingredient-or-equipment)
                # the triple training also can actually use batches, while the graph training only does 1 graph at a time
                log = kge_model.train_step(kge_model, optimizer, triple_train_iterator, args)
            else:
                log = kge_model.train_step(kge_model, optimizer, graph_train_iterator, args)

            training_logs.append(log)
            
            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 10
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, kge_model.parameters()), 
                    lr=current_learning_rate,
                    eps=1e-4
                )
                warm_up_steps = warm_up_steps * 3
            
            if step % args.save_checkpoint_steps == 0:
                save_variable_list = {
                    'step': step, 
                    'current_learning_rate': current_learning_rate,
                    'warm_up_steps': warm_up_steps
                }
                save_model(kge_model, optimizer, save_variable_list, args)
                
            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                log_metrics('Training average', step, metrics)
                training_logs = []
                
            if args.do_valid and step % args.valid_steps == 0:
                logging.info('Evaluating on Valid Dataset...')
                metrics = kge_model.test_step(kge_model, valid_graphs, all_true_graphs, valid_triples, all_true_triples, args)
                log_metrics('Valid', step, metrics)
        
        save_variable_list = {
            'step': step,
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }
        save_model(kge_model, optimizer, save_variable_list, args)
        
    if args.do_valid:
        logging.info('Evaluating on Valid Dataset...')
        metrics = kge_model.test_step(kge_model,
                                      valid_graphs, all_true_graphs, valid_triples, all_true_triples,
                                      args)
        log_metrics('Valid', step, metrics)
    
    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        metrics = kge_model.test_step(kge_model,
                                      test_graphs, all_true_graphs, test_triples, all_true_triples,
                                      args)
        log_metrics('Test', step, metrics)
    
    if args.evaluate_train:
        logging.info('Evaluating on Training Dataset...')
        metrics = kge_model.test_step(kge_model, train_graphs, all_true_graphs, args)
        log_metrics('Test', step, metrics)


if __name__ == '__main__':
    main(parse_args())
