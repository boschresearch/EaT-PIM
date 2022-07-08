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

import json
import pickle
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import argparse
from eatpim.utils import path

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default=None)
args = parser.parse_args()

# point to your data here, probably at ../data/YOURDIR/recipe_tree_data.json
my_parse = (path.DATA_DIR / f'{args.data_path}/recipe_tree_data.json').resolve()
ground_parse = "transformed_recipe_data.pkl"

with open(my_parse, 'r') as f:
    eval_data = json.load(f)

with open(ground_parse, 'rb') as f:
    ground_data = pickle.load(f)

eval_id_to_graph = {}
ground_id_to_graph = {}

for k in eval_data.keys():
    edges = eval_data[k]['edges']
    output_node = eval_data[k]['output_node']
    clean_graph = nx.DiGraph()
    for (e_in, e_out) in edges:
        if e_out == output_node:
            continue
        if e_in[:5] == "pred_":
            mod_ein = e_in.split("_")[0]+" "+e_in.split("_")[1]
        else:
            mod_ein = e_in
        mod_eout = e_out.split("_")[0]+" "+e_out.split("_")[1]
        clean_graph.add_edge(mod_ein, mod_eout)

    graph_err = False
    target_ground_ind = -1
    for ind in range(len(ground_data['recipe_ids'])):
        if str(ground_data['recipe_ids'][ind]) == str(k):
            target_ground_ind = ind
            break
    ground_clean_graph = nx.DiGraph()
    for (e_in, e_out) in ground_data['recipe_graphs'][target_ground_ind].edges():
        if e_in[:5] == "pred ":
            mod_ein = e_in.split(" ")[0]+" "+e_in.split(" ")[1].split("_")[0]
        else:
            mod_ein = e_in
        if e_out[:5] != "pred ":
            graph_err = True
            break
        mod_eout = e_out.split(" ")[0]+" "+e_out.split(" ")[1].split("_")[0]
        ground_clean_graph.add_edge(mod_ein, mod_eout)
    if graph_err:
        continue

    eval_id_to_graph[k] = clean_graph
    ground_id_to_graph[k] = ground_clean_graph

print('number of graphs to evaluate against: ', len(eval_id_to_graph.keys()))
precs = []
recs = []
for k in eval_id_to_graph.keys():
    eg = eval_id_to_graph[k]
    gg = ground_id_to_graph[k]
    rec_edge_count = 0
    prec_edge_count = 0
    for e in gg.edges():
        if e in eg.edges():
            rec_edge_count += 1
    for e in eg.edges():
        if e in gg.edges():
            prec_edge_count += 1
    precs.append(prec_edge_count/len(eg.edges()))
    recs.append(rec_edge_count/len(gg.edges()))

import numpy as np
avg_precision = np.mean(precs)
avg_recall = np.mean(recs)
print('av precision: ', avg_precision)
print('av recall: ', avg_recall)
print('f1: ', 2*((avg_precision*avg_recall)/(avg_precision+avg_recall)))