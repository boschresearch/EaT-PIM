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
from eatpim.utils import path
import numpy as np
import argparse
from collections import defaultdict
import networkx as nx
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import lil_matrix, vstack


def main(*, input_file: str, output_file: str, output_file_2: str, cleanup_file: str, processes: int,
         format_output_file: str):
    all_edges = []
    all_edge_labels = []
    G = nx.DiGraph()
    with open(output_file, 'r') as f:
        recipe_data = json.load(f)
    for recipe, data in recipe_data.items():
        edges = data['edges']
        edge_labels = data['edge_labels']
        for ind, e in enumerate(edges):
            G.add_edge(e[0], e[1])
            all_edges.append(e)
            all_edge_labels.append(edge_labels[ind])
            to_node = e[1]
            if to_node[:5] == "pred_":
                node_content = to_node.split("_")
                to_node_pred_type = f'{node_content[0]} {node_content[1]}'
                G.add_edge(to_node_pred_type, e[1])
                all_edges.append([e[1], to_node_pred_type])
                all_edge_labels.append('class_pred_type')


    relation_G = nx.DiGraph()
    with open(output_file_2, 'r') as f:
        relation_data = json.load(f)
    for ind, e in enumerate(relation_data['subclass_edges']):
        # Swap the order of edges for relations
        # right now these are subclass relations, so we have something like 'red potato'->'potato', 'potato'->'...'
        # the foods used in the flowgraphs direct towards the final recipe output, like 'stepX'->'recipe output'
        # so for training the embeddings I want the direction of the relation information to also follow a
        # similar direction
        relation_G.add_edge(e[1], e[0])
        all_edges.append([e[1], e[0]])

        all_edge_labels.append(relation_data['subclass_edge_labels'][ind])

    edge_label_index = {}
    for edge_name in set(all_edge_labels):
        edge_label_index[edge_name] = len(edge_label_index)

    nodes = list(G.nodes())
    node_count = len(nodes)
    # add node features
    # to keep things simple, node features will be a one-hot encoding to represent whether the node
    # is an input leaf, an intermediate node, an intermediate node type (i.e., 'pred mix' is the type for an
    # intermediate node where thinns are being mixed), a recipe output node, or external knowledge
    # order is [leaf, intermediate, intermediate_type, output, external]
    print("processing node features")
    feature_length = 5
    node_features = lil_matrix((node_count, feature_length))
    for node_ind, node in enumerate(nodes):
        node_content = node.split("_")
        if node[:5] == "pred_":
            node_features[node_ind, 1] = 1
        elif node[:5] == "pred ":
            node_features[node_ind, 2] = 1
        elif node[:13] == "RECIPE_OUTPUT":
            node_features[node_ind, 3] = 1
        else:
            node_features[node_ind, 0] = 1

    relation_nodes= []
    seen_nodes = set(nodes)
    for node in relation_G.nodes():
        if node not in seen_nodes:
            nodes.append(node)
            relation_nodes.append(node)

    relation_node_feats = lil_matrix((len(relation_nodes), feature_length))
    relation_node_feats[:, 4] = 1
    node_features = vstack((node_features, relation_node_feats))

    formatted_data = {'nodes': nodes,
                      'node_features': node_features,
                      'edges': all_edges,
                      'edge_labels': all_edge_labels}
    with open(format_output_file, 'wb') as f:
        pickle.dump(formatted_data, f)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--n_cpu", type=int, default=1)

    args = parser.parse_args()

    main_dir = (path.DATA_DIR / args.input_dir).resolve()
    input_file = (main_dir / "parsed_recipes.pkl").resolve()
    output_file = (main_dir / "recipe_tree_data.json").resolve()
    output_file_2 = (main_dir / "entity_relations.json").resolve()
    format_output_file = (main_dir / "formatted_graph_data.pkl").resolve()
    cleanup_file = (main_dir / "word_cleanup_linking.json").resolve()

    main(input_file=str(input_file),
         output_file=str(output_file),
         output_file_2=str(output_file_2),
         cleanup_file=str(cleanup_file),
         processes=args.n_cpu,
         format_output_file=str(format_output_file))
