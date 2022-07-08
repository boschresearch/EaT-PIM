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
import os
from pathlib import Path
from eatpim.utils import path
from typing import List
import numpy as np
import argparse
from collections import defaultdict
import networkx as nx
from sklearn.model_selection import train_test_split
import rdflib
from scipy.sparse import dok_matrix
import numpy as np
import time
import random
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from frozendict import frozendict


class FGCalculator:
    def __init__(self,
                 ent_embs, entity2id,
                 rel_embs, relation2id
                 ):
        self.ent_embs = ent_embs
        self.entity2id = entity2id
        self.id2entity = dict()
        for k,v in self.entity2id.items():
            self.id2entity[v] = k
        self.rel_embs = rel_embs
        self.relation2id = relation2id
        self.id2relation = dict()
        for k,v in self.relation2id.items():
            self.id2relation[v] = k

    def GOpTranseCalcOperation(self, *,
                               ops,
                               rem_ing, rep_ing=None):
        if isinstance(ops, frozendict):
            # we expect only 1 key/val pair in this dict. the key is the relation type, the val is
            # a list of entities or other operations that are performed
            for k, v in ops.items():
                relation_name = k
                entity_list = v
        else:
            # otherwise, ops is a tensor representing the ID of an entity
            if ops == rem_ing and rep_ing is not None:
                ops = rep_ing
            head = self.ent_embs[self.entity2id[ops]]
            return head

        relation = self.rel_embs[self.relation2id[relation_name]]

        entity_list_mod = []
        for ent in entity_list:
            inner_ent = self.GOpTranseCalcOperation(ops=ent, rem_ing=rem_ing, rep_ing=rep_ing)
            entity_list_mod.append(inner_ent)

        # the aggregation strategy here is to just use sum
        head_content = np.mean(np.array(entity_list_mod), axis=0)

        stacked_score = head_content+relation

        return stacked_score

    def ingredient_operation_sim(self, *, target_recipe, recipe_ops, replace_ing, ing_list):
        original_calc_vec = self.GOpTranseCalcOperation(ops=recipe_ops,
                                                                rem_ing=replace_ing)
        original_recipe_vec = self.ent_embs[self.entity2id[target_recipe]]
        calculated_sim = dict()
        original_sim = dict()
        for ing in ing_list:
            calc_recipe_vec = self.GOpTranseCalcOperation(ops=recipe_ops, rem_ing=replace_ing, rep_ing=ing)

            calculated_sim[ing] = cosine_similarity(
                original_calc_vec.reshape(1, -1), calc_recipe_vec.reshape(1, -1))[0][0]
            original_sim[ing] = cosine_similarity(
                original_recipe_vec.reshape(1, -1), calc_recipe_vec.reshape(1, -1))[0][0]

        sorted_sim_to_calc = sorted(calculated_sim.items(), key=lambda x: x[1], reverse=True)
        sorted_sim_to_og = sorted(original_sim.items(), key=lambda x: x[1], reverse=True)
        return sorted_sim_to_calc, sorted_sim_to_og

    def ingredient_sim(self, *, target_ing, ing_set):
        target_ing_emb = self.ent_embs[self.entity2id[target_ing]]
        sims = cosine_similarity(target_ing_emb.reshape(1,-1), self.ent_embs)[0]
        print(sims.shape)
        sim_dict = dict()
        for i in range(sims.shape[0]):
            if self.id2entity[i] in ing_set:
                sim_dict[self.id2entity[i]] = sims[i]
        sorted_sim = sorted(sim_dict.items(), key=lambda x: x[1], reverse=True)
        return sorted_sim


def get_ing_cooc_cosine_sims(target_ing, ing_to_index, cooc_matrix):
    return cosine_similarity(cooc_matrix[ing_to_index[target_ing]].reshape(1,-1), cooc_matrix)[0]


def get_prob_ing_exists_with_recipe(*, target_ing, ing_to_index, cooc_matrix,
                                    recipe_ingredients, ing_total_occ_arr):
    prob_matrix = cooc_matrix / ing_total_occ_arr
    total_ings = prob_matrix.shape[0]
    filter_ings = np.zeros((total_ings, total_ings))
    for ing in recipe_ingredients:
        filter_ings[:, ing_to_index[ing]] = 1
    prob_matrix = np.multiply(prob_matrix, filter_ings)

    prob_sim = cosine_similarity(prob_matrix[ing_to_index[target_ing]].reshape(1,-1), prob_matrix)[0]
    return prob_sim


def simple_visualize(G):
    pos = nx.nx_pydot.graphviz_layout(G, prog="dot") # dot or neato format
    plt.figure(1, figsize=(11, 11))

    nx.draw(G, pos, node_size=2000)
    node_labels = {}
    for n in G.nodes():
        if n[:5] == "pred_":
            node_labels[n] = n.split("_")[1]
        else:
            node_labels[n] = n
    nx.draw_networkx_labels(G, pos, labels=node_labels)
    plt.show()


def main(*,
         input_file: Path,
         main_dir: Path,
         valid_ingredient_list: List,
         kge_calc: FGCalculator,
         recipe_operations,
         target_recipe: str,
         target_ing: str):
    valid_ingredient_set = set(valid_ingredient_list)
    all_ingredients = set()
    leaf_ocurrence_count = defaultdict(lambda: 0)
    recipe_leafs = {}
    with open(input_file, 'r') as f:
        recipe_data = json.load(f)
    graphs = {}
    start = time.time()

    recipe_intermediatenode_count = {}
    for recipe, data in recipe_data.items():
        nodecount = 0
        G = nx.DiGraph()
        for e in data['edges']:
            G.add_edge(e[0], e[1])
        graphs[data['output_node']] = G
        recipe_leafs[data['output_node']] = set()
        for node in G.nodes():
            if node in valid_ingredient_set:
                all_ingredients.add(node)
                recipe_leafs[data['output_node']].add(node)
                leaf_ocurrence_count[node] += 1
            elif node != data['output_node']:
                nodecount += 1
        recipe_intermediatenode_count[data['output_node']] = nodecount


    print(f'{len(graphs)} graphs loaded')
    print(f'{len(all_ingredients)} distinct leaf nodes')
    print('')
    all_ingredients_list = list(all_ingredients)

    occ_matrix_files = (main_dir / "ing_occ_data.pkl").resolve()
    if (occ_matrix_files.is_file()):
        print('loading co-occ counts')
        with open(occ_matrix_files, 'rb') as f:
            matrix_data = pickle.load(f)
        ing_to_index = matrix_data['ing_to_index']
        index_to_ing = matrix_data['index_to_ing']
        ing_cooc_matrix = matrix_data['ing_cooc_matrix']
        ing_total_occ_count_arr = matrix_data['ing_total_occ_count_arr']
    else:

        ing_to_index, index_to_ing, \
        ing_cooc_matrix, ing_total_occ_count_arr = \
            compute_ing_cooc_matrix(all_ingredients_list,
                                    recipe_leafs,
                                    leaf_ocurrence_count)
        matrix_data = {
            'ing_to_index': ing_to_index ,
            'index_to_ing': index_to_ing ,
            'ing_cooc_matrix': ing_cooc_matrix ,
            'ing_total_occ_count_arr': ing_total_occ_count_arr
        }
        with open(occ_matrix_files, 'wb') as f:
            pickle.dump(matrix_data, f)

    if target_recipe == '':
        target_recipe_output = random.choice(list(graphs.keys()))
    else:
        target_recipe = f"RECIPE_OUTPUT_{target_recipe}"
        if target_recipe in graphs.keys():
            target_recipe_output = target_recipe
        else:
            print("The recipe you specified is not contained in the data.")
            return
        if target_ing != '':
            if target_ing in recipe_leafs[target_recipe_output]:
                target_replace_ing = target_ing
            else:
                print("The replacement ingredient you specified is not contained in the recipe")
                return
    if target_ing == '':
        target_replace_ing = random.choice(list(recipe_leafs[target_recipe_output]))


    print(f'recipe choice: {target_recipe_output}, replacing {target_replace_ing}')
    print(f'number of intermediate nodes: {recipe_intermediatenode_count[target_recipe_output]}')
    target_recipe_graph = graphs[target_recipe_output]
    simple_visualize(target_recipe_graph)

    t1sim = get_ing_cooc_cosine_sims(
        target_ing=target_replace_ing,
        ing_to_index=ing_to_index,
        cooc_matrix=ing_cooc_matrix)
    top_sims = np.argsort(t1sim)[::-1]
    print('')
    print("top 10 cosine similarity of ingredient-coocurrences: ")
    for i in top_sims[:10]:
        print(index_to_ing[i], t1sim[i])

    t2sim = get_prob_ing_exists_with_recipe(
        target_ing=target_replace_ing,
        ing_to_index=ing_to_index,
        cooc_matrix=ing_cooc_matrix,
        ing_total_occ_arr=ing_total_occ_count_arr,
        recipe_ingredients=recipe_leafs[target_recipe_output]
    )
    top_sims2 = np.argsort(t2sim)[::-1]
    print('')
    print("top 10 cosine similarity of ingredient-coocurrences, only considering ingredients in the target recipe: ")
    for i in top_sims2[:10]:
        print(index_to_ing[i], t2sim[i])

    target_actions_in_recipe = set()
    for n in nx.dfs_successors(target_recipe_graph, target_replace_ing):
        if n == target_replace_ing:
            continue
        action_name = n.split("_")[1]
        target_actions_in_recipe.add(action_name)

    recipe_ops = recipe_operations[target_recipe_output]
    calc_sim, og_sim = kge_calc.ingredient_operation_sim(target_recipe=target_recipe_output,
                                                             recipe_ops=recipe_ops,
                                                             replace_ing=target_replace_ing,
                                                             ing_list=all_ingredients_list)
    ing_sim = kge_calc.ingredient_sim(target_ing=target_replace_ing, ing_set=all_ingredients)
    print("")
    print(f"{target_replace_ing} occ frequency in data: ", ing_total_occ_count_arr[ing_to_index[target_replace_ing]])

    print('')
    print('top 30 ingredients that produce outputs most similar to the original calculated output')
    for i in range(30):
        print("rank ",i, ": ", calc_sim[i])
    print('')
    print('top 20 ingredients that produce outputs most similar to the original recipe\'s learned embedding')
    for i in range(20):
        print(og_sim[i])
    print('')
    print('top 20 ingredients whose embedding is most similar to the original ingredient\'s embedding')
    for i in range(20):
        print(ing_sim[i])

    # ing_rank_vote_v1 = defaultdict(lambda: 0)
    # ing_rank_vote_v2 = defaultdict(lambda: 0)
    # for rank, tup in enumerate(calc_sim):
    #     ing_rank_vote_v1[tup[0]] += rank+1
    #     ing_rank_vote_v2[tup[0]] += rank+1
    # for rank, tup in enumerate(og_sim):
    #     ing_rank_vote_v1[tup[0]] += rank+1
    #     ing_rank_vote_v2[tup[0]] += rank+1
    # for rank, tup in enumerate(ing_sim):
    #     ing_rank_vote_v1[tup[0]] += rank+1
    # sorted_vote_v1 = sorted(ing_rank_vote_v1.items(), key=lambda x: x[1])
    # sorted_vote_v2 = sorted(ing_rank_vote_v2.items(), key=lambda x: x[1])
    # print('')
    # print('voting using all 3 sims:')
    # for i in range(25):
    #     print(sorted_vote_v1[i])
    # print('')
    # print('voting using calc and og recipe sims')
    # for i in range(25):
    #     print(sorted_vote_v2[i])


def load_embedding_data(main_dir, model_dir) -> FGCalculator:
    ent_embs = np.load((main_dir / model_dir / 'entity_embedding.npy').resolve())
    rel_embs = np.load((main_dir / model_dir / 'relation_embedding.npy').resolve())

    with open((main_dir / 'eatpim_triple_data/entities.dict').resolve()) as fin:
        id2entity = dict()
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)
            id2entity[int(eid)] = entity

    with open((main_dir / 'eatpim_triple_data/relations.dict').resolve()) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)

    calc = FGCalculator(ent_embs=ent_embs, rel_embs=rel_embs, entity2id=entity2id, relation2id=relation2id)
    return calc


def content_to_ids(input_dict):
    output_dict = dict()
    for k, v_list in input_dict.items():
        id_content = []
        for v in v_list:
            if isinstance(v, dict):
                id_content.append(content_to_ids(v))
            else:
                id_content.append(str(v))
        output_dict[str(k)] = frozenset(id_content)
    return frozendict(output_dict)


def compute_ing_cooc_matrix(all_ingredients_list, recipe_leafs, leaf_ocurrence_count):
    print("processing to compute ingredient co-occurence counts")
    ing_to_index = {}
    index_to_total_occ_count = []
    for ing in all_ingredients_list:
        ing_to_index[ing] = len(ing_to_index.keys())
        index_to_total_occ_count.append(leaf_ocurrence_count[ing])
    index_to_ing = {v:k for k,v in ing_to_index.items()}
    ing_cooc_matrix = dok_matrix((len(index_to_ing), len(index_to_ing)), dtype=np.float16)
    ing_total_occ_count_arr = np.array(index_to_total_occ_count).reshape(-1, 1)
    for recipe, ing_set in recipe_leafs.items():
        ing_index_list = [ing_to_index[ing] for ing in ing_set]
        for i, ing1 in enumerate(ing_index_list):
            for ing2 in ing_index_list[i + 1:]:
                ing_cooc_matrix[ing1, ing2] += 1
                ing_cooc_matrix[ing2, ing1] += 1
    ing_cooc_matrix = ing_cooc_matrix.tocsr()

    return ing_to_index, index_to_ing, ing_cooc_matrix,ing_total_occ_count_arr


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)

    parser.add_argument("--target_recipe", type=str, default='')
    parser.add_argument("--target_ingredient", type=str, default='')

    args = parser.parse_args()

    main_dir = (path.DATA_DIR / args.data_dir).resolve()
    input_file = (main_dir / "recipe_tree_data.json").resolve()
    relation_input_file = (main_dir / "entity_relations.json").resolve()
    with open((main_dir / "ingredient_list.json").resolve(), 'r') as f:
        ingredient_list = json.load(f)

    calc = load_embedding_data(main_dir, args.model_dir)

    recipe_operations = {}
    import random

    for dset in ['train.txt', 'valid.txt', 'test.txt']:
        with open((main_dir / f'eatpim_triple_data/{dset}').resolve()) as fin:
            for line in fin:
                graph_dict = json.loads(line)
                # there should only be one item in the first depth of this dict
                # the key is the output recipe node, the value is the dictionary representation of the flowgraph
                for k, v in graph_dict.items():
                    recipe_operations[str(k)] = content_to_ids(v)

    main(input_file=input_file,
         main_dir=main_dir,
         valid_ingredient_list=ingredient_list,
         kge_calc=calc, recipe_operations=recipe_operations,
         target_recipe=args.target_recipe, target_ing=args.target_ingredient)


#RECIPE_OUTPUT_365375, replacing sugar
# good example?

#v--target_recipe 421747 --target_ingredient "white bread flour"
#

# --target_recipe 129656 --target_ingredient "bacon" running recipe
# --target_recipe 219233 --target_ingredient "pork" pork example

# red potato sub - in 172744 mashed potato recipe
# ('sweet potato', 0.9997594) : rank  58
# ('carrot', 0.9997389) : rank  219
# ('parsnip', 0.99973124) : rank  328
# ('cauliflower', 0.99968296) : rank  1351
# ('jicama', 0.9996819) : rank  1371
# ('rutabaga', 0.99962944) : rank  2201
# car pars caul jic
# jic cauli pars ruta
# learned embedding rank order - carr, jic, cauli, parsnip, ruta, sp

# - in 15202, potatoes with garlic and cheese  pommes de terre a l ail
# ('sweet potato', 0.99958664) : rank  47
# ('parsnip', 0.9995422) : rank  260
# ('carrot', 0.9995282) : rank  391
# ('cauliflower', 0.99945205) : rank  1389
# ('jicama', 0.99944115) : rank  1524
# ('rutabaga', 0.9993692) : rank  2244
# pars car caul jic
# pars caul jic ruta
# learned embedding rank order - carr, sp, parsnip, cauli, jic, ruta

# - in 266081, "healthy soup"
# ('sweet potato', 0.8984294) : rank  56
# ('parsnip', 0.89423525) : rank  105
# ('carrot', 0.8884514) : rank  256
# ('jicama', 0.8687967) : rank  1343
# ('cauliflower', 0.86774457) : rank  1401
# ('rutabaga', 0.86402225) : rank  1617
# pars car jic caul
# pars ruta cauli jic
# learned embedding rank order - carr, parsnip, sp, ruta, cauli, jicama

###########################################################
###########################################################
###########################################################


#('rutabaga', 0.9999989) : rank  2086
# ('parsnip', 0.99999917) : rank  492
# ('sweet potato', 0.9999993) : rank  91
# ('cauliflower', 0.99999905) : rank  1396
# ('carrot', 0.9999992) : rank  336
# ('jicama', 0.99999905) : rank  1329

#('sweet potato', 0.7401762) : rank  702
# ('carrot', 0.7401529) : rank  1166
# ('jicama', 0.74013805) : rank  1485
# ('cauliflower', 0.74013656) : rank  1521
# ('parsnip', 0.74010676) : rank  2143
# ('rutabaga', 0.74009234) : rank  2448