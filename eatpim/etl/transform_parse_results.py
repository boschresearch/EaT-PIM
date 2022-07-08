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
from urllib.parse import quote
import rdflib
from rdflib import Literal
from rdflib.namespace import RDFS, RDF
from typing import Dict
import argparse
from collections import defaultdict
import networkx as nx
from copy import copy
import time
import multiprocessing as mp
import numpy as np
import os
import matplotlib.pyplot as plt
from eatpim.etl.recipe_graph import RecipeGraph


class SemSimCalculator:
    """
    Distance calculations based on wpath similarity from Zhu 2016
    "Computing Semantic Similarity of Concepts in Knowledge Graphs"
    """
    def __init__(self, *, G: nx.DiGraph, shared_ic_dict, k:float=0.5):
        self.G = G
        self.rev_G = nx.reverse(G, copy=True)
        self.shared_ic_dict = shared_ic_dict
        self.max_D = nx.dag_longest_path_length(G)
        self.total_N = len(G.nodes())
        self.k = k
        self.root_node = 'http://purl.obolibrary.org/obo/FOODON_00001002'

    def get_node_information_content(self, n):
        if n in self.shared_ic_dict.keys():
            return self.shared_ic_dict[n]
        else:
            ic = -np.log10((len(nx.descendants(self.G, n))+1)/self.total_N)
            self.shared_ic_dict[n] = ic
            return ic

    def sim(self, a, b):
        # bfs predecessors outputs a bunch of edges, as (to_node, from_node)
        # we just want to to_node part here
        a_preds = [a]
        a_preds.extend([tup[0] for tup in nx.bfs_predecessors(self.rev_G, a)])
        b_preds = [b]
        b_preds.extend([tup[0] for tup in nx.bfs_predecessors(self.rev_G, b)])
        common_preds = set([n for n in a_preds]).intersection(set([n for n in b_preds]))
        common_preds = [p for p in common_preds if nx.has_path(self.G, self.root_node, p)]

        if len(common_preds) == 0:
            return 0

        max_depth = 0
        max_ic = 0
        lca_node = common_preds[0]
        # networkx's built in lowest common ancestor algorithm is quite slow
        # check the common ancestors, and choose the one with the greatest depth
        # if there is a tie in depth, choose the one with greater information content.
        for p in common_preds:
            depth = nx.shortest_path_length(self.G, self.root_node, p)
            if depth > max_depth:
                max_depth = depth
                max_ic = self.get_node_information_content(n=p)
                lca_node = p
            elif depth == max_depth:
                p_ic = self.get_node_information_content(n=p)
                if p_ic > max_ic:
                    max_ic = p_ic
                    lca_node = p
        lca_pathdist = nx.shortest_path_length(self.G, lca_node, a)+nx.shortest_path_length(self.G, lca_node, b)
        sim = 1 / (1 + lca_pathdist * (self.k ** max_ic))
        return sim


def format_graph_to_triples(recipe_graph: nx.DiGraph, recipe_id: int):
    node_to_inputs = defaultdict(lambda: [])
    output_triples = []
    leaves = set(n for n in recipe_graph.nodes() if recipe_graph.in_degree(n) == 0)
    for node in recipe_graph.nodes():
        if node in leaves:
            continue
        for e in recipe_graph.in_edges(node):
            in_node = e[0]
            if in_node in leaves:
                node_to_inputs[node].append(in_node.split("_")[0])
            else:
                node_to_inputs[node].append(f'{recipe_id}_{in_node}_OUTPUT')
    for k, v in node_to_inputs.items():
        if k != f'RECIPE_OUTPUT_{recipe_id}':
            action_name = k.split("_")[0]
            output_triples.append((v, action_name, f'{recipe_id}_{k}_OUTPUT'))
    return output_triples

def visualize(G, ingredient_nodes, external_ingredient_nodes, foodon_nodes, equipment_nodes):

    pos = nx.nx_pydot.graphviz_layout(G, prog="neato")
    plt.figure(1, figsize=(11, 11))

    nx.draw(G, pos, node_size=2000)
    nx.draw_networkx_nodes(G, pos, nodelist=list(ingredient_nodes), node_color='r')
    nx.draw_networkx_nodes(G, pos, nodelist=list(external_ingredient_nodes), node_color='white')
    nx.draw_networkx_nodes(G, pos, nodelist=list(foodon_nodes), node_color='green')
    nx.draw_networkx_nodes(G, pos, nodelist=list(equipment_nodes), node_color='y')
    # nx.draw_networkx_nodes(G, pos, nodelist=list(true_ing_nodes), node_color='black')
    nx.draw_networkx_labels(G, pos)
    plt.show()

def process_single_recipe(q: mp.Queue, recipe_tree_dict,
                          entity_subclass_dig: nx.DiGraph,
                          reversed_entity_subclass_dig: nx.DiGraph,
                          ing_connection_dict: Dict,
                          sem_dist_calculator: SemSimCalculator):
    # a single CPU process for converting a single recipe into a graph.
    # takes in data from a multiprocessing Queue
    full_graphs = 0
    has_specialcase = 0
    has_allings = 0
    while True:
        recipe_id, recipe_data = q.get()
        if recipe_data is None:
            break

        ing_checks = copy(recipe_data['ingredients'])
        path_to_ing = defaultdict(lambda: [])
        objset = set(obj for step in recipe_data['parsed_steps'].values() for obj in step['noun_chunks'])
        for obj in objset:
            if obj in ing_checks:
                ing_checks.remove(obj)
                path_to_ing[obj].append(obj)
            else:
                if not entity_subclass_dig.has_node(obj):
                    continue
                for ing in ing_checks:
                    if entity_subclass_dig.has_node(ing) and (
                            nx.has_path(entity_subclass_dig, obj, ing) or nx.has_path(entity_subclass_dig, ing, obj)):
                        ing_checks.remove(ing)
                        path_to_ing[obj].append(ing)
                        break

        recipe_graph = RecipeGraph(recipe_data=recipe_data, recipe_id=recipe_id,
                                   entity_subclass_dig=entity_subclass_dig,
                                   reversed_entity_subclass_dig=reversed_entity_subclass_dig,
                                   ing_connection_dict=ing_connection_dict,
                                   sim_calc=sem_dist_calculator)

        for step_id, step in recipe_data['parsed_steps'].items():
            recipe_graph.parse_step_into_graph(step_id=step_id, step=step)

        # post processing to connect into a single graph
        connect_graph_success = recipe_graph.connect_graph_content()
        if not connect_graph_success:
            continue

        recipe_graph.clean_nodes()

        if recipe_graph.is_fully_connected():

            if recipe_graph.is_acyclic():

                # add links from nodes specific to this recipe to slightly more 'general' node names

                recipe_tree_dict[recipe_id] = recipe_graph.format_recipe_tree_output()

                #####
                full_graphs += 1
                # g, a, b, c, d = recipe_graph.get_vis_data()
                # visualize(g, a, b, c, d)

    print(f"{full_graphs} graphs")

def convert_recipe_data_to_kg(*, data, cleanup_links, ing_list, processes):
    # use queue and manager for multiprocessing
    q = mp.Queue(maxsize=processes)
    m = mp.Manager()
    recipe_to_tree = m.dict()
    obj_connection_info = m.dict()

    print("setting up nx graph for substrings/cleanup")
    unique_objects = set()
    entity_subclass_dig = nx.DiGraph()
    # SPECIAL CASE for 'ingredient', to help with nouns referring to groups of ingredients
    # eg 'add the first 5 ingredients', 'sauce ingredients'
    # http://purl.obolibrary.org/obo/FOODON_00001002 is the root for food products in foodon
    entity_subclass_dig.add_edge('ingredient', 'http://purl.obolibrary.org/obo/FOODON_00001002')
    # entity_subclass_dig.add_edge('ingredient', 'OBJ_REACHES_FOOD')
    for ing in ing_list:
        entity_subclass_dig.add_edge(ing, 'ING_ENTITY')

    # for dict_name in relevant_dicts:
    for _, match_dict in cleanup_links.items():
        for k, v_list in match_dict.items():
            if v_list:
                for v in v_list:
                    if k != v:
                        entity_subclass_dig.add_edge(k, v)
                    unique_objects.add(v)
                unique_objects.add(k)

    # set up relations in external data sources (foodon and wikidata subclasses)
    foodon_onto = rdflib.ConjunctiveGraph()
    foodon_onto.parse(str((path.DATA_DIR / "foodon_ontologies/foodon_subclasses.nq").resolve()), format='nquads')

    # add entities from FoodOn and wikidata to the graph
    for s, o in foodon_onto.subject_objects(predicate=rdflib.RDFS.subClassOf):
        entity_subclass_dig.add_edge(str(s), str(o))
        entity_subclass_dig.add_edge(str(s), 'FOODON_ENTITY')
        entity_subclass_dig.add_edge(str(o), 'FOODON_ENTITY')

    for wiki_file in ['cul_equip.nq', 'food_prep.nq']:
        wiki_kg = rdflib.ConjunctiveGraph()
        wiki_kg.parse(str((path.DATA_DIR / f"wikidata_cooking/{wiki_file}").resolve()), format='nquads')
        for pred in [rdflib.URIRef('http://www.wikidata.org/prop/direct/P31'), #instance of
                     rdflib.RDFS.subClassOf,
                     rdflib.URIRef('http://www.wikidata.org/prop/direct/P361'), # part of
                     ]:
            for s, o in wiki_kg.subject_objects(predicate=pred):
                entity_subclass_dig.add_edge(str(s), str(o))
                entity_subclass_dig.add_edge(str(s), 'WIKIDATA_ENTITY')
                entity_subclass_dig.add_edge(str(o), 'WIKIDATA_ENTITY')

    print("finished setting up nx graph")

    stime = time.time()
    print("preprocessing to collect all ingredient/object subclass info ahead of time")

    def calc_shortest_path(obj, target):
        asp = list(nx.all_shortest_paths(entity_subclass_dig, obj, target))
        sp = asp[0]
        if len(asp) > 1:
            # having multiple shortest paths should only occur if the object is not directly connected
            # and contains multiple obj/ingredients of smaller size, e.g. '1 % low fat milk' has connections
            # to 'fat' and 'milk'. here, we break the tie by choosing the shortest path using the word that is
            # furthest to the right within the word. i.e., for 'low fat milk', if it has connections to 'fat'
            # and 'milk', we choose milk since it occurrs further to the right in the object name
            min_index = -1
            for c_sp in asp:
                closest_name = c_sp[1]
                try:
                    c_index = obj.index(closest_name)
                    if c_index > min_index:
                        min_index = c_index
                        sp = c_sp
                except ValueError:
                    # if the string isn't found, that means the object is directly connected to a
                    # foodon/wikidata entity
                    # (i.e. we're looking at a path like, "salt"->"FOODON_12345"->"FOOD_ENTITY")
                    # so just set the min_index to 0 since the object is directly connected to an external entity
                    min_index = 0
                    sp = c_sp
        return sp

    for obj in unique_objects:
        ing_closest, food_closest, eqp_closest = None, None, None
        reach_ing = nx.has_path(entity_subclass_dig, obj, 'ING_ENTITY')
        min_sp = -1
        closest_link = None
        if reach_ing:
            shortest_path = calc_shortest_path(obj, 'ING_ENTITY')
            ing_closest = shortest_path[-2]
            if min_sp == -1 or len(shortest_path) < min_sp:
                min_sp = len(shortest_path)
                closest_link = ing_closest
        reach_food = nx.has_path(entity_subclass_dig, obj, 'FOODON_ENTITY')
        if reach_food:
            shortest_path = calc_shortest_path(obj, 'FOODON_ENTITY')
            food_closest = shortest_path[-2]
            if min_sp == -1 or len(shortest_path) < min_sp:
                min_sp = len(shortest_path)
                closest_link = food_closest
        reach_eqp = nx.has_path(entity_subclass_dig, obj, 'WIKIDATA_ENTITY')
        if reach_eqp:
            shortest_path = calc_shortest_path(obj, 'WIKIDATA_ENTITY')
            eqp_closest = shortest_path[-2]
            if min_sp == -1 or len(shortest_path) < min_sp:
                min_sp = len(shortest_path)
                closest_link = eqp_closest
        obj_connection_info[obj] = {
            'r_ing': reach_ing,
            'r_food': reach_food,
            'r_eqp': reach_eqp,
            'c_ing': ing_closest,
            'c_food': food_closest,
            'c_eqp': eqp_closest,
            'closest_link': closest_link,
        }

    reversed_entity_subclass_dig = entity_subclass_dig.reverse(copy=True)
    remove_nodes = ['ING_ENTITY', 'WIKIDATA_ENTITY', 'FOODON_ENTITY']
    for n in remove_nodes:
        reversed_entity_subclass_dig.remove_node(n)
    print("finished preprocessing: ", round(time.time()-stime, 5), 's')

    starttime = time.time()

    recipe_ids = data.keys()

    total_items = len(recipe_ids)
    print(f"total items: {total_items}")
    progress = 0

    aps_dict = m.dict()
    sem_dist_calc = SemSimCalculator(G=reversed_entity_subclass_dig, shared_ic_dict=aps_dict)

    pool = mp.Pool(processes, initializer=process_single_recipe, initargs=(q, recipe_to_tree,
                                                                           entity_subclass_dig,
                                                                           reversed_entity_subclass_dig,
                                                                           obj_connection_info,
                                                                           sem_dist_calc))

    for recipe_id in recipe_ids:
        recipe_data = data[recipe_id]
        q.put((recipe_id, recipe_data))

        progress += 1
        if progress % 1000 == 0:
            print(f'progress: {round(progress/total_items, 3)}, time: {time.time()-starttime}')
        # if progress == 20:#00:
        #     break

    for _ in range(processes):
        q.put((None, None))
    pool.close()
    pool.join()

    recipe_to_tree = dict(recipe_to_tree)

    print(f"{len(recipe_to_tree.keys())} graphs produced")
    return recipe_to_tree, entity_subclass_dig, dict(obj_connection_info)

def load_parsed_data(*, data_file: Path):
    with open(data_file, 'rb') as f:
        data = pickle.load(f)

    return data

def main(*, input_file: str, ingredient_file: str, output_file: str, output_file_2: str, cleanup_file: str, processes: int):
    data = load_parsed_data(data_file=input_file)

    with open(cleanup_file, 'r') as f:
        parsed_links = json.load(f)
    with open(ingredient_file, 'r') as f:
        ing_list = json.load(f)

    recipe_tree_data, entity_subclass_dig, obj_connection_info = \
        convert_recipe_data_to_kg(data=data, ing_list=ing_list, cleanup_links=parsed_links, processes=processes)
    print("finished data processing")

    print("saving data...")
    with open(output_file, 'w') as f:
        json.dump(recipe_tree_data, f)

    # a bit of a waste of time redoing this whole process, but just doing this for simplicity. todo refactor.
    print("formatting and saving subclass/relation data")
    print("nodes that can't reach any ingredient/food/equipment will be removed.")

    edges = []
    edge_labels = []
    relabels = {}
    remove_nodes = ['ING_ENTITY', 'WIKIDATA_ENTITY', 'FOODON_ENTITY']
    edges = set(entity_subclass_dig.edges())

    for nodename, data in obj_connection_info.items():
        if data['closest_link'] is None:
            remove_nodes.append(nodename)

    for node in remove_nodes:
        entity_subclass_dig.remove_node(node)
    nx.relabel_nodes(entity_subclass_dig, mapping=relabels, copy=False)
    entity_subclass_dig.remove_edges_from(nx.selfloop_edges(entity_subclass_dig))

    edges = list(entity_subclass_dig.edges())
    checkonce = True
    for e in edges:
        edge_labels.append('sub')

    print("saving data...")
    output_data = {'subclass_edges': edges, 'subclass_edge_labels': edge_labels}
    with open(output_file_2, 'w') as f:
        json.dump(output_data, f)

    print("finished")
    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--n_cpu", type=int, default=1)

    args = parser.parse_args()

    main_dir = (path.DATA_DIR / args.input_dir).resolve()
    input_file = (main_dir / "parsed_recipes.pkl").resolve()
    ingredient_file = (main_dir / "ingredient_list.json").resolve()
    output_file = (main_dir / "recipe_tree_data.json").resolve()
    output_file_2 = (main_dir / "entity_relations.json").resolve()
    cleanup_file = (main_dir / "word_cleanup_linking.json").resolve()

    main(input_file=str(input_file),
         ingredient_file=ingredient_file,
         output_file=str(output_file),
         output_file_2=str(output_file_2),
         cleanup_file=str(cleanup_file),
         processes=args.n_cpu)
