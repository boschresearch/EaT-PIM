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
import numpy as np
import argparse
from collections import defaultdict
import networkx as nx
from sklearn.model_selection import train_test_split
import rdflib
import time


def main(*, input_file: Path,
         ingredient_file: Path,
         output_dir: Path, cleanup_file: Path):
    # currently restricting to only use ingredients for the final training of embeddings
    with open(ingredient_file, 'r') as f:
        ingredient_set = set(json.load(f))

    entity_names = set()
    entity_names.add("Recipe")
    entity_names.add("Ingredient")
    entity_names.add("FoodOnFood")
    edge_names = set()
    edge_names.add('hasNode')
    edge_names.add('RECIPE OUTPUT')
    edge_names.add(str(rdflib.RDFS.subClassOf))
    edge_names.add(str(rdflib.RDF.type))

    formatted_triple_data = []

    with open(input_file, 'r') as f:
        recipe_data = json.load(f)
    print(f"processing {len(recipe_data.items())} recipes")
    progress = 0
    starttime = time.time()
    for recipe, data in recipe_data.items():
        recG = nx.DiGraph()
        edges = data['edges']
        entity_names.add(data['output_node'])
        node_set = set()
        for e in edges:
            # order is reversed for more easy traversal
            recG.add_edge(e[1], e[0])

        # remove any leafs in the graph that aren't ingredients.
        # performing this step to remove all equipment for current experimental setup
        while True:
            remove_nodes = []
            for n in recG.nodes():
                # this is checking for out degree here, since the graph's edges are all swapped in this setup
                if recG.out_degree(n) == 0 and n not in ingredient_set:
                    remove_nodes.append(n)
            if not remove_nodes:
                break
            for n in remove_nodes:
                recG.remove_node(n)

        # empty graph - possibly due to removing non-ingredient nodes.
        # in this case, just move on
        if data['output_node'] not in recG:
            continue

        def get_pred_items(G, n):
            output = []
            pred_name = " ".join(n.split("_")[:2])
            edge_names.add(pred_name)
            for to_node, from_node in recG.out_edges(n):
                entity_names.add(to_node)
                entity_names.add(to_node)
                formatted_triple_data.append((from_node, pred_name, to_node))
                if from_node[:4] == 'pred':
                    output.append(get_pred_items(G, n=from_node))
                else:
                    output.append(from_node)
                    entity_names.add(from_node)
                    node_set.add(from_node)
                    formatted_triple_data.append((str(from_node), str(rdflib.RDF.type), str("Ingredient")))
        get_pred_items(recG, data['output_node'])

        for node in node_set:
            formatted_triple_data.append((data['output_node'], 'hasNode', node))
        formatted_triple_data.append((data['output_node'], str(rdflib.RDF.type), str("Recipe")))

        progress += 1
        if progress % 1000 == 0:
            print(f"progress: {round(progress/len(recipe_data.keys()), 4)} - {round(time.time()-starttime,3)}seconds")
    print("finished processing recipes")

    print("processing subclass/relation data")
    with open(cleanup_file, 'r') as f:
        cleanup_links = json.load(f)

    # currently only using information about ingredients
    # other lines can be uncommented if equipment is also being considered more properly
    relevant_dicts = [
        # 'obj_to_subobj', 'obj_to_ing',
        # 'obj_to_foodon', 'obj_to_equipment',
        'ing_to_foodon', 'ing_to_ing'
    ]
    for dict in relevant_dicts:
        for k, v_content in cleanup_links[dict].items():
            if k not in entity_names:
                continue
            if isinstance(v_content, list):
                if not v_content:
                    continue
                for v in v_content:
                    if v not in entity_names:
                        continue
                    formatted_triple_data.append((k, str(rdflib.RDFS.subClassOf), v))
            else:
                v = v_content
                if v not in entity_names:
                    continue
                if k != v:
                    formatted_triple_data.append((k, str(rdflib.RDFS.subClassOf), v))
    # set up relations in external data sources (foodon and wikidata subclasses)
    foodon_onto = rdflib.ConjunctiveGraph()
    foodon_onto.parse(str((path.DATA_DIR / "foodon_ontologies/foodon_subclasses.nq").resolve()), format='nquads')

    for s, o in foodon_onto.subject_objects(predicate=rdflib.RDFS.subClassOf):
        formatted_triple_data.append((str(s), str(rdflib.RDFS.subClassOf), str(o)))
        formatted_triple_data.append((str(s), str(rdflib.RDF.type), str("FoodOnFood")))
        formatted_triple_data.append((str(o), str(rdflib.RDF.type), str("FoodOnFood")))
        entity_names.add(str(s))
        entity_names.add(str(o))

    formatted_triple_data = list(set(formatted_triple_data))
    print("finished processing relation data")

    trip_train, trip_test = train_test_split(formatted_triple_data, test_size=0.15)
    trip_train, trip_val = train_test_split(trip_train, test_size=0.17647)
    print(f"writing results, {len(formatted_triple_data)} total triple data ")


    entity_names = list(entity_names)
    edge_names = list(edge_names)
    entity_to_index = {}
    edge_to_index = {}
    for ind, ent in enumerate(entity_names):
        entity_to_index[ent] = ind
    for ind, edge in enumerate(edge_names):
        edge_to_index[edge] = ind

    sorted_entities = sorted(entity_to_index.items(), key=lambda kv: kv[1])
    print(f'saving entity dict - {len(entity_to_index.keys())} entities')
    with open((output_dir / "entities.dict").resolve(), 'w') as f:
        for i in range(len(sorted_entities)-1):
            f.write(f"{sorted_entities[i][1]}\t{sorted_entities[i][0]}\n")
        f.write(f"{sorted_entities[-1][1]}\t{sorted_entities[-1][0]}")

    sorted_relations = sorted(edge_to_index.items(), key=lambda kv: kv[1])
    print(f'saving relation dict - {len(edge_to_index.keys())} relation types')
    with open((output_dir / "relations.dict").resolve(), 'w') as f:
        for i in range(len(sorted_relations)-1):
            f.write(f"{sorted_relations[i][1]}\t{sorted_relations[i][0]}\n")
        f.write(f"{sorted_relations[-1][1]}\t{sorted_relations[-1][0]}")

    print("saving triples")
    #### for some reason some duplicate entries seem to leak into test/validation sets
    #### so do this to ensure that entries going into the test/val datasets do not contain
    #### any of the same entries as the triples in train.
    train_set = set(trip_train)
    test_set = set(trip_test)
    val_set = set(trip_val)
    test_overlap = test_set.intersection(train_set)
    val_overlap = val_set.intersection(train_set)
    with open((output_dir / "trip_test.txt").resolve(), 'w') as f:
        for i in range(len(trip_test)-1):
            if trip_test[i] not in test_overlap:
                f.write(f"{trip_test[i][0]}\t{trip_test[i][1]}\t{trip_test[i][2]}\n")
        f.write(f"{trip_test[-1][0]}\t{trip_test[-1][1]}\t{trip_test[-1][2]}")

    with open((output_dir / "trip_valid.txt").resolve(), 'w') as f:
        for i in range(len(trip_val)-1):
            if trip_val[i] not in val_overlap:
                f.write(f"{trip_val[i][0]}\t{trip_val[i][1]}\t{trip_val[i][2]}\n")
        f.write(f"{trip_val[-1][0]}\t{trip_val[-1][1]}\t{trip_val[-1][2]}")

    with open((output_dir / "trip_train.txt").resolve(), 'w') as f:
        for i in range(len(trip_train)-1):
            f.write(f"{trip_train[i][0]}\t{trip_train[i][1]}\t{trip_train[i][2]}\n")
        f.write(f"{trip_train[-1][0]}\t{trip_train[-1][1]}\t{trip_train[-1][2]}")

    print("finished setting up formatting as triples for embedding code")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--n_cpu", type=int, default=1)

    args = parser.parse_args()

    main_dir = (path.DATA_DIR / args.input_dir).resolve()
    input_file = (main_dir / "recipe_tree_data.json").resolve()
    ingredient_file = (main_dir / "ingredient_list.json").resolve()
    output_dir = (main_dir / "baseline_only_triple_data/").resolve()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cleanup_file = (main_dir / "word_cleanup_linking.json").resolve()

    main(input_file=input_file,
         ingredient_file=ingredient_file,
         output_dir=output_dir,
         cleanup_file=cleanup_file)
