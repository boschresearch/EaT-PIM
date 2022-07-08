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
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import spacy
spacy.prefer_gpu()
nlp = spacy.load("en_core_web_trf")


def visualize(G):
    pos = nx.nx_pydot.graphviz_layout(G, prog="neato")
    plt.figure(1, figsize=(11, 11))

    nx.draw(G, pos, node_size=600)
    # nx.draw_networkx_nodes(G, pos, nodelist=list(true_ing_nodes), node_color='black')
    nx.draw_networkx_labels(G, pos)
    plt.show()

data_dir_1 = Path('data/r-100/')
data_dir_2 = Path('data/r-200/')

recipe_ids = []
recipe_texts = []
recipe_graphs = []
recipe_names = []
recipe_ings = []

recipe_files = list(data_dir_1.rglob("*.list"))
recipe_files.extend(list(data_dir_2.rglob("*.list")))

for rf in recipe_files:
    recipe_id = len(recipe_ids)
    print(rf)
    flow_file = str(rf)[:-5]+".flow"
    print(flow_file)
    id_to_word = {}
    id_to_sentencenum = {}
    id_to_pos = {}
    id_to_type = {}
    edge_to_link_type = {}
    recipe_directions = []
    prev_sentence_step = 0
    prev_sentence_num = 0
    next_sentence = ""
    cont_ing_id = 0

    with open(rf, 'r', encoding='utf-8') as f:
        for line in f:
            if not line:
                continue
            contents = line[:-1].split(" ")
            s_step = int(contents[0])
            s_num = int(contents[1])
            word_id = f"{contents[0]} {contents[1]} {contents[2]}"
            word = contents[3]
            id_to_word[word_id] = word
            id_to_sentencenum[word_id] = len(recipe_directions)
            id_to_pos[word_id] = contents[4]
            wordtype = contents[5]
            id_to_type[word_id] = wordtype
            if wordtype == "F-B":
                cont_ing_id = word_id
            elif wordtype == "F-I":
                id_to_word[cont_ing_id] = id_to_word[cont_ing_id]+" "+word
            else:
                cont_ing_id = 0
            if s_step != prev_sentence_step or s_num != prev_sentence_num:
                if next_sentence != "":
                    recipe_directions.append(next_sentence)
                prev_sentence_step = s_step
                prev_sentence_num = s_num
                next_sentence = word
            else:
                next_sentence += " "+word
        recipe_directions.append(next_sentence)

    for step in recipe_directions:
        print(step)
    id_G = nx.DiGraph()
    G = nx.DiGraph()
    with open(flow_file, 'r', encoding="ISO-8859-1") as f:
        for line in f:

            contents = line[:-1].split(" ")
            if len(contents) != 7 or contents[0] == "#":
                print("SKIP: ", contents)
                continue
            link_type = contents[3]
            from_word = f"{contents[0]} {contents[1]} {contents[2]}"
            to_word = f"{contents[4]} {contents[5]} {contents[6]}"

            from_word_txt = f"{id_to_word[from_word]}_{recipe_id}_{id_to_sentencenum[from_word]}"
            to_word_txt = f"{id_to_word[to_word]}_{recipe_id}_{id_to_sentencenum[to_word]}"
            G.add_edge(from_word_txt, to_word_txt)
            id_G.add_edge(from_word, to_word)
            edge_to_link_type[(from_word, to_word)] = link_type
            # print(from_word, id_to_word[from_word], to_word, id_to_word[to_word], link_type)
    if len(list(nx.simple_cycles(G))) > 0:
        print(f"WARNING cycle exists in {flow_file}")
    if len(G.nodes()) == 0:
        print("no nodes in ", flow_file)
        continue
    print(len(G))

    while True:
        prev_len = len(id_G)
        remove_nodes = []
        for node in id_G.nodes():
            if id_G.in_degree(node) == 0:
                if id_to_type[node] not in {"F", "F-I", "F-B"}:
                    remove_nodes.append(node)
        for rn in remove_nodes:
            id_G.remove_node(rn)

        remove_nodes = []
        for node in id_G.nodes():
            if id_to_type[node][:2] in {"Sf", "St"}:
                remove_nodes.append(node)
        for n in remove_nodes:
            in_edges = id_G.in_edges(n)
            out_edges = id_G.out_edges(n)
            if len(in_edges) > 0 and len(out_edges) > 0:
                print(f"! ERROR CASE 1 for {id_to_word[n]}, ID {n}, flow graph {flow_file}")
            id_G.remove_node(n)

        remove_nodes = []
        for node in id_G.nodes():
            if id_to_type[node][0] == "T":
                remove_nodes.append(node)
        for n in remove_nodes:
            in_edges = id_G.in_edges(n)
            out_edges = id_G.out_edges(n)
            if len(in_edges) > 1 and len(out_edges) > 1:
                print(f"! ERROR CASE 2 for {id_to_word[n]}, ID {n}, flow graph {flow_file}")
            elif len(in_edges) == 1 and len(out_edges) == 1:
                in_edge = list(id_G.in_edges(n))[0][0]
                out_edge = list(id_G.out_edges(n))[0][1]
                id_G.add_edge(in_edge, out_edge)
            id_G.remove_node(n)

        remove_nodes = []
        rem_edges = []
        for e in id_G.edges():
            if edge_to_link_type.get(e, '') == "f-eq":
                remove_nodes.append(e[1])
        for n in remove_nodes:
            if n not in id_G:
                continue
            in_edges = [i for (i,_) in id_G.in_edges(n)]
            out_edges = [o for (_, o) in id_G.out_edges(n)]
            for in_n in in_edges:
                for out_n in out_edges:
                    if in_n in id_G.nodes() and out_n in id_G.nodes():
                        id_G.add_edge(in_n, out_n)
            id_G.remove_node(n)

        remove_nodes = []
        for e in id_G.edges():
            if edge_to_link_type.get(e, '_')[0] == "v":
                remove_nodes.append(e[0])
        for n in remove_nodes:
            id_G.remove_node(n)

        remove_nodes = []
        for e in id_G.edges():
            if edge_to_link_type.get(e, '_') == "f-part-of":
                remove_nodes.append(e[1])
        for n in remove_nodes:
            if n not in id_G.nodes():
                continue
            in_edges = id_G.in_edges(n)
            out_edges = id_G.out_edges(n)
            if len(in_edges) > 1:
                print(f"! ERROR CASE 311 for {id_to_word[n]}, ID {n}, flow graph {flow_file}")
            if len(out_edges) > 1:
                print(f"! ERROR CASE 322 for {id_to_word[n]}, ID {n}, flow graph {flow_file}")
            if len(in_edges) > 1 and len(out_edges) > 1:
                print(f"! ERROR CASE 3 for {id_to_word[n]}, ID {n}, flow graph {flow_file}")
            in_edges = [e[0] for e in id_G.in_edges(n)]
            out_edges = [e[1] for e in id_G.out_edges(n)]
            for i_n in in_edges:
                for o_n in out_edges:
                    if in_n in id_G.nodes() and o_n in id_G.nodes():
                        id_G.add_edge(i_n, o_n)
            id_G.remove_node(n)

        remove_nodes = []
        for node in id_G.nodes():
            if id_to_type[node][0] == "Q":
                remove_nodes.append(node)
        for n in remove_nodes:
            id_G.remove_node(n)

        if len(id_G) == prev_len:
            break

    ancestor_node = None
    most_ancestors = 0
    for node in id_G.nodes:
        if id_G.out_degree(node) == 0:
            anc_count = len(nx.ancestors(id_G, node))
            if anc_count > most_ancestors:
                most_ancestors = anc_count
                ancestor_node = node
    remove_nodes = []
    for node in id_G:
        if not nx.has_path(id_G, node, ancestor_node):
            remove_nodes.append(node)
    for n in remove_nodes:
        id_G.remove_node(n)

    action_nodes = []
    for node in id_G.nodes():
        if id_to_type[node][0] == "A":
            action_nodes.append(node)
    inter_id = 0
    for n in action_nodes:
        in_edges = id_G.in_edges(n)
        out_edges = id_G.out_edges(n)
        doc = nlp(f"you {id_to_word[n]}")
        pred_lem = str(doc[1].lemma_).lower()

        new_node = f"pred {pred_lem}_{n}"
        inter_id += 1
        for (in_n, _) in in_edges:
            id_G.add_edge(in_n, new_node)
        for (_, out_n) in out_edges:
            id_G.add_edge(new_node, out_n)
        id_G.remove_node(n)


    G = nx.DiGraph()
    for (from_word, to_word) in id_G.edges:
        if from_word in id_to_word.keys():
            from_word_txt = f"{id_to_word[from_word]}_{recipe_id}_{id_to_sentencenum[from_word]}"
        else:
            from_word_txt = from_word
        if to_word in id_to_word.keys():
            to_word_txt = f"{id_to_word[to_word]}_{recipe_id}_{id_to_sentencenum[to_word]}"
        else:
            to_word_txt = to_word
        G.add_edge(from_word_txt, to_word_txt)
    if len(G.nodes()) == 0:
        print("no nodes remaining in ", flow_file)
        continue

    ingredient_names = []
    ing_node_relabels = {}
    for node in G.nodes():
        if G.in_degree(node) == 0:
            ing_name = node.split("_")[0]
            doc = nlp(ing_name)
            word = ""
            for w in doc[:-1]:
                word += str(w).lower() + " "
            word += str(doc[-1].lemma_).lower()
            ing_node_relabels[node] = word
            ingredient_names.append(word)
    for k,v in ing_node_relabels.items():
        for (_, o_n) in G.out_edges(k):
            G.add_edge(v, o_n)
        G.remove_node(k)
    ingredient_names = list(set(ingredient_names))

    # visualize(G)
    recipe_graphs.append(G)
    recipe_texts.append(recipe_directions)
    recipe_ids.append(recipe_id)
    recipe_ings.append(ingredient_names)
    recipe_names.append(str(rf).split("\\")[-1][:-5])

print(f'number of valid recipe graphs: {len(recipe_graphs)}')
output_data = {
    "recipe_texts": recipe_texts,
    "recipe_ids": recipe_ids,
    "recipe_graphs": recipe_graphs
}

import pickle
with open('transformed_recipe_data.pkl', 'wb') as f:
    pickle.dump(output_data, f)

import pandas as pd
recipe_data = []
for i in range(len(recipe_ids)):
    recipe_data.append([recipe_names[i], recipe_ids[i], recipe_texts[i], recipe_ings[i], ["none"],'',''])
df = pd.DataFrame(recipe_data, columns=["name", "id", "steps", "ingredients", "tags", "contributor_id", "submitted"])
df.to_csv("flowgraph_recipes.csv", index=False)
