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

import networkx as nx
from collections import defaultdict
from typing import *


SPECIAL_CASES = {"all ingredient", "remaining ingredient", "ingredient", "everything"}


class RecipeGraph:

    def __init__(self, *, recipe_data: Dict, recipe_id, entity_subclass_dig: nx.DiGraph,
                 reversed_entity_subclass_dig: nx.DiGraph,
                 ing_connection_dict: Dict[str, Dict[str, str]], sim_calc):
        self.G = nx.DiGraph()
        self.recipe_data = recipe_data
        self.recipe_id = recipe_id
        self.entity_subclass_dig = entity_subclass_dig
        self.reversed_entity_subclass_dig = reversed_entity_subclass_dig
        self.ing_connection_dict = ing_connection_dict
        self.sim_calc = sim_calc

        self.recipe_output_name = f'RECIPE_OUTPUT_{self.recipe_id}'

        self.recipe_ingredients = recipe_data['ingredients']
        self.step_nodes = []
        self.step_node_ingredient_content = defaultdict(lambda: set())
        self.step_node_equipment_content = defaultdict(lambda: set())
        self.step_node_ing_leaves = {}
        self.step_node_eqp_leaves = {}
        self.step_node_special_leaves = {}
        self.step_node_instruction_string = {}

        self.ingredient_nodes = set()
        self.external_ingredient_nodes = set()
        self.equipment_nodes = set()
        self.foodon_nodes = set()
        self.seen_ingredients = set()

    def get_vis_data(self):
        return self.G, self.ingredient_nodes, self.external_ingredient_nodes, self.foodon_nodes, self.equipment_nodes

    def is_fully_connected(self):
        if not self.G.has_node(self.recipe_output_name):
            return False
        for node in self.G.nodes():
            if node == self.recipe_output_name:
                continue
            if not nx.has_path(self.G, node, self.recipe_output_name):
                return False
        return True

    def is_acyclic(self):
        return list(nx.simple_cycles(self.G)) == []

    def format_recipe_tree_output(self):

        edges = []
        for tup in self.G.edges():
            in_node = tup[0]
            split_in_node = in_node.split("_")
            # leaf nodes have the recipeid/stepid attached up to this point.
            # keeping the IDs helps with fixing the connections when constructing the flow graph
            # at this point, we want to rename the edges so that leaf nodes are using normal names
            # like 'potato' instead of 'potato_12345_0'.
            # nodes associated with actions (like mixing, etc) start with 'pred', and we dont want
            # to rename those.
            if split_in_node[0] == 'pred':
                edges.append((tup[0], tup[1]))
            else:
                nodename = split_in_node[0]
                edges.append((nodename, tup[1]))
        edge_labels = [" ".join(e[1].split("_")[:2]) for e in edges]

        return {'edges': edges,
                'edge_labels': edge_labels,
                'output_node': self.recipe_output_name}

    def format_subject_name(self, *, subj_name, step_id):
        return f'{subj_name}_{self.recipe_id}_{step_id}'

    def format_predicate_name(self, *, pred_name, step_id):
        return f'pred_{pred_name}_{self.recipe_id}_{step_id}'

    def update_node_linkage(self, *, leaf_node, link_node, src_node, updating_node):
        # replace the node with the input node -- i.e., if 'flour mix' is getting
        # linked to some outputs from earlier, remove 'flour mix' and replace
        # it with the output of the previous steps that are linking in
        removelist = []
        for e in self.G.out_edges(leaf_node):
            self.G.add_edge(link_node, e[1])
            removelist.append(e)
        for e in removelist:
            self.G.remove_edge(e[0], e[1])
        self.G.add_edge(link_node, leaf_node)

        self.update_node_contents(src_node=src_node, updating_node=updating_node)

    def update_node_contents(self, *, src_node, updating_node):
        self.step_node_ingredient_content[updating_node] = \
            self.step_node_ingredient_content[updating_node].union(
                self.step_node_ingredient_content[src_node])
        self.step_node_equipment_content[updating_node] = \
            self.step_node_equipment_content[updating_node].union(
                self.step_node_equipment_content[src_node])
        return

    def check_leaf_links(self, *,
                         node_raw,
                         node,
                         ingredients_in_step, ingredient_leaf_nodes,
                         equipment_in_step, equipment_leaf_nodes):
        data = self.ing_connection_dict.get(node_raw, [])
        if not data:
            return
        if data['r_ing'] or data['r_food'] or node_raw in self.recipe_ingredients:
            ingredient_leaf_nodes.add((node_raw, node))
            ingredients_in_step.add(node_raw)
        if data['r_eqp']:
            equipment_leaf_nodes.add((node_raw, node))
            equipment_in_step.add(node_raw)
        return

    def add_connection(self, *, subj_name: str, pred_name: str, word_ind: int, step_id,
                       ingredients_in_step, ingredient_leaf_nodes, equipment_in_step, equipment_leaf_nodes, special_ing_leaves,
                       ent_act_links, added_preds
                       ):
        subj = self.format_subject_name(subj_name=subj_name, step_id=step_id)
        pred = self.format_predicate_name(pred_name=pred_name, step_id=step_id)
        ent_act_links[subj][word_ind] = pred
        added_preds.add(pred)
        if subj_name in SPECIAL_CASES:
            special_ing_leaves.add(subj)

        self.check_leaf_links(node_raw=subj_name, node=subj,
                              ingredients_in_step=ingredients_in_step, ingredient_leaf_nodes=ingredient_leaf_nodes,
                              equipment_in_step=equipment_in_step, equipment_leaf_nodes=equipment_leaf_nodes)
        return subj, pred

    def parse_step_into_graph(self, *, step_id, step,):
        ingredient_leaf_nodes = set()
        equipment_leaf_nodes = set()
        ingredients_in_step = set()
        equipment_in_step = set()
        special_ing_leaves = set()
        prev_len = len(self.G.edges())

        final_step_index = -1
        final_step = None
        dobj_found = False
        ent_act_links = defaultdict(lambda: {})
        orphaned_actions = {}
        added_preds = set()
        low_prio_edges = set()

        # start by going through parsed steps that involve a subject and predicate or predicate and object.
        # also, check to see if any direct object (dobj) relation has been detected. this is relevant to
        # help with linking together the graphs with other steps.

        for a, b, i, d in step['subj_pred']:
            # acomp is an adjective complement, which we are not interested in for now
            if d == 'acomp':
                continue

            ############
            subj, pred = self.add_connection(subj_name=a, pred_name=b, word_ind=i, step_id=step_id,
                                             ingredients_in_step=ingredients_in_step,
                                             ingredient_leaf_nodes=ingredient_leaf_nodes,
                                             equipment_in_step=equipment_in_step,
                                             equipment_leaf_nodes=equipment_leaf_nodes,
                                             special_ing_leaves=special_ing_leaves,
                                             ent_act_links=ent_act_links,
                                             added_preds=added_preds
                                             )

            if i > final_step_index:
                final_step_index = i
                final_step = pred
            ############

            if d == 'dobj':
                dobj_found = True

        for a, b, i, d in step['pred_obj']:
            if d == 'acomp':
                continue

            # note subj is b and pred is a for pred_obj content
            subj, pred = self.add_connection(subj_name=b, pred_name=a, word_ind=i, step_id=step_id,
                                             ingredients_in_step=ingredients_in_step,
                                             ingredient_leaf_nodes=ingredient_leaf_nodes,
                                             equipment_in_step=equipment_in_step,
                                             equipment_leaf_nodes=equipment_leaf_nodes,
                                             special_ing_leaves=special_ing_leaves,
                                             ent_act_links=ent_act_links,
                                             added_preds=added_preds
                                             )

            if i > final_step_index:
                final_step_index = i
                final_step = pred
            ############

            if d == 'dobj':
                dobj_found = True

        apo_to_add = []
        for a, b, c, i in step['action_prep_obj']:
            # handle action_pred_obj slightly differently than the other subject-action type content
            subj_name = c
            pred_name = a

            subj = self.format_subject_name(subj_name=subj_name, step_id=step_id)
            pred = self.format_predicate_name(pred_name=pred_name, step_id=step_id)

            if a == c:
                continue

            added_preds.add(pred)
            if not self.G.has_node(pred):
                orphaned_actions[i] = pred
            apo_to_add.append((subj, pred))
            # to help avoid cycles
            low_prio_edges.add((subj, pred))
            if i > final_step_index:
                final_step_index = i
                final_step = pred

            if subj_name in SPECIAL_CASES:
                special_ing_leaves.add(subj)

            self.check_leaf_links(node_raw=subj_name, node=subj,
                                  ingredients_in_step=ingredients_in_step,
                                  ingredient_leaf_nodes=ingredient_leaf_nodes,
                                  equipment_in_step=equipment_in_step,
                                  equipment_leaf_nodes=equipment_leaf_nodes)
        for tup in apo_to_add:
            self.G.add_edge(tup[0], tup[1])

        no_final_step = False
        if final_step_index == -1:
            no_final_step = True

        # modifying_... come up when the relation is a type of adverbial clause modifier.
        # for the purposes of parsing dependency relations into flow graphs, these are not as direclty relevant as
        # non-modifying ones, so we give them slightly lower priority for things like setting the action as
        # the final node for the step

        for a, b, i in step['modifying_subj_pred']:
            # note subj is b and pred is a for pred_obj content
            subj, pred = self.add_connection(subj_name=a, pred_name=b, word_ind=i, step_id=step_id,
                                             ingredients_in_step=ingredients_in_step,
                                             ingredient_leaf_nodes=ingredient_leaf_nodes,
                                             equipment_in_step=equipment_in_step,
                                             equipment_leaf_nodes=equipment_leaf_nodes,
                                             special_ing_leaves=special_ing_leaves,
                                             ent_act_links=ent_act_links,
                                             added_preds=added_preds
                                             )

            if no_final_step and i > final_step_index:
                final_step_index = i
                final_step = pred
            ############

        for a, b, i in step['modifying_pred_obj']:
            # note subj is b and pred is a for pred_obj content
            subj, pred = self.add_connection(subj_name=b, pred_name=a, word_ind=i, step_id=step_id,
                                             ingredients_in_step=ingredients_in_step,
                                             ingredient_leaf_nodes=ingredient_leaf_nodes,
                                             equipment_in_step=equipment_in_step,
                                             equipment_leaf_nodes=equipment_leaf_nodes,
                                             special_ing_leaves=special_ing_leaves,
                                             ent_act_links=ent_act_links,
                                             added_preds=added_preds
                                             )

            if no_final_step and i > final_step_index:
                final_step_index = i
                final_step = pred
            ############

        for ent, action_dict in ent_act_links.items():
            if len(action_dict) == 1:
                s = ent
                p = list(action_dict.values())[0]
                self.G.add_edge(s, p)
                if self.G.has_edge(p, s):
                    if (p, s) in low_prio_edges:
                        self.G.remove_edge(p, s)
            else:
                sorted_actions = sorted(k for k in action_dict.keys())
                self.G.add_edge(ent, action_dict[sorted_actions[0]])
                for ind, action in enumerate(sorted_actions[1:]):
                    s = action_dict[sorted_actions[ind]]
                    p = action_dict[action]
                    self.G.add_edge(s, p)
                    if self.G.has_edge(p, s):
                        if (p, s) in low_prio_edges:
                            self.G.remove_edge(p, s)

        add_tups = set()
        for ing in self.recipe_ingredients:
            ing_data = self.ing_connection_dict.get(ing, [])

            for (leaf, leaf_node) in ingredient_leaf_nodes:
                if self.entity_subclass_dig.has_node(ing) and \
                        (nx.has_path(self.entity_subclass_dig, leaf, ing) or
                         nx.has_path(self.entity_subclass_dig, ing, leaf)):
                    ingredients_in_step.add(ing)
                    add_tups.add((ing, leaf_node))
                elif ing_data:
                    if self.ing_connection_dict[leaf]['r_food'] and ing_data['r_food']:
                        leaf_foodon = self.ing_connection_dict[leaf]['c_food']
                        ing_foodon = ing_data['c_food']
                        if (nx.has_path(self.entity_subclass_dig, leaf_foodon, ing_foodon)
                                or nx.has_path(self.entity_subclass_dig, ing_foodon, leaf_foodon)):
                            ingredients_in_step.add(ing)
                            add_tups.add((ing, leaf_node))
        for tup in add_tups:
            ingredient_leaf_nodes.add(tup)

        if len(self.G.edges()) == prev_len:
            # no new edges added
            return

        if len(self.step_nodes) > 0 and not dobj_found:
            if orphaned_actions:
                sorted_orphans = sorted(k for k in orphaned_actions.keys())
                self.G.add_edge(self.step_nodes[-1], orphaned_actions[sorted_orphans[0]])
                for ind, action_ind in enumerate(sorted_orphans[1:]):
                    self.G.add_edge(orphaned_actions[sorted_orphans[ind]], orphaned_actions[action_ind])
            else:
                self.G.add_edge(self.step_nodes[-1], final_step)
            self.step_node_ingredient_content[final_step] = ingredients_in_step.union(
                self.step_node_ingredient_content[self.step_nodes[-1]])
            self.step_node_equipment_content[final_step] = equipment_in_step.union(
                self.step_node_equipment_content[self.step_nodes[-1]])

        while True:
            final_step_out_deg = self.G.out_degree(final_step)
            if final_step_out_deg == 0:
                break
            elif final_step_out_deg == 1:
                new_final_step = [tup for tup in self.G.out_edges(final_step)][0][1]
                if nx.has_path(self.G, new_final_step, final_step):
                    # a cycle exists...
                    break
                final_step = new_final_step
            else:
                break

        for p in added_preds:
            if p != final_step and self.G.out_degree(p) == 0:
                self.G.add_edge(p, final_step)

        if final_step:
            self.step_nodes.append(final_step)
            if final_step not in self.step_node_ingredient_content.keys():
                self.step_node_ingredient_content[final_step] = ingredients_in_step
                self.step_node_equipment_content[final_step] = equipment_in_step
            self.step_node_ing_leaves[final_step] = ingredient_leaf_nodes
            self.step_node_eqp_leaves[final_step] = equipment_leaf_nodes

            self.step_node_special_leaves[final_step] = special_ing_leaves
            # step string will be used in some limited instances to break some ties
            self.step_node_instruction_string[final_step] = step['step_string']

    def connect_graph_content(self):
        for n_ind, node in enumerate(self.step_nodes):
            for ing in self.step_node_ingredient_content[node]:
                self.seen_ingredients.add(ing)
        for n_ind, node in enumerate(self.step_nodes):
            if self.step_node_special_leaves[node]:
                found_special = False
                for sl in self.step_node_special_leaves[node]:
                    raw_name = sl.split("_")[0]
                    if raw_name in SPECIAL_CASES:
                        found_special = True
                        removelist = []
                        for e in self.G.out_edges(sl):
                            removelist.append(e)
                            for ing in self.recipe_ingredients - self.seen_ingredients:
                                formatted_node = self.format_subject_name(subj_name=ing, step_id=n_ind)
                                self.G.add_edge(formatted_node, e[1])
                                self.step_node_ingredient_content[node].add(ing)
                                self.step_node_ing_leaves[node].add((ing, formatted_node))
                        for e in removelist:
                            self.G.remove_edge(e[0], e[1])
                # update the list of seen ingredients after checking all the special nodes
                # to handle edge cases where a step contains multiple instances of
                # 'all ingredients', 'ingredients', etc.
                if found_special:
                    self.seen_ingredients = self.recipe_ingredients
            if self.G.out_degree(node) >= 1:
                continue
            linked = False

            for nn_ind, next_node in enumerate(self.step_nodes[n_ind + 1:]):
                if linked:
                    break
                link_candidates = []
                # prioritize linking to ingredients
                for (leaf, leaf_node) in self.step_node_ing_leaves[next_node]:
                    for ing in self.step_node_ingredient_content[node]:
                        if not self.entity_subclass_dig.has_node(ing) and leaf != ing:
                            continue
                        if nx.has_path(self.entity_subclass_dig, leaf, ing) or \
                                nx.has_path(self.entity_subclass_dig, ing, leaf):
                            link_candidates.append((leaf_node, ing, self.sim_calc.sim(a=ing, b=leaf)))

                if not link_candidates:
                    for (leaf, leaf_node) in self.step_node_eqp_leaves[next_node]:
                        for eqp in self.step_node_equipment_content[node]:
                            if not self.entity_subclass_dig.has_node(eqp) and leaf != eqp:
                                continue
                            if nx.has_path(self.entity_subclass_dig, leaf, eqp) or \
                                    nx.has_path(self.entity_subclass_dig, eqp, leaf):
                                link_candidates.append((leaf_node, eqp, self.sim_calc.sim(a=eqp, b=leaf)))
                # if no direct ingredient/equipment links, check for links through foodon/wikidata
                if not link_candidates:
                    for (leaf, leaf_node) in self.step_node_ing_leaves[next_node]:
                        if not self.ing_connection_dict[leaf]['r_food']:
                            continue
                        leaf_closest_food = self.ing_connection_dict[leaf]['c_food']
                        for ing in self.step_node_ingredient_content[node]:
                            ing_data = self.ing_connection_dict.get(ing, [])
                            if not ing_data or not ing_data['r_food']:
                                continue
                            ing_closest_food = ing_data['c_food']

                            if nx.has_path(self.entity_subclass_dig, leaf_closest_food, ing_closest_food) or \
                                    nx.has_path(self.entity_subclass_dig, ing_closest_food, leaf_closest_food):
                                link_candidates.append((leaf_node, ing, self.sim_calc.sim(a=ing_closest_food, b=leaf_closest_food)))

                if link_candidates:
                    if len(link_candidates) > 1:
                        link_candidates.sort(key=lambda x: x[2])
                        link_candidates = link_candidates[::-1]
                        best_score = link_candidates[0][2]
                        # if multiple best links exist with the same similarity, perform the linking for all of them
                        for cand in link_candidates:
                            if cand[2] != best_score:
                                break
                            else:
                                link_leaf = cand[0]
                                self.update_node_linkage(leaf_node=link_leaf, link_node=node,
                                                         src_node=node, updating_node=next_node)

                    else:
                        link_leaf = link_candidates[0][0]
                        self.update_node_linkage(leaf_node=link_leaf, link_node=node,
                                                 src_node=node, updating_node=next_node)

                    linked = True
                    break
            if not linked and n_ind < len(self.step_nodes) - 1:
                self.G.add_edge(node, self.step_nodes[n_ind + 1])
        if not self.step_nodes:
            return False

        self.G.add_edge(self.step_nodes[-1], self.recipe_output_name)
        return True

    def clean_nodes(self):
        # cleanup to remove nodes that aren't contributing
        remove_nodes = []
        for node in self.G.nodes():
            if node == self.recipe_output_name:
                continue
            if self.G.out_degree(node) == 0:
                remove_nodes.append(node)
        for node in remove_nodes:
            self.G.remove_node(node)

        # replace nodes connecting to ingredients with the ingredient nodes
        # i.e. if the ingredients specify "horseradish" and the graph has "prepared horseradish",
        # replace so that "horseradish" takes its place
        # similarly, check through and connect all remaining ingredients that haven't been linked yet.
        leftover_ings = self.recipe_ingredients - self.seen_ingredients
        name_to_node = defaultdict(lambda: [])
        for node in self.step_nodes:
            for (leaf, leaf_node) in self.step_node_ing_leaves[node]:
                name_to_node[leaf].append(leaf_node)

        ing_to_closest = {}
        for ing in leftover_ings:
            ing_to_closest[ing] = {'sim': -1, 'link': []}
        for n in name_to_node.keys():
            if n not in self.reversed_entity_subclass_dig.nodes():
                continue
            for ing in leftover_ings:
                if ing not in self.reversed_entity_subclass_dig.nodes():
                    continue
                sim = self.sim_calc.sim(a=ing, b=n)

                if ing_to_closest[ing]['sim'] == -1 or sim > ing_to_closest[ing]['sim']:
                    ing_to_closest[ing]['sim'] = sim
                    ing_to_closest[ing]['link'] = [n]
                elif sim == ing_to_closest[ing]['sim']:
                    ing_to_closest[ing]['link'].append(n)

        for k in ing_to_closest.keys():
            closest_links = ing_to_closest[k]['link']
            if not closest_links:
                continue
            for closest_link in closest_links:
                for node in name_to_node[closest_link]:
                    node_edges = self.G.out_edges(node)

                    for e in node_edges:
                        self.G.add_edge(k, e[1])

        remove_nodes = []
        for node in self.step_nodes:
            for (leaf, leaf_node) in self.step_node_ing_leaves[node]:
                if not self.G.has_node(leaf_node):
                    continue
                leaf_edges = self.G.out_edges(leaf_node)
                for e in leaf_edges:
                    if leaf in self.recipe_ingredients:
                        self.G.add_edge(leaf, e[1])
                remove_nodes.append(leaf_node)
            for (leaf, leaf_node) in self.step_node_eqp_leaves[node]:
                if not self.G.has_node(leaf_node):
                    continue
                leaf_edges = self.G.out_edges(leaf_node)
                for e in leaf_edges:
                    self.G.add_edge(leaf, e[1])
                remove_nodes.append(leaf_node)

        for n in remove_nodes:
            if self.G.has_node(n):
                self.G.remove_node(n)

        # remove leaf nodes that aren't ingredients #or equipment
        good_leaves = set()
        while True:
            remove_nodes = []
            for node in self.G.nodes():
                if self.G.in_degree(node) == 0 and node not in good_leaves:
                    node_name = node.split("_")[0]
                    if node_name in SPECIAL_CASES:
                        continue
                    # remove nodes if they (1) aren't in the entity_subclass_dig, which should contain
                    # all ingredient/equipment/foods, (2) can't connect to either an ingredient
                    # or equipment, or (3) aren't equipment and aren't connected to an ingredient
                    # in this recipe
                    if not self.entity_subclass_dig.has_node(node_name):
                        remove_nodes.append(node)
                    elif not (
                            node in self.recipe_ingredients or
                            nx.has_path(self.entity_subclass_dig, node_name, 'ING_ENTITY')
                                 # or nx.has_path(self.entity_subclass_dig, node_name, 'WIKIDATA_ENTITY')
                                     # or nx.has_path(self.entity_subclass_dig, node_name, 'FOOD_ENTITY')
                            ):
                        remove_nodes.append(node)
                    else:

                        # one last case - remove ingredients that only are connecting to external nodes
                        # and not ingredients in this recipe
                        connection_data = self.ing_connection_dict.get(node, [])
                        if not connection_data:
                            remove_nodes.append(node)
                            continue
                        # this node is an ingredient - check that it is an ingredient in this recipe.
                        # if not, remove.
                        if connection_data['r_ing'] and\
                                connection_data['closest_link'] == connection_data['c_ing'] and\
                                connection_data['closest_link'] != connection_data['c_eqp']:
                            if node not in self.recipe_ingredients:
                                remove_nodes.append(node)
                                continue
                        good_leaves.add(node)
                        # link_prio mostly just used for visualization for now
                        link_prio = -1
                        for ing in self.recipe_ingredients:
                            if not self.entity_subclass_dig.has_node(ing):
                                continue
                            if nx.has_path(self.entity_subclass_dig, node_name, ing):
                                link_prio = 3
                                break
                        if link_prio != 3 and self.ing_connection_dict.get(node_name, []):
                            icd = self.ing_connection_dict[node_name]
                            if icd['closest_link']:
                                cl = icd['closest_link']
                                if icd['c_ing'] == cl:
                                    link_prio = 2
                                elif icd['c_eqp'] == cl:
                                    link_prio = 1
                                # elif icd['c_food'] == cl:
                                #     link_prio = 0
                        if link_prio == 3:
                            self.ingredient_nodes.add(node)
                        elif link_prio == 2:
                            self.external_ingredient_nodes.add(node)
                        elif link_prio == 1:
                            self.equipment_nodes.add(node)
                        # elif link_prio == 0:
                        #     self.foodon_nodes.add(node)
            for node in remove_nodes:
                self.G.remove_node(node)
            if len(remove_nodes) == 0:
                break
