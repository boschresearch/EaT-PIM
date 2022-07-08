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

from eatpim.utils import path
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import defaultdict
import rdflib
from typing import Tuple
import networkx as nx
from eatpim.etl.linkers import Matcher
from typing import List


class FoodOnMatcher(Matcher):

    def __init__(self, *,
                 min_confidence: float = 0.75):
        self.min_confidence_level = min_confidence
        g = rdflib.Graph()
        onto_files = [str(f) for f in (path.DATA_DIR / "foodon_ontologies/").resolve().glob("*.owl")]
        for onto_file in onto_files:
            g.parse(onto_file)

        gene_o_ns = rdflib.Namespace('http://www.geneontology.org/formats/oboInOwl#')
        obo_ns = rdflib.Namespace('http://purl.obolibrary.org/obo/')

        valid_uris = set()
        query_food_products = g.query("""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX obo: <http://purl.obolibrary.org/obo/> 
        SELECT ?s
        WHERE {
        ?s (rdfs:subClassOf)* obo:FOODON_00001002 .
        }""")
        for res in query_food_products:
            valid_uris.add(res.s)

        # subclass_graph will be used to measure the shortest distance between entities as the root note
        subclass_graph = nx.DiGraph()
        root_uri = obo_ns['FOODON_00001002']
        subclass_graph_distances = {}

        self.label_to_uris = defaultdict(lambda: [])
        self.uris = []
        self.labels = []
        self.label_priority = []
        self.hierarchy_priority = []

        # predicates, ordered by priority of the relation (ie if two things have the same word as 'label' and 'synonym', the
        # entity with the word as it's label is deemed "more correct" than the synonym, since the label is its primary name)
        predicates = [rdflib.RDFS.label,
                      gene_o_ns['hasExactSynonym'],
                      gene_o_ns['hasSynonym'],
                      ]

        for prio_ind, pred in enumerate(predicates):
            for s, v in g.subject_objects(predicate=pred):
                if s not in valid_uris:
                    # we only want subclasses of food products
                    continue

                if not isinstance(v, rdflib.Literal) or v.language != "en":
                    # error case, a couple thing have labels that arent strings
                    # also, get rid of things that aren't english labels
                    continue

                self.label_priority.append(prio_ind)

                self.label_to_uris[v.value].append(s)

                self.uris.append(s)
                self.labels.append(v.value)

                # special case - in foodon, common food products often are labeled as "x food product"
                # e.g., 'potato food product' encompasses all potato subclasses
                # colloquial usage of a term like 'potato' can encompass a variety of different types of potatos,
                # but 'potato food product' is the super class of 'potato', 'red potato', etc, so if potato gets linked
                # to a entity that's a subclass of potato food product then we won't be able to see that 'red potato' is
                # a subclass of potato, and so on with similar ingredients that have names like 'x food product'
                # thus, if we find 'food product' in the label, we also want to add another label which removes the
                # 'food product' portion of the name.
                if len(v.value) > 13 and v.value[-13:] == ' food product':
                    # everything besides the label is the same as the original
                    self.label_priority.append(prio_ind)
                    self.label_to_uris[v.value].append(s)
                    self.uris.append(s)
                    self.labels.append(v.value[:-13])


        # add subclasses to the subclass_graph for processing later
        seen_uris = set()
        fg = rdflib.ConjunctiveGraph(identifier=rdflib.URIRef("http://bproj.com/RecipeInfoKG/FoodOnRelations"))
        def get_subclasses_of(obj):
            if obj in seen_uris:
                return
            seen_uris.add(obj)
            fg.add((obj, rdflib.RDF.type, rdflib.URIRef("http://bproj.com/RecipeInfoOnto/FoodOnEntity")))
            for s in g.subjects(predicate=rdflib.RDFS.subClassOf, object=obj):
                get_subclasses_of(s)
                subclass_graph.add_edge(s, obj)
                fg.add((s, rdflib.RDFS.subClassOf, obj))
        get_subclasses_of(root_uri)
        fg.serialize(str((path.DATA_DIR / "foodon_ontologies/foodon_subclasses.nq").resolve()), format='nquads')

        for ind, foodon_uri in enumerate(self.uris):
            if foodon_uri in subclass_graph_distances.keys():
                self.hierarchy_priority.append(subclass_graph_distances[foodon_uri])
            else:
                dist_to_root = nx.shortest_path_length(subclass_graph,source=foodon_uri,target=root_uri)
                subclass_graph_distances[foodon_uri] = dist_to_root
                self.hierarchy_priority.append(dist_to_root)

        self.vectorizer = TfidfVectorizer()
        self.label_vectors = self.vectorizer.fit_transform(self.labels)
        self.label_vec_norm = self.l2_norm(self.label_vectors)

    def match(self, *, input_str) -> Tuple[rdflib.URIRef, float]:
        new_vec = self.vectorizer.transform([input_str])
        new_vec_l2 = self.l2_norm(new_vec)[0]
        if new_vec_l2 == 0:
            return None, 0
        # compute cosine similarity
        dotprod = new_vec.dot(self.label_vectors.T)
        div = (new_vec_l2 * self.label_vec_norm.T)
        cosine_sim = dotprod / div

        max_score = np.max(cosine_sim)

        # using a cutoff, which would indicate that something really isn't matching well
        if max_score < self.min_confidence_level:
            return None, np.max(cosine_sim)

        sorted_indexes = np.argsort(cosine_sim).tolist()[0][::-1]

        top_match_labels = [self.labels[sorted_indexes[0]]]
        top_match_indexes = [sorted_indexes[0]]

        # check the rest of the sorted matches for equal match scores
        for i in range(1, len(self.labels)):
            if cosine_sim[0, sorted_indexes[i]] == max_score:
                top_match_labels.append(self.labels[sorted_indexes[i]])
                top_match_indexes.append(sorted_indexes[i])
            else:
                break

        # no ties
        if len(top_match_labels) == 1:
            return self.uris[top_match_indexes[0]], max_score

        # when ties occur, resolve ties by (1) choosing the URI who's text match came from a more preferable
        # predicate type, like rdfs:label, and if that doesn't resolve the tie (2) choose the entity that is
        # higher up on the hierarchy.
        min_prio = self.label_priority[top_match_indexes[0]]
        min_prio_indexes = []

        for ind in top_match_indexes:
            if self.label_priority[ind] < min_prio:
                min_prio = self.label_priority[ind]
        for ind in top_match_indexes:
            if self.label_priority[ind] == min_prio:
                min_prio_indexes.append(ind)

        if len(min_prio_indexes) == 1:
            return self.uris[min_prio_indexes[0]], max_score

        min_prio = self.hierarchy_priority[top_match_indexes[0]]
        min_prio_index = top_match_indexes[0]

        # if there's any more ties past this point we don't really have a good way to choose the winner
        for ind in top_match_indexes:
            if self.hierarchy_priority[ind] < min_prio:
                min_prio = self.hierarchy_priority[ind]
                min_prio_index = ind
        return self.uris[min_prio_index], max_score
