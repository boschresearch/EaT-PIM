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
import spacy
from spacy.parts_of_speech import VERB


class WikidataMatcher(Matcher):

    def __init__(self, *, mode: str, nlp: spacy.Language, min_confidence: float = 0.75):
        self.min_confidence_level = min_confidence

        lemmatizer = nlp.get_pipe("lemmatizer")
        wiki_ns = rdflib.Namespace('http://www.wikidata.org/entity/')
        g = rdflib.ConjunctiveGraph()
        if mode == 'equipment':
            filename = (path.DATA_DIR / "wikidata_cooking/cul_equip.nq").resolve()
            root_uri = wiki_ns['Q26037047']
            root_uri_cm = wiki_ns['Q57583712'] # for cooking appliance
        elif mode == 'preparation':
            filename = (path.DATA_DIR / "wikidata_cooking/food_prep.nq").resolve()
            root_uri = wiki_ns['Q16920758']
            root_uri_cm = wiki_ns['Q1039303'] # for cooking method
        else:
            return

        g.parse(str(filename), format='nquads')

        valid_uris = set()

        # subclass_graph will be used to measure the shortest distance between entities as the root note
        relation_graph = nx.DiGraph()
        subclass_graph_distances = {}

        self.label_to_uris = defaultdict(lambda: [])
        self.uris = []
        self.labels = []
        self.label_priority = []
        self.hierarchy_priority = []

        # predicates, ordered by priority of the relation (ie if two things have the same word as 'label' and 'synonym', the
        # entity with the word as it's label is deemed "more correct" than the synonym, since the label is its primary name)
        predicates = [rdflib.RDFS.label,
                      rdflib.URIRef("http://www.w3.org/2004/02/skos/core#/altLabel")
                      ]

        # RDFS label has a higher priority than altLabels. we want to get these in order so that we can use this
        # to break ties in cases where some entities share the main label with another one's alt label
        for prio_ind, pred in enumerate(predicates):
            for s, v in g.subject_objects(predicate=pred):

                if not isinstance(v, rdflib.Literal) or v.value == '':
                    # error case, a couple thing have labels that arent strings
                    # also, get rid of things that aren't english labels
                    continue

                raw_label = v.value.lower().replace(".", "")
                doc = [d for d in nlp.pipe([raw_label])][0]

                if len(doc) == 1:
                    label = str(doc[0].lemma_)
                else:
                    label_prefix = " ".join([str(w) for w in doc[:-1] if not w.is_punct])
                    label = f"{label_prefix} {str(doc[-1].lemma_)}"

                self.label_priority.append(prio_ind)
                self.label_to_uris[label].append(s)

                self.uris.append(s)
                self.labels.append(label)

                # special case for when verbs aren't correctly parsed
                # e.g. 'mixing' gets tagged as a noun when it's just on its own, so check
                # if the lemmatization of the label changes with a different POS tag.
                # if it does, add this alternate label.
                if doc[-1].pos != VERB:
                    mod_word = doc[-1]
                    mod_word.pos = VERB
                    lem_words = [lemmatizer.lookup_lemmatize(mod_word)[0], lemmatizer.rule_lemmatize(mod_word)[0]]
                    for lem_word in lem_words:
                        if lem_word != doc[-1].lemma_:
                            if len(doc) > 1:
                                alt_label = f"{label_prefix} {lem_word}"
                            else:
                                alt_label = str(lem_word)
                            self.label_priority.append(prio_ind)
                            self.label_to_uris[alt_label].append(s)
                            self.uris.append(s)
                            self.labels.append(alt_label)


        # add subclasses to the subclass_graph for processing later
        seen_uris = set()
        def get_subclasses_of(obj):
            if obj in seen_uris:
                return
            seen_uris.add(obj)

            for s in g.subjects(predicate=rdflib.RDFS.subClassOf, object=obj):
                get_subclasses_of(s)
                relation_graph.add_edge(s, obj)
            for s in g.subjects(predicate=rdflib.URIRef("http://www.wikidata.org/prop/direct/P31"), object=obj):
                get_subclasses_of(s)
                relation_graph.add_edge(s, obj)
            for s in g.subjects(predicate=rdflib.URIRef("http://www.wikidata.org/prop/direct/P361"), object=obj):
                get_subclasses_of(s)
                relation_graph.add_edge(s, obj)

        get_subclasses_of(root_uri)
        if root_uri_cm:
            get_subclasses_of(root_uri_cm)

        for ind, wiki_uri in enumerate(self.uris):
            if wiki_uri in subclass_graph_distances.keys():
                self.hierarchy_priority.append(subclass_graph_distances[wiki_uri])
            else:
                if nx.has_path(relation_graph, wiki_uri, root_uri):
                    dist_to_root = nx.shortest_path_length(relation_graph,source=wiki_uri,target=root_uri)
                else:
                    dist_to_root = nx.shortest_path_length(relation_graph, source=wiki_uri, target=root_uri_cm)
                subclass_graph_distances[wiki_uri] = dist_to_root
                self.hierarchy_priority.append(dist_to_root)

        self.vectorizer = TfidfVectorizer()
        self.label_vectors = self.vectorizer.fit_transform(self.labels)
        self.label_vec_norm = self.l2_norm(self.label_vectors)

    def match(self, *, input_str) -> Tuple[rdflib.URIRef, float]:
        new_vec = self.vectorizer.transform([input_str])
        for s in input_str.split(" "):
            if s not in self.vectorizer.vocabulary_:
                return None, 0

        new_vec_l2 = self.l2_norm(new_vec)[0]
        if new_vec_l2 == 0:
            return None, 0
        # compute cosine similarity
        dotprod = new_vec.dot(self.label_vectors.T)
        div = (new_vec_l2 * self.label_vec_norm.T)
        cosine_sim = dotprod / div

        max_score = np.max(cosine_sim)

        # matching is lower than the minimum confidence level we want to consider, so no link
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
