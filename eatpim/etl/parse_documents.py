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

import spacy
from spacy.tokens.doc import Doc
from spacy.symbols import *
import pickle
from pathlib import Path
import pandas as pd
from typing import List, Tuple, Dict, Set, Union, Any
from eatpim.utils import path
from collections import defaultdict
import time
import argparse
import os
spacy.prefer_gpu()


NON_STOP_WORDS = {"all", "everything"}


def parse_documents(*, texts: List[Tuple[List[str], Dict[str, int]]]) -> Dict[int, Dict[int, Doc]]:
    nlp = spacy.load("en_core_web_trf")
    start_time = time.time()

    # dictionary of the form {recipe_id: {step_num: doc}}
    output_dict = defaultdict(lambda: {})

    count = 0
    total_count = len(texts)

    for doc, context in nlp.pipe(texts, disable=["ner", "textcat", "token2vec"], as_tuples=True):
        res = process_doc(doc)
        if (not res['subj_pred'] and
            not res['pred_obj'] and
            not res['modifying_subj_pred'] and
            not res['modifying_pred_obj']):
            # spaCy apparently has trouble parsing imperative sentences due to the lack of training data.
            # so when no verb is found in the sentence, we can try to modify it to make it easier to parse
            # e.g., "add water" can end up with a parse where "add water" is identified as a noun
            # instead, we add "you" to make the sentence be like "you add water"
            # this sometimes makes sentences look awkward grammatically, but it helps the parser
            # especially in cases where the first word is a verb
            modified_text = [('you '+res['step_string'], context)]
            newtry, context = list(nlp.pipe(modified_text, disable=["ner", "textcat", "token2vec"], as_tuples=True))[0]
            newres = process_doc(newtry)
            # if the updated parse changed enough that verbs were added, save the new result. otherwise,
            # we just keep the original.
            if len(newres['verbs']) > 0:
                # make the step's text be the same as the original, rather than the weird 'you ...' form
                newres['step_string'] = res['step_string']
                res = newres
        output_dict[context["recipe_id"]][context["step_num"]] = res
        count += 1
        if count % 1000 == 0:
            print(f"recipe progress: {round(count/total_count, 4)} : {round(time.time()-start_time, 2)}s elapsed")

    return output_dict


def parse_ings(*, texts: List[Tuple[List[str], Dict[str, int]]]) -> Dict[int, List[str]]:
    nlp = spacy.load("en_core_web_trf")
    start_time = time.time()

    count = 0
    total_count = len(texts)

    output_dict = defaultdict(lambda: set())

    for doc, context in nlp.pipe(texts, disable=["ner", "textcat", "token2vec"], as_tuples=True):
        output_dict[context["recipe_id"]].add(process_ing(doc))

        count += 1
        if count % 1000 == 0:
            print(f"ingredient progress: {round(count/total_count, 4)} : {round(time.time()-start_time, 2)}s elapsed")

    return dict(output_dict)


# convert everything from Spacy tokens to strings, and also get rid of stopwords
def clean_str(input, verb_modifier=None):

    if isinstance(input, spacy.tokens.token.Token):
        if input.lemma_ in {'be', 'let', 'use'}:
            return ''
        cleaned_str = str(input.lemma_).replace("-", "").replace("{", "").replace("}", "")
        if verb_modifier and input in verb_modifier.keys():
            cleaned_str += " "+str(verb_modifier[input]).replace("-", "").replace("{", "").replace("}", "")
        return cleaned_str

    if input[0].lemma_ in {'be', 'let', 'use'}:
        return ''

    relevant_words = [w for w in input
                      if ((not w.is_stop or str(w) in NON_STOP_WORDS)
                          and not str(w) == "-" and
                          not w.pos in {punct, PUNCT})]
    if not relevant_words:
        return ''
    elif len(relevant_words) == 1:
        cleaned_str = str(relevant_words[0].lemma_).replace("-", "").replace("{", "").replace("}", "")
        return cleaned_str

    # get rid of stay dashes, which seem relatively common
    cleaned_words = " ".join([str(w).replace("-", "").replace("{", "").replace("}","") for w in relevant_words[:-1]])
    # lemmatize the last word
    cleaned_words += f" {str(relevant_words[-1].lemma_).replace('-', '')}"
    return cleaned_words


def process_ing(doc):
    return clean_str(doc)

def process_doc(doc):
    xcomp_connector = defaultdict(lambda: set())
    prep_connector_process = defaultdict(lambda: set())
    prep_connector_obj = defaultdict(lambda: set())

    verbs = set()
    verb_modifier = {}
    all_noun_chunks = set([chunk for chunk in doc.noun_chunks])
    nounchunk_deps = {}

    subj_pred = defaultdict(lambda: set())
    pred_obj = defaultdict(lambda: set())

    conj_tups = []
    process_spec_contents = set()

    for word in doc:
        if word.dep == punct:
            continue

        if word.pos == VERB and not word.is_stop:
            verbs.add(word)

        if word.dep == conj and word.pos in {VERB}:
            # save conjunctions and deal with them later
            # information might either all fall under the 'object' for a pred_obj
            # - e.g., ('slice', 'the onions and carrots'), where carrots/onions are connected by the conjunction
            # or it might be separated out if the conjunction is a verb
            # - e.g., ('chop', 'the carrots'), ('peel', 'the carrots'), ('dice', 'the carrots'), where
            # chop, peel, and dice are connected by the conjunction.
            # conjunctions for Nouns are handled differently, by checking for noun_chunks in a subtree
            conj_tups.append((word, word.head))
        elif word.dep in {prep}:
            # connect prepositions together at a later point if the PoS tag of the head is a verb.
            # e.g., this case occurs for "add to bowl" - "to" has a prep dependency to "add"
            if word.head.pos in {VERB, AUX}:
                prep_connector_process[word].add(word.head)
        elif word.dep in {prt}:
            # prt is a phrasal verb modifier - e.g., this case occurs for "mix in the butter", where
            # "mix in" should be considered a single action. "in" has the prt relation to "stir".
            # this differentiates between a case like "mix in a bowl", where "mix" is taking place in a bowl
            # rather than having the action "mix in" occur to a bowl.
            if word.head.pos in {VERB, AUX}:
                verb_modifier[word.head] = word
        elif word.dep == xcomp:
            # xcomp is an open clausal complement dependency.
            # this situation occurs in e.g., "use the knife to cut ...", where "cut" has the xcomp relation to
            # "use". we want to be able to make a connection from "knife" to "cut", so we store this information
            # in the xcomp_connector dict and make such connections later
            if word.pos in {VERB, AUX}:
                xcomp_connector[word].add(word.head)
        elif word.dep == advcl:
            # advcl is an adverbial clause modifier dependency.
            # this is a similar situation to the xcomp dep above - in a sentence like "using a knife, cut ...",
            # "using" has the advcl relation to "cut", and we once again want to make the connection between "knife"
            # and "cut" later on.
            if word.head.pos in {VERB, AUX}:
                xcomp_connector[word.head].add(word)
        elif word.head == word:
            # word is the root of the sentence, so it has no head relation for us to consider.
            continue
        elif word.pos in {AUX}:
            # auxiliary verbs, like Tense auxiliaries: has (done), is (doing), will (do).
            # ignoring these for now.
            continue
        elif word.head.pos in {VERB, AUX}:
            if word.dep in {mark, prep, prt} and word.head.head and word.head.head != word.head:
                process_spec_contents.add((word.head.head, word, word.head))
                continue
            if word.pos in {CONJ, CCONJ, ADV, ADP, PART}:
                # ignore certain PoS tags to ignore making direct pred-obj or sub-pref connections
                # for cases like "to cut", "and cut", and also throw away adverbs
                continue
            if word.dep in {dep}:
                # 'dep' is an unspecified dependency/unable to determine
                continue

            # use spacy's built-in nounchunks to get the whole part of the noun instead of a single word
            # e.g. get the noun chunk "ground meat" instead of just "meat"
            subtree = doc[word.left_edge.i:word.right_edge.i + 1]
            noun_chunks = [nc for nc in subtree.noun_chunks]
            if noun_chunks:
                for nc in noun_chunks:
                    # pass being in the dependency indicates it is a passive voice
                    if 'subj' in word.dep_ and 'pass' not in word.dep_:
                        subj_pred[nc].add(word.head)
                    else:
                        pred_obj[word.head].add(nc)
                    nounchunk_deps[nc] = nc[-1].dep_
            else:
                # no noun chunks, just assume it's one big noun or something that's not a noun
                if 'subj' in word.dep_ and 'pass' not in word.dep_:
                    subj_pred[subtree].add(word.head)
                else:
                    pred_obj[word.head].add(subtree)
                nounchunk_deps[subtree] = subtree[-1].dep_
        elif word.head.dep == advcl:
            if word.dep in {mark, prep} and word.head.head and word.head.head != word.head:
                process_spec_contents.add((word.head.head, word, word.head))
        elif word.head.dep in {prep}:
            # the head's dependency is a preposition
            # this indicates a situation like "brown the meat in a pot", where the current word we're looking at is
            # "pot", and its head is "in", which has the prep relation to "meat"
            # we want to be able to later make the connection ("brown", "in", "pot")
            subtree = doc[word.left_edge.i:word.right_edge.i + 1]
            noun_chunks = [nc for nc in subtree.noun_chunks]
            if len(noun_chunks):
                for nc in noun_chunks:
                    prep_connector_obj[word.head].add(nc)
                    nounchunk_deps[nc] = nc[-1].dep_
            else:
                prep_connector_obj[word.head].add(subtree)
                nounchunk_deps[subtree] = subtree[-1].dep_
        elif word.head.dep in {prt} and word.head.head:
            # this case occurs in cases like "cut in the butter with a fork ...", where we want to use the action
            # as "cut in". "in" has the prt relation to "cut", and "butter" is the pobj of "in".
            subtree = doc[word.left_edge.i:word.right_edge.i + 1]
            noun_chunks = [nc for nc in subtree.noun_chunks]
            if len(noun_chunks):
                for nc in noun_chunks:
                    if 'subj' in word.dep_ and 'pass' not in word.dep_:
                        subj_pred[nc].add(word.head.head)
                    else:
                        pred_obj[word.head.head].add(nc)
                    nounchunk_deps[nc] = nc[-1].dep_
            else:
                if 'subj' in word.dep_ and 'pass' not in word.dep_:
                    subj_pred[subtree].add(word.head.head)
                else:
                    pred_obj[word.head.head].add(subtree)
                nounchunk_deps[subtree] = subtree[-1].dep_

    changed = True
    # use a while loop so that we don't accidentally fail to add info from conjunctions due to unlucky ordering
    while changed:
        changed = False
        for tup in conj_tups:
            word = tup[0]
            head = tup[1]
            if word in subj_pred.keys():
                old_len = len(subj_pred[head])
                subj_pred[head] = subj_pred[head].union(subj_pred[word])
                if len(subj_pred[head]) != old_len:
                    changed = True
            if head in subj_pred.keys():
                old_len = len(subj_pred[word])
                subj_pred[word] = subj_pred[word].union(subj_pred[head])
                if len(subj_pred[word]) != old_len:
                    changed = True

            if word in pred_obj.keys():
                old_len = len(pred_obj[head])
                pred_obj[head] = pred_obj[head].union(pred_obj[word])
                if len(pred_obj[head]) != old_len:
                    changed = True
            if head in pred_obj.keys():
                old_len = len(pred_obj[word])
                pred_obj[word] = pred_obj[word].union(pred_obj[head])
                if len(pred_obj[word]) != old_len:
                    changed = True

    for xcomp_verb, pred_set in xcomp_connector.items():
        for pred in pred_set:
            if pred in pred_obj.keys():
                for obj in pred_obj[pred]:
                    # passive subjects - don't follow through with the logic
                    if "pass" in obj[-1].dep_:
                        continue
                    subj_pred[obj].add(xcomp_verb)

                    # loop through conjunctions again to make sure all relevant conjunctions are added
                    # this probably will only loop once
                    # this logic can very likely be greatly simplified/moved elsewhere to avoid this kind of loop
                    while True:
                        xc_verbs = subj_pred[obj]
                        prev_len = len(subj_pred[obj])
                        for tup in conj_tups:
                            if tup[0] in xc_verbs:
                                subj_pred[obj].add(tup[1])
                            elif tup[1] in xc_verbs:
                                subj_pred[obj].add(tup[0])
                        if len(subj_pred[obj]) == prev_len:
                            break

    for prep_word, prep_processes in prep_connector_process.items():
        if prep_word in prep_connector_obj.keys():
            prep_objs = prep_connector_obj[prep_word]
            for obj in prep_objs:
                for process in prep_processes:
                    process_spec_contents.add((process, prep_word, obj))

    # convert everything into strings before outputting
    output_strings: Dict[str, Any] = {
        "subj_pred": set(),
        "pred_obj": set(),
        "modifying_subj_pred": set(),
        "modifying_pred_obj": set(),
        "verbs": set(),
        "root_verb": "",
        "noun_chunks": set(),
        "action_prep_obj": set(),
        "step_string": str(doc)
    }
    for v in verbs:
        if v.dep_ == 'ROOT':
            output_strings['root_verb'] = clean_str(v, verb_modifier=verb_modifier)

    # verbs are all single words.
    for k, val_set in subj_pred.items():
        for v in val_set:
            cleaned_subj = clean_str(k)
            cleaned_pred = clean_str(v, verb_modifier=verb_modifier)
            if not (cleaned_subj and cleaned_pred):
                continue
            if v.dep in {advcl}:
                output_strings["modifying_subj_pred"].add((cleaned_subj,
                                                           cleaned_pred,
                                                           v.i))
            else:
                output_strings["subj_pred"].add((cleaned_subj,
                                                 cleaned_pred,
                                                 v.i, nounchunk_deps[k]))
    for k, val_set in pred_obj.items():
        for v in val_set:
            cleaned_obj = clean_str(v)
            cleaned_pred = clean_str(k, verb_modifier=verb_modifier)
            if not (cleaned_obj and cleaned_pred):
                continue
            if k.dep in {advcl}:
                output_strings["modifying_pred_obj"].add((cleaned_pred,
                                                          cleaned_obj,
                                                          k.i))
            else:
                output_strings["pred_obj"].add((cleaned_pred,
                                                cleaned_obj,
                                                k.i, nounchunk_deps[v]))

    output_strings["verbs"] = set(clean_str(verb, verb_modifier=verb_modifier) for verb in verbs)-{''}
    output_strings["noun_chunks"] = set(clean_str(np) for np in all_noun_chunks)-{''}
    # many prepositions also are considered stopwords, but don't filter them out.
    for (s,p,o) in process_spec_contents:
        clean_s = clean_str(s, verb_modifier=verb_modifier)
        clean_p = clean_str(p)
        clean_o = clean_str(o)
        if clean_s and clean_p and clean_o:
            output_strings["action_prep_obj"].add((clean_s, clean_p, clean_o, s.i))

    return output_strings


def load_recipe_data(*, data_file: Path, recipe_choices: str, limit_n_recipes: int = -1):
    df = pd.read_csv(data_file.resolve())
    df.set_index('id', inplace=True, drop=True)
    df.drop(columns=['contributor_id', 'submitted'], inplace=True)
    # if a recipe limit is specified, randomly sample limit_n_recipes recipes from the loaded dataframe
    if limit_n_recipes != -1:
        df = df.sample(n=limit_n_recipes)
        df.to_csv(recipe_choices, index=False)
    return df


def save_process_results(*, data, output_file: Path):
    with open(output_file.resolve(), 'wb') as f:
        pickle.dump(data, f)


def main(*, input_file: str, output_file: str, output_texts: str, n_recipes: int = 0):
    raw_recipe_data = load_recipe_data(data_file=(path.DATA_DIR / input_file), limit_n_recipes=n_recipes, recipe_choices=output_texts)

    id_steps = list(zip(raw_recipe_data.index.values.tolist(), raw_recipe_data["steps"].values.tolist()))
    id_ings = list(zip(raw_recipe_data.index.values.tolist(), raw_recipe_data["ingredients"].values.tolist()))
    id_ings = [(ing, {"recipe_id": tup[0]})
               for tup in id_ings
               for ing in eval(tup[1])]

    print(f'{len(id_steps)} recipes total to be processed')
    formatted_id_steps = [(step_str, {"recipe_id": id_steps_tup[0], "step_num": step_index})
                          for id_steps_tup in id_steps
                          for step_index, step_str in enumerate(eval(id_steps_tup[1]))]
    fist = []
    for tup in formatted_id_steps:
        tup_mod = tup[0]+"."
        tup_mod = tup_mod.replace(" ,", ",")
        fist.append((tup_mod, tup[1]))
    formatted_id_steps = fist

    recipe_parsed_ings = parse_ings(texts=id_ings)
    recipe_parsed_data = parse_documents(texts=formatted_id_steps)

    output_data = {}
    for ids in id_steps:
        id = ids[0]
        output_data[id] = {}

        data_row = raw_recipe_data.loc[id]
        parsed_steps = recipe_parsed_data[id]

        output_data[id]["recipe_name"] = data_row["name"]
        output_data[id]["tags"] = eval(data_row.tags)
        output_data[id]["parsed_steps"] = parsed_steps
        output_data[id]["ingredients"] = recipe_parsed_ings[id]

    save_process_results(data=output_data, output_file=output_file)


if __name__ == "__main__":
    # nlp = spacy.load("en_core_web_trf")
    #
    # from spacy import displacy
    #
    # doc = nlp("Place bacon in a large, deep skillet.")
    # options = {"compact": True, "font": "Source Sans Pro"}
    # displacy.serve(doc, style="dep", options=options)
    # quit()

    # nlp = spacy.load("en_core_web_trf")
    # start_time = time.time()
    #
    # texts = ["cook over medium high heat until crispy", "asdf foo"]
    #
    # for doc in nlp.pipe(texts, disable=["ner", "textcat", "token2vec"]):
    #     res = process_doc(doc)
    #     print(res)
    # quit()

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--input_file", type=str, default="RAW_recipes.csv")
    parser.add_argument("--n_recipes", type=int, default=-1)

    args = parser.parse_args()

    n_rec = args.n_recipes
    input_file = args.input_file

    if not os.path.exists((path.DATA_DIR / args.output_dir).resolve()):
        os.makedirs((path.DATA_DIR / args.output_dir).resolve())
    output_file = (path.DATA_DIR / args.output_dir / "parsed_recipes.pkl")
    output_texts = (path.DATA_DIR / args.output_dir / "selected_recipes.csv")

    main(input_file=input_file, output_file=output_file, output_texts=output_texts, n_recipes=n_rec)
