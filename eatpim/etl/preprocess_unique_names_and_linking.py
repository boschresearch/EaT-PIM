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

import rdflib
import argparse
from eatpim.utils import path
import json
import pickle
from linkers import *
import numpy as np
import time
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import spacy
spacy.prefer_gpu()


def match_objects_and_ings(*, noun_names, noun_name_CV, noun_name_vector,
                           ing_names, ing_name_CV, ing_name_vector):
    noun_to_ing_candidates = {}
    noun_to_subnoun_candidates = {}
    noun_remaining_modifiers = {}

    checked = 0
    starttime = time.time()
    print("handling all noun linking...")
    for noun in noun_names:
        checked += 1
        if checked % 1000 == 0:
            print(f'progress: {round(checked / len(noun_names), 5)}, {round(time.time() - starttime, 5)}s')

        nounlen = len(noun)
        # vector approach below
        noun_vector = ing_name_CV.transform([noun])
        noun_ing_dotprod = noun_vector.dot(ing_name_vector.T).toarray()
        nonzero_match_ing_indexes = np.argwhere(noun_ing_dotprod > 0)[:, 1]
        candidate_ings = [ing_names[i] for i in nonzero_match_ing_indexes]

        ing_match_candidates = {}
        for ing_name in candidate_ings:
            if len(ing_name) <= nounlen:
                ing_match_candidates[ing_name] = len(ing_name)

        stripped_noun = noun
        if len(ing_match_candidates):
            ingmatch_sorted_by_len = sorted(ing_match_candidates.items(), key=lambda kv: kv[1], reverse=True)
            ing_matches = []
            # iteratively get ingredient matches, assuming that the longest match in the string is the most correct
            # e.g., in "tablespoon olive oil", the ingredient "olive oil" is longer than "oil" so it should match first
            # we also want to be checking the very end of the words so that we don't catch situations like
            # matching "cookie" in "cookie sheet".
            matches_updated = True
            while matches_updated:
                matches_updated = False
                for (ingstr, inglen) in ingmatch_sorted_by_len:
                    ingstr_index = stripped_noun.find(ingstr)
                    if ingstr_index != -1 and ingstr_index + inglen == len(stripped_noun):
                        # if ingstr_index is not at the beginning of the noun, it should be right after a space
                        # this should help avoid situations like matching "all" into "ball"
                        if ingstr_index != 0 and noun[ingstr_index-1] != " ":
                            continue
                        ing_matches.append(ingstr)
                        stripped_noun = stripped_noun.replace(ingstr, "", 1).rstrip()
                        if len(stripped_noun) and stripped_noun[-1] == "/":
                            stripped_noun = stripped_noun[:-1].strip()
                        matches_updated = True
            noun_to_ing_candidates[noun] = [ing for ing in ing_matches]
        else:
            noun_to_ing_candidates[noun] = []
        stripped_noun = stripped_noun.strip()

        # the first and last characters are digits - this probably is a situation like "1 to 2 potatoes", where
        # the potatoes was removed and we are left with "1 to 2". this isn't anything of interest for now, so continue.
        # also continue on if the stripped noun is now empty.
        if len(stripped_noun) <= 1 or (stripped_noun[0].isdigit() and stripped_noun[-1].isdigit()):
            continue

        noun_vector = noun_name_CV.transform([stripped_noun])
        noun_noun_dotprod = noun_vector.dot(noun_name_vector.T).toarray()
        nonzero_match_noun_indexes = np.argwhere(noun_noun_dotprod > 0)[:, 1]
        candidate_nouns = [noun_names[i] for i in nonzero_match_noun_indexes]
        # check for nouns that are substrings of this noun
        subnoun_match = {}
        for noun_2 in candidate_nouns:
            if noun_2 != stripped_noun and len(noun_2) < nounlen:
                subnoun_match[noun_2] = len(noun_2)

        if len(subnoun_match):
            nounmatch_sorted_by_len = sorted(subnoun_match.items(), key=lambda kv: kv[1], reverse=True)
            noun_matches = []
            matches_updated = True
            while matches_updated:
                matches_updated = False
                # iteratively get noun matches, in a similar fashion to how ingredient matches
                # if the first part of a noun match is a digit, make sure that the noun matche starts at index 0
                # i.e., prevent "1 / 2 cup ..." from matching with "2 cup" - it'll either match with "1 / 2 cup", or "cup".
                # if a noun match corresponds to the occurrence of a digit in the string, get rid of the digits.
                # i.e., if "cup" matches in "1 / 2 cup ...", get rid of the "1 / 2" portion as well
                for (noun_2, nlen) in nounmatch_sorted_by_len:
                    if noun_2 not in stripped_noun:
                        continue
                    noun_2_index = stripped_noun.find(noun_2)
                    if noun_2_index != -1 and noun_2_index + nlen == len(stripped_noun):
                        # we dont want to be matching numbers of things together
                        if noun_2[0].isdigit() and noun_2_index != 0:
                            continue
                        # if noun_2_index is not at the beginning of the noun, it should be right after a space
                        # this should help avoid situations like matching "all" into "ball"
                        if noun_2_index != 0 and noun[noun_2_index-1] != " ":
                            continue
                        noun_matches.append(noun_2)
                        if noun_2_index != 0 and noun_2_index > 1 and \
                                (stripped_noun[noun_2_index - 2].isdigit() or stripped_noun[noun_2_index - 1].isdigit()):
                            stripped_noun = stripped_noun[noun_2_index + nlen:].strip()
                        else:
                            stripped_noun = stripped_noun.replace(noun_2, "", 1).strip()
                        if len(stripped_noun) and stripped_noun[-1] == "/":
                            stripped_noun = stripped_noun[:-1].strip()
                        matches_updated = True

            noun_to_subnoun_candidates[noun] = [n for n in noun_matches]
        else:
            noun_to_subnoun_candidates[noun] = []

        noun_remaining_modifiers[noun] = stripped_noun

    return noun_to_ing_candidates, noun_to_subnoun_candidates, noun_remaining_modifiers

def match_ings_and_ings(*, ing_names, ing_name_CV, ing_name_vector):
    # similar to match_objects_and_ings, but slightly different logic since we're just checking
    # substring matches within ingredient names
    ing_to_subing_candidates = {}

    checked = 0
    starttime = time.time()
    print("handling all ing linking...")
    for ing in ing_names:
        checked += 1
        if checked % 1000 == 0:
            print(f'progress: {round(checked / len(ing_names), 5)}, {round(time.time() - starttime, 5)}s')

        inglen = len(ing)
        ing_vector = ing_name_CV.transform([ing])
        noun_noun_dotprod = ing_vector.dot(ing_name_vector.T).toarray()
        nonzero_match_noun_indexes = np.argwhere(noun_noun_dotprod > 0)[:, 1]
        candidate_ings = [ing_names[i] for i in nonzero_match_noun_indexes]
        # check for nouns that are substrings of this noun
        subing_match = {}

        for ing_2 in candidate_ings:
            if ing_2 != ing and len(ing_2) < inglen:
                subing_match[ing_2] = len(ing_2)

        if len(subing_match):
            stripped_noun = ing
            ingmatch_sorted_by_len = sorted(subing_match.items(), key=lambda kv: kv[1], reverse=True)
            ing_matches = []
            matches_updated = True
            while matches_updated:
                matches_updated = False
                # iteratively ingredient matches
                for (ing_2, nlen) in ingmatch_sorted_by_len:
                    if ing_2 in ing_matches:
                        continue
                    if ing_2 not in ing:
                        continue
                    ing_2_index = ing.find(ing_2)
                    if ing_2_index != -1 and ing_2_index + nlen == len(stripped_noun):
                        if ing_2[0].isdigit() and ing_2_index != 0:
                            continue
                        if ing_2_index != 0 and ing[ing_2_index-1] != " ":
                            continue
                        stripped_noun = stripped_noun.replace(ing_2, "", 1).strip()
                        ing_matches.append(ing_2)
                        matches_updated = True

            ing_to_subing_candidates[ing] = [n for n in ing_matches]
        else:
            ing_to_subing_candidates[ing] = []

    return ing_to_subing_candidates


def match_names_to_external(*, name_list, matcher: Matcher):
    links = {}
    checked = 0
    max_count = len(name_list)
    starttime = time.time()
    no_foodon_match = 0
    for name in name_list:
        match_uri, score = matcher.match(input_str=name)
        if match_uri is not None:
            links[name] = [match_uri]
        else:
            no_foodon_match += 1
        checked += 1
        if checked % 1000 == 0:
            print(f'progress: {round(checked / max_count, 5)}, {round(time.time() - starttime, 5)}s')
    print(f"names with no match: {no_foodon_match}")
    return links

def main(*,
         input_file,
         wiki_dir,
         foodon_dir,
         output_dir):
    nlp = spacy.load("en_core_web_trf", disable=["ner", "textcat", "token2vec"])

    with open(input_file.resolve(), 'rb') as f:
        recipe_data = pickle.load(f)

    wiki_p_matcher = WikidataMatcher(mode='preparation', nlp=nlp)

    objlist = list(set(obj
              for recipe in recipe_data.values()
              for step in recipe['parsed_steps'].values()
              for obj in step['noun_chunks']))

    verblist = list(set(obj
              for recipe in recipe_data.values()
              for step in recipe['parsed_steps'].values()
              for obj in step['verbs']))

    inglist = list(set(ingr
                 for recipe in recipe_data.values()
                 for ingr in recipe['ingredients']))

    with open((output_dir / "ingredient_list.json").resolve(), 'w') as f:
        json.dump(inglist, f)

    print('unique ingredients: ', len(inglist))
    print('unique objects: ', len(objlist))
    print('unique verbs: ', len(verblist))

    ing_name_CV = TfidfVectorizer()
    ing_name_vector = ing_name_CV.fit_transform(inglist)

    noun_name_CV = CountVectorizer()
    noun_name_vector = noun_name_CV.fit_transform(objlist)

    obj_to_ing, obj_to_subobj, obj_leftovers = match_objects_and_ings(
        noun_names=objlist, noun_name_CV=noun_name_CV, noun_name_vector=noun_name_vector,
        ing_names=inglist, ing_name_CV=ing_name_CV, ing_name_vector=ing_name_vector)

    ing_to_ing = match_ings_and_ings(ing_names=inglist, ing_name_CV=ing_name_CV, ing_name_vector=ing_name_vector)

    foodon_m = FoodOnMatcher(min_confidence=0.6)
    print("linking ingredient names to foodon")
    ing_to_foodon = match_names_to_external(name_list=inglist, matcher=foodon_m)
    foodon_m = FoodOnMatcher(min_confidence=0.9)
    print("linking object names to foodon")
    obj_to_foodon = match_names_to_external(name_list=objlist, matcher=foodon_m)

    wiki_e_matcher = WikidataMatcher(mode='equipment', nlp=nlp, min_confidence=0.85)
    print("linking objects to wikidata cooking equipment")
    obj_to_equipment = match_names_to_external(name_list=objlist, matcher=wiki_e_matcher)

    wiki_p_matcher = WikidataMatcher(mode='preparation', nlp=nlp, min_confidence=0.85)
    print("linking verbs to wikidata cooking preparation methods")
    verb_to_preparations = match_names_to_external(name_list=verblist, matcher=wiki_p_matcher)

    save_data = {
        'obj_to_ing': obj_to_ing,
        'obj_to_subobj': obj_to_subobj,
        'obj_leftovers': obj_leftovers,
        'ing_to_ing': ing_to_ing,
        'ing_to_foodon': ing_to_foodon,
        'obj_to_foodon': obj_to_foodon,
        'obj_to_equipment': obj_to_equipment,
        'verb_to_preparations': verb_to_preparations
    }

    with open((output_dir / 'word_cleanup_linking.json').resolve(), 'w') as f:
        json.dump(save_data, f)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir")

    parser.add_argument("--wiki_dir", type=str, default="wikidata_cooking")
    parser.add_argument("--foodon_dir", type=str, default="foodon_ontologies")

    args = parser.parse_args()

    input_file = (path.DATA_DIR / args.input_dir / "parsed_recipes.pkl")
    wiki_dir = (path.DATA_DIR / args.wiki_dir)
    foodon_dir = (path.DATA_DIR / args.foodon_dir)
    output_dir = (path.DATA_DIR / args.input_dir)

    main(input_file=input_file,
         wiki_dir=wiki_dir,
         foodon_dir=foodon_dir,
         output_dir=output_dir)
