#   Copyright (c) 2020.  CSIRO Australia.
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
#  documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
#  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
#  and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of
#  the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
#  THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
#  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#
import argparse
import re
import sys
from pathlib import Path

import json

import csv
from pprint import pprint

import logging

from tqdm import tqdm

REVERB = "reverb"
NYT = "nyt"
NYT_NA = "nyt_NA"
NYT_NEW_NA = "nyt_new_NA"

parser = argparse.ArgumentParser()
parser.add_argument("--log-level", default="WARNING")
parser.add_argument("--max-rels", default=None, type=int, help="Stop after this many freebase triples")
parser.add_argument("--max-wiki", default=None, type=int, help="Stop after this many wiki entities recognised")
parser.add_argument("--stop-on-all", action="store_true", help="Stop once all useful relation types have been seen")
parser.add_argument("--dataset", choices=(REVERB, NYT, NYT_NA, NYT_NEW_NA), help="which dataset to get info for")
parser.add_argument("--merge-info-with", default=None, help="e2info.json file of previously extracted entity info")
parser.add_argument("--clobber", action="store_true", help="Overwrite existing output file, otherwise quit if it exists"
                                                           " (default is to quit).")
args = parser.parse_args()
logging.basicConfig(level=args.log_level)

# NOTE: freebase doesn't seem to have extra relation metadata, at least not for all relations.
#       eg: '<http://rdf.freebase.com/ns/education.school_category.schools_of_this_kind>' has no metadata at all

freebase_dir = Path("/media/ian/Data2/freebase")
freebase_file = "freebase-rdf-latest.gz"  # format: ent1_uri \t rel_uri \t ent2_uri \t notes? mostly "."
# freebase_to_wikidata_file = "fb2w.nt.gz"  # format: fbent_uri \t w3/SameAs \t wd_ent_uri

if args.dataset == REVERB:
    data_dir = Path("/media/ian/Data2/OpenKI-relation-data/data_reverb")

    # Read in the entity freebase ids
    reverb_entities_file = "e2name.json"
    with open(data_dir / reverb_entities_file) as f:
        e2name = json.load(f)
    fbid_to_e2name = {v: int(k) for k, v in e2name.items()}
    num_entities = len(e2name)
    #     "1": "<http://rdf.freebase.com/ns/m.01nz1q6>",
    # "253868": "<http://rdf.freebase.com/ns/m.03mcbg2>"
    # reverb_entities_wd_ids_file = "e2wd.json"
elif args.dataset.startswith("nyt"):
    variant = args.dataset[3:]
    data_dir = Path(f"/media/ian/Data2/riedel/data-nyt-2010/openki_data{variant}")

    with open(data_dir / "entities.json") as f:
        ent_in_data = json.load(f)
    fbid_to_e2name = {f"<http://rdf.freebase.com/ns/{next(iter(info['fb_id']))}>": int(info["id"])
                      for info in ent_in_data if type(info['fb_id']) is dict}
    num_entities = len(ent_in_data)
    # for n, (fb_id, eid) in enumerate(fbid_to_e2name.items()):
    #     print(f"{eid:3d} : {fb_id}")
    #     if n >= 20:
    #         break
else:
    raise NotImplementedError(f"I don't know what to do with dataset {args.dataset}")

entities_info_file = "e2info_fb.json"  # None   ... None to use stdout (eg: piped to gzip!)
entities_extra_info_file = "e2info_extra_fb.json"  # when an entity comes up twice, we store the unused info here
entities_broken_info_file = "e2info_with_missing_fb.pickle"  # when one of name, desc, enwiki are missing in wd

if (data_dir / entities_info_file).exists():
    if args.clobber:
        logging.warning(f"Going to clobber {data_dir / entities_info_file} aftr processing!")
    else:
        logging.critical(f"--clobber not set! Cowardly refusing to clobber file {data_dir / entities_info_file}!")
        exit(1)

# <http://rdf.freebase.com/ns/american_football.football_player.footballdb_id>    <http://rdf.freebase.com/ns/type.object.name>   "footballdb ID"@en      .
ENTITY_GET_LABEL_ORDER = {
    b'<http://rdf.freebase.com/ns/type.object.name>': ('name', 'alias'),  # this appears to be present in all entries
    b'<http://rdf.freebase.com/ns/common.topic.alias>': ('alias', 'alias'),  # works as name+lang. eg: "Yuko Ootaki"@ja
    b'<http://rdf.freebase.com/ns/common.topic.description>': ('desc', 'alias'),
    b'<http://rdf.freebase.com/key/en>': ('name_wiki', 'wikipedia'),   # as wikipedia. eg: "yuko_ootaki"
    b'<http://rdf.freebase.com/key/wikipedia.en>': ('name_wiki_', 'wikipedia'),  # name w. _ and $-enc. eg: "Nancy$0027s_Pizza"
    b'<http://rdf.freebase.com/key/wikipedia.en_title>': ('name_wiki_t', 'wikipedia'),  # ditto
    b'<http://www.w3.org/2000/01/rdf-schema#label>': ('name_label', 'alias')  # looks like the same fomat as 'alias': spaces, @lang
}
e_info_keys = set(ENTITY_GET_LABEL_ORDER.values())
e_info_rel_seen = {k.decode(): False for k in ENTITY_GET_LABEL_ORDER.keys()}

# read lines from the freebase dump, piped here from gzip with a command like:
#    zcat freebase-rdf-latest.gz | tr -d '\000' | python get-freebase-entity-info.py
# freebase_stream = csv.reader(sys.stdin, delimiter='\t')
freebase_stream = sys.stdin.buffer
wiki_unicode_re = re.compile(rb'\$([0-9A-F]{4})')
underscore_trans = str.maketrans('_', ' ')
# null_byte_trans = bytes.maketrans(b'\0', b'')
e2info = [None for _ in range(num_entities)]
if args.merge_info_with:
    with open(data_dir / args.merge_info_with) as f_info:
        for ent_id_str, ent_info in json.load(f_info):
            e2info[int(ent_id_str)] = ent_info
num_fb_processed = 0
ent_found = 0
wiki_processed = 0
unrecognised_rels = set()
for fb_entry in tqdm(freebase_stream, total=3130753066):  # (line.translate({b'\n': None})
    fb_entry = fb_entry.split(b'\t')
    num_fb_processed += 1
    if args.max_rels and num_fb_processed >= args.max_rels:
        logging.debug(f"processed {args.max_rels} fb lines, quitting")
        break
    try:
        subject_val, rel, object_val, notes = fb_entry
    except ValueError:
        print(f"{fb_entry}")
        raise
    info_key = ENTITY_GET_LABEL_ORDER.get(rel, None)
    if info_key is None:
        # logging.debug(f"Rel not recognised: {rel}")
        if args.log_level == "DEBUG":
            unrecognised_rels.add(rel)  # note these are still bytes objects, not str!
        continue
    subject_val, rel, object_val, notes = subject_val.decode(), rel.decode(), object_val.decode(), notes.decode()
    if not subject_val.startswith("<http://rdf.freebase.com/ns/m."):
        # logging.debug(f"Subject not recognised: {subject_val} with relation {rel}")
        continue
    e_id = fbid_to_e2name.get(subject_val, None)
    if e_id is None:
        logging.debug(f"Entity not recognised: {subject_val}")
        continue
    ent_found += 1
    log_msg = f"Entity {ent_found} recognised: {subject_val}"
    info_key, process_as = info_key
    if process_as is 'alias':
        try:
            bits = object_val.split('@')
            lang = bits[-1]
            info = '@'.join(bits[:-1])
        except ValueError:
            print(f"Bad @ expr {object_val} with relation {rel}")
            raise
        if lang != 'en':
            logging.debug(log_msg + f", but lang is {lang} for {info}")
            continue
        logging.debug(log_msg + f": {info}")
    elif process_as is 'wikipedia':
        info = wiki_unicode_re.sub(b'\\\\u\\1', bytes(object_val, 'ascii'))\
            .decode('unicode-escape')\
            .translate(underscore_trans)
        logging.debug(log_msg + f", found wiki info: {info}")
        wiki_processed += 1
        if args.max_wiki and wiki_processed >= args.max_wiki:
            print("processed 100 wiki lines, quititng")
            break
    else:
        raise ValueError(f"bad process_as {process_as} for {info_key} after " + log_msg)
    if args.stop_on_all:
        e_info_rel_seen[rel] = True
        if all(e_info_rel_seen.values()):
            print(f"seen all name-ish relations, quitting!")
            break
    info_record = e2info[e_id]
    if info_record is None:
        info_record = {key: [] for key, _ in e_info_keys}
        e2info[e_id] = info_record
    info_record[info_key].append(info.strip('" \n'))

print(f"Processed {num_fb_processed} records, found {ent_found} useful records, {wiki_processed} wiki entries, "
      f", {sum(1 for info_record in e2info if info_record is not None)} distinct entities "
      f"and missed {sum(1 for info_record in e2info if info_record is None)} entities.")
if unrecognised_rels:
    print("Unrecognised relations:")
    for rel in sorted(unrecognised_rels):
        print(rel)

e2info_dict = {i: info for i, info in enumerate(e2info) if info is not None}
if entities_info_file is not None:
    with open(data_dir / entities_info_file, "w") as f_ent_info:
        json.dump(e2info_dict, f_ent_info, indent='    ')
else:
    json.dump(e2info_dict, sys.stdout, indent='    ')
# for i, info_record in enumerate(e2info):
#     if info_record is not None:
#         print(f"Record for entity number {i}:")
#         pprint(info_record)