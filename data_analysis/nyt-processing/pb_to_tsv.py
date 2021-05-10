import sys
import csv
from pathlib import Path

# from src.python.Utilities import message_iterator, guid_to_mid
from Utilities import message_iterator, guid_to_mid


def create_guid_dict(file_names):
    '''
    Create a dictionary with guids as key and entity name as value
    '''
    print(f"Loading entity lists from {', '.join(file_names)}")
    guid_dict = {}
    for file_name in file_names:
        with open(file_name, 'r') as f_in:
            tsvin = csv.reader(f_in, delimiter='\t')
            for row in tsvin:
                assert row[0] not in guid_dict, f"guid {row[0]} is present twice!"
                guid_dict[row[0]] = row[1]
    return guid_dict


input_file = Path(sys.argv[1])
out_file = input_file.with_suffix(".tsv")

print("Starting converting files in ", input_file)
guid_mapping_files = sys.argv[2:]
guid_dict = create_guid_dict(guid_mapping_files)


with open(input_file, "rb") as fin, open(out_file, 'w') as fout:
    f_write = csv.writer(fout, delimiter='\t')
    # f_write = None
    # print("not writing tsv...")
    print("starting iterator")
    for rel in message_iterator(fin):
        # print("processing a relation")
        # print(rel)
        # break
        sourceId_ = rel.sourceGuid
        destId_ = rel.destGuid

        e1_name = guid_dict.get(sourceId_, None)
        if e1_name is None:
            print(f".... Could'nt find guid {sourceId_}")
            print(rel)
            break
        e1_name = guid_dict[sourceId_]
        e2_name = guid_dict[destId_]
        # e1_name_new = e1_name.replace(' ', '_')
        # e2_name_new = e2_name.replace(' ', '_')

        relation = rel.relType
        sourceId = guid_to_mid(sourceId_)
        destId = guid_to_mid(destId_)
        if f_write is None:
            print(f"got {e1_name}  {relation}  {e2_name}  ...  {sourceId}, {destId} ... {sourceId_}, {destId_}")

        n = 0
        for mention in rel.mention:
            n += 1

            sentence = mention.sentence
            # sentence = sentence.replace(e1_name, e1_name_new)
            # sentence = sentence.replace(e2_name, e2_name_new)
            e_spans = ([], [])
            for ei, ei_spans in zip((e1_name, e2_name), e_spans):
                end_index = 0
                try:
                    while len(ei_spans) < 10:
                        start_index = sentence.index(ei, end_index)
                        end_index = start_index + len(ei)
                        ei_spans.append((start_index, end_index))
                    if len(ei_spans) >= 10:
                        print(f"too many spans for {ei} in: {sentence}")
                except ValueError:
                    pass

            types = mention.feature[0].split('->')

            if f_write:
                f_write.writerow([sourceId, destId, e1_name, e2_name, str(e_spans[0]), str(e_spans[1]),
                    types[0], types[1], relation, sentence])
            else:
                print(f"    {mention.filename}")
                print(f"    {mention.sourceId}, {mention.destId}")
                print(f"    {types}")
                print(f"    {sentence}")
                break
        if f_write is None:
            break
        # print(file_name"    with {n} mentions")
