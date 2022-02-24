#  * This code is provided solely for the personal and private use of students
#  * taking the CSC401 course at the University of Toronto. Copying for purposes
#  * other than this use is expressly prohibited. All forms of distribution of
#  * this code, including but not limited to public repositories on GitHub,
#  * GitLab, Bitbucket, or any other online platform, whether as given or with
#  * any changes, are expressly prohibited.
#  
#  * All of the files in this directory and all subdirectories are:
#  * Copyright (c) 2020 Frank Rudzicz


import re
import os
import sys
import json
import argparse

import spacy
# import unicodedata

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
nlp.add_pipe('sentencizer')


def political_category(term: str):
    lefts = ["TwoXChromosomes", "occupyWallStreet", "lateStageCapitalism",
             "progressive", "socialism", "demsocialist", "Liberal"]
    centers = ["news", "politics", "energy", "canada", "worldnews", "law"]
    rights = ["TheNewRight", "WhiteRights", "Libertarian", "AskTrumpSupporters",
              "The_Donald", "new_right", "Conservative", "tea_party"]
    alts = ["conspiracy", "911truth"]

    if term in lefts:
        return "Left"
    elif term in centers:
        return "Center"
    elif term in rights:
        return "Right"
    elif term in alts:
        return "Alt"
    else:
        raise ValueError(term)


def preproc1(comment, steps=range(1, 6)):
    """ This function pre-processes a single comment
    Parameters:
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step
    Returns:
        modComm : string, the modified comment
    """
    modified_comment = comment
    if 1 in steps: # modify this to handle other whitespace chars. replace newlines with spaces
        modified_comment = re.sub(r"\s+", " ", modified_comment)

    if 2 in steps:  # unescape html
        # modified_comment = unicodedata.normalize('NFKD', modified_comment)
        pass

    if 3 in steps:  # remove URLs
        modified_comment = re.sub(r"\b(http:\/\/|https:\/\/|www\.)\S+", "", modified_comment)
        
    if 4 in steps:  # remove duplicate spaces.
        modified_comment = re.sub(" +", " ", modified_comment)

    if 5 in steps:
        """ Tagging """
        utt = nlp(modified_comment)

        """ Lemmatization """
        sentences_n_tags = []
        for sent in utt.sents:
            tokens_n_tags = []
            for token in sent:
                if token.lemma_.startswith("-") and not token.text.startswith("-"):
                    text = token.text
                else:
                    text = token.lemma_

                if token.text.isupper():
                    text = token.text
                else:
                    text = text.lower()

                token_n_tag = text + "/" + token.tag_  # Write "/POS" after each token.
                tokens_n_tags.append(token_n_tag)
            joint_token = " ".join(tokens_n_tags)  # Split tokens with spaces.
            sentences_n_tags.append(joint_token)
        """ Sentence segmentation """
        modified_comment = "\n".join(sentences_n_tags)+"\n"  # Insert "\n" between sentences.

    return modified_comment


def main(args):
    all_output = []
    for subdir, dirs, files in os.walk(indir):
        print(subdir)
        for file in files:
            full_file = os.path.join(subdir, file)
            print("Processing " + full_file)
            with open(full_file, "r", encoding="utf-8") as data_f:
                js = json.load(data_f)

            """ Step 1: select appropriate args.max lines """
            i_line = 0
            for j in js:
                """ Step 2: read those lines with something like `j = json.loads(line)` """
                """ Step 3: choose to retain fields from those lines that are relevant to you. """
                j = json.loads(j)
                output = {"id": j["id"], "body": str(j["body"])}

                """ Step 4: add a field to each selected line called 'cat' (e.g., 'Right', ) """
                output["cat"] = political_category(j["subreddit"])

                """ Step 5: process the body field (j['body']) with preproc1(...) using default for `steps` argument """
                """ Step 6: replace the 'body' field with the processed text """
                output["body"] = preproc1(output["body"])
                print(output)

                """ Step 7: append the result to 'allOutput' """
                all_output.append(output)

                i_line += 1
                if i_line > args.max:
                    break

    fout = open(args.output, 'w')
    fout.write(json.dumps(all_output, indent=2))
    fout.close()


if __name__ == "__main__":
    r"""
    python a1_preproc.py 999123456 -o preproc.json --a1_dir=.
    python a1_preproc.py 999123456 -o preproc.json --a1_dir=C:\Users\wanglx\Downloads\Compressed
    """
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", type=int, help="The maximum number of comments to read from each file", default=10000)
    parser.add_argument("--a1_dir", default='/u/cs401/A1',
                        help="The directory for A1. Should contain subdir data. "
                             "Defaults to the directory for A1 on cdf.")
    
    args = parser.parse_args()

    if args.max > 200272:
        print("Error: If you want to read more than 200,272 comments per file, you have to read them all.")
        sys.exit(1)
    
    indir = os.path.join(args.a1_dir, 'data')
    main(args)
