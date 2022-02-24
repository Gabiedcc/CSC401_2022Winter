#  * This code is provided solely for the personal and private use of students
#  * taking the CSC401 course at the University of Toronto. Copying for purposes
#  * other than this use is expressly prohibited. All forms of distribution of
#  * this code, including but not limited to public repositories on GitHub,
#  * GitLab, Bitbucket, or any other online platform, whether as given or with
#  * any changes, are expressly prohibited.
#  
#  * All of the files in this directory and all subdirectories are:
#  * Copyright (c) 2020 Frank Rudzicz

import json
import argparse
import os.path

import numpy as np

from numpy import genfromtxt

# Provided word lists.
FIRST_PERSON_PRONOUNS = ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']
SECOND_PERSON_PRONOUNS = ['you', 'your', 'yours', 'u', 'ur', 'urs']
THIRD_PERSON_PRONOUNS = ['he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them', 'their', 'theirs']
FUTURE_TENSES = ["'ll", "will", "gonna"]
COMMON_NOUNS = ["NN", "NNS"]
PROPER_NOUNS = ["NNP", "NNPS"]
ADVERBS = ["RB", "RBR", "RBS"]
WH_WORDS = ["WDT", "WP", "WP$", "WRB"]
SLANG = ['smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh', 'rofl', 'wtf', 'bff', 'wyd', 'lylc', 'brb', 'atm', 'imao',
         'sml', 'btw', 'bw', 'imho', 'fyi', 'ppl', 'sob', 'ttyl', 'imo', 'ltr', 'thx', 'kk', 'omg', 'omfg', 'ttys',
         'afn', 'bbs', 'cya', 'ez', 'f2f', 'gtr', 'ic', 'jk', 'k', 'ly', 'ya', 'nm', 'np', 'plz', 'ru', 'so',
         'tc', 'tmi', 'ym', 'ur', 'u', 'sol', 'fml']

CAT_DICT = {"Left": 0, "Center": 1, "Right": 2, "Alt": 3}

bgl_csv = genfromtxt('/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv', delimiter=',', dtype="str", encoding="utf-8")
w_csv = genfromtxt('/u/cs401/Wordlists/Ratings_Warriner_et_al.csv', delimiter=',', dtype="str", encoding="utf-8")

feat_dir = r"/u/cs401/A1/feats/"  # todo change it

alt_ids = genfromtxt(os.path.join(feat_dir, "Alt_IDs.txt"), dtype=None, encoding="utf-8")
left_ids = genfromtxt(os.path.join(feat_dir, "Left_IDs.txt"), dtype=None, encoding="utf-8")
right_ids = genfromtxt(os.path.join(feat_dir, "Right_IDs.txt"), dtype=None, encoding="utf-8")
center_ids = genfromtxt(os.path.join(feat_dir, "Center_IDs.txt"), dtype=None, encoding="utf-8")

alt_feat = np.load(os.path.join(feat_dir, "Alt_feats.dat.npy"))
left_feat = np.load(os.path.join(feat_dir, "Left_feats.dat.npy"))
right_feat = np.load(os.path.join(feat_dir, "Right_feats.dat.npy"))
center_feat = np.load(os.path.join(feat_dir, "Center_feats.dat.npy"))

CAT_ID_DICT = {
    "Left": left_ids,
    "Center": center_ids,
    "Right": right_ids,
    "Alt": alt_ids
}

CAT_FEAT_DICT = {
    "Left": left_feat,
    "Center": center_feat,
    "Right": right_feat,
    "Alt": alt_feat
}


def word_list_feats(tokens, col, word_list_name):
    if word_list_name == "BGL":
        word_csv = bgl_csv
    else:
        word_csv = w_csv

    wl_feat = [list(word_csv[np.where(word_csv[:, 1] == token), np.where(word_csv[0, :] == col)]) for token in tokens]
    wl_feat = [feat[0][0] for feat in wl_feat if len(feat[0]) > 0]
    wl_feat = [float(feat) for feat in wl_feat if feat != ""]
    return wl_feat


def extract1(comment):
    """ This function extracts features from a single comment
    Parameters:
        comment : string, the body of a comment (after preprocessing)
    Returns:
        feats : numpy Array, a 173-length vector of floating point features
        (only the first 29 are expected to be filled, here)
    """
    comment_feats = np.zeros(173+1)
    sentences = [sent for sent in comment.split("\n")]
    terms = [term for sent in sentences for term in sent.split(" ") if term != ""]
    tokens = [term.split("/")[0] for term in terms]
    tags = [term.split("/")[1] for term in terms]

    """ Extract features that rely on capitalization. """
    # 1. Number of tokens in uppercase (≥ 3 letters long)
    comment_feats[0] = len([token for token in tokens if token.upper()])

    """ Lowercase the text in comment. Be careful not to lowercase the tags. (e.g. "Dog/NN" -> "dog/NN")."""
    """ Extract features that do not rely on capitalization. """
    tokens = [token.lower() for token in tokens]

    # 2. Number of first-person pronouns
    comment_feats[1] = len([token for token in tokens if token in FIRST_PERSON_PRONOUNS])

    # 3. Number of second-person pronouns
    comment_feats[2] = len([token for token in tokens if token in SECOND_PERSON_PRONOUNS])

    # 4. Number of third-person pronouns
    comment_feats[3] = len([token for token in tokens if token in THIRD_PERSON_PRONOUNS])

    # 5. Number of coordinating conjunctions
    comment_feats[4] = len([tag for tag in tags if tag == "CC"])

    # 6. Number of past-tense verbs
    comment_feats[5] = len([tag for tag in tags if tag == "VBD"])

    # 7. Number of future-tense verbs
    comment_feats[6] = len([token for token in tokens if token in FUTURE_TENSES])

    # 8. Number of commas
    comment_feats[7] = len([token for token in tokens if token == ","])

    # 9. Number of multi-character punctuation tokens todo

    # 10. Number of common nouns
    comment_feats[9] = len([tag for tag in tags if tag in COMMON_NOUNS])

    # 11. Number of proper nouns
    comment_feats[10] = len([tag for tag in tags if tag in PROPER_NOUNS])

    # 12. Number of adverbs
    comment_feats[11] = len([tag for tag in tags if tag in ADVERBS])

    # 13. Number of wh-words
    comment_feats[12] = len([tag for tag in tags if tag in WH_WORDS])

    # 14. Number of slang acronyms
    comment_feats[13] = len([tag for tag in tags if tag in SLANG])

    # 15. Average length of sentences, in tokens
    comment_feats[14] = len(tokens)

    # 16. Average length of tokens, excluding punctuation-only tokens, in characters todo

    # 17. Number of sentences.
    comment_feats[16] = len(sentences)

    bgl_aoa = word_list_feats(tokens, "AoA (100-700)", "BGL")
    bgl_img = word_list_feats(tokens, "IMG", "BGL")
    bgl_fam = word_list_feats(tokens, "FAM", "BGL")

    # 18. Average of AoA (100-700) from Bristol, Gilhooly, and Logie norms
    comment_feats[17] = np.mean(bgl_aoa) if bgl_aoa != [] else 0

    # 19. Average of IMG from Bristol, Gilhooly, and Logie norms
    comment_feats[18] = np.mean(bgl_img) if bgl_img != [] else 0

    # 20. Average of FAM from Bristol, Gilhooly, and Logie norms
    comment_feats[19] = np.mean(bgl_fam) if bgl_fam != [] else 0

    # 21. Standard deviation of AoA (100-700) from Bristol, Gilhooly, and Logie norms
    comment_feats[20] = np.std(bgl_aoa) if bgl_aoa != [] else 0

    # 22. Standard deviation of IMG from Bristol, Gilhooly, and Logie norms
    comment_feats[21] = np.std(bgl_img) if bgl_img != [] else 0

    # 23. Standard deviation of FAM from Bristol, Gilhooly, and Logie norms
    comment_feats[22] = np.std(bgl_fam) if bgl_fam != [] else 0

    w_vms = word_list_feats(tokens, "V.Mean.Sum", "W")
    w_ams = word_list_feats(tokens, "A.Mean.Sum", "W")
    w_dms = word_list_feats(tokens, "D.Mean.Sum", "W")

    # 24. Average of V.Mean.Sum from Warringer norms
    comment_feats[23] = np.mean(w_vms) if w_vms!= [] else 0

    # 25. Average of A.Mean.Sum from Warringer norms
    comment_feats[24] = np.mean(w_ams) if w_ams!= [] else 0

    # 26. Average of D.Mean.Sum from Warringer norms
    comment_feats[25] = np.mean(w_dms) if w_dms!= [] else 0

    # 27. Standard deviation of V.Mean.Sum from Warringer norms
    comment_feats[26] = np.std(w_vms) if w_vms!= [] else 0

    # 28. Standard deviation of A.Mean.Sum from Warringer norms
    comment_feats[27] = np.std(w_ams) if w_ams!= [] else 0

    # 29. Standard deviation of D.Mean.Sum from Warringer norms
    comment_feats[28] = np.std(w_dms) if w_dms!= [] else 0

    return comment_feats


def extract2(feat, comment_class, comment_id):
    """ This function adds features 30-173 for a single comment.
    Parameters:
        feat: np.array of length 173
        comment_class: str in {"Alt", "Center", "Left", "Right"}
        comment_id: int indicating the id of a comment
    Returns:
        feat : numpy Array, a 173-length vector of floating point features (this function adds feature 30-173).
        This should be a modified version of the parameter feats.
    """
    cat_id = CAT_DICT[comment_class]
    liwc_feats = CAT_FEAT_DICT[comment_class]
    liwc_ids = CAT_ID_DICT[comment_class]
    feat[29: 173] = liwc_feats[np.where(liwc_ids == comment_id)]
    feat[173] = cat_id
    return feat


def main(args):
    # Load data
    with open(args.input, "r") as f:
        data = json.load(f)

    feats = np.zeros((len(data), 173+1))

    for i, comment in enumerate(data):

        print(comment["body"])

        """ Call extract1 for each datapoint to find the first 29 features. """
        feat = extract1(comment["body"])

        """ Call extract2 for each feature vector to copy LIWC features (features 30-173) into feats. """
        # Note that these rely on each data point's class, which is why we can't add them in extract1.
        feat = extract2(feat, comment["cat"], comment["id"])

        # Add these to feats.
        feats[i] = feat

    np.savez_compressed(args.output, feats)


if __name__ == "__main__":
    """
    python a1_extractFeatures.py -i preproc.json -o feats.npz --a1_dir=.
    """
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    parser.add_argument("-p", "--a1_dir", default="/u/cs401/A1/",
                        help="Path to csc401 A1 directory. "
                             "By default it is set to the cdf directory for the assignment.")
    args = parser.parse_args()

    main(args)
