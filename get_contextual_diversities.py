"""
Get a context co-occurrence matrix for tokens, computing raw contextual diversities.
Uses simple space-based tokenization for CHILDES.

Sample usage:
python3 get_contextual_diversities.py \
--childes_directory="cogsci_data/childes_data" --childes_lang="eng-na" \
--window_size=5 --lemmas="true" --spacy_lang="en_core_web_sm" --common_tokens=5000 \
--wordbank_lang="English (American)" \
--wordbank_file="cogsci_data/all_prop_data_WS_cleaned.tsv" \
--output="eng-na_diversities_window5.txt"

"""
import argparse
import codecs
import torch
import operator
import random
from collections import Counter
import spacy
import os


def create_parser():
    parser = argparse.ArgumentParser()
    # Window size in each direction. Default 5 in Hills et al. (2010).
    parser.add_argument('--window_size', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    # Whether to split in-degree and out-degree.
    # split (sum forward and backwards counts), bidirectional (position unaware),
    # unidirectional (backwards only), or uni_two (sum the previous two positions).
    # Equivalent value was "split" in Hills et al. (2010).
    parser.add_argument('--split_in_out', default="bidirectional")
    # Maximum size of the dataset to consider. Default ~2M in Hills et al. (2010).
    parser.add_argument('--max_token_count', type=int, default=1e9)
    # Set nonempty to use lemmas, treating all forms of each lemma as a single token.
    # Uses spaCy, not supported for Mandarin.
    parser.add_argument('--lemmas', default="true")
    # How many of the most frequent tokens to consider in co-occurrences.
    # Default 5000 in Hills et al. (2010).
    parser.add_argument('--common_tokens', type=int, default=5000)

    # File settings.
    parser.add_argument('--output', default='output.txt')
    parser.add_argument('--wordbank_file', default="cogsci_data/all_prop_data_WS_cleaned.tsv")
    # English (American), German, French (French), Mandarin (Beijing), Spanish (Mexican).
    parser.add_argument('--wordbank_lang', default="English (American)")
    # Should contain CHILDES txt and tsv files (from get_childes_data.py).
    parser.add_argument('--childes_directory', default="")
    # eng-na, spanish, mandarin, german, french.
    parser.add_argument('--childes_lang', default="eng-na")
    # Note: you may need to first run: python3 -m spacy download [lang]
    # en: en_core_web_sm
    # de: de_core_news_sm
    # fr: fr_core_news_sm
    # es: es_core_news_sm
    # Only used if using lemmas.
    parser.add_argument('--spacy_lang', default="en_core_web_sm")
    return parser


def lemmatize(token, spacy_nlp):
    # Only to use for individual tokens.
    token = spacy_nlp(token)[0].lemma_
    return token


def update_co_occurrences(args, occurrence_matrix, example, token_id_to_idx):
    # Update the co-occurrence matrix with a single example.
    # example should be a list of token ids.
    for idx, id in enumerate(example):
        if id not in token_id_to_idx:
            continue
        # As in original paper, goes from idx+1 to idx+size-1 (inclusive).
        start_window = min(len(example)-1, idx+1) # Range is inclusive of start_window.
        end_window = min(len(example), idx+args.window_size) # Range is exclusive of end_window.
        # If bidirectional, window goes from idx-size+1 to idx+size-1 (inclusive).
        if args.split_in_out == "bidirectional":
            start_window = max(0, idx-args.window_size+1)
        for idx2 in range(start_window, end_window):
            if idx2 == idx: # Ignore the current token.
                continue
            id2 = example[idx2]
            if id2 in token_id_to_idx:
                occurrence_matrix[token_id_to_idx[id], token_id_to_idx[id2]] += 1
    return occurrence_matrix


def get_co_occurrences(args, token_to_id, token_id_to_idx, spacy_nlp=None):
    # Get co-occurrence matrix.
    # Map token ids to indices in the matrix.
    occurrences = torch.zeros(len(token_id_to_idx), len(token_id_to_idx), dtype=torch.int32)
    token_count = 0
    childes_txt = os.path.join(args.childes_directory, "childes_{}.txt".format(args.childes_lang))
    infile = codecs.open(childes_txt, 'rb', encoding='utf-8')
    lines = infile.readlines()
    infile.close()
    random.shuffle(lines)
    print("Total lines: {}".format(len(lines)))
    for line_count, line in enumerate(lines):
        if line_count % 10000 == 0:
            print("Finished line {}".format(line_count))
        example_tokens = line.lower().strip().split()
        if args.lemmas:
            # Using SpaCy, tokenize and lemmatize entire sentence.
            nlp_sentence = spacy_nlp(" ".join(example_tokens))
            example_tokens = [nlp_token.lemma_ for nlp_token in nlp_sentence]
        token_count += len(example_tokens)
        if token_count >= args.max_token_count:
            print("Reached max token count. Stopping.")
            break
        example = [] # List of token ids.
        for token in example_tokens:
            if token in token_to_id:
                example.append(token_to_id[token])
            else:
                example.append(-1)
        # Process example.
        occurrences = update_co_occurrences(args, occurrences, example, token_id_to_idx)
    print("Token count: {}".format(token_count))
    return occurrences


def get_tokens(args, spacy_nlp=None):
    # Get tokens to consider.
    # Load tokens/words to consider from WordBank list.
    print("Loading wordbank tokens.")
    # Only used if using lemmas. Maps wordbank wordforms to lemmas, so that
    # wordbank wordforms can still appear in the final output (instead of
    # just lemmas).
    wordform_to_lemma = dict()
    # Assume that wordbank words are cleaned in the final column.
    infile = codecs.open(args.wordbank_file, 'rb', encoding='utf-8')
    tokens = set()
    for line in infile:
        split_line = line.strip().split('\t')
        if split_line[5] == args.wordbank_lang:
            wordform = split_line[-1]
            # Convert wordform to token form.
            if args.lemmas:
                lemma = lemmatize(wordform, spacy_nlp)
                wordform_to_lemma[wordform] = lemma
                token = lemma # The lemma is used as the token.
            else:
                token = wordform
            tokens.add(token)
    infile.close()

    # Count token frequencies.
    print("Loading common tokens.")
    token_frequencies = dict()
    token_mlus = dict()
    # CHILDES data.
    token_counts = Counter()
    total_utterance_lengths = Counter()
    childes_tsv = os.path.join(args.childes_directory, "childes_{}.tsv".format(args.childes_lang))
    infile = codecs.open(childes_tsv, 'rb', encoding='utf-8')
    for line_count, line in enumerate(infile):
        if line_count == 0: # Ignore header.
            continue
        split_line =  line.strip().split('\t')
        if len(split_line) != 5 or len(split_line[0]) == 0:
            continue
        token = split_line[0].lower()
        if args.lemmas:
            # The token is the lemmatized form.
            # Thus, frequencies and MLUs are aggregated over wordforms for each lemma.
            token = lemmatize(token, spacy_nlp)
        count = float(split_line[1]) # Raw count.
        token_counts[token] += count
        utterance_length = float(split_line[2])*count # Average MLU * count.
        total_utterance_lengths[token] += utterance_length
    infile.close()
    # Convert total utterance lengths to MLU averaged over lemmas.
    token_mlus = dict()
    for token, count in token_counts.items():
        token_mlus[token] = total_utterance_lengths[token]/float(count)
    del total_utterance_lengths
    # Convert counts to frequencies per 1000 tokens.
    total_token_count = sum(token_counts.values())
    for token, count in token_counts.items():
        token_frequencies[token] = count*1000.0/total_token_count
    del token_counts
    # Add additional tokens (n most common).
    if args.common_tokens > 0:
        # Sort by counts, get most common words.
        sorted_tokens = sorted(token_frequencies.items(), key=operator.itemgetter(1), reverse=True)
        common_tokens = [pair[0] for pair in sorted_tokens]
        common_tokens = common_tokens[:args.common_tokens]
        tokens.update(common_tokens) # Add to set of tokens.

    # Map tokens to token_ids. Assigns ids in order.
    token_to_id = dict(zip(tokens, range(len(tokens))))

    # Note that wordform_to_lemma only maps wordbank wordforms to lemmas (tokens).
    # wordform_to_lemma is empty if not using lemma.
    return (token_to_id, wordform_to_lemma, token_frequencies, token_mlus)


def compute_diversity(context_tokens):
    # Compute diversity from a vector of co-occurrences.
    diversity = torch.sum(context_tokens > 0).item()
    # Compute the entropy of the distribution over co-occurring tokens.
    context_tokens = context_tokens.float()
    p = context_tokens/torch.sum(context_tokens.float())
    p += 0.000000001 # Smooth with (1e-9).
    logp = torch.log2(p)
    diversity_entropy = torch.sum(-p*logp).item()
    return (diversity, diversity_entropy)


def main(args):
    random.seed(args.seed)
    spacy_nlp = None
    if args.lemmas:
        print("Using lemmatization.")
        spacy_nlp = spacy.load(args.spacy_lang, disable=['parser', 'ner',
            'entity_linker', 'entity_ruler', 'textcat', 'textcat_multilabel',
            'senter', 'sentencizer', 'tok2vec', 'transformer'])
    else:
        print("Not using lemmatization.")
    print("Getting tokens.")
    token_to_id, wordform_to_lemma, token_frequencies, token_mlus = get_tokens(args, spacy_nlp)
    # This is redundant because token_ids are already assigned in order.
    # This is useful if using a tokenizer, i.e. token_ids are not necessarily in order.
    token_id_to_idx = dict(zip(token_to_id.values(), range(len(token_to_id))))
    print("Computing co-occurrences.")
    if args.split_in_out in ["split", "unidirectional", "bidirectional"]:
        occurrences = get_co_occurrences(args, token_to_id, token_id_to_idx, spacy_nlp)
        # Optional:
        # torch.save(occurrences, 'occurrences.pt')
    elif args.split_in_out == "uni_two":
        # Special case for separating the next position, and the position two away.
        random.seed(args.seed)
        args.window_size = 2 # Window includes only the previous/next position.
        # Occurrences in the next position.
        occurrences1 = get_co_occurrences(args, token_to_id, token_id_to_idx, spacy_nlp)
        random.seed(args.seed) # Use the same random seed to ensure that the same set of sentences is used.
        args.window_size = 3
        occurrences2 = get_co_occurrences(args, token_to_id, token_id_to_idx, spacy_nlp)
        # Occurrences in next two positions, not counting the immediate next position.
        occurrences2 -= occurrences1
    else:
        print("Unsupported split_in_out value.")

    # Write outputs.
    print("Writing outputs.")
    outfile = codecs.open(args.output, 'w', encoding='utf-8')
    outfile.write("Token\tDiversity\tDiversityEntropy\tCountPerThousand\tMLU\n")
    # If necessary, add back wordbank tokens from before lemmatization.
    if args.lemmas:
        for wordform, lemma in wordform_to_lemma.items():
            if wordform not in token_to_id: # Wordform was otherwise not considered.
                token_to_id[wordform] = token_to_id[lemma] # Add the wordform, token_id pair.
            # Update the frequency of the wordform based on the lemma.
            if lemma in token_frequencies: # Otherwise the frequency is 0.0.
                token_frequencies[wordform] = token_frequencies[lemma]
            if lemma in token_mlus: # Same for MLUs, but will impute NA.
                token_mlus[wordform] = token_mlus[lemma]
    # Compute diversity for each token.
    for token, token_id in token_to_id.items():
        occurrence_idx = token_id_to_idx[token_id]
        # Get co-occurrence diversity metrics for the desired token.
        if args.split_in_out == "split":
            # Counts the number of distinct tokens appearing after, plus the number of
            # distinct tokens appearing before the target token (given the window size).
            # Qualitatively, these results were similar to the results for "bidirectional".
            # Note: in occurrences, only tokens appearing after the current token were counted in rows.
            # Then, the tokens appearing before the current token are counted in columns.
            out_diversity, out_entropy = compute_diversity(occurrences[occurrence_idx])
            in_diversity, in_entropy = compute_diversity(torch.transpose(occurrences, 0, 1)[occurrence_idx])
            diversity = in_diversity + out_diversity
            diversity_entropy = (in_entropy + out_entropy)/2.0
        elif args.split_in_out == "unidirectional":
            # Counts the number of distinct tokens appearing before the target token (given the window size).
            # Note: in occurrences, only tokens appearing after the current token were counted in rows.
            diversity, diversity_entropy = compute_diversity(torch.transpose(occurrences, 0, 1)[occurrence_idx])
        elif args.split_in_out == "bidirectional":
            # Counts the number of distinct tokens appearing within the window of the target token.
            # This is the default used in the paper.
            # Note: in occurrences, tokens appearing before and after were counted in rows.
            diversity, diversity_entropy = compute_diversity(occurrences[occurrence_idx])
        elif args.split_in_out == "uni_two":
            # Counts the number of distinct tokens appearing one position before the target token, plus
            # the number of distinct tokens appearing two positions before the target token.
            # Note: in occurrences, only tokens appearing after the current token were counted in rows.
            in_diversity1, in_entropy1 = compute_diversity(torch.transpose(occurrences1, 0, 1)[occurrence_idx])
            in_diversity2, in_entropy2 = compute_diversity(torch.transpose(occurrences2, 0, 1)[occurrence_idx])
            diversity = in_diversity1 + in_diversity2
            diversity_entropy = (in_entropy1 + in_entropy2)/2.0
        # Note: uses data from original frequency files, but sums over wordforms for lemmas.
        # Same for MLUs.
        frequency = token_frequencies[token] if token in token_frequencies else 0.0
        mlu = token_mlus[token] if token in token_mlus else "NA"
        # Write outputs for the given token.
        outfile.write("{0}\t{1}\t{2}\t{3}\t{4}\n".format(token, diversity, diversity_entropy, frequency, mlu))
    outfile.close()
    print("Done.")


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    if args.lemmas in ["", "False", "false", "no", "No", "None"]:
        args.lemmas = False
    else:
        args.lemmas = True
    main(args)
