# contextual-diversity
Code and data for the paper [Does Contextual Diversity Hinder Early Word Acquisition?](https://tylerachang.github.io/paper_pdfs/cogsci_2022_contextual_diversity_word_acquisition.pdf) (CogSci 2022).
Includes code for computing contextual diversities for words in a corpus.
Run on Python 3.9.13 and R 4.2.0.
Data is in cogsci_data/adjusted_diversities.

## Child age of acquisition (AoA) data.
Child age of acquisition data is pulled from Wordbank (Frank et al., 2017).
Child-directed speech data is pulled from CHILDES (MacWhinney, 2000).
1. Get age of acquisition data (r_code/get_child_aoa.R). This pulls multilingual AoA data.
2. Clean the AoA token data from the previous step (clean_wordbank_words.py).
This is beause the UTF-8 bytes can sometimes get offset.
This adds a CleanedWord and CleanedSingle column to the dataset, where CleanedSingle represents each word's single canonical form.
The output should be saved as the child AoA file (e.g. child_aoa.tsv).

## Raw contextual diversities.
1. Get the CHILDES data (get_childes_data.py).
This outputs a txt and tsv file for each language.
The txt file contains the raw sentences, and the tsv file contains statistics (word counts and mean sentence lengths) for each wordform.
2. Compute the raw contextual diversities for each language (get_contextual_diversities.py).
Frequency-adjusted contextual diversities are computed later in R (see below).
Sample usage for English:
<pre>
python3 get_contextual_diversities.py \
--childes_directory="cogsci_data/childes_data" \
--childes_lang="eng-na" \
--window_size=5 \
--lemmas="true" --spacy_lang="en_core_web_sm" \
--common_tokens=5000 \
--wordbank_lang="English (American)" \
--wordbank_file="cogsci_data/all_prop_data_WS_cleaned.tsv" \
--output="cogsci_data/diversities/eng-na_diversities_window5.txt"
</pre>

## Analyses.
All analyses can be reproduced using cogsci_analyses.Rmd.
This includes computing the frequency-adjusted contextual diversities and running the statistical tests.
Concreteness norms from Brysbaert et al. (2014).

## Citation.
<pre>
@inproceedings{chang-bergen-2022-contextual,
  title={Does Contextual Diversity Hinder Early Word Acquisition?},
  author={Tyler Chang and Benjamin Bergen},
  booktitle={CogSci 2022},
  year={2022},
}
</pre>