import copy

from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import corpus_bleu


def rouge_helper_prepare_results(m, p, r, f):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(m, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)


def remove_sub_strings(predicted_txt, tokens=['ᐛ ', ' ✬', '<unk>']):
    """
    remove the list of strings (tokens) from predicted_txt
    """
    for token in tokens:
        predicted_txt = predicted_txt.replace(token, "")
    return predicted_txt


def remove_sub_strings_chinese(predicted_txt, tokens=['ᐛ', '✬', '<unk>']):
    """
    remove the list of strings (tokens) from predicted_txt
    """
    for token in tokens:
        predicted_txt = predicted_txt.replace(token, "")
    return predicted_txt


def translation_paraphrase_evaluation_english_tagpa(sources, hypos, refs, print_scores=True, max_n=4, rouge_alpha=0.5,
                                                    rouge_weight_factor=1.2, rouge_stemming=True):
    """
    to evalute generated paraphrase or translations with BlEU and ROUGE scores.
    Nothing should be tokenized here.
    :param sources: source sentence to start with. e.g. ['Young woman with sheep on straw covered floor .', 'A man who is walking across the street .']
    :param hypos: generated hypotheses. should share the same shape with sources. (each source, generate one list of hypothesis sentence.) e.g. ['Young woman with sheep on straw covered floor .', 'a little girl with sheep on straw covered floor .'] for 'Young woman with sheep on straw covered floor .'
    :param refs: list of list of sentences. For each source, given a list of possible references. e.g. [['Young woman with sheep on straw covered floor .', 'Young woman on the floor .'] ['A man who is walking across the street now.', 'A man walking across the street.']]
    :return: a dictionary of scores.
    """
    import rouge  # pip install git+https://github.com/Mohan-Zhang-u/py-rouge.git
    sources_refs = [[sentence] for sentence in
                    sources]  # we use source as the reference to compute a negative score, in order to measure the diversity of paraphrasing.

    metrics_dict = {}

    for aggregator in ['Avg', 'Best']:
        apply_avg = aggregator == 'Avg'
        apply_best = aggregator == 'Best'

        evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                                max_n=max_n,
                                apply_avg=apply_avg,
                                apply_best=apply_best,
                                alpha=rouge_alpha,  # Default F1_score
                                weight_factor=rouge_weight_factor,
                                stemming=rouge_stemming)

        compare_dict = {'hypos': hypos, 'sources': sources, 'sources_refs_diversity_negative': hypos}
        for key in compare_dict:
            if key == 'sources_refs_diversity_negative':
                scores = evaluator.get_scores(compare_dict[key], sources_refs)
            else:
                scores = evaluator.get_scores(compare_dict[key], refs)
            metrics_dict[key + '_rouge_' + aggregator] = scores

            if print_scores:
                print('Evaluation with {} with {}'.format(key, aggregator))
                for metric, results in sorted(scores.items(), key=lambda x: x[0]):
                    if not apply_avg and not apply_best:  # value is a type of list as we evaluate each summary vs each reference
                        for hypothesis_id, results_per_ref in enumerate(results):
                            nb_references = len(results_per_ref['p'])
                            for reference_id in range(nb_references):
                                print('\tHypothesis #{} & Reference #{}: '.format(hypothesis_id, reference_id))
                                print('\t' + rouge_helper_prepare_results(metric, results_per_ref['p'][reference_id],
                                                                          results_per_ref['r'][reference_id],
                                                                          results_per_ref['f'][reference_id]))
                        print()
                    else:
                        print(rouge_helper_prepare_results(metric, results['p'], results['r'], results['f']))
                print()

    bleu_sources = []
    for source in sources:
        bleu_sources.append(word_tokenize(source))
    bleu_hypos = []
    for hypo in hypos:
        bleu_hypos.append(word_tokenize(hypo))
    bleu_refs = copy.deepcopy(refs)
    for sub_ref in bleu_refs:
        for i in range(len(sub_ref)):
            sub_ref[i] = word_tokenize(sub_ref[i])
    for sources_ref in sources_refs:
        for i in range(len(sources_ref)):
            sources_ref[i] = word_tokenize(sources_ref[i])

    # metrics_dict["bleu_no_weights"] = corpus_bleu(refs, hypos)
    metrics_dict["bleu_1"] = corpus_bleu(bleu_refs, bleu_hypos, weights=(1, 0, 0, 0))
    metrics_dict["bleu_2"] = corpus_bleu(bleu_refs, bleu_hypos, weights=(0.5, 0.5, 0, 0))
    metrics_dict["bleu_3"] = corpus_bleu(bleu_refs, bleu_hypos, weights=(0.33, 0.33, 0.34, 0))
    metrics_dict["bleu_4"] = corpus_bleu(bleu_refs, bleu_hypos, weights=(0.25, 0.25, 0.25, 0.25))

    metrics_dict["source_sentence_bleu_1"] = corpus_bleu(bleu_refs, bleu_sources, weights=(1, 0, 0, 0))
    metrics_dict["source_sentence_bleu_2"] = corpus_bleu(bleu_refs, bleu_sources, weights=(0.5, 0.5, 0, 0))
    metrics_dict["source_sentence_bleu_3"] = corpus_bleu(bleu_refs, bleu_sources, weights=(0.33, 0.33, 0.34, 0))
    metrics_dict["source_sentence_bleu_4"] = corpus_bleu(bleu_refs, bleu_sources, weights=(0.25, 0.25, 0.25, 0.25))

    metrics_dict["sources_as_refs_diversity_negative_bleu_1"] = corpus_bleu(sources_refs, bleu_hypos,
                                                                            weights=(1, 0, 0, 0))
    metrics_dict["sources_as_refs_diversity_negative_bleu_2"] = corpus_bleu(sources_refs, bleu_hypos,
                                                                            weights=(0.5, 0.5, 0, 0))
    metrics_dict["sources_as_refs_diversity_negative_bleu_3"] = corpus_bleu(sources_refs, bleu_hypos,
                                                                            weights=(0.33, 0.33, 0.34, 0))
    metrics_dict["sources_as_refs_diversity_negative_bleu_4"] = corpus_bleu(sources_refs, bleu_hypos,
                                                                            weights=(0.25, 0.25, 0.25, 0.25))

    if print_scores:
        for sc in ["bleu_1", "bleu_2", "bleu_3", "bleu_4", "source_sentence_bleu_1", "source_sentence_bleu_2",
                   "source_sentence_bleu_3", "source_sentence_bleu_4", "sources_as_refs_diversity_negative_bleu_1",
                   "sources_as_refs_diversity_negative_bleu_2", "sources_as_refs_diversity_negative_bleu_3",
                   "sources_as_refs_diversity_negative_bleu_4"]:
            print(sc, "(percents):", round(metrics_dict[sc], 4) * 100)

    return metrics_dict


def translation_paraphrase_evaluation(sources, hypos, refs, sentence_preproce_function=None, print_scores=True, max_n=4,
                                      rouge_alpha=0.5, rouge_weight_factor=1.2, rouge_stemming=True,
                                      hypo_style='first'):
    """
    to evalute generated paraphrase or translations with BlEU and ROUGE scores.
    Nothing should be tokenized here.
    :param sources: source sentence to start with. e.g. sources = ['Young woman with sheep on straw covered floor.', 'A man who is walking across the street.', 'A brightly lit kitchen with lots of natural light.']
    :param hypos: generated hypotheses. should share the same shape with sources. (each source, generate one list of hypothesis sentence.) e.g. [['A child places his hands on the head and neck of a sheep while another sheep looks at his face.', 'A person petting the head of a cute fluffy sheep.', 'A child is petting a sheep while another sheep watches.', 'A woman kneeling to pet animals while others wait. '], ['A busy intersection with an ice cream truck driving by.', 'a man walks behind an ice cream truck ', 'A man is crossing a street near an icecream truck.', 'The man is walking behind the concession bus.'], ['A modern kitchen in white with stainless steel lights.', 'A kitchen filled with lots of white counter space.', 'A KITCHEN IN THE ROOM WITH WHITE APPLIANCES ', 'A modern home kitchen and sitting area looking out towards the back yard']]
    :param refs: list of list of sentences. For each source, given a list of possible references. e.g. [['A woman standing next to a sheep in a pen .<unk>', 'A woman standing next to a sheep on a farm .<unk>', 'A woman standing next to a sheep in a barn .<unk>', 'A woman standing next to a sheep in a field .<unk>', 'A woman standing next to a sheep in a barn<unk>'], ['A man crossing the street in front of a store .<unk>', 'A man crossing the street in a city .<unk>', 'A person crossing the street in a city .<unk>', 'A man crossing the street in the middle of a city<unk>', 'A man crossing the street in the middle of a city street<unk>'], ['a kitchen with a stove a microwave and a sink<unk>', 'a kitchen with a stove a sink and a microwave<unk>', 'a kitchen with a stove a sink and a refrigerator<unk>', 'A kitchen with a sink , stove , microwave and window .<unk>', 'a kitchen with a stove a sink and a window<unk>']]
    :param hypo_style: how to evaluate the generated hypotheses. Pick the first? Choose the one with best evalution score? Average the scores on all hypotheses? Should be one of ['first', 'best', 'average']
    :param sentence_preproce_function: a function that will be applied to all sentences in sources, hypos, refs
    :return: a dictionary of scores.
    """
    import rouge  # pip install git+https://github.com/Mohan-Zhang-u/py-rouge.git
    assert (isinstance(sources, list))
    assert (isinstance(sources[0], str))
    assert (isinstance(hypos, list))
    assert (isinstance(hypos[0], list))
    assert (isinstance(hypos[0][0], str))
    assert (isinstance(refs, list))
    assert (isinstance(refs[0], list))
    assert (isinstance(refs[0][0], str))

    if hypo_style == 'first':
        hypos = [hypo[0] for hypo in hypos]
    else:
        raise NotImplementedError

    # apply sentence_preproce_function, e.g. remove_tokens
    if sentence_preproce_function is not None:
        sources = [sentence_preproce_function(source) for source in sources]
        if hypo_style == 'first':
            hypos = [sentence_preproce_function(hypo) for hypo in hypos]
        else:
            raise NotImplementedError
            hypos = [[sentence_preproce_function(hypo) for hypo in hypo_list] for hypo_list in hypos]
        refs = [[sentence_preproce_function(ref) for ref in refs_list] for refs_list in refs]

    sources_refs = [[sentence] for sentence in
                    sources]  # we use source as the reference to compute a negative score, in order to measure the diversity of paraphrasing.
    metrics_dict = {}

    for aggregator in ['Avg', 'Best']:
        apply_avg = aggregator == 'Avg'
        apply_best = aggregator == 'Best'

        evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                                max_n=max_n,
                                apply_avg=apply_avg,
                                apply_best=apply_best,
                                alpha=rouge_alpha,  # Default F1_score
                                weight_factor=rouge_weight_factor,
                                stemming=rouge_stemming)

        compare_dict = {'hypos': hypos, 'sources': sources, 'sources_refs_diversity_negative': hypos}
        for key in compare_dict:
            if key == 'sources_refs_diversity_negative':
                scores = evaluator.get_scores(compare_dict[key], sources_refs)
            else:
                scores = evaluator.get_scores(compare_dict[key], refs)
            metrics_dict[key + '_rouge_' + aggregator] = scores

            if print_scores:
                print('Evaluation with {} with {}'.format(key, aggregator))
                for metric, results in sorted(scores.items(), key=lambda x: x[0]):
                    if not apply_avg and not apply_best:  # value is a type of list as we evaluate each summary vs each reference
                        for hypothesis_id, results_per_ref in enumerate(results):
                            nb_references = len(results_per_ref['p'])
                            for reference_id in range(nb_references):
                                print('\tHypothesis #{} & Reference #{}: '.format(hypothesis_id, reference_id))
                                print('\t' + rouge_helper_prepare_results(metric, results_per_ref['p'][reference_id],
                                                                          results_per_ref['r'][reference_id],
                                                                          results_per_ref['f'][reference_id]))
                        print()
                    else:
                        print(rouge_helper_prepare_results(metric, results['p'], results['r'], results['f']))
                print()

    bleu_sources = []
    for source in sources:
        bleu_sources.append(word_tokenize(source))
    bleu_hypos = []
    if hypo_style == 'first':
        for hypo in hypos:
            bleu_hypos.append(word_tokenize(hypo))
    else:
        raise NotImplementedError
        bleu_hypos = copy.deepcopy(hypos)
        for sub_hypo in bleu_hypos:
            for i in range(len(sub_hypo)):
                sub_hypo[i] = word_tokenize(sub_hypo[i])
    bleu_refs = copy.deepcopy(refs)
    for sub_ref in bleu_refs:
        for i in range(len(sub_ref)):
            sub_ref[i] = word_tokenize(sub_ref[i])
    for sources_ref in sources_refs:
        for i in range(len(sources_ref)):
            sources_ref[i] = word_tokenize(sources_ref[i])

    # print(corpus_bleu(bleu_refs, bleu_hypos, weights=(1, 0, 0, 0)))
    # return
    metrics_dict["bleu_1"] = corpus_bleu(bleu_refs, bleu_hypos, weights=(1, 0, 0, 0))
    metrics_dict["bleu_2"] = corpus_bleu(bleu_refs, bleu_hypos, weights=(0.5, 0.5, 0, 0))
    metrics_dict["bleu_3"] = corpus_bleu(bleu_refs, bleu_hypos, weights=(0.33, 0.33, 0.34, 0))
    metrics_dict["bleu_4"] = corpus_bleu(bleu_refs, bleu_hypos, weights=(0.25, 0.25, 0.25, 0.25))

    metrics_dict["source_sentence_bleu_1"] = corpus_bleu(bleu_refs, bleu_sources, weights=(1, 0, 0, 0))
    metrics_dict["source_sentence_bleu_2"] = corpus_bleu(bleu_refs, bleu_sources, weights=(0.5, 0.5, 0, 0))
    metrics_dict["source_sentence_bleu_3"] = corpus_bleu(bleu_refs, bleu_sources, weights=(0.33, 0.33, 0.34, 0))
    metrics_dict["source_sentence_bleu_4"] = corpus_bleu(bleu_refs, bleu_sources, weights=(0.25, 0.25, 0.25, 0.25))

    metrics_dict["sources_as_refs_diversity_negative_bleu_1"] = corpus_bleu(sources_refs, bleu_hypos,
                                                                            weights=(1, 0, 0, 0))
    metrics_dict["sources_as_refs_diversity_negative_bleu_2"] = corpus_bleu(sources_refs, bleu_hypos,
                                                                            weights=(0.5, 0.5, 0, 0))
    metrics_dict["sources_as_refs_diversity_negative_bleu_3"] = corpus_bleu(sources_refs, bleu_hypos,
                                                                            weights=(0.33, 0.33, 0.34, 0))
    metrics_dict["sources_as_refs_diversity_negative_bleu_4"] = corpus_bleu(sources_refs, bleu_hypos,
                                                                            weights=(0.25, 0.25, 0.25, 0.25))

    if print_scores:
        for sc in ["bleu_1", "bleu_2", "bleu_3", "bleu_4", "source_sentence_bleu_1", "source_sentence_bleu_2",
                   "source_sentence_bleu_3", "source_sentence_bleu_4", "sources_as_refs_diversity_negative_bleu_1",
                   "sources_as_refs_diversity_negative_bleu_2", "sources_as_refs_diversity_negative_bleu_3",
                   "sources_as_refs_diversity_negative_bleu_4"]:
            print(sc, "(percents):", round(metrics_dict[sc], 4) * 100)

    return metrics_dict


def translation_paraphrase_evaluation_chinese(sources, hypos, refs, sentence_preproce_function=None, print_scores=True,
                                              max_n=4, rouge_alpha=0.5, rouge_weight_factor=1.2, rouge_stemming=True,
                                              hypo_style='first', word_segmentor='character'):
    """
    to evalute generated paraphrase or translations with BlEU and ROUGE scores.
    Nothing should be tokenized here.
    :param sources: source sentence to start with. 
    :param hypos: generated hypotheses. should share the same shape with sources. (each source, generate one list of hypothesis sentence.)
    :param refs: list of list of sentences. For each source, given a list of possible references.
    :param hypo_style: how to evaluate the generated hypotheses. Pick the first? Choose the one with best evalution score? Average the scores on all hypotheses? Should be one of ['first', 'best', 'average']
    :param sentence_preproce_function: a function that will be applied to all sentences in sources, hypos, refs
    :param word_segmentor: 'character' means seperate each character to be a word, 'hanlp' means an hanlp chinese tokenizer.
    :return: a dictionary of scores.
    """
    import rouge  # pip install git+https://github.com/Mohan-Zhang-u/py-rouge.git
    assert (isinstance(sources, list))
    assert (isinstance(sources[0], str))
    assert (isinstance(hypos, list))
    assert (isinstance(hypos[0], list))
    assert (isinstance(hypos[0][0], str))
    assert (isinstance(refs, list))
    assert (isinstance(refs[0], list))
    assert (isinstance(refs[0][0], str))

    # apply sentence_preproce_function, e.g. remove_tokens
    if sentence_preproce_function is not None:
        sources = [sentence_preproce_function(source) for source in sources]
        hypos = [[sentence_preproce_function(hypo) for hypo in hypo_list] for hypo_list in hypos]
        refs = [[sentence_preproce_function(ref) for ref in refs_list] for refs_list in refs]

    sources_refs = [[sentence] for sentence in
                    sources]  # we use source as the reference to compute a negative score, in order to measure the diversity of paraphrasing.
    metrics_dict = {}

    # tokenize chinese sentences.
    if word_segmentor == 'character':
        sources = [' '.join(source) for source in sources]
        refs = [[' '.join(ref) for ref in ref_list] for ref_list in refs]
        sources_refs = [[' '.join(ref) for ref in ref_list] for ref_list in sources_refs]
        hypos = [[' '.join(hypo) for hypo in hypo_list] for hypo_list in hypos]

        def word_tokenize(sentence):
            return sentence.split(' ')

        bleu_sources = []
        for source in sources:
            bleu_sources.append(word_tokenize(source))
        bleu_hypos = copy.deepcopy(hypos)
        for sub_hypo in bleu_hypos:
            for i in range(len(sub_hypo)):
                sub_hypo[i] = word_tokenize(sub_hypo[i])
        bleu_refs = copy.deepcopy(refs)
        for sub_ref in bleu_refs:
            for i in range(len(sub_ref)):
                sub_ref[i] = word_tokenize(sub_ref[i])
        bleu_sources_refs = copy.deepcopy(sources_refs)
        for sources_ref in bleu_sources_refs:
            for i in range(len(sources_ref)):
                sources_ref[i] = word_tokenize(sources_ref[i])

    elif word_segmentor == 'hanlp':
        import hanlp
        word_tokenize = hanlp.load('LARGE_ALBERT_BASE')

        bleu_sources = []
        for source in sources:
            bleu_sources.append(word_tokenize(source))
        bleu_hypos = copy.deepcopy(hypos)
        for sub_hypo in bleu_hypos:
            for i in range(len(sub_hypo)):
                sub_hypo[i] = word_tokenize(sub_hypo[i])
        bleu_refs = copy.deepcopy(refs)
        for sub_ref in bleu_refs:
            for i in range(len(sub_ref)):
                sub_ref[i] = word_tokenize(sub_ref[i])
        bleu_sources_refs = copy.deepcopy(sources_refs)
        for bleu_sources_ref in bleu_sources_refs:
            for i in range(len(bleu_sources_ref)):
                bleu_sources_ref[i] = word_tokenize(bleu_sources_ref[i])

        sources = [' '.join(source) for source in bleu_sources]
        refs = [[' '.join(ref) for ref in ref_list] for ref_list in bleu_refs]
        sources_refs = [[' '.join(ref) for ref in ref_list] for ref_list in bleu_sources_refs]
        hypos = [[' '.join(hypo) for hypo in hypo_list] for hypo_list in bleu_hypos]

    if hypo_style == 'first':
        hypos = [hypo[0] for hypo in hypos]
        bleu_hypos = [hypo[0] for hypo in bleu_hypos]
    else:
        raise NotImplementedError

    for aggregator in ['Avg', 'Best']:
        apply_avg = aggregator == 'Avg'
        apply_best = aggregator == 'Best'

        evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                                max_n=max_n,
                                apply_avg=apply_avg,
                                apply_best=apply_best,
                                alpha=rouge_alpha,  # Default F1_score
                                weight_factor=rouge_weight_factor,
                                stemming=rouge_stemming,
                                language='chinese')

        compare_dict = {'hypos': hypos, 'sources': sources, 'sources_refs_diversity_negative': hypos}
        for key in compare_dict:
            if key == 'sources_refs_diversity_negative':
                scores = evaluator.get_scores(compare_dict[key], sources_refs)
            else:
                scores = evaluator.get_scores(compare_dict[key], refs)
            metrics_dict[key + '_rouge_' + aggregator] = scores

            if print_scores:
                print('Evaluation with {} with {}'.format(key, aggregator))
                for metric, results in sorted(scores.items(), key=lambda x: x[0]):
                    if not apply_avg and not apply_best:  # value is a type of list as we evaluate each summary vs each reference
                        for hypothesis_id, results_per_ref in enumerate(results):
                            nb_references = len(results_per_ref['p'])
                            for reference_id in range(nb_references):
                                print('\tHypothesis #{} & Reference #{}: '.format(hypothesis_id, reference_id))
                                print('\t' + rouge_helper_prepare_results(metric, results_per_ref['p'][reference_id],
                                                                          results_per_ref['r'][reference_id],
                                                                          results_per_ref['f'][reference_id]))
                        print()
                    else:
                        print(rouge_helper_prepare_results(metric, results['p'], results['r'], results['f']))
                print()

    metrics_dict["bleu_1"] = corpus_bleu(bleu_refs, bleu_hypos, weights=(1, 0, 0, 0))
    metrics_dict["bleu_2"] = corpus_bleu(bleu_refs, bleu_hypos, weights=(0.5, 0.5, 0, 0))
    metrics_dict["bleu_3"] = corpus_bleu(bleu_refs, bleu_hypos, weights=(0.33, 0.33, 0.34, 0))
    metrics_dict["bleu_4"] = corpus_bleu(bleu_refs, bleu_hypos, weights=(0.25, 0.25, 0.25, 0.25))

    metrics_dict["source_sentence_bleu_1"] = corpus_bleu(bleu_refs, bleu_sources, weights=(1, 0, 0, 0))
    metrics_dict["source_sentence_bleu_2"] = corpus_bleu(bleu_refs, bleu_sources, weights=(0.5, 0.5, 0, 0))
    metrics_dict["source_sentence_bleu_3"] = corpus_bleu(bleu_refs, bleu_sources, weights=(0.33, 0.33, 0.34, 0))
    metrics_dict["source_sentence_bleu_4"] = corpus_bleu(bleu_refs, bleu_sources, weights=(0.25, 0.25, 0.25, 0.25))

    metrics_dict["sources_as_refs_diversity_negative_bleu_1"] = corpus_bleu(bleu_sources_refs, bleu_hypos,
                                                                            weights=(1, 0, 0, 0))
    metrics_dict["sources_as_refs_diversity_negative_bleu_2"] = corpus_bleu(bleu_sources_refs, bleu_hypos,
                                                                            weights=(0.5, 0.5, 0, 0))
    metrics_dict["sources_as_refs_diversity_negative_bleu_3"] = corpus_bleu(bleu_sources_refs, bleu_hypos,
                                                                            weights=(0.33, 0.33, 0.34, 0))
    metrics_dict["sources_as_refs_diversity_negative_bleu_4"] = corpus_bleu(bleu_sources_refs, bleu_hypos,
                                                                            weights=(0.25, 0.25, 0.25, 0.25))

    if print_scores:
        for sc in ["bleu_1", "bleu_2", "bleu_3", "bleu_4", "source_sentence_bleu_1", "source_sentence_bleu_2",
                   "source_sentence_bleu_3", "source_sentence_bleu_4", "sources_as_refs_diversity_negative_bleu_1",
                   "sources_as_refs_diversity_negative_bleu_2", "sources_as_refs_diversity_negative_bleu_3",
                   "sources_as_refs_diversity_negative_bleu_4"]:
            print(sc, "(percents):", round(metrics_dict[sc], 4) * 100)

    return metrics_dict
