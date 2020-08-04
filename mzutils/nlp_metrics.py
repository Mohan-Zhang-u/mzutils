import copy
from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize import word_tokenize
import rouge  # from https://github.com/Mohan-Zhang-u/py-rouge.git


def rouge_helper_prepare_results(m, p, r, f):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(m, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)


def translation_paraphrase_evaluation(sources, hypos, refs, print_rouge=True, max_n=4, rouge_alpha=0.5, rouge_weight_factor=1.2, rouge_stemming=True):
    """
    to evalute generated paraphrase or translations with BlEU and ROUGE scores.
    :param sources: source sentence to start with. e.g. ['Young woman with sheep on straw covered floor .', 'A man who is walking across the street .']
    :param hypos: generated hypotheses. should share the same shape with sources. (each source, generate one hypothesis sentence.) e.g. ['Young woman with sheep on straw covered floor .', 'A man who is walking across the street now.']
    :param refs: list of list of sentences. For each source, given a list of possible references. e.g. [['Young woman with sheep on straw covered floor .', 'Young woman on the floor .'] ['A man who is walking across the street now.', 'A man walking across the street.']]
    :return:
    """
    
    metrics_dict = {}

    for aggregator in ['Avg', 'Best']:
        apply_avg = aggregator == 'Avg'
        apply_best = aggregator == 'Best'

        evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                            max_n=max_n,
                            apply_avg=apply_avg,
                            apply_best=apply_best,
                            alpha=rouge_alpha, # Default F1_score
                            weight_factor=rouge_weight_factor,
                            stemming=rouge_stemming)
        
        scores = evaluator.get_scores(hypos, refs)
        metrics_dict['rouge_'+aggregator] = scores

        if print_rouge:
            print('Evaluation with {}'.format(aggregator))
            for metric, results in sorted(scores.items(), key=lambda x: x[0]):
                if not apply_avg and not apply_best: # value is a type of list as we evaluate each summary vs each reference
                    for hypothesis_id, results_per_ref in enumerate(results):
                        nb_references = len(results_per_ref['p'])
                        for reference_id in range(nb_references):
                            print('\tHypothesis #{} & Reference #{}: '.format(hypothesis_id, reference_id))
                            print('\t' + rouge_helper_prepare_results(metric,results_per_ref['p'][reference_id], results_per_ref['r'][reference_id], results_per_ref['f'][reference_id]))
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

    # metrics_dict["bleu_no_weights"] = corpus_bleu(refs, hypos)
    metrics_dict["bleu_1"] = corpus_bleu(bleu_refs, bleu_hypos, weights=(1, 0, 0, 0))
    metrics_dict["bleu_2"] = corpus_bleu(bleu_refs, bleu_hypos, weights=(0.5, 0.5, 0, 0))
    metrics_dict["bleu_3"] = corpus_bleu(bleu_refs, bleu_hypos, weights=(0.33, 0.33, 0.34, 0))
    metrics_dict["bleu_4"] = corpus_bleu(bleu_refs, bleu_hypos, weights=(0.25, 0.25, 0.25, 0.25))

    metrics_dict["ground_truth_bleu_1"] = corpus_bleu(bleu_refs, bleu_sources, weights=(1, 0, 0, 0))
    metrics_dict["ground_truth_bleu_2"] = corpus_bleu(bleu_refs, bleu_sources, weights=(0.5, 0.5, 0, 0))
    metrics_dict["ground_truth_bleu_3"] = corpus_bleu(bleu_refs, bleu_sources, weights=(0.33, 0.33, 0.34, 0))
    metrics_dict["ground_truth_bleu_4"] = corpus_bleu(bleu_refs, bleu_sources, weights=(0.25, 0.25, 0.25, 0.25))


    return metrics_dict