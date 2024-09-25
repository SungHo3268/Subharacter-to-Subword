# import datasets
import evaluate
import numpy as np
from tqdm import tqdm
from konlpy.tag import Mecab
from transformers import *
from typing import List
import os
import sys
sys.path.append(os.getcwd())
from nlg_tasks.srcs.korouge import Rouge
from nlg_tasks.srcs.kobert_score import BERTScore


## For heatmap
data = {
    'bleu3': [],
    'bleu4': [],
    'rouge2': [],
    'rougeL': [],
    'meteor': [],
    'mbert': [],
    'kobert': [],
    'coverage': [],
    'total_avg': {}
}


bleu_metric = evaluate.load('bleu')
meteor_metric = evaluate.load('meteor')
rouge = Rouge()

def coverage_score(preds, concept_sets, tokenizer):
    covs = []
    for p, cs in tqdm(zip(preds, concept_sets), total=len(preds)):
        cs = set(cs)
        if '#' in cs:
            cs.remove('#')
        lemmas = set()
        for token in tokenizer(p):
            lemmas.add(token)
        cov = len(lemmas & cs) / len(cs)
        covs.append(cov)
    data['coverage'].extend(covs)
    return sum(covs) / len(covs)


def scoring(preds, concept_sets, tokenizer):
    Coverage = coverage_score(preds, concept_sets, tokenizer)
    # print(f"System level Concept Coverage: {Coverage * 100:.2f}")


def eval_main(refs_list: List[List[str]], preds_list: List[str], concepts_list: List[str]):
    """
    :param refs_list: [[ref1, ref2, ref3], [ref1, ref2, ref3], ...]
    :param preds_list: [pred1, pred2, pred3, ...]
    :param concepts_list: [concept1, concept2, concept3, ...]

    e.g.,
    refs_list = [['창문 앞에 핫도그 두개가 진열되어 있다.', '진열 창문 앞에 두 개의 핫도그가 있다.', '창문 앞에 진열된 핫도그는 두 개다.'],
                 [...], ...]
    preds_list = ['그 창문은 두개의 핫도그가 진열되어 있다.',
                 ...]
    concepts_list = ['진열#두#창문 앞#그#핫',
                     ...]
    """
    # print("Start KommonGen Evaluation")

    num_label_set = len(refs_list[0])

    mmodel = "bert-base-multilingual-cased"
    kmodel = "monologg/kobert"
    kbertscore = BERTScore(kmodel, best_layer=2)
    mbertscore = BERTScore(mmodel, best_layer=2)
    concept_sets = []

    rouge2 = []
    rougeL = []
    bleu_predictions = []
    bleu_references = []
    met_references = []
    met_predictions = []
    mecab_tokenizer = Mecab().morphs

    bert_prds = []
    bert_refs = []

    preds_list = [prd.strip() for prd in preds_list]

    for ref, prd, cpt in zip(refs_list, preds_list, concepts_list):
        ref = [r.strip() for r in ref]

        concept_set = mecab_tokenizer(cpt.strip())
        concept_sets.append(concept_set)

        # For BLEU score
        bleu_references.extend([' '.join(mecab_tokenizer(rs)) for rs in ref])             # give 3x stride to refs index per one prediction
        bleu_predictions.append(' '.join(mecab_tokenizer(prd)))

        # For METEOR score
        met_references.extend(ref)              # give 3x stride to refs index per one prediction
        met_predictions.append(prd)

        # For Rouge score
        max_rouge_2_score = rouge.calc_score_2(prd, ref)
        max_rouge_L_score = rouge.calc_score_L(prd, ref)
        rouge2.append(max_rouge_2_score)
        rougeL.append(max_rouge_L_score)

        bert_prds.extend([prd for _ in range(len(ref))])
        bert_refs.extend(ref)

    cur_mBS = mbertscore(bert_refs, bert_prds, batch_size=128)
    cur_kBS = kbertscore(bert_refs, bert_prds, batch_size=128)


    # BLEU 3/4 - max
    bleu3, bleu4, meteors = [], [], []
    for i in range(len(bleu_predictions)):
        if num_label_set == 1:
            # print("")
            # print(f"bleu_predictions: {bleu_predictions[i]}")
            # print(f"bleu_references: {bleu_references[i]}")
            bleu_ref = bleu_metric.compute(predictions=[bleu_predictions[i]], references=[bleu_references[i]], max_order=4)

            bleu3.append(round(bleu_ref['precisions'][2], 4))
            bleu4.append(round(bleu_ref['precisions'][3], 4))
            met_ref = meteor_metric.compute(predictions=[met_predictions[i]], references=[met_references[i]])
            meteors.append(round(met_ref['meteor'], 4))
        else:
            bleu_ref_first = bleu_metric.compute(predictions=[bleu_predictions[i]], references=[bleu_references[num_label_set*i]], max_order=4)
            bleu_ref_second = bleu_metric.compute(predictions=[bleu_predictions[i]], references=[bleu_references[num_label_set*i+1]], max_order=4)
            bleu_ref_third = bleu_metric.compute(predictions=[bleu_predictions[i]], references=[bleu_references[num_label_set*i+2]], max_order=4)

            bleu3_max = max(round(bleu_ref_first['precisions'][2], 4), round(bleu_ref_second['precisions'][2], 4), round(bleu_ref_third['precisions'][2], 4))
            bleu3.append(bleu3_max)
            bleu4_max = max(round(bleu_ref_first['precisions'][3], 4), round(bleu_ref_second['precisions'][3], 4), round(bleu_ref_third['precisions'][3], 4))
            bleu4.append(bleu4_max)

            meteor_ref_first = meteor_metric.compute(predictions=[met_predictions[i]], references=[met_references[3 * i]])
            meteor_ref_second = meteor_metric.compute(predictions=[met_predictions[i]], references=[met_references[3 * i + 1]])
            meteor_ref_third = meteor_metric.compute(predictions=[met_predictions[i]], references=[met_references[3 * i + 2]])

            meteor_max = max(round(meteor_ref_first['meteor'], 4),
                             round(meteor_ref_second['meteor'], 4),
                             round(meteor_ref_third['meteor'], 4))
            meteors.append(meteor_max)

    ktmp = []
    mtmp = []
    cur_kBS_heatmap = []
    cur_mBS_heatmap = []
    for i, (k, m) in enumerate(zip(cur_kBS, cur_mBS)):
        if num_label_set == 1:
            cur_kBS_heatmap.append(k)
            cur_mBS_heatmap.append(m)
        else:
            if (i+1) % num_label_set == 0:
                kscore = np.mean(ktmp)
                mscore = np.mean(mtmp)
                cur_kBS_heatmap.append(kscore)
                cur_mBS_heatmap.append(mscore)
                ktmp = []
                mtmp = []
            else:
                ktmp.append(k)
                mtmp.append(m)

    # Coverage
    scoring(preds_list, concept_sets, mecab_tokenizer)

    # print("BLEU 3: ", round(np.mean(bleu3), 4))
    # print("BLEU 4: ", round(np.mean(bleu4), 4))
    #
    # print("ROUGE-2: ", round(np.mean(rouge2), 4))
    # print("ROUGE-L: ", round(np.mean(rougeL), 4))
    #
    # print("METEOR: ", round(np.mean(meteors), 4))
    #
    # print("mBERTScore: ", round(np.mean(cur_mBS), 4))
    # print("KoBERTScore: ", round(np.mean(cur_kBS), 4))

    data['bleu3'] = bleu3
    data['bleu4'] = bleu4
    data['rouge2'] = rouge2
    data['rougeL'] = rougeL
    data['meteor'] = meteors
    data['kobert'] = cur_kBS_heatmap
    data['mbert'] = cur_mBS_heatmap

    data['total_avg']['bleu3'] = round(np.mean(bleu3), 4)
    data['total_avg']['bleu4'] = round(np.mean(bleu4), 4)
    data['total_avg']['rouge2'] = round(np.mean(rouge2), 4)
    data['total_avg']['rougeL'] = round(np.mean(rougeL), 4)
    data['total_avg']['meteor'] = round(np.mean(meteors), 4)
    data['total_avg']['mbert'] = round(np.mean(cur_mBS), 4)
    data['total_avg']['kobert'] = round(np.mean(cur_kBS), 4)
    data['total_avg']['coverage'] = round(np.mean(data['coverage']), 4)

    return data
