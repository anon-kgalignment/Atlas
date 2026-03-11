
from multiprocessing import Pool
import numpy as np
import heapq


def _one_thread(start, end, embeds1, embeds2, top_k, metric, normalize):
        result = []
        for i in range(start, end):
            e1 = embeds1[i]
            if normalize:
                e1 = e1 / np.linalg.norm(e1)

            sims = []
            for j in range(len(embeds2)):
                e2 = embeds2[j]
                if normalize:
                    e2 = e2 / np.linalg.norm(e2)

                if metric == 'cosine':
                    score = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
                elif metric == 'euclidean':
                    score = -np.linalg.norm(e1 - e2)
                else: 
                    score = np.dot(e1, e2)

                sims.append((score, j))

            top_k_sim = heapq.nlargest(max(top_k), sims)
            result.append([idx for _, idx in top_k_sim])
        return result


def greedy_alignment(embeds1, embeds2, top_k, threads_num, metric='inner', normalize=False, csls_k=0, accurate=False):
    entity_num = embeds1.shape[0]
    step = (entity_num + threads_num - 1) // threads_num
    ranges = [(i, min(i + step, entity_num)) for i in range(0, entity_num, step)]

    # Add embeds1, embeds2, top_k, etc., to each argument tuple
    args = [(start, end, embeds1, embeds2, top_k, metric, normalize) for start, end in ranges]

    with Pool(threads_num) as p:
        parts = p.starmap(_one_thread, args)

    alignment_result = [x for part in parts for x in part]

    hits = []
    for k in top_k:
        hit_k = np.mean([1 if i in alignment_result[i][:k] else 0 for i in range(len(alignment_result))])
        hits.append(hit_k)

    ranks = [alignment_result[i].index(i) + 1 if i in alignment_result[i] else len(alignment_result[i]) for i in range(len(alignment_result))]
    mr = np.mean(ranks)
    mrr = np.mean([1.0 / r for r in ranks])

    return alignment_result, hits, mr, mrr


def test(embeds1, embeds2, mapping, top_k, threads_num, metric='inner', normalize=False, csls_k=0, accurate=True):
    if mapping is None:
        alignment_rest_12, hits1_12, mr_12, mrr_12 = greedy_alignment(embeds1, embeds2, top_k, threads_num,
                                                                            metric, normalize, csls_k, accurate)
    else:
        test_embeds1_mapped = np.matmul(embeds1, mapping)
        alignment_rest_12, hits1_12, mr_12, mrr_12 = greedy_alignment(test_embeds1_mapped, embeds2, top_k, threads_num,
                                                                            metric, normalize, csls_k, accurate)
    return alignment_rest_12, hits1_12, mr_12, mrr_12
    
    