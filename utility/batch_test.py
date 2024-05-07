'''
PyTorch Implementation of ID-based Recommender System
This file is used to set init seed, shuffle dataset and construct the mini-batch data
'''
import torch
import numpy as np
from utility.data_loader import Data
from utility.tools import mini_batch
import utility.metrics


def Test(dataset: Data, model, device, config):
    model = model.eval()

    topK = eval(config.top_K)

    model_results = {'precision': np.zeros(len(topK)),
                     'recall': np.zeros(len(topK)),
                     'hit': np.zeros(len(topK)),
                     'ndcg': np.zeros(len(topK))}

    with torch.no_grad():
        users = list(dataset.test_dict.keys())  # get user list to test
        users_list, rating_list, ground_true_list = [], [], []
        num_batch = len(users) // int(config.test_batch_size) + 1

        for batch_users in mini_batch(users, batch_size=int(config.test_batch_size)):
            exclude_users, exclude_items = [], []
            all_positive = dataset.get_user_pos_items(batch_users)
            ground_true = [dataset.test_dict[u] for u in batch_users]

            batch_users_device = torch.Tensor(batch_users).long().to(device)

            rating = model.get_rating_for_test(batch_users_device)

            # Positive items are excluded from the recommended list
            for i, items in enumerate(all_positive):
                exclude_users.extend([i] * len(items))
                exclude_items.extend(items)
            rating[exclude_users, exclude_items] = -1

            # get the top-K recommended list for all users
            _, rating_k = torch.topk(rating, k=max(topK))

            rating = rating.cpu()
            del rating

            users_list.append(batch_users)
            rating_list.append(rating_k.cpu())
            ground_true_list.append(ground_true)

        assert num_batch == len(users_list)
        enum_list = zip(rating_list, ground_true_list)

        results = []
        for single_list in enum_list:
            results.append(test_one_batch(single_list, topK))

        for result in results:
            model_results['recall'] += result['recall']
            model_results['precision'] += result['precision']
            model_results['ndcg'] += result['ndcg']

        model_results['recall'] /= float(len(users))
        model_results['precision'] /= float(len(users))
        model_results['ndcg'] /= float(len(users))

        return model_results


def test_one_batch(X, topK):
    recommender_items = X[0].numpy()
    ground_true_items = X[1]
    r = utility.metrics.get_label(ground_true_items, recommender_items)
    precision, recall, ndcg = [], [], []

    for k_size in topK:
        recall.append(utility.metrics.recall_at_k(r, k_size, ground_true_items))
        precision.append(utility.metrics.precision_at_k(r, k_size, ground_true_items))
        ndcg.append(utility.metrics.ndcg_at_k(r, k_size, ground_true_items))

    return {'recall': np.array(recall), 'precision': np.array(precision), 'ndcg': np.array(ndcg)}

   
def sparsity_test(dataset: Data, model, device, config):
    sparsity_results = []
    model = model.eval()
    # top-20, 40, ..., 100
    topK = eval(config.top_K)

    with torch.no_grad():
        for users in dataset.split_test_dict:
            model_results = {
                'precision': np.zeros(len(topK)),
                'recall': np.zeros(len(topK)),
                'hit': np.zeros(len(topK)),
                'ndcg': np.zeros(len(topK))
            }
            users_list, rating_list, ground_true_list = [], [], []
            num_batch = len(users) // int(config.test_batch_size) + 1

            for batch_users in mini_batch(users, batch_size=int(config.test_batch_size)):
                exclude_users, exclude_items = [], []
                all_positive = dataset.get_user_pos_items(batch_users)
                ground_true = [dataset.test_dict[u] for u in batch_users]

                batch_users_device = torch.Tensor(batch_users).long().to(device)

                rating = model.get_rating_for_test(batch_users_device)

                # Positive items are excluded from the recommended list
                for i, items in enumerate(all_positive):
                    exclude_users.extend([i] * len(items))
                    exclude_items.extend(items)

                rating[exclude_users, exclude_items] = -1

                # get the top-K recommended list for all users
                _, rating_k = torch.topk(rating, k=max(topK))

                rating = rating.cpu()
                del rating

                users_list.append(batch_users)
                rating_list.append(rating_k.cpu())
                ground_true_list.append(ground_true)

            assert num_batch == len(users_list)
            enum_list = zip(rating_list, ground_true_list)

            results = []
            for single_list in enum_list:
                results.append(test_one_batch(single_list, topK))

            for result in results:
                model_results['recall'] += result['recall']
                model_results['precision'] += result['precision']
                model_results['ndcg'] += result['ndcg']

            model_results['recall'] /= float(len(users))
            model_results['precision'] /= float(len(users))
            model_results['ndcg'] /= float(len(users))
            sparsity_results.append(model_results)

    return sparsity_results