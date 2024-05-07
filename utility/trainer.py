import torch
import utility.batch_test
import utility.tools
from time import time

def universal_trainer(model, args, config, dataset, device):
    model.to(device)

    Optim = torch.optim.Adam(model.parameters(), lr=float(config['learn_rate']))

    best_report_recall = 0.
    best_report_epoch = 0

    for epoch in range(int(config['training_epochs'])):
        start_time = time()

        model.train()

        sample_data = dataset.sample_data_to_train_all()
        users = torch.Tensor(sample_data[:, 0]).long()
        pos_items = torch.Tensor(sample_data[:, 1]).long()
        neg_items = torch.Tensor(sample_data[:, 2]).long()

        users = users.to(device)
        pos_items = pos_items.to(device)
        neg_items = neg_items.to(device)

        users, pos_items, neg_items = utility.tools.shuffle(users, pos_items, neg_items)
        num_batch = len(users) // int(config['batch_size']) + 1

        total_loss_list = []

        for batch_i, (batch_users, batch_positive, batch_negative) in \
                enumerate(utility.tools.mini_batch(users, pos_items, neg_items, batch_size=int(config['batch_size']))):
            loss_list = model(batch_users, batch_positive, batch_negative)

            if batch_i == 0:
                assert len(loss_list) >= 1
                total_loss_list = [0.] * len(loss_list)

            total_loss = 0.
            for i in range(len(loss_list)):
                loss = loss_list[i]
                total_loss += loss
                total_loss_list[i] += loss.item()

            Optim.zero_grad()
            total_loss.backward()
            Optim.step()

        end_time = time()

        loss_strs = str(round(sum(total_loss_list) / num_batch, 6)) \
                    + " = " + " + ".join([str(round(i / num_batch, 6)) for i in total_loss_list])

        print("\t Epoch: %4d| train time: %.3f | train_loss: %s" % (epoch + 1, end_time - start_time, loss_strs))

        if epoch % args.verbose == 0:
            if int(config["sparsity_test"]) == 0:
                result = utility.batch_test.Test(dataset, model, device, config)
                if result['recall'][0] > best_report_recall:
                    best_report_epoch = epoch + 1
                    best_report_recall = result['recall'][0]
#                     user_emb, item_emb = model.get_embedding()
#                     torch.save(item_emb, str(epoch) + "_item_embedding"+".pth")
#                     torch.save(user_emb, str(epoch) + "_user_embedding"+".pth")  

                print("\t Recall:", result['recall'], "NDCG:", result['ndcg'], "precision:", result['precision'])
            else:
                result = utility.batch_test.sparsity_test(dataset, model, device, config)
                if result[0]['recall'][0] > best_report_recall:
                    best_report_epoch = epoch + 1
                    best_report_recall = result[0]['recall'][0]
                print("\t level_1: recall:", result[0]['recall'],  ',ndcg:',
                      result[0]['ndcg'])
                print("\t level_2: recall:", result[1]['recall'],  ',ndcg:',
                      result[1]['ndcg'])
                print("\t level_3: recall:", result[2]['recall'],  ',ndcg:',
                      result[2]['ndcg'])
                print("\t level_4: recall:", result[3]['recall'],  ',ndcg:',
                      result[3]['ndcg'])

    print("\t Model training process completed.")

    print("\t best epoch:", best_report_epoch)
    print("\t best recall:", best_report_recall)