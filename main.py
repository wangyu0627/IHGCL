import torch
import time
import utility.batch_test
import utility.parser
import utility.losses
import utility.tools
import utility.trainer
from utility.data_loader import Data
from time import time
from model.model import GCRec
from utility.load_hete_data import load_data


def main():
    args = utility.parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # load train and test
    dataset = Data(args.data_path + args.dataset, args)
    # load hete model
    uu_graph, ii_graph = load_data(args.dataset, device)

    model = GCRec(args, dataset, uu_graph, ii_graph, device=device)

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_report_recall = 0.
    best_report_epoch = 0

    for epoch in range(args.epochs):
        start_time = time()

        model.train()
        sample_data = dataset.sample_data_to_train_all()
        users = torch.Tensor(sample_data[:, 0]).long()
        pos_items = torch.Tensor(sample_data[:, 1]).long()
        neg_items = torch.Tensor(sample_data[:, 2]).long()

        users = users.to(device)
        pos_items = pos_items.to(device)
        neg_items = neg_items.to(device)
        num_batch = len(users) // args.batch_size + 1
        total_loss_list = []

        for batch_i, (batch_users, batch_positive, batch_negative) in \
                enumerate(utility.tools.mini_batch(users, pos_items, neg_items,
                                                   batch_size=int(args.batch_size))):
            loss_list = model(batch_users, batch_positive, batch_negative)

            if batch_i == 0:
                assert len(loss_list) >= 1
                total_loss_list = [0.] * len(loss_list)

            total_loss = 0.
            for i in range(len(loss_list)):
                loss = loss_list[i]
                total_loss += loss
                total_loss_list[i] += loss.item()

            opt.zero_grad()
            total_loss.backward()
            opt.step()

            # print("\t Batch:", batch_i, " loss_list:", loss_list, "total_loss:", total_loss, total_loss_list)

        end_time = time()

        loss_strs = str(round(sum(total_loss_list) / num_batch, 6)) \
                    + " = " + " + ".join([str(round(i / num_batch, 6)) for i in total_loss_list])
        print("\t Epoch: %4d| train time: %.3f | train_loss: %s" % (epoch + 1, end_time - start_time, loss_strs))

        if epoch % args.verbose == 0:
            if not args.sparsity_test:
                result = utility.batch_test.Test(dataset, model, device, args)
                if result['recall'][0] > best_report_recall:
                    best_report_epoch = epoch + 1
                    best_report_recall = result['recall'][0]

                print("\t Recall:", result['recall'], "NDCG:", result['ndcg'])
            else:
                result = utility.batch_test.sparsity_test(dataset, model, device, args)
                if result[0]['recall'][0] > best_report_recall:
                    best_report_epoch = epoch + 1
                    best_report_recall = result[0]['recall'][0]
                print("\t level_1: recall:", result[0]['recall'], ',ndcg:', result[0]['ndcg'])
                print("\t level_2: recall:", result[1]['recall'], ',ndcg:', result[1]['ndcg'])
                print("\t level_3: recall:", result[2]['recall'], ',ndcg:', result[2]['ndcg'])
                print("\t level_4: recall:", result[3]['recall'], ',ndcg:', result[3]['ndcg'])
    # vis_embedding
    # user_emb = torch.tensor(model.user_embedding.weight, dtype=torch.float32)
    # item_emb = torch.tensor(model.item_embedding.weight, dtype=torch.float32)
    # all_embedding = torch.cat([user_emb, item_emb])
    # torch.save(all_embedding, 'data/emb/all_embedding.pth')
    # torch.save(user_emb, 'data/dbook_user_emb.pth')
    # torch.save(item_emb, 'data/dbook_item_emb.pth')

    # vis_noise
    # hete_user_embedding = []
    # hete_item_embedding = []
    # for i in range(len(uu_graph)):
    #     hete_user_embedding.append(model.user_autoencoder[i](model.uu_graph[i], user_emb).flatten(1))
    # for i in range(len(ii_graph)):
    #     hete_item_embedding.append(model.item_autoencoder[i](model.ii_graph[i], item_emb).flatten(1))
    # torch.save(hete_user_embedding[0], 'data/emb/dbmovie_user_noise_uu.pth')
    # torch.save(hete_user_embedding[1], 'data/emb/dbmovie_user_noise_ugu.pth')
    # torch.save(hete_item_embedding[0], 'data/emb/dbmovie_item_noise_mam.pth')
    # torch.save(hete_item_embedding[1], 'data/emb/dbmovie_item_noise_mdm.pth')

    print("\t Model training process completed.")
    print("\t best epoch:", best_report_epoch)
    print("\t best recall:", best_report_recall)

if __name__ == '__main__':
    main()