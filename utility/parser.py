import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="GC-HGNNRec")

    parser.add_argument("--seed", type=int, default=2024, help="random seed for init")
    parser.add_argument("--dataset", default="movielens-1m", help="Dataset to use, default: Movielens")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning Rate")
    parser.add_argument('--ssl_lambda', type=float, default=0.02, help='cl_rate_2')
    parser.add_argument('--intra_lambda', type=float, default=0.001, help='cl_rate_3')
    parser.add_argument('--ib_lambda', type=float, default=0.001, help='cl_rate_4')
    parser.add_argument('--IB_size', type=int, default=8, help='IB_size')
    parser.add_argument("--reg_lambda", type=float, default=0.0001, help="Regularizations")
    parser.add_argument("--temperature", type=float, default=0.2, help="temperature")
    parser.add_argument("--sparsity_test", type=int, default=0, help="sparsity_test")
    parser.add_argument('--dim', type=int, default=64, help='embedding size')
    parser.add_argument('--top_K', type=str, default="[20, 10, 5]", help='size of Top-K')
    parser.add_argument('--epochs', type=int, default=60, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=4096, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=200, help='batch size')
    parser.add_argument("--verbose", type=int, default=1, help="Test interval")
    parser.add_argument('--GCN_layer', type=int, default=2, help="the layer number of GCN")
    parser.add_argument("--data_path", nargs="?", default="data/", help="Input data path.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0")
    # hete_hyper
    parser.add_argument("--in_size", default=64, type=int, help="Initial dimension size for entities.")
    parser.add_argument("--out_size", default=64, type=int, help="Output dimension size for entities.")
    parser.add_argument('--enc_num_layer', type=int, default=1)
    parser.add_argument('--dec_num_layer', type=int, default=1)
    parser.add_argument('--mask_rate', type=float, default=0.1)
    parser.add_argument('--remask_rate', type=float, default=0.1)
    parser.add_argument('--num_remasking', type=int, default=1)
    
    return parser.parse_args()
