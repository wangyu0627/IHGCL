import scipy.sparse as sp
import dgl
import numpy as np
def load_data(dataset, device):
    if dataset == 'douban-movie':
        uuuuu_graph = sp.load_npz('data/douban-movie/uuuuu.npz')
        # umamu_graph = sp.load_npz('data/douban-movie/umamu.npz')
        # uumuu_graph = sp.load_npz('data/douban-movie/uumuu.npz')
        # umu_graph = sp.load_npz('data/douban-movie/umu.npz')
        ugu_graph = sp.load_npz('data/douban-movie/ugu_processed.npz')
        mam_graph = sp.load_npz('data/douban-movie/mam.npz')
        # mum_graph = sp.load_npz('data/douban-movie/mum.npz')
        # mtm_graph = sp.load_npz('data/douban-movie/mtm.npz')
        mdm_graph = sp.load_npz('data/douban-movie/mdm.npz')

        uuuuu_graph = get_graph(uuuuu_graph).to(device)
        # umamu_graph = get_graph(umamu_graph).to(device)
        # umu_graph = get_graph(umu_graph).to(device)
        # uumuu_graph = get_graph(uumuu_graph).to(device)
        ugu_graph = get_graph(ugu_graph).to(device)
        mam_graph = get_graph(mam_graph).to(device)
        # mum_graph = get_graph(mum_graph).to(device)
        # mtm_graph = get_graph(mtm_graph).to(device)
        mdm_graph = get_graph(mdm_graph).to(device)

        user_mps, item_mps = [uuuuu_graph, ugu_graph], [mam_graph, mdm_graph]

    elif dataset == 'lastfm':
        uu_graph = sp.load_npz('data/lastfm/uu.npz')
        # uau_graph = sp.load_npz('data/lastfm/uau.npz')
        uatau_graph = sp.load_npz('data/lastfm/uatau_processed.npz')
        aa_graph = sp.load_npz('data/lastfm/aa.npz')
        # aua_graph = sp.load_npz('data/lastfm/aua.npz')
        ata_graph = sp.load_npz('data/lastfm/ata_processed.npz')

        uu_graph = get_graph(uu_graph).to(device)
        # uau_graph = get_graph(uau_graph).to(device)
        uatau_graph = get_graph(uatau_graph).to(device)
        aa_graph = get_graph(aa_graph).to(device)
        # aua_graph = get_graph(aua_graph).to(device)
        ata_graph = get_graph(ata_graph).to(device)
        user_mps, item_mps = [uu_graph, uatau_graph], [aa_graph, ata_graph]  # , uatau_graph, aua_graphE

    elif dataset == 'yelp':
        uu_graph = sp.load_npz('data/yelp/uu.npz')
        ubu_graph = sp.load_npz('data/yelp/ubu_processed.npz')
        # ucu_graph = sp.load_npz('data/yelp/ucu.npz')
        # bub_graph = sp.load_npz('data/yelp/bub.npz')
        bcab_graph = sp.load_npz('data/yelp/bcab_processed.npz')
        bcib_graph = sp.load_npz('data/yelp/bcib_processed.npz')

        uu_graph = get_graph(uu_graph).to(device)
        ubu_graph = get_graph(ubu_graph).to(device)
        # ucu_graph = get_graph(ucu_graph).to(device)
        # bub_graph = get_graph(bub_graph).to(device)
        bcab_graph = get_graph(bcab_graph).to(device)
        bcib_graph = get_graph(bcib_graph).to(device)

        user_mps, item_mps = [uu_graph, ubu_graph], [bcib_graph, bcab_graph] # , ubu_graph, bcib_graph

    elif dataset == 'douban-book':
        uu_graph = sp.load_npz('data/douban-book/uu.npz')
        ugu_graph = sp.load_npz('data/douban-book/ugu_processed.npz')
        # ulu_graph = sp.load_npz('data/douban-book/ulu.npz')
        bab_graph = sp.load_npz('data/douban-book/bab.npz')
        # bpb_graph = sp.load_npz('data/douban-book/bpb.npz')
        byb_graph = sp.load_npz('data/douban-book/byb_processed.npz')

        uu_graph = get_graph(uu_graph).to(device)
        ugu_graph = get_graph(ugu_graph).to(device)
        # ulu_graph = get_graph(ulu_graph).to(device)
        bab_graph = get_graph(bab_graph).to(device)
        # bpb_graph = get_graph(bpb_graph).to(device)
        byb_graph = get_graph(byb_graph).to(device)
        user_mps, item_mps = [uu_graph, ugu_graph], [bab_graph, byb_graph]  # , ulu_graph, byb_graph

    elif dataset == 'amazon':
        uibiu_graph = sp.load_npz('data/amazon/uibiu_processed.npz')
        # uiviu_graph = sp.load_npz('data/amazon/uiviu.npz')
        uiu_graph = sp.load_npz('data/amazon/uiu_processed.npz')
        ibi_graph = sp.load_npz('data/amazon/ibi_processed.npz')
        # ivi_graph = sp.load_npz('data/amazon/ivi.npz')
        # iui_graph = sp.load_npz('data/amazon/iui.npz')
        ici_graph = sp.load_npz('data/amazon/ici_processed.npz')

        uibiu_graph = get_graph(uibiu_graph).to(device)
        # uiviu_graph = get_graph(uiviu_graph).to(device)
        uiu_graph = get_graph(uiu_graph).to(device)
        ibi_graph = get_graph(ibi_graph).to(device)
        # ivi_graph = get_graph(ivi_graph).to(device)
        # iui_graph = get_graph(iui_graph).to(device)
        ici_graph = get_graph(ici_graph).to(device)
        user_mps, item_mps = [uibiu_graph, uiu_graph], [ibi_graph, ici_graph]  #

    elif dataset == 'movielens-1m':
        # umgmu_graph = sp.load_npz('data/movielens-1m/umgmu.npz')
        # uu_graph = sp.load_npz('data/movielens-1m/uu.npz')
        uou_graph = sp.load_npz('data/movielens-1m/uou_processed.npz')
        # uau_graph = sp.load_npz('data/movielens-1m/uau_processed.npz')
        umu_graph = sp.load_npz('data/movielens-1m/umu_processed.npz')
        # uumuu_graph = sp.load_npz('data/movielens-1m/uumuu.npz')
        # mm_graph = sp.load_npz('data/movielens-1m/mm.npz')
        # mmumm_graph = sp.load_npz('data/movielens-1m/mmumm.npz')
        mum_graph = sp.load_npz('data/movielens-1m/mum_processed.npz')
        mgm_graph = sp.load_npz('data/movielens-1m/mgm_processed.npz')

        # umgmu_graph = get_graph(umgmu_graph).to(device)
        # uu_graph = get_graph(uu_graph).to(device)
        uou_graph = get_graph(uou_graph).to(device)
        # uau_graph = get_graph(uau_graph).to(device)
        umu_graph = get_graph(umu_graph).to(device)
        # uumuu_graph = get_graph(uumuu_graph).to(device)
        # mm_graph = get_graph(mm_graph).to(device)
        # mmumm_graph = get_graph(mmumm_graph).to(device)
        mum_graph = get_graph(mum_graph).to(device)
        mgm_graph = get_graph(mgm_graph).to(device)
        user_mps, item_mps = [umu_graph, uou_graph], [mgm_graph, mum_graph]  #

    return user_mps, item_mps

def get_graph(graph):
    new_row = graph.row
    new_col = graph.col
    g = dgl.graph((new_row, new_col), num_nodes=graph.shape[0])
    return g
