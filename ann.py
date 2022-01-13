import os
import pickle
import time
from tqdm import tqdm
import torch

import nmslib
import numpy as np
import pandas as pd
import hydra

from models.metric_net import MetricNet
from datasets.jpo_dataset import JPODataset

@hydra.main(config_path="config", config_name="ann")
def main(cfg):
    test_df = pd.read_csv(hydra.utils.get_original_cwd()+"/data/input/test.csv")
    test_df["path"] = hydra.utils.get_original_cwd()+"/data/input/crop_test_apply_images/" + test_df["path"]
    test_image_paths = test_df["path"].values
    
    cite_df = pd.read_csv(hydra.utils.get_original_cwd()+"/data/input/cite_v2.csv")
    cite_df["path"] = hydra.utils.get_original_cwd()+"/data/input/crop_cite_images/" + cite_df["path"]
    cite_image_paths = cite_df["path"].values

    n_classes = cite_df["gid"].nunique()

    if not os.path.exists(f'{hydra.utils.get_original_cwd()}/data/features/test_image_embeddings.npy'):
        query_arrays = get_image_embeddings(test_image_paths, n_classes) # size(1542, 1280)
        np.save(f'{hydra.utils.get_original_cwd()}/data/features/test_image_embeddings.npy', query_arrays)
    else:
        query_arrays = np.load(f'{hydra.utils.get_original_cwd()}/data/features/test_image_embeddings.npy')
    if not os.path.exists(f'{hydra.utils.get_original_cwd()}/data/features/cite_image_embeddings.npy'):
        search_arrays = get_image_embeddings(cite_image_paths, n_classes) # size(799175, 1280)
        np.save(f'{hydra.utils.get_original_cwd()}/data/features/cite_image_embeddings.npy', search_arrays)
    else:
        search_arrays = np.load(f'{hydra.utils.get_original_cwd()}/data/features/cite_image_embeddings.npy')
    # [参照]https://note.nkmk.me/python-dict-create/
    test_idx2id = dict(zip(test_df.index, test_df['gid']))
    cite_idx2id = dict(zip(cite_df.index, cite_df['gid']))

    # create/load index
    index = nmslib.init(method='hnsw', space='cosinesimil')
    if os.path.exists(f'{hydra.utils.get_original_cwd()}/models/index.ann'):
        print("load index starts")
        start = time.time()
        index.loadIndex(f'{hydra.utils.get_original_cwd()}/models/index.ann')
        end = time.time()
        print(f"load index taked {end-start} sec")
    else:
        print(f'create index starts')
        start = time.time()
        index.addDataPointBatch(search_arrays)
        index.createIndex({'post': 2}, print_progress=True)
        end = time.time()
        print(f'create index takes {end-start} sec')
        index.saveIndex(f'{hydra.utils.get_original_cwd()}/models/index.ann')
    
    # get neighbors
    start = time.time()

    sim_ids_dict = {}
    sim_dists_dict = {}

    

    for i in range(query_arrays.shape[0]): #range(1542)
        idxs, dists = index.knnQuery(query_arrays[i], k=1000)
        #print(f'idxs: {idxs}')
        #print(idxs.shape)
        #print(f'dists: {dists}')
        #print(dists.shape)
        ids = [cite_idx2id[idx] for idx in idxs]
        #print(f'ids: {ids}')
        base_id = test_idx2id[i]
        sim_ids_dict[base_id] = ids
        sim_dists_dict[base_id] = dists

    # save results
    print(f'save idxs starts')
    with open(f'{hydra.utils.get_original_cwd()}/data/output/results/similar_ids.pickle', 'wb') as f:
        pickle.dump(sim_ids_dict, f)
    print(f'save idxs ends')

    print(f'save dists starts')
    with open(f'{hydra.utils.get_original_cwd()}/data/output/results/similar_dists.pickle', 'wb') as f:
        pickle.dump(sim_dists_dict, f)
    print(f'save dists ends')

def get_image_embeddings(image_paths, n_classes):
    embeds = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MetricNet(n_classes=n_classes)
    model.load_state_dict(torch.load(f"{hydra.utils.get_original_cwd()}/models/model_params/best_param_of_{model.__class__.__name__}.pt"), strict=False)
    model.eval()
    test_dataset = JPODataset(pd.Series(image_paths), is_train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16)

    with torch.no_grad():
        for img in tqdm(test_loader):
            img = img.to(device)
            feat = model.extract_feat(img)
            image_embeddings = feat.detach().cpu().numpy()
            embeds.append(image_embeddings)

            del img
    
    image_embeddings = np.concatenate(embeds)
    return image_embeddings

if __name__ == "__main__":
    main()