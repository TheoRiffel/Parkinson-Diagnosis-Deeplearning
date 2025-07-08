from tqdm import tqdm
import torch
import os

def load_graphs(path: str) -> list[torch.Tensor]:
    """
    Load graphs from path
    """
    graphs = []
    for file in tqdm(os.listdir(path)):
        if file.endswith('.pt'):
            graph_path = os.path.join(path, file)
            # Add weights_only=False to allow loading networkx.Graph objects
            graph = torch.load(graph_path, weights_only=False)
            graphs.append(graph)
    
    print(f'Loaded {len(graphs)} graphs from {path}')
    return graphs

def save_graphs(graphs, path: str):

    if not os.path.exists(path):
        os.makedirs(path)

    for idx, graph in enumerate(tqdm(graphs)):
        graph_path = os.path.join(path, f'graph_{idx}.pt')
        torch.save(graph, graph_path)
        print(f'Saved {len(graphs)} graphs to {path}')

