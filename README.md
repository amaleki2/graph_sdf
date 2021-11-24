# Graph SDF

This project implement a model to learn SDF from mesh-based geometries using graph neural networks.
In particular, we use the encode-process-decode introduced in [MeshGraphNet](https://arxiv.org/pdf/2010.03409.pdf).
The encode-process-decode was implemented separately in another repo called [graph_networks](https://github.com/amaleki2/graph_networks)

### Dependencies
- torch >= 1.8 (We use Lazy layers introduced in pytorch since 1.8 version)
- torch-geometric
- tensorflow (no cuda is necessary)
- scipy, matplotlib, numpy, 
- json, joblib, tqdm
- trimesh, meshio

as well as [graph_networks](https://github.com/amaleki2/graph_networks). 
By default, the implementation assumes graph_networks folder exists in the parent folder of graph_sdf. 
Otherwise, you need to pass the address to graph_network parent folder using `--extra-path` command argument

The example folder should have three json script files
- `data_configs.json`: containing parameters for converting mesh data to graph data.
- `network_configs.json`: containing parameters of encode-process-decode network.
- `training_configs.json`: containing training parameters

The folder `script` contains a set of such config files. Notice that configs in the `script` folder assume the code is running on four GPUs.
If only a single GPU is available or the course must be run on CPU, 
- `train_configs["train"]["device"]="cuda"  or "cpu"` and 
- `data_configs["train_data"]["dataloader_params"]["device"]=false`. 

### Usage
- train: `python run.py -e D:/experiments/graph_sdf_experiments/e1`
- train consistency network: `python main.py -e D:/experiments/graph_sdf_experiments/e1`

