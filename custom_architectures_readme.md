## Creating custom network architectures

To create a custom neural network architecture, create a new class that extends the `BaseNet` in [asvtorch/src/networks/architectures.py](asvtorch/src/networks/architectures.py).

In the constructor of your new class, initialize
`self.feat_dim` and `self.n_speakers`, which, in practice, represent the sizes of the input and output layers of your network (see examples from [asvtorch/src/networks/architectures.py](asvtorch/src/networks/architectures.py)).

To execute experiments using your custom network, change the network class in the `run_configs.py` of your recipe. For example in your run configs you could specify:
``` txt
network.network_class = 'asvtorch.src.networks.myarchitectures.MyCustomClass'
```
