# Kerbitat

Kerbitat is a profiling tool for pridicting running time of a DNN model on NVIDIA GPUs.
It can predict the performance of the target GPU from the profiling result of the source GPU.


# Install
``` bash
pip install ./
```

# How it works

1. Kerbitat use KernelProfiler from habitat to get mangled kernel names of running DNN model.
2. Kerbitat convert the mangled kernel names to feature vector.
3. Kerbitat use a trained model to predict the performance of the target GPU from the 
   feature vector.
4. Kerbitat use the predicted performance to estimate the execution time of the target GPU.
   A Big Thanks to habitat team for providing KernelProfiler and all the other great thoughts.

# How to use

``` python
from kerbitat import Kerbitat, GPU_TYPE
... # your own code
kerbitat_instance = Kerbitat(GPU_TYPE.A40)
kerbitat_instance.profiling(lambda: train_one_step(model, optimizer, x, y))
kerbitat_instance.get_target_time(GPU_TYPE.RTX2080Ti)
# 0.03143 sec per mini batch predicted on GPU RTX 2080 Ti
kerbitat_instance.convert_rate(GPU_TYPE.M40)
# The execution time on GPU M40 is 4.722 times longer than on GPU A40
```

# Acknowledgement

This work is based on CentML's [Habitat](https://github.com/CentML/DeepView.Predict), A very big thanks to them.

DeepView.Predict (aka Habitat) began as a research project in the EcoSystem Group at the University of Toronto. The accompanying research paper appeared in the proceedings of USENIX ATC'21. If you are interested, you can read a preprint of the paper here.

``` bib
@inproceedings{habitat-yu21,
  author = {Yu, Geoffrey X. and Gao, Yubo and Golikov, Pavel and Pekhimenko,
    Gennady},
  title = {{Habitat: A Runtime-Based Computational Performance Predictor for
    Deep Neural Network Training}},
  booktitle = {{Proceedings of the 2021 USENIX Annual Technical Conference
    (USENIX ATC'21)}},
  year = {2021},
}
```
