![Github](https://github.com/alirezaniki/OBSTransformer/assets/24492517/3676bc18-663c-4bc8-97ab-fbeea89c3273)
![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2Falirezaniki%2FOBSTransformer&labelColor=%2337d67a&countColor=%23263759&style=plastic)

---
Discription
--

**OBSTransformer** is a transfer-learned seismic phase picker for Ocean Bottom Seismometer (OBS) data adopted from the EqTransformer model. 
**OBSTransformer** has been trained on an auto-labeled tectonically inclusive OBS dataset comprising ~36k earthquake and 25k noise samples.
**OBSTransformer** is now integrated with the “hands-free” earthquake location workflow-[LOCFLOW](https://github.com/Dal-mzhang/LOC-FLOW).
You can download the training data via [this link](https://drive.google.com/uc?id=1J2LtLA067S6FeyS-lD1fEquBLEVVa_vC&export=download). Below
snippet code demonstrates how to deal with the training data:
```
import h5py as hp
import matplotlib.pyplot as plt

dataset = "OBSTransformer_training_data.hdf5"
d = hp.File(dataset, 'r')
for item in d['data']:
  dd = d.get(f'data/{item}')
  p_arr = dd.attrs['p_arrival_sample']
  s_arr = dd.attrs['s_arrival_sample']
  print(p_arr, s_arr)
  plt.plot(dd)
  plt.show()
```

---
Installation
--

OBSTransformer is a variant of EqTransformer optimized for OBS data. Visit the [parent repository](https://github.com/smousavi05/EQTransformer) for detailed installation guidelines.

To start with (anaconda):
--

```
conda create -n obst python=3.7
conda activate obst
pip install EQTransformer
```
You may encounter version conflicts between packages, such as numpy or protobuf. Try to install the suggested versions to eliminate the issue.

---
Quick Instruction
--


  1. Create a text file including instrument information (stations.dat)
  2. Build the required station metadata using the provided shell script (build_json.sh; needs two addresses)
  3. Prepare the dataset directory (dataset/)
  4. Run the detection.py code (python detection.py dataset/)

---
Links
--

**Publication(s)**: will be added shortly ...

---
Reference
--

Niksejel, A. & Zhang, M., 2023. OBSTransformer: A Deep-Learning Seismic Phase Picker for OBS Data Using Automated Labelling and Transfer Learning. https://doi.org/10.48550/arXiv.2306.04753

```
@misc{niksejel2023obstransformer,
      title={OBSTransformer: A Deep-Learning Seismic Phase Picker for OBS Data Using Automated Labelling and Transfer Learning}, 
      author={Alireza Niksejel and Miao Zhang},
      year={2023},
      eprint={2306.04753},
      archivePrefix={arXiv},
      primaryClass={physics.geo-ph}
}
```
