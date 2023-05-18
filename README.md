<div id="header" align="center">
  <img src="https://drive.google.com/uc?id=1hRSnfqbzKX2kneOxO00L609oABGPtIhc" width="800"/>
</div>

---
Discription
--

**OBSTransformer** is a transfer-learned seismic phase picker for Ocean Bottom Seismometer (OBS) data adopted from the EqTransformer model. 
**OBSTransformer** has been trained on an auto-labeled tectonically inclusive OBS dataset comprising ~36k earthquake and 25k noise samples.

---
Installation
--

OBSTransformer is a variant of EqTransformer optimized for OBS data. Visit the [parent](https://github.com/smousavi05/EQTransformer) repository for installation guidlines.

---
A Quick Instruction
--

  1. Create a text file including instrument information (stations.dat)
  2. Build the required station-metadata file using the provided shell code (build_json.sh; needs two addresses)
  3. Prepare the dataset directory (dataset/)
  4. Run the detection.py code (python detection.py dataset/)

---
Links
--

**Paper list**: will be added shortly ...

---
Reference
--

Will be added shortly ...
