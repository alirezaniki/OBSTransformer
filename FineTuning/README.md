
Transfer Leaning
---

Don't have sufficient labeled samples to train your model from scratch? No worries, tranfer learning is here to help. You can simply fine-tune OBSTranformer or EqTransformer to further optimize them for your region.

1. Provide the input database and metada for training (see sample.csv and sample.hdf5) 
2. Modify the parameters in TL.py (fine-tuning controls and "trainer" function) based on the size of your dataset 
3. Run the TL.py code (python TL.py)
