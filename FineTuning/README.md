
Transfer Leaning
---

Don't have sufficient labeled samples to train your model from scratch? No worries, transfer learning is here to help. You can simply fine-tune OBSTranformer or EqTransformer to further optimize them for your region.

1. Provide the input database and metadata for transfer learning (see sample.csv and sample.hdf5) 
2. Modify the parameters in TL.py (transfer learning controls and the "trainer" function) based on the size of your dataset 
3. Run the TL.py code (python TL.py)

Note that the EqT_utils.py has been slightly modified to use OBS noise samples for data augmentation. Replace it with the original one if necessary
