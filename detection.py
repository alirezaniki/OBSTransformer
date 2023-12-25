import os, sys
from EQTransformer.utils.hdf5_maker import preprocessor


input = sys.argv[1].replace('/', '')


json_basepath = os.path.join(os.getcwd(),"json/station_list.json")
if not os.path.isdir(f'./{input}_processed_hdfs'):
    preprocessor(preproc_dir="./preproc",
                mseed_dir=input,
                stations_json=json_basepath, 
                overlap=0.0,
                n_processor=10)

from EQTransformer.core.predictor import predictor
predictor(input_dir=f"./{input}_processed_hdfs",
         input_model="./models/OBSTransformer/OBSTransformer.h5",
         output_dir="output/OBSTransformer",
          estimate_uncertainty=False,
          output_probabilities=False,
          number_of_sampling=5,
          loss_weights=[0.02, 0.40, 0.58],
          detection_threshold=0.4,                
          P_threshold=0.3,
          S_threshold=0.3, 
          number_of_plots=50,
          plot_mode='time',
          batch_size=500,
          number_of_cpus=10,
          keepPS=False,
          spLimit=60)
