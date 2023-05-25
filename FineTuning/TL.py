'''
Alireza Niksejel, Aug 2022 
(alireza.niksejel@dal.ca)

Modified from:
    https://github.com/smousavi05/EQTransformer

'''


from __future__ import print_function
from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from keras.models import load_model
from keras.optimizers import Adam
from EqT_utils import f1, SeqSelfAttention, FeedForward, LayerNormalization
import tensorflow as tf
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import pickle
from csv import reader
import os
import shutil
import multiprocessing
from EqT_utils import DataGenerator, _lr_schedule
import datetime
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


base_model = "EqT_original_model.h5"
# define the target layers for freezing: count/name/None(full fine-tuning)
freeze_mode = None      
# works with freeze_mode="count". A range of target layers as [start, end] to be freezed before TL. (e.g., [0,100])
freeze_layers = None   
# works with freeze_mode="name". A file including the list of layer names to be freezed before TL.
freeze_list = "freeze_these"



def trainer(input_hdf5='./training_data.hdf5',
            input_csv='./metadata.csv',
            output_name='fine_tuned_model',                
            input_dimention=(6000, 3),
            padding='same',
            activation = 'relu',            
            drop_rate=0.2,
            shuffle=True, 
            label_type='gaussian',
            normalization_mode='std',
            augmentation=True,
            add_event_r=0.6,
            shift_event_r=0.99,
            add_noise_r=0.6, 
            drop_channel_r=0.5,
            add_gap_r=0.2,
            scale_amplitude_r=None,
            pre_emphasis=False,                
            loss_weights=[0.05, 0.4, 0.55],
            loss_types=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'],
            train_valid_test_split=[0.7, 0.15, 0.15],
            mode='generator',
            batch_size=200,
            epochs=100, 
            monitor='val_loss',
            patience=8,
            multi_gpu=True,
            number_of_gpus=1,
            gpuid=0,
            gpu_limit=None,
            use_multiprocessing=True):
         


    args = {
    "input_hdf5": input_hdf5,
    "input_csv": input_csv,
    "output_name": output_name,
    "input_dimention": input_dimention,
    "padding": padding,
    "activation": activation,
    "drop_rate": drop_rate,
    "shuffle": shuffle,
    "label_type": label_type,
    "normalization_mode": normalization_mode,
    "augmentation": augmentation,
    "add_event_r": add_event_r,
    "shift_event_r": shift_event_r,
    "add_noise_r": add_noise_r,
    "add_gap_r": add_gap_r,
    "drop_channel_r": drop_channel_r,
    "scale_amplitude_r": scale_amplitude_r,
    "pre_emphasis": pre_emphasis,
    "loss_weights": loss_weights,
    "loss_types": loss_types,
    "train_valid_test_split": train_valid_test_split,
    "mode": mode,
    "batch_size": batch_size,
    "epochs": epochs,
    "monitor": monitor,
    "patience": patience,           
    "multi_gpu": multi_gpu,
    "number_of_gpus": number_of_gpus,           
    "gpuid": gpuid,
    "gpu_limit": gpu_limit,
    "use_multiprocessing": use_multiprocessing
    }
                       
    def train(args):

        save_dir, save_models=_make_dir(args['output_name'])
        training, validation=_split(args, save_dir)
        callbacks=_make_callback(args, save_models)
        model = loader(input_model = base_model)
        model.summary()        

        if args['gpuid']:           
            os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpuid)
            tf.Session(config=tf.ConfigProto(log_device_placement=True))
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = False
            config.gpu_options.per_process_gpu_memory_fraction = float(args['gpu_limit']) 
            K.tensorflow_backend.set_session(tf.Session(config=config))
            
        start_training = time.time()                  


        params_training = {'file_name': str(args['input_hdf5']), 
                            'dim': args['input_dimention'][0],
                            'batch_size': args['batch_size'],
                            'n_channels': args['input_dimention'][-1],
                            'shuffle': args['shuffle'],  
                            'norm_mode': args['normalization_mode'],
                            'label_type': args['label_type'],
                            'augmentation': args['augmentation'],
                            'add_event_r': args['add_event_r'],
                            'add_gap_r': args['add_gap_r'],
                            'shift_event_r': args['shift_event_r'],    
                            'add_noise_r': args['add_noise_r'],
                            'drop_channe_r': args['drop_channel_r'],
                            'scale_amplitude_r': args['scale_amplitude_r'],
                            'pre_emphasis': args['pre_emphasis']}
                    
        params_validation = {'file_name': str(args['input_hdf5']),  
                                'dim': args['input_dimention'][0],
                                'batch_size': args['batch_size'],
                                'n_channels': args['input_dimention'][-1],
                                'shuffle': False,  
                                'norm_mode': args['normalization_mode'],
                                'augmentation': False}          


        training_generator = DataGenerator(training, **params_training)
        validation_generator = DataGenerator(validation, **params_validation) 

        print('Started training in generator mode ...') 
        class_weights = {0: 0.11, 1: 0.89}
        history = model.fit_generator(generator=training_generator,
                                        validation_data=validation_generator,
                                        use_multiprocessing=args['use_multiprocessing'],
                                        workers=multiprocessing.cpu_count(),    
                                        callbacks=callbacks, 
                                        epochs=args['epochs'],
                                        class_weight=class_weights)
        end_training = time.time()
            

        
        return history, model, start_training, end_training, save_dir, save_models, len(training), len(validation)
                  
    history, model, start_training, end_training, save_dir, save_models, training_size, validation_size=train(args)  
    _document_training(history, model, start_training, end_training, save_dir, save_models, training_size, validation_size, args)


def loader(
           input_model=None,
           loss_weights=[0.05, 0.4, 0.55],
           loss_types=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'],
           ):

              
         
    argus = {
    "input_model": input_model,
    "loss_weights": loss_weights,
    "loss_types": loss_types,
    }  

    print('Loading the model ...', flush=True)        
    eqt_model = load_model(argus['input_model'], custom_objects={'SeqSelfAttention': SeqSelfAttention, 
                                                         'FeedForward': FeedForward,
                                                         'LayerNormalization': LayerNormalization, 
                                                         'f1': f1                                                                            
                                                         })

    if freeze_mode == "count":
        for layer in eqt_model.layers[freeze_layers[0]:freeze_layers[1]]:
            layer.trainable = False
    elif freeze_mode == "name":       
        with open(freeze_list, 'r') as f:
            line_reader = reader(f, delimiter = " ")
            for line in line_reader:
                _, eqtlayer = line
                for layer in eqt_model.layers:
                    if layer.name == eqtlayer:
                        layer.trainable = False
    else:
        pass


    eqt_model.compile(loss = argus['loss_types'],
                  loss_weights =  argus['loss_weights'],           
                  optimizer = Adam(lr=_lr_schedule(0)),
                  metrics = [f1])
    
    print('Loading is complete!', flush=True)
    return eqt_model  



def _make_dir(output_name):

    if output_name == None:
        print('Please specify output_name!') 
        return
    else:
        save_dir = os.path.join(os.getcwd(), str(output_name)+'_outputs')
        save_models = os.path.join(save_dir, 'models')      
        if os.path.isdir(save_dir):
            shutil.rmtree(save_dir)  
        os.makedirs(save_models)
    return save_dir, save_models



def _split(args, save_dir):
     
    df = pd.read_csv(args['input_csv'])
    ev_list = df.trace_name.tolist()    
    np.random.shuffle(ev_list)     
    training = ev_list[:int(args['train_valid_test_split'][0]*len(ev_list))]
    validation =  ev_list[int(args['train_valid_test_split'][0]*len(ev_list)):
                            int(args['train_valid_test_split'][0]*len(ev_list) + args['train_valid_test_split'][1]*len(ev_list))]
    test =  ev_list[ int(args['train_valid_test_split'][0]*len(ev_list) + args['train_valid_test_split'][1]*len(ev_list)):]
    np.save(save_dir+'/test', test)  
    return training, validation 



def _make_callback(args, save_models):
    
    m_name=str(args['output_name'])+'_{epoch:03d}.h5'   
    filepath=os.path.join(save_models, m_name)  
    early_stopping_monitor=EarlyStopping(monitor=args['monitor'], 
                                           patience=args['patience']) 
    checkpoint=ModelCheckpoint(filepath=filepath,
                                 monitor=args['monitor'], 
                                 mode='auto',
                                 verbose=1,
                                 save_best_only=True)  
    lr_scheduler=LearningRateScheduler(_lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=args['patience']-2,
                                   min_lr=0.5e-6)

    callbacks = [checkpoint, lr_reducer, lr_scheduler, early_stopping_monitor]
    return callbacks
 


def _document_training(history, model, start_training, end_training, save_dir, save_models, training_size, validation_size, args): 

    """ 
    
    Write down the training results.

    Parameters
    ----------
    history: dic
        Training history.  
   
    model: 
        Trained model.  

    start_training: datetime
        Training start time. 

    end_training: datetime
        Training end time.    
         
    save_dir: str
        Path to the output directory. 

    save_models: str
        Path to the folder for saveing the models.  
      
    training_size: int
        Number of training samples.    

    validation_size: int
        Number of validation samples. 

    args: dic
        A dictionary containing all of the input parameters. 
              
    Returns
    -------- 
    ./output_name/history.npy: Training history.    

    ./output_name/X_report.txt: A summary of parameters used for the prediction and perfomance.

    ./output_name/X_learning_curve_f1.png: The learning curve of Fi-scores.         

    ./output_name/X_learning_curve_loss.png: The learning curve of loss.  
        
        
    """   
    
    with open(f'{save_dir}/history', 'wb') as his:
        pickle.dump(history.history, his)

    model.save(save_dir+'/final_model.h5')
    model.to_json()   
    model.save_weights(save_dir+'/model_weights.h5')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(history.history['loss'])
    ax.plot(history.history['detector_loss'])
    ax.plot(history.history['picker_P_loss'])
    ax.plot(history.history['picker_S_loss'])
    try:
        ax.plot(history.history['val_loss'], '--')
        ax.plot(history.history['val_detector_loss'], '--')
        ax.plot(history.history['val_picker_P_loss'], '--')
        ax.plot(history.history['val_picker_S_loss'], '--') 
        ax.legend(['loss', 'detector_loss', 'picker_P_loss', 'picker_S_loss', 
               'val_loss', 'val_detector_loss', 'val_picker_P_loss', 'val_picker_S_loss'], loc='upper right')
    except Exception:
        ax.legend(['loss', 'detector_loss', 'picker_P_loss', 'picker_S_loss'], loc='upper right')  
        
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid(visible=True, which='major', color='#666666', linestyle='-')
    fig.savefig(os.path.join(save_dir,str('X_learning_curve_loss.png')), dpi=600) 
       
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(history.history['detector_f1'])
    ax.plot(history.history['picker_P_f1'])
    ax.plot(history.history['picker_S_f1'])
    try:
        ax.plot(history.history['val_detector_f1'], '--')
        ax.plot(history.history['val_picker_P_f1'], '--')
        ax.plot(history.history['val_picker_S_f1'], '--')
        ax.legend(['detector_f1', 'picker_P_f1', 'picker_S_f1', 'val_detector_f1', 'val_picker_P_f1', 'val_picker_S_f1'], loc='lower right')
    except Exception:
        ax.legend(['detector_f1', 'picker_P_f1', 'picker_S_f1'], loc='lower right')        
    plt.ylabel('F1')
    plt.xlabel('Epoch')
    plt.grid(visible=True, which='major', color='#666666', linestyle='-')
    fig.savefig(os.path.join(save_dir,str('X_learning_curve_f1.png')), dpi=600) 

    delta = end_training - start_training
    hour = int(delta / 3600)
    delta -= hour * 3600
    minute = int(delta / 60)
    delta -= minute * 60
    seconds = delta    
    
    trainable_count = int(np.sum([K.count_params(p) for p in model.trainable_weights]))
    non_trainable_count = int(np.sum([K.count_params(p) for p in model.non_trainable_weights]))
    
    with open(os.path.join(save_dir,'X_report.txt'), 'a') as the_file: 
        the_file.write('================== Overal Info =============================='+'\n')               
        the_file.write('date of report: '+str(datetime.datetime.now())+'\n')         
        the_file.write('input_hdf5: '+str(args['input_hdf5'])+'\n')            
        the_file.write('input_csv: '+str(args['input_csv'])+'\n')
        the_file.write('output_name: '+str(args['output_name']+'_outputs')+'\n')  
        the_file.write('================== Model Parameters ========================='+'\n')   
        the_file.write('input_dimention: '+str(args['input_dimention'])+'\n')
        the_file.write('padding_type: '+str(args['padding'])+'\n')
        the_file.write('activation_type: '+str(args['activation'])+'\n')        
        the_file.write('drop_rate: '+str(args['drop_rate'])+'\n')            
        the_file.write(str('total params: {:,}'.format(trainable_count + non_trainable_count))+'\n')    
        the_file.write(str('trainable params: {:,}'.format(trainable_count))+'\n')    
        the_file.write(str('non-trainable params: {:,}'.format(non_trainable_count))+'\n') 
        the_file.write('================== Training Parameters ======================'+'\n')  
        the_file.write('mode of training: '+str(args['mode'])+'\n')   
        the_file.write('loss_types: '+str(args['loss_types'])+'\n')
        the_file.write('loss_weights: '+str(args['loss_weights'])+'\n')
        the_file.write('batch_size: '+str(args['batch_size'])+'\n')
        the_file.write('epochs: '+str(args['epochs'])+'\n')   
        the_file.write('train_valid_test_split: '+str(args['train_valid_test_split'])+'\n')           
        the_file.write('total number of training: '+str(training_size)+'\n')
        the_file.write('total number of validation: '+str(validation_size)+'\n')
        the_file.write('monitor: '+str(args['monitor'])+'\n')
        the_file.write('patience: '+str(args['patience'])+'\n') 
        the_file.write('gpuid: '+str(args['gpuid'])+'\n')
        the_file.write('gpu_limit: '+str(args['gpu_limit'])+'\n')             
        the_file.write('use_multiprocessing: '+str(args['use_multiprocessing'])+'\n')  
        the_file.write('================== Training Performance ====================='+'\n')  
        the_file.write('finished the training in:  {} hours and {} minutes and {} seconds \n'.format(hour, minute, round(seconds,2)))                         
        the_file.write('stoped after epoche: '+str(len(history.history['loss']))+'\n')
        the_file.write('last loss: '+str(history.history['loss'][-1])+'\n')
        the_file.write('last detector_loss: '+str(history.history['detector_loss'][-1])+'\n')
        the_file.write('last picker_P_loss: '+str(history.history['picker_P_loss'][-1])+'\n')
        the_file.write('last picker_S_loss: '+str(history.history['picker_S_loss'][-1])+'\n')
        the_file.write('last detector_f1: '+str(history.history['detector_f1'][-1])+'\n')
        the_file.write('last picker_P_f1: '+str(history.history['picker_P_f1'][-1])+'\n')
        the_file.write('last picker_S_f1: '+str(history.history['picker_S_f1'][-1])+'\n')
        the_file.write('================== Other Parameters ========================='+'\n')
        the_file.write('label_type: '+str(args['label_type'])+'\n')
        the_file.write('augmentation: '+str(args['augmentation'])+'\n')
        the_file.write('shuffle: '+str(args['shuffle'])+'\n')               
        the_file.write('normalization_mode: '+str(args['normalization_mode'])+'\n')
        the_file.write('add_event_r: '+str(args['add_event_r'])+'\n')
        the_file.write('add_noise_r: '+str(args['add_noise_r'])+'\n')   
        the_file.write('shift_event_r: '+str(args['shift_event_r'])+'\n')                            
        the_file.write('drop_channel_r: '+str(args['drop_channel_r'])+'\n')
        the_file.write('add_gap_r: '+str(args['add_gap_r'])+'\n')
        the_file.write('scale_amplitude_r: '+str(args['scale_amplitude_r'])+'\n')            
        the_file.write('pre_emphasis: '+str(args['pre_emphasis'])+'\n')




OBS_train = trainer()
OBS_train         




