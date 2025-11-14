from Code.dir_definitions import *
from Code.plotter import get_ser_data, plot_ser_by_block_index, plot_ser_by_snr, plot_summary_table
from Code.trainer import Trainer
import torch
import gc
import multiprocessing

from Code.csv_reporter import ModelPerformanceTracker

#import os
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

def execute_model(model, detector_method, self_supervised, all_curves, current_params, run_over,num_of_rep,perf_tracker,HYPERPARAMS_DICT):
    if model == 'ClassicViterbi':
        execute_and_plot(model, 'Statistical', False, all_curves, current_params, run_over,num_of_rep,perf_tracker,HYPERPARAMS_DICT)  # Classic Viterbi Alg with Perfect-CSI
    else:
        execute_and_plot(model, detector_method, self_supervised, all_curves, current_params, run_over,num_of_rep,perf_tracker,HYPERPARAMS_DICT)

#def execute_and_plot(model_name, detector_method, self_supervised, all_curves, current_params, run_over,num_rep,HYPERPARAMS_DICT):
def execute_and_plot(model_name, detector_method, self_supervised, all_curves, current_params, run_over, num_rep, perf_tracker,HYPERPARAMS_DICT):
    method_name = model_name + "_" + detector_method

    # Start timing the model execution
    perf_tracker.start_timing(model_name)

    trainer = Trainer(
                    model_name=model_name,
                    detector_method=detector_method,
                    self_supervised=self_supervised,
                    weights_dir=os.path.join(WEIGHTS_DIR,
                    f'{method_name}_training_{HYPERPARAMS_DICT["val_block_length"]}_{HYPERPARAMS_DICT["n_symbols"]}_channel1_{HYPERPARAMS_DICT["channel_coefficients"]}'),
                    **HYPERPARAMS_DICT)
    
    # Get model size (number of parameters)
    model_size = 0
    if hasattr(trainer.detector, "model"):
        model_parameters = filter(lambda p: p.requires_grad, trainer.detector.model.parameters())
        model_size = sum([torch.numel(p) for p in model_parameters])
    
    # ser = get_ser_data`(trainer=trainer,run_over=run_over,num_of_rep=1,method_name=method_name + '_' + current_params)
    ser = get_ser_data(trainer, run_over=run_over, num_of_rep=num_rep, method_name=method_name + '_' + current_params)
    
    # ser = get_ser_data(trainer=trainer,run_over=run_over,num_of_rep=num_rep,method_name=method_name + '_' + current_params)
    # ser = get_ser_data(trainer, run_over=run_over, method_name=method_name + '_' + current_params)
    
    # End timing
    run_time = perf_tracker.end_timing(model_name)
    
    # Calculate final SER (mean of the SER values)
    final_ser = np.mean(ser)
    
    # Record metrics
    perf_tracker.record_metrics(
        model_name=model_name,
        snr=HYPERPARAMS_DICT['curr_SNR'],
        final_ser=final_ser,
        model_size=model_size,
        run_time=run_time
    )
    perf_tracker.save_to_csv(f"model_performance_snr_{snr}_parallel.csv")
    all_curves.append((ser, model_name, HYPERPARAMS_DICT['val_block_length'], HYPERPARAMS_DICT['n_symbols']))


HYPERPARAMS_DICT = {
                    'noisy_est_var': 0,
                    'fading_taps_type': 1,  # 1 / 2  for time decay only
                    'fading_in_channel': True,
                    'fading_in_decoder': True,
                    'gamma': 0.25,
                    'channel_type': 'ISI_AWGN',
                    'val_frames': 12,  # up to 12 for cost2100
                    'subframes_in_frame': 25,  # up to 25 for cost2100
                    'self_supervised_iterations': 200,
                    'ser_thresh': 0.02,  # ser threshold for online training
                    'train_minibatch_num': 25,  # 25
                    }

if __name__ == '__main__':
  parameters = [(15,120*5,25*5),
                (16,120*7,25*7),  
                (17,120*10,25*10)]
  perf_tracker = ModelPerformanceTracker()  
  for mc in range(20):
    # torch.cuda.empty_cache()
    # gc.collect()
    num_of_reps = [5,7,10]
    # main flags
    if (mc<1):
      run_over = 2 # 0 - load plots from previous runs if exists / 1 - load trained weights and start online evaluation / 2 - clear all and start training  from scratch
    else:
      run_over = 2
    plot_by_block = False  # False / True either plot by SNR or by block index
    block_length = 120  # determine the transmission length
    channel_coefficients = 'cost2100'  # 'time_decay' / 'cost2100'
    n_symbol = 2
    snr_start , snr_end = 15,17
    # deep learning models list 'ADNN', 'Sionna', 'SionnaPlus', 'Transformer', 'LSTM', 'ViterbiNet'
    #models_list = ['ADNN', 'Sionna', 'SionnaPlus', 'Transformer', 'LSTM', 'ViterbiNet', 'ClassicViterbi']
    # models_list = ['Sionna','SionnaPlus','Transformer','ViterbiNet', 'ClassicViterbi']    # Gil Zukerman : Jun/03/2023
    # models_list = ['Sionna','Transformer']
    models_list = ['Transformer','ViterbiNet']
    # models_list = ['Mamba']
    # models_list = ['ViterbiNet']
    # models_list = ['Transformer']
    # models_list = ['SionnaPlus']
    # models_list = ['Mamba']
    detector_method = 'ModelBased'  # ModelBased / EndToEnd / Statistical
    self_supervised = False  # True / False for online evaluation enablement
    all_curves = []
    processes = []

    # for snr in range(snr_start, snr_end+1):
    for snr, val_block_length,pilot_length in parameters:
      print(f'snr={snr}, val_block_length={val_block_length}, pilots_length={pilot_length},n_symbols={n_symbol}')    
      print(f'iteration: {mc},snr={snr}, block-length={block_length}, num-symbols={n_symbol}')
      HYPERPARAMS_DICT['n_symbols'] = n_symbol
      HYPERPARAMS_DICT['curr_SNR'] = snr
      HYPERPARAMS_DICT['val_block_length'] = val_block_length
      HYPERPARAMS_DICT['train_block_length'] = val_block_length
      HYPERPARAMS_DICT['fading_in_channel'] = True if channel_coefficients == 'time_decay' else False
      HYPERPARAMS_DICT['pilots_num'] = pilot_length
      HYPERPARAMS_DICT['channel_coefficients'] = channel_coefficients
      current_params = HYPERPARAMS_DICT['channel_coefficients'] + '_' + str(HYPERPARAMS_DICT['curr_SNR']) + '_' + \
                      str(HYPERPARAMS_DICT['val_block_length']) + '_' + str(HYPERPARAMS_DICT['n_symbols'])
      # for model in models_list:
      #     if model == 'ClassicViterbi':
      #         execute_and_plot(model, 'Statistical', False, all_curves, current_params, run_over)  # Classic Viterbi Alg with Perfect-CSI
      #     else:
      #         execute_and_plot(model, detector_method, self_supervised, all_curves, current_params, run_over)

      for model in models_list:
          print(f'iteration: {mc},snr={snr}, block-length={block_length}, num-symbols={n_symbol}, model={model}')
          process = multiprocessing.Process(target=execute_model, args=(model, detector_method, self_supervised, all_curves, current_params, run_over,num_of_reps[snr-snr_start], perf_tracker,HYPERPARAMS_DICT))
          processes.append(process)
          process.start()
      
      for process in processes:
        process.join()   

    # snr_values = [s for s in range(snr_start, snr_end+1)]
    
  
  # if not plot_by_block:
  #     plot_ser_by_snr(all_curves, snr_values)
  # else:
  #     plot_ser_by_block_index(all_curves, block_length, n_symbol, snr)

  # plot_summary_table(all_curves, models_list, snr_values)
  # Save final metrics after all iterations
  perf_tracker.save_to_csv(f"model_performance_final_mc_{mc}_parallel.csv")


