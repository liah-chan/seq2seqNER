import os
import argparse
from argparse import RawTextHelpFormatter
import configparser
import sys
from pprint import pprint

from data_loader import *
# from model import *
from train import *
from evaluate import *
import distutils
import random

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Parse command line arguments
def parse_arguments(arguments=None):
    ''' Parse the arguments

    arguments:
        arguments the arguments, optionally given as argument
    '''
    argparser = argparse.ArgumentParser(description='''seq2seq for NER''',
                                        formatter_class=RawTextHelpFormatter)
    argparser.add_argument('--parameters_filepath', required=False, 
        default='/home/liah/ner/seq2seq_for_ner/parameters.ini', help='The parameters file')

    try:
        arguments = argparser.parse_args(args=arguments)
        
    except:
        argparser.print_help()
        sys.exit(0)

    arguments = vars(arguments) 
    # arguments['argument_default_value'] = argument_default_value
    return arguments

def load_parameters(parameters_filepath, arguments={}, verbose=True):
    parameters = {}
    # If a parameter file is specified, load it
    if len(parameters_filepath) > 0:
        conf_parameters = configparser.ConfigParser()
        conf_parameters.read(parameters_filepath)
        # nested_parameters = utils.convert_configparser_to_dictionary(conf_parameters)
        my_config_parser_dict = {s:dict(conf_parameters.items(s)) for s in conf_parameters.sections()}

        for k,v in my_config_parser_dict.items():
            parameters.update(v)
    # print(parameters)
    
    for k,v in parameters.iteritems():
        if k in ['n_epochs', 'n_iters','hidden_size','n_layers_encoder','n_layers_decoder', 'max_len', 
                 'n_layers','plot_every', 'save_every', 'print_every', 'evaluate_every', 'batch_size',
                 'patience', 'step', 'char_emb_size']:
            parameters[k] = int(v)
        elif k in ['dropout_p', 'learning_rate', 'teacher_forcing_ratio', 'clip']:
            parameters[k] = float(v)
        elif k in ['attn_mode', 'lang_in', 'lang_out', 'output_model_dir', 'mode',
                   'encoder_path', 'decoder_path', 'conll_filepath', 'pred_output_dir',
                   'embedding_filepath', 'nn_model', 'main_eval_method']:
            parameters[k] = str(v)
        elif k in ['resume', 'use_pretrained_word_embedding', 'save_best_epoch','add_class',
                   'use_char_embedding', 'tune_hyperparam']:
            parameters[k] = distutils.util.strtobool(v)

    if verbose: pprint(parameters)
    return parameters, conf_parameters

def mkdir_if_not_exist(path):
    if not os.path.isdir(path):
        print('Creating directory: {:s}'.format(path))
        os.mkdir(path)
    else:
        print('Directory exist: {:s}'.format(path))
        
def main():
    if len(sys.argv) > 1:
        arguments = parse_arguments(sys.argv[1:])
    else:
        arguments = parse_arguments()
    parameters, conf_parameters = load_parameters(arguments['parameters_filepath'],
                                                  arguments=arguments)
    
    mkdir_if_not_exist(parameters['output_model_dir'])
    mkdir_if_not_exist(parameters['pred_output_dir'])
    
    if parameters['baseline_folder'] != 'none':
        parameters['output_model_filepath'] = os.path.join(parameters['output_model_dir'],
                                                           parameters['baseline_folder'])
        parameters['pred_output_folder'] = os.path.join(parameters['pred_output_dir'],
                                                        parameters['baseline_folder'])
    else:
        parameters['output_model_filepath'] = os.path.join(parameters['output_model_dir'],
                                                           'step'+str(parameters['step']))
        parameters['pred_output_folder'] = os.path.join(parameters['pred_output_dir'],
                                                        'step'+str(parameters['step']))
    
    mkdir_if_not_exist(parameters['output_model_filepath'])
    mkdir_if_not_exist(parameters['pred_output_folder'])
    
    
    lang_in, lang_out = parameters['lang_in'], parameters['lang_out']
    
    lang_obj_file = os.path.join(parameters['output_model_dir'], parameters['lang_obj_file_name'])
    
    input_lang, output_lang, \
    all_pairs, all_max_len, all_lengths = prepareData(lang_in, lang_out,
                                                   parameters['dataset_filepath'],
                                                   parameters['mode'], lang_obj_file=lang_obj_file,
                                                   reverse=False, exp_step = parameters['step'],
                                                   add_class = parameters['add_class'],
                                                   resume_training = parameters['resume'])
    print('Example training pair:')
    for data_type in all_max_len.keys():
        print('Example {:s} sentence: '.format(data_type), all_pairs[data_type][0][0])
        print('Example {:s} tags: '.format(data_type), all_pairs[data_type][0][1])

    # torch.cuda.empty_cache()

    if parameters['tune_hyperparam']:
        best_f1 = 0.0
        best_param = {}
        
        def create_subpath(old_path, i, old_paths_dict):
            new_subpath = os.path.join(old_paths_dict[old_path], 'trial'+str(i))
            mkdir_if_not_exist(new_subpath)
            return new_subpath

        old_paths = {}
        for p in ['output_model_filepath', 'pred_output_folder']:
                old_paths[p] = parameters[p]
        parameters_cp = parameters.copy()
        for i in range(10):
            print('Hyperparameter optimization trial {:d}'.format(i+1))
            for p in ['output_model_filepath', 'pred_output_folder']:
                parameters_cp[p] = create_subpath(p, i+1, old_paths)
            
            #need to set a consistent epoch value which is smaller than 100
            new_parameters = tune_param(parameters_cp)
            new_parameters['n_epochs'] = 20
            fpath = os.path.join(new_parameters['output_model_filepath'], 'info') 
            with open(fpath, 'w', encoding='utf-8') as f:
                for k,v in new_parameters.iteritems():
                    # f.writelines(str(k) + ' : ' + str(v) + '\n')
                    f.writelines(unicode(k)+u' : '+unicode(v)+u'\n')

            f1_valid, f1s_conll, f1s_sklearn = train(new_parameters, input_lang, output_lang, all_pairs,
                             all_max_len, all_lengths, delete_after_train = True)
            with open(fpath, 'a', encoding='utf-8') as f:
                # pprint(unicode(new_parameters), f)
                for f1s in [f1s_conll, f1s_sklearn]:
                    for item in f1s.items():
                        f.writelines(unicode(item[0])+u':\n')
                        for k,v in item[1].iteritems():
                            line = u'\t'+ unicode(k) + u':\n\t\t'+ u', '.join([unicode(x) for x in v])+ u'\n'
                            f.writelines(line)

            if f1_valid > best_f1:
                best_param = new_parameters.copy()
                best_f1 = f1_valid
        print('Hyper-param selection finished! The parameters results in the best validation F1 are:')
        best_param['n_epochs'] = 100
        if best_param['learning_rate'] > 0.0025:
            best_param['patience'] = 30
        for key in old_paths.keys():
            best_param[key] = old_paths[key]           
        pprint(best_param)
        _ = train(best_param, input_lang, output_lang, all_pairs, all_max_len, all_lengths)
        
    else: #not searching the best hyper param
        _ = train(parameters, input_lang, output_lang, all_pairs, all_max_len, all_lengths)

def tune_param(parameters):
    # current_choices = []
    param_choices = {
        'batch_size': [16, 32, 64, 128, 256],
        'clip': [5.,25.,50.],
        'char_emb_size' : [25, 12],
        'hidden_size' : [64, 72, 128],
        'dropout_p' :[0.2, 0.3, 0.4, 0.5],
        'learning_rate' : [ 0.05, 0.02, 0.01, 0.0075, 0.005]#, 0.0025, 0.001]
    }
    param_exist = True
    while param_exist:
        for key in param_choices.keys():
            r = len(param_choices[key])
            rand = random.randint(0,r-1)
            old_val = parameters[key]
            if old_val != param_choices[key][rand]:
                parameters[key] = param_choices[key][rand]
                param_exist = False
                # current_choices.append(param_choices[key][rand])
                print('change parameter {:s} from {:s} to {:s}'.format(
                        key, str(old_val), str(parameters[key])))
    print('new parameters:')
    pprint(parameters)
    return parameters

if __name__ == "__main__":
    main()