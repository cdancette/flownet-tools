from train import train
import sys
import argparse
from run_model import run_model, run_model_multiples, run_model_lmdb

def call_train(parser):    
    train(parser.prototxt, parser.input_weights, parser.output_weights, parser.iterations, 
          solver=None, disp_interval=10, log_file=parser.log_file)
    
def call_run(parser):
    run_model(parser.prototxt, parser.input_weights, parser.image0, parser.image1, parser.out_flow, verbose=False)

def call_run_multiples(parser):
    run_model_multiples(parser.prototxt, parser.weights, parser.listfile, parser.output_dir, parser.outfile)

def main():
    parser = argparse.ArgumentParser(description='Use flownet 2')
          
    subparsers = parser.add_subparsers(help='command')
          
    parser_train = subparsers.add_parser('train', help='a help')
    parser_run = subparsers.add_parser('run', help='a help')
    parser_run_multiple = subparsers.add_parser('run_test', help='a help')    

    parser_train.add_argument('prototxt')
    parser_train.add_argument('input_weights')
    parser_train.add_argument('output_weights')
    parser_train.add_argument('iterations')
    parser_train.add_argument('log_file')
    parser_train.set_defaults(func=call_train)
    
    parser_run.add_argument('prototxt')
    parser_run.add_argument('input_weights')
    parser_run.add_argument('image0')
    parser_run.add_argument('image1')
    parser_run.add_argument('out_flow')
    parser_run.set_defaults(func=call_run)

    parser_run_multiple.add_argument('prototxt')
    parser_run_multiple.add_argument('weights')
    parser_run_multiple.add_argument('listfile')
    parser_run_multiple.add_argument('output_dir')
    parser_run_multiple.add_argument('outfile', help="csv file where the loss will be saved")
    parser_run_multiple.set_defaults(func=call_run_multiples)

    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
    
    
    
    
    
