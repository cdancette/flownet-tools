from train import train
import sys
import argparse


def call_train(parser):    
    train(parser.prototxt, parser.input_weights, parser.output_weights, parser.iterations, 
          solver=None, disp_interval=10, log_file=parser.log_file)

def main():
 
    parser = argparse.ArgumentParser(description='Use flownet 2')
          
    subparsers = parser.add_subparsers(help='command')
          
    parser_train = subparsers.add_parser('train', help='a help')
       
    parser_train.add_argument('prototxt')
    parser_train.add_argument('input_weights')
    parser_train.add_argument('output_weights')
    parser_train.add_argument('iterations')
    parser_train.add_argument('log_file')
    parser_train.set_defaults(func=call_train)
    
    args = parser.parse_args()
    
    args.func(args)


if __name__ == '__main__':
    main()
    
    
    
    
    
