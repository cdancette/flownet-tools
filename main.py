from train import train
import sys
import argparse
from run_model import run_model, run_model_multiples, run_model_lmdb
from video import make_consecutive


def call_train(parser):    
    train(parser.prototxt, parser.input_weights, parser.output_weights, int(parser.iterations), 
          solver=None, disp_interval=10, log_file=parser.log_file)
    
def call_run(parser):
    run_model(parser.prototxt, parser.input_weights, parser.image0, parser.image1, parser.out_flow, verbose=False)

def call_run_multiples(parser):
    run_model_multiples(parser.prototxt, parser.weights, parser.listfile, parser.output_dir, flow_loss=parser.flow_loss, save_images=parser.save_images)

def call_consecutive(parser):
    make_consecutive(parser.prototxt, parser.weights, parser.listfile, parser.output_dir, parser.start)

def main():
    parser = argparse.ArgumentParser(description='Use flownet 2')
          
    subparsers = parser.add_subparsers(help='command')
          
    parser_train = subparsers.add_parser('train', help='a help')
    parser_run = subparsers.add_parser('run', help='a help')
    parser_run_multiple = subparsers.add_parser('run_multiple', help='a help')
    parser_consecutive = subparsers.add_parser('consecutive', help='a help')


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
    parser_run_multiple.add_argument('--flow-loss', dest="flow_loss",action="store_true", help="use flow_loss. For this you need the flow in the input file")
    parser_run_multiple.add_argument('--save-images', dest="save_images",action="store_true", help="use save_images. For this you need the flow in the input file")
    parser_run_multiple.set_defaults(flow_loss=False, save_warp=False, func=call_run_multiples)

    parser_consecutive.add_argument('prototxt')
    parser_consecutive.add_argument('weights')
    parser_consecutive.add_argument('listfile')
    parser_consecutive.add_argument('output_dir')
    parser_consecutive.add_argument('--start', type=int, default=0, help='start at image')
    
    parser_consecutive.set_defaults(func=call_consecutive)
    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
    
    
    
    
    
