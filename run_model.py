import os, sys, numpy as np
from scipy import misc
#import cv2
import caffe
import tempfile
from math import ceil
flush = sys.stdout.flush
sys.path.append("OpticalFlowToolkit/")

from lib.flowlib import save_flow_image

def readFlow(name):
    f = open(name, 'rb')

    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')

    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()

    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))

    return flow.astype(np.float32)

def writeFlow(name, flow):
    f = open(name, 'wb')
    f.write('PIEH'.encode('utf-8'))
    np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
    flow = flow.astype(np.float32)
    flow.tofile(f)
    f.flush()
    f.close() 


def run_model(prototxt, weights, img0_p, img1_p, out_p, verbose=False):

    if(not os.path.exists(weights)): raise BaseException('caffemodel does not exist: '+weights)
    if(not os.path.exists(prototxt)): raise BaseException('deploy-proto does not exist: '+prototxt)
    if(not os.path.exists(img0_p)): raise BaseException('img0 does not exist: '+img0_p)
    if(not os.path.exists(img1_p)): raise BaseException('img1 does not exist: '+img1_p)
    
    print("starting run_model")
    num_blobs = 2
    input_data = []
    img0 = misc.imread(img0_p)
    if len(img0.shape) < 3: input_data.append(img0[np.newaxis, np.newaxis, :, :])
    else:                   input_data.append(img0[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :])
    img1 = misc.imread(img1_p)
    if len(img1.shape) < 3: input_data.append(img1[np.newaxis, np.newaxis, :, :])
    else:                   input_data.append(img1[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :])

    width = input_data[0].shape[3]
    height = input_data[0].shape[2]
    vars = {}
    vars['TARGET_WIDTH'] = width
    vars['TARGET_HEIGHT'] = height
    print(width, height)

    divisor = 64.
    vars['ADAPTED_WIDTH'] = int(ceil(width/divisor) * divisor)
    vars['ADAPTED_HEIGHT'] = int(ceil(height/divisor) * divisor)

    vars['SCALE_WIDTH'] = width / float(vars['ADAPTED_WIDTH']);
    vars['SCALE_HEIGHT'] = height / float(vars['ADAPTED_HEIGHT']);

    tmp = tempfile.NamedTemporaryFile(mode='w', delete=True)

    proto = open(prototxt).readlines()
    for line in proto:
        for key, value in vars.items():
            tag = "$%s$" % key
            line = line.replace(tag, str(value))

        tmp.write(line)

    tmp.flush()

    if not verbose:
        caffe.set_logging_disabled()
       
    #caffe.set_device(args.gpu)
    
    caffe.set_mode_gpu()
    
    net = caffe.Net(tmp.name, weights, caffe.TEST)

    input_dict = {}
    for blob_idx in range(num_blobs):
        input_dict[net.inputs[blob_idx]] = input_data[blob_idx]
    
   
    #
    # There is some non-deterministic nan-bug in caffe
    # it seems to be a race-condition 
    #
    print('Network forward pass using %s.' % weights)
    i = 1
    while i<=5:
        i+=1

        net.forward(**input_dict)

        containsNaN = False
        for name in net.blobs:
            blob = net.blobs[name]
            has_nan = np.isnan(blob.data[...]).any()

            if has_nan:
                print('blob %s contains nan' % name)
                containsNaN = True

        if not containsNaN:
            print('Succeeded.')
            break
        else:
            print('**************** FOUND NANs, RETRYING ****************')

    blob = np.squeeze(net.blobs['predict_flow_final'].data).transpose(1, 2, 0)
    
    writeFlow(out_p, blob)
    
    
def run_model_multiples(prototxt, weights, listfile, output_dir, blobs=['warp_loss2'], save_image=False):
    """
    Input : prototxt, weights
    listfile : list of (image1, image2)
    
    output_dir : place where the computed flows will be saved
    outfile : csv file with losses for each image pair will be saved
    
        Call this function with a prototxt that work without LMDB, and that has only two images in input
    
    """
    import os, sys, numpy as np
    import argparse
    from scipy import misc
    import caffe
    import tempfile
    from math import ceil

       
    if(not os.path.exists(weights)): raise BaseException('caffemodel does not exist: '+weights)
    if(not os.path.exists(prototxt)): raise BaseException('deploy-proto does not exist: '+prototxt)
    if(not os.path.exists(listfile)): raise BaseException('listfile does not exist: '+listfile)

    def readTupleList(filename):
        list = []
        for line in open(filename).readlines():
            if line.strip() != '':
                list.append(line.split())

        return list

    ops = readTupleList(listfile)

    width = -1
    height = -1
    
    output = []
    output.append(['image0', 'image1', 'real_flow', 'estimated_flow'] + blobs)
    n = len(ops)
    for i, ent in enumerate(ops):
        print("processing", i)
        flush()
        print('Processing tuple:', ent)

        num_blobs = 2
        input_data = []
        img0 = misc.imread(ent[0])
        if len(img0.shape) < 3: input_data.append(img0[np.newaxis, np.newaxis, :, :])
        else:                   input_data.append(img0[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :])
        img1 = misc.imread(ent[1])
        if len(img1.shape) < 3: input_data.append(img1[np.newaxis, np.newaxis, :, :])
        else:                   input_data.append(img1[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :])

        if width != input_data[0].shape[3] or height != input_data[0].shape[2]:
            width = input_data[0].shape[3]
            height = input_data[0].shape[2]

            vars = {}
            vars['TARGET_WIDTH'] = width
            vars['TARGET_HEIGHT'] = height

            divisor = 64.
            vars['ADAPTED_WIDTH'] = int(ceil(width/divisor) * divisor)
            vars['ADAPTED_HEIGHT'] = int(ceil(height/divisor) * divisor)

            vars['SCALE_WIDTH'] = width / float(vars['ADAPTED_WIDTH']);
            vars['SCALE_HEIGHT'] = height / float(vars['ADAPTED_HEIGHT']);

            tmp = tempfile.NamedTemporaryFile(mode='w', delete=False)

            proto = open(prototxt).readlines()
            for line in proto:
                for key, value in vars.items():
                    tag = "$%s$" % key
                    line = line.replace(tag, str(value))

                tmp.write(line)

            tmp.flush()

        caffe.set_logging_disabled()
        caffe.set_mode_gpu()
        net = caffe.Net(tmp.name, weights, caffe.TEST)

        input_dict = {}
        for blob_idx in range(num_blobs):
            input_dict[net.inputs[blob_idx]] = input_data[blob_idx]

        #input_data
        # There is some non-deterministic nan-bug in caffe
        #
        print('Network forward pass using %s.' % weights)
        j = 1
        while j<=5:
            j+=1

            net.forward(**input_dict)

            containsNaN = False
            for name in net.blobs:
                blob = net.blobs[name]
                has_nan = np.isnan(blob.data[...]).any()

                if has_nan:
                    print('blob %s contains nan' % name)
                    containsNaN = True

            if not containsNaN:
                print('Succeeded.')
                break
            else:
                print('**************** FOUND NANs, RETRYING ****************')
        
        flow_blob = np.squeeze(net.blobs['predict_flow_final'].data).transpose(1, 2, 0)
        output_flow_file = output_dir + "flow_%04d" % i
        print("saving flow at %s" % output_flow_file)
        if save_image:
            save_flow_image(flow_blob, output_flow_file + ".png")
        writeFlow(output_flow_file + ".flo", flow_blob)
        out_line = [ent[0], ent[1], output_flow_file]
        for blob in blobs:
            out_line.append(net.blobs[blob].data)
            
        output.append(out_line)
        
    with open(output_dir + '/list_out.txt', 'w') as f:
        f.write("\n".join((",".join(line) for line in output)))
    
        
        
            
def run_model_lmdb(prototxt, weights, output_dir, blobs=['flow_loss', 'warp_loss'], batch_size=1, 
                   num_inputs=100, save_input=False):
    caffe.set_mode_gpu()

    print("loading network")
    net = caffe.Net(prototxt, weights, caffe.TEST)
    print("done loading network")

    iterations = num_inputs / batch_size
    
    output = []
    if save_input:
        output.append(['image0', 'image1', 'real_flow', 'estimated_flow'] + blobs)
    else:
        output.append(['real_flow', 'estimated_flow'] + blobs)

    for i in range(iterations):
        print("iteration", i)
        flush()
        net.forward()
        imgs0 = net.blobs["blob0"].data.copy()
        imgs1 = net.blobs["blob1"].data.copy()
        flows = net.blobs["blob2"].data.copy()

        losses = {loss: net.blobs[loss].data for loss in blobs}

        for b in range(batch_size):
            out_line = []
            current_id = i * batch_size + b
            flow = np.squeeze(flows[b]).transpose(1, 2, 0)
            flow_path = output_dir + "/predict_flow_%s" % current_id
            writeFlow(flow_path, flow)

            # if save_input:
            #     img0 = 

            out_line.append(flow_path)
            for l in losses:
                out_line.append(str(losses[l] / batch_size))
        output.append(out_line)

    with open(output_dir + '/losses.txt', 'w') as f:
        f.write("\n".join((",".join(line) for line in output)))
    

