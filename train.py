from caffe.proto import caffe_pb2
import tempfile

import os
import numpy as np
import caffe
caffe.set_mode_gpu()
caffe_root = "/home/gpu_user/corentin/flownet2/"
import sys
flush = sys.stdout.flush

def create_solver(train_net_path, snapshot_prefix=None, test_net_path=None, base_lr=0.00001):
    s = caffe_pb2.SolverParameter()

    # Specify locations of the train and (maybe) test networks.
    s.train_net = train_net_path
    if test_net_path is not None:
        s.test_net.append(test_net_path)
        s.test_interval = 1000  # Test after every 1000 training iterations.
        s.test_iter.append(100) # Test on 100 batches each time we test.

    # The number of iterations over which to average the gradient.
    # Effectively boosts the training batch size by the given factor, without
    # affecting memory utilization.
    s.iter_size = 1
    
    s.max_iter = 500000     # # of times to update the net (training iterations)
    
    # Solve using the stochastic gradient descent (SGD) algorithm.
    # Other choices include 'Adam' and 'RMSProp'.
    s.type = 'Adam'

    # Set the initial learning rate for SGD.
    s.base_lr = base_lr

    # Set `lr_policy` to define how the learning rate changes during training.
    # Here, we 'step' the learning rate by multiplying it by a factor `gamma`
    # every `stepsize` iterations.
    s.lr_policy = 'multistep'
    s.gamma = 0.5
    s.stepsize = 20000

    # Set other SGD hyperparameters. Setting a non-zero `momentum` takes a
    # weighted average of the current gradient and previous gradients to make
    # learning more stable. L2 weight decay regularizes learning, to help prevent
    # the model from overfitting.
    s.momentum = 0.9
    s.weight_decay = 1e-4

    # Display the current training loss and accuracy every 1000 iterations.
    s.display = 50
    s.momentum2 = 0.999

    # Snapshots are files used to store networks we've trained.  Here, we'll
    # snapshot every 10K iterations -- ten times during training.
    s.snapshot = 10000
    s.snapshot_prefix = snapshot_prefix or '/home/gpu_user/corentin/flownet2/models/random-prefix-%s' % time.time()
    
    # Train on the GPU.  Using the CPU to train large networks is very slow.
    s.solver_mode = caffe_pb2.SolverParameter.GPU
    
    # Write the solver to a temporary file and return its filename.
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(str(s))
        return f.name
    
    
import time
def run_solvers(niter, solvers, disp_interval=10, log_file=None, test_interval=20):
    """Run solvers for niter iterations,
   returning the loss and accuracy recorded each iteration.
   `solvers` is a list of (name, solver) tuples."""
    print("Starting training")
    if log_file:
        logs = dict()
        for name, _ in solvers:
            log = open(log_file + '_' + name + '-' + time.strftime('%Y-%m-%d-%H:%M:%S'), 'a')
            logs[name] = log
        
    blobs = ('net2_flow_loss2', 'net2_flow_loss3', 'net2_flow_loss4', 'net2_flow_loss5', 'net2_flow_loss6')
    loss = [{name: np.zeros(niter) for name, _ in solvers}
                 for _ in blobs]
    test_loss = [{name: np.zeros(niter) for name, _ in solvers}
                 for _ in blobs]
    for it in range(niter):
        for name, s in solvers:
            s.step(1)  # run a single step in Caffe
	    for i, blob in enumerate(blobs):
                loss[i][name][it] = s.net.blobs[blob].data.copy() 
            ## TEST 
            if it % test_interval == 0 or it == niter - 1:
                correct = 0
                mean_loss = 0
                s.test_nets[0].forward()
		for i, b in enumerate(blobs):
	                test_loss[i][name][it] = s.test_nets[0].blobs[blob].data
                if log_file:
                    for b in blobs:
                        logs[name].write("%s,%s,%s\n" % (it, ','.join(str(loss[i][name][it]) for i, b in enumerate(blobs)), ','.join(str(test_loss[i][name][it]) for i, b in enumerate(blobs)))) 
                        logs[name].flush()
            # DONE TESTING
                loss_disp = '; '.join('%s: loss=%.2f, test_loss=%.2f' % (n, loss[0][n][it],test_loss[0][n][it]) for n, _ in solvers)
                print '%3d) %s' % (it, loss_disp)
                flush()
                
    # Save the learned weights from both nets.
    weight_dir = tempfile.mkdtemp()
    weights = {}
    for name, s in solvers:
        filename = 'weights.%s.caffemodel' % name
        print("saving in %s" % os.path.join(weight_dir, filename))
        weights[name] = os.path.join(weight_dir, filename)
        s.net.save(weights[name])
    return loss, test_loss, weights


from matplotlib import pyplot as plt

def deprocess_net_image(image):
    image = image.copy()              # don't modify destructively
    image = image[::-1]               # BGR -> RGB
    image = image.transpose(1, 2, 0)  # CHW -> HWC
    #image += [123, 117, 104]          # (approximately) undo mean subtraction

    # clamp values in [0, 255]
    image[image < 0], image[image > 255] = 0, 255

    # round and cast from float32 to uint8
    image = np.round(image)
    image = np.require(image, dtype=np.uint8)

    return image
def disp_preds(net, image, labels, k=5, name='ImageNet'):
    input_blob = net.blobs['data']
    net.blobs['data'].data[0, ...] = image
    probs = net.forward(start='conv1')['probs'][0]
    top_k = (-probs).argsort()[:k]
    print 'top %d predicted %s labels =' % (k, name)
    print '\n'.join('\t(%d) %5.2f%% %s' % (i+1, 100*probs[p], labels[p])
                    for i, p in enumerate(top_k))

def disp_imagenet_preds(net, image):
    disp_preds(net, image, imagenet_labels, name='ImageNet')

def disp_style_preds(net, image):
    disp_preds(net, image, style_labels, name='style')
    
    
def train(prototxt, initial_weights, output_weights, iterations, 
         solver=None, disp_interval=10, log_file=None):
    print "Loading net and solver"
    if solver is None:
        solver_filename = create_solver(prototxt, snapshot_prefix=output_weights, test_net_path=prototxt)
        solver = caffe.get_solver(solver_filename)
        solver.net.copy_from(initial_weights)
    solvers = [('solver', solver)]
    loss, test_loss, weights = run_solvers(iterations, solvers, disp_interval=disp_interval, log_file=log_file)
    print('Done.')
    train_loss = loss['solver']
    #weights = weights['pretrained']
    solver.net.save(output_weights)
    print("Weights saved in %s" % output_weights)
    return solver, train_loss, test_loss
