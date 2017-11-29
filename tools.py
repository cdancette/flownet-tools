import cv2
import sys
sys.path.append("OpticalFlowToolkit/")
import lib.flowlib as optical_flow_lib
import numpy as np
import matplotlib.pyplot as plt


## FLOW TOOLS

def writeFlow(name, flow):
    f = open(name, 'wb')
    f.write('PIEH'.encode('utf-8'))
    np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
    flow = flow.astype(np.float32)
    flow.tofile(f)
    f.flush()
    f.close()

def open_flow(flow_path):
    return optical_flow_lib.read_flo_file(flow_path)

def open_image(image_path):
    img = cv2.imread(image_path)
    return img

def apply_flow_reverse(image, flow):
    h, w = flow.shape[:2]
    # openCV coordinates are inversed / numpy
    map_x = flow[:,:,0] + np.arange(w)
    map_x = map_x.astype('float32')
    map_y = flow[:,:,1] + np.arange(h)[:,np.newaxis]
    map_y = map_y.astype('float32')
    new_image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
    return new_image

def apply_flow_reverse_path(second_image_path, incoming_flow_path, output_path):
    second_image = open_image(second_image_path)
    incoming_flow = open_flow(incoming_flow_path)
    expected_first_image = apply_flow_reverse(second_image, incoming_flow)
    cv2.imwrite(output_path, expected_first_image)
    
def calculate_loss(image1, image2):
    return cv2.norm(image1, image2, cv2.NORM_L2)

def show_image(image):
    image_show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_show)
    
    
def new_loss(image_prev, image_after, flow):
    h, w = flow.shape[:2]
    flow_x = flow[:,:,0]
    flow_y = flow[:,:,1]
    sum_abs_flow = abs(flow_x) + abs(flow_y)
    nonzeros = np.nonzero(sum_abs_flow) # indices where the flow is not nul in x or y direction
    nonzeros_after =  nonzeros[0] + flow_x[nonzeros].astype('int'), nonzeros[1] + flow_y[nonzeros].astype('int')
    
    ## keep only positive indexes
    # filter negative x values
    nx = (nonzeros_after[0] >= 0).nonzero()
    nonzeros_after = [nonzeros_after[0][nx], nonzeros_after[1][nx]]
    nonzeros = [nonzeros[0][nx], nonzeros[1][nx]]
    # filter negative y values
    ny = (nonzeros_after[1] >= 0).nonzero()
    nonzeros_after = [nonzeros_after[0][ny], nonzeros_after[1][ny]]
    nonzeros = [nonzeros[0][ny], nonzeros[1][ny]]
    
    ## Discard values that are too high
    # filter too high x values
    nx = (nonzeros_after[0] < image_prev.shape[0]).nonzero()
    nonzeros_after = [nonzeros_after[0][nx], nonzeros_after[1][nx]]
    nonzeros = [nonzeros[0][nx], nonzeros[1][nx]]
    ## filter too high y values
    ny = (nonzeros_after[1] < image_prev.shape[1]).nonzero()
    nonzeros_after = [nonzeros_after[0][ny], nonzeros_after[1][ny]]
    nonzeros = [nonzeros[0][ny], nonzeros[1][ny]]
    
    return cv2.norm(image_prev[nonzeros], image_after[nonzeros_after], cv2.NORM_L2), nonzeros, nonzeros_after

    ## TODO : what to do when the flow is nul ?
    
    
#### DATASET

def generate_translation_flow(height, width, dx, dy):
    """
    dx <=> columns 
    dy <=> rows
    """
    flow_x = np.full((height, width), dx)
    flow_y = np.full((height, width), dy)
    flow = np.stack((flow_x, flow_y), axis=2)
    return flow    

def translate_image(img, x, y):
    rows,cols = img.shape[0], img.shape[1]
    M = np.float32([[1,0,x],[0,1,y]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst


def crop_image(image, up, left, down, right):
    return image[up:image.shape[0] - down,left:image.shape[1] - right]


def generate_two_images_and_flow(image, dx, dy):
    """
    Returns img1, img2, flow : two images that are a transition of each other by factor dx, dy
    img1 and img2 have shape (x - dx, y - dy)
    /!\ 
    dx = column
    dy = row
    """
    dxf, dyf = dx, dy
    x, y = image.shape[1], image.shape[0]
    
    dx2, dy2 = 0, 0
    if dx < 0:
        dx2 = abs(dx)
        dx = 0
    if dy < 0:
        dy2 = abs(dy)
        dy = 0
    img1 = image[dy:y - dy2, dx:x - dx2]
    img2 = image[dy2:y - dy, dx2:x - dx]
    flow = generate_translation_flow(img1.shape[0], img1.shape[1], dxf, dyf)
    return img1, img2, flow


import os
from os import listdir, makedirs
import os
from random import randint, choice
import numpy as np

def generate_dataset_simple(directory, output_directory, n_images=32):
    files = listdir(directory)
    image_paths = [directory + "/" + file for file in files]
    list_files = ""
    makedirs(output_directory)
    i = 0
    for path in image_paths:
        print(path)
        if i == n_images:
            break
        i += 1
        if not os.path.isfile(os.path.join(path)):
            continue
        image_name = path.split('/')[-1].split('.')[0]
        print(image_name)
        image = open_image(path)
        print(image.shape)
        image = cv2.resize(image, None, image, 1.5, 1.5)
        img1, img2, flow = generate_two_images_and_flow(image, dx=choice([-40, 40]), dy=choice([-40, 40]))
        writeFlow(output_directory + image_name + '.flo', flow)
        cv2.imwrite(output_directory + image_name + '-1.ppm', img1)
        cv2.imwrite(output_directory + image_name + '-2.ppm', img2)
        list_files += output_directory + image_name + '-1.ppm' + " " + \
            output_directory + image_name + '-2.ppm' + " "  + \
            output_directory + image_name + '.flo' + "\n"

    with open(output_directory + 'list.txt', 'w') as f:
        f.write(list_files)
    make_lmdb(os.path.join(output_directory,"list.txt"), os.path.join(output_directory,"data_lmdb"))


def generate_dataset(directory, output_directory, n_images=32):
    image_paths = []
    folders = listdir(directory)
    for folder in folders:
	image_paths += [folder + "/" + path for path in listdir(directory + "/" + folder)]
    list_files = ""
    makedirs(output_directory)
    i = 0
    for path in image_paths:
        print(path)
        if i == n_images:
            break
        i += 1
        if not os.path.isfile(os.path.join(directory, path)):
            continue
        image_name = path.split('.')[0].replace("/", "-")
        image = open_image(directory + path)
        print(image.shape)
        image = cv2.resize(image, None, image, 1.5, 1.5)
        img1, img2, flow = generate_two_images_and_flow(image, dx=choice([-40, 40]), dy=choice([-40, 40]))
        writeFlow(output_directory + image_name + '.flo', flow)
        cv2.imwrite(output_directory + image_name + '-1.ppm', img1)
        cv2.imwrite(output_directory + image_name + '-2.ppm', img2)
        list_files += output_directory + image_name + '-1.ppm' + " " + \
            output_directory + image_name + '-2.ppm' + " "  + \
            output_directory + image_name + '.flo' + "\n"

    with open(output_directory + 'list.txt', 'w') as f:
        f.write(list_files)
    make_lmdb(os.path.join(output_directory,"list.txt"), os.path.join(output_directory,"data_lmdb"))
        
import subprocess  

def make_lmdb(list_path, output_path):
    bash_command = "/opt/flownet2/build/tools/convert_imageset_and_flow.bin %s %s 0 lmdb" % (list_path, output_path)
    print("executing command")
    print(bash_command)
    process = subprocess.check_output(bash_command.split(), stderr=subprocess.STDOUT)
    #while True:
    #    line = process.stdout.readline()
    #    print(line)
    #    if not line: break

### TESTING


def test_model_on_image_pair(prototxt,
              model_weights, img0_p, img1_p, prefix):
    from run_model import run_model
    """
    img0_p, img1_p : paths
    prefix : path where the output files will be saved (flow, and expected image)
    """
    flow_p = prefix + "-out.flo"
    
    it1 = cv2.imread(img1_p)
    it0 = cv2.imread(img0_p)
   
    run_model(prototxt, model_weights, img0_p, img1_p, flow_p, verbose=False)
    
    flow = optical_flow_lib.read_flo_file(flow_p)    
    expected_it0 = apply_flow_reverse(it1, flow)
    cv2.imwrite(prefix + "-img0-expected.jpg", expected_it0)
    cv2.imwrite(prefix + "-img0.jpg", it0)
    cv2.imwrite(prefix + "-img1.jpg", it1)

    optical_flow_lib.save_flow_image(flow, prefix + "-flo.png")
    return flow_p
    
    
