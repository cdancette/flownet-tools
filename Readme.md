# Documentation Flownet Project


## FLowNet2 installation 

Follow the documentation in https://github.com/lmb-freiburg/flownet2/

## Run the network

Everything is done through the main.py script

You can run `python main.py -h` for help, or `python main.py command -h` for a specific command help.

* Run the network on an image pair, to get the flow

`python main.py run <prototxt> <input_weights> <image0> <image1> <output_path_flow>`

* To run the network on multiple pairs

Create a text file containing a list of (path_image1, path_image2)

and call `main.py run_multiple <prototxt> <weights> <listfile> <output_dir>`

Use the `--save-images` flag to save the flow, the two images and the warped image1, all as image files.

* If you have consecutive image, you can provide a list file containing only the list of single images, 
and run `main.py consecutive <prototxt> <weights> <listfile> <output_dir>`.


### Make a video
To make a video, with a list of images or flow images, you can use ffmpeg. See `my make_video.sh` script.

## Train the network

* To train the network
Basically, follow the flownet2 documentation. You have to : 
- create the dataset (in the lmdb format), using the command `/build/tools/convert_imageset_and_flow.bin` (see make_lmdbs.sh script in flownet2/data directory)
- create your train.prototxt file, from one of the flownet2 models (you have to change the training / testing ldmb paths)
- create your solver.prototxt (it is provided in flownet2/models)

and train with caffe (`caffe train --solver solver.prototxt`)

You can also train using my function `main.py train`

usage: `main.py train [-h]
                     prototxt input_weights output_weights iterations log_file`

.

## Tools

I have a few python scripts to run computations on optical flow, you can check in tools.py.
The most interesting functions are : 
- `apply_flow_reverse` : warp an image, given a flow.
- tools to build a dataset : `generate_dataset` that uses `generate_two_images_and_flow`, which, from one image, generates two sub-images that are translated, and the corresponding flow.



TODO : calculer sur les datasets pour chaque image le flot par rapport a l'image de depart, et le warp. 
Ensuite calculer la L2 entre le warp et l'originale
- avec les bandes noires
- sans les bandes noires (normalisé sur le nombre de pixels)
- en remplaçant les bandes noires par la moyenne de l'image


## FlowNet2 models
They are listed in the flownet2 paper : https://arxiv.org/pdf/1612.01925.pdf

- FlowNet2 : The biggest model, best results but the slowest (~5s for 1 flow, but should be much faster according to paper).
- FlowNet2-s : the fastest model, but bad results

And between those two : FlowNet2-CSS, FlowNet2-SS


## 3D reconstruction
You need the point cloud library for this.

You can use the optical flow to perform 3D reconstruction.

I copied three files from the RECONSTRUCT[1] library and adapted them to work with optical flows obtained with flownet. 

To use it : modify the paths in the `reconstruct.py` file, at the end.
You have to give the two images, the flow, and the output paths.
Then run `python reconstruct.py`. It will save a 3D point cloud in pcd format.

You can then vizualize it with pcl_viewer[2] from Point Cloud Library.

You can also change sensor parameter in the `processing.py` file, line 49 (sensor size, and focal length).

## Summary of results.

#### Fine tuning

I tried to fine-tune FlowNet2-S and FlowNet2-SS on the translation dataset, but I never got better results than with the default weights (except on translated images). They were very bad visually.

I didn't manage to fine-tune train the big FlowNet2 network.

Other approachs we could use to fine tune to our dataset : 
- Use the same method they used, but with the lake images as a background, and copy paste natural objects on the foreground (trees, houses).
- We can even do something like having 2 or three layers of depth, that move relatively to one another, to simulate a translation of the boat.
- Try to train using the warp loss. I almost did this but had no time to finish.



[1] https://github.com/contactjiayi/RECONSTRUCT
[2] https://github.com/PointCloudLibrary/pcl/blob/master/visualization/tools/pcd_viewer.cpp


