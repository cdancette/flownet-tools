Flownet-s architecture with warp loss between original image and expected image
after warping.

## Finetuning

Use weights provided by flownet2 repo in models/Flownet2-s/
    
    caffe train --solver solver.prototxt --weights Flownet-s/FlowNet2-S_weights.caffemodel.h5
