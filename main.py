deploy_model ="/home/gpu_user/corentin/flownet2/models/FlowNet2-s/FlowNet2-s_deploy.prototxt.template"
weights ="/home/gpu_user/corentin/flownet2/models/FlowNet2-s/coco-weights-3.caffemodel"
weights_original="/home/gpu_user/corentin/flownet2/models/FlowNet2-s/coco-FlowNet2-s_weights.caffemodel"
img0_p = "/home/gpu_user/corentin/lake-dataset/140606f/0036/0390.jpg"
img1_p = "/home/gpu_user/corentin/lake-dataset/140606f/0036/0391.jpg"
flow_p ="/home/gpu_user/corentin/FlowNet2-s/model3/140606f-0036-0390-0391-out.flo"
prefix = "/home/gpu_user/corentin/FlowNet2-s/model3/140606f-0036-0390-0391"
prefix_original ="/home/gpu_user/corentin/FlowNet2-s/model-original/140606f-0036-0390-0391"

from tools import *

test_model_on_image_pair(deploy_model, weights, img0_p, img1_p, prefix)
