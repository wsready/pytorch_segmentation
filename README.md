# pytorch-segmentation


**Update on 2023/12/24. Release the first version code

### Introduction
This is a PyTorch(2.0.0) implementation of medical segmentation.

### Installation
The code was tested with Anaconda and Python 3.9. After installing the Anaconda environment:

0. Clone the repo:
    ```Shell
    git clone https://github.com/wsready/pytorch_segmentation.git
    cd pytorch_segmentation
    ```

1. Install dependencies:
   ```Shell
   conda activate ***
   pip install -r requirements.txt
   ```
    
### Training
Follow steps below to train your model:
1. Input arguments: (see full input arguments via python train.py --help):
   ```Shell
    usage: train.py [-h] [--backbone {resnet,xception,drn,mobilenet}]
            [--out-stride OUT_STRIDE] [--dataset {pascal,coco,cityscapes}]
            [--use-sbd] [--workers N] [--base-size BASE_SIZE]
            [--crop-size CROP_SIZE] [--sync-bn SYNC_BN]
            [--freeze-bn FREEZE_BN] [--loss-type {ce,focal}] [--epochs N]
            [--start_epoch N] [--batch-size N] [--test-batch-size N]
            [--use-balanced-weights] [--lr LR]
            [--lr-scheduler {poly,step,cos}] [--momentum M]
            [--weight-decay M] [--nesterov] [--no-cuda]
            [--gpu-ids GPU_IDS] [--seed S] [--resume RESUME]
            [--checkname CHECKNAME] [--ft] [--eval-interval EVAL_INTERVAL]
            [--no-val]
   ```
2. To train unet using custom medical dataset and ResNet as backbone:
    ```Shell
    bash train.sh
    ```   


