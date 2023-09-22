Environment<br>
python 3.9.7<br>
torch 2.0.0<br>
torchvision 0.15.1<br>
numpy 1.20.3<br>

To train the model:

python train.py -root /data/E2H_source/ -mode rgb -save_model I3D_RGB_HMDB -dset H -CLASS_NUM 13 -trainfile ./targetlistname_My_H.txt -r1 0.2 -r2 0.3 -th 0.9 -lr 0.025 -sset E -droot /data/HMDB51-frame -PRE 5 -method Pro -r3 0.5

where /data/E2H_source/ is the directory containing synthesized video frames and /data/HMDB51-frame is directory containing extracted video frames from HMDB dataset. The synthesized video frames of E2H can be found in https://drive.google.com/file/d/1m2_C2C-7b-geoD46W9Mh2Kdtd_A6X15Y/view?usp=sharing.

The rgb_imagenet.pt can be found in https://github.com/piergiaj/pytorch-i3d/tree/master/models.

Some logs are placed in logs/. In each testing phase, the former number is the accuracy of trained model and the later one is the accuracy of refined pseudo labels.
