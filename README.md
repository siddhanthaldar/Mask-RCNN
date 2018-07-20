# Mask-RCNN
This is an implementation of Mask RCNN for carrying out nuclei segmentation. The code has been referred from the Mask RCNN implementation by matterport.

## Dataset
The dataset for [Kaggle Data Science Bowl 2018](https://www.kaggle.com/c/data-science-bowl-2018) has been used here. The dataset must be put into the folder **samples/nucleus/data/**.

## Training 
Train a new model starting from ImageNet weights
```
cd samples/nucleus/
python3 nucleus.py train --dataset=data/ --subset=train --weights=imagenet
```
Train a new model starting from specific weights file
```
cd samples/nucleus/
python3 nucleus.py train --dataset=data/ --subset=train --weights=/path/to/weights.h5
```

Resume training a model that you had trained earlier
```
cd samples/nucleus/
python3 nucleus.py train --dataset=data/ --subset=train --weights=last
```

Generate submission file
```
cd samples
python3 nucleus_test.py 
```

## Results
![1](https://user-images.githubusercontent.com/25313941/42985989-6aef75f8-8c11-11e8-88f7-8a061fe2d7b7.jpg)
![nucleus3_mrcnn](https://user-images.githubusercontent.com/25313941/42986007-89cddc62-8c11-11e8-8d23-0952bb95c6a3.png)
![4](https://user-images.githubusercontent.com/25313941/42986010-8ea5ecf2-8c11-11e8-9009-f3177caf881a.png)
