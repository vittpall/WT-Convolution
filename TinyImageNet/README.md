For running the test

Choose which dataset to use:

    1. TinyImageNet: The input size is 64x64.
    2. ImageNet: The input size is 224x224.

Choose which network to use:

    1. HT-ResNet50: The input size is 224x224.
    2. HT-ResNet101: The input size is 224x224.

WHTResNet50x3 is the 3-path HT-ResNet50, and the input size is 224x224.



To train the network:
        
    python main.py -a wht_resnet50 -b 128 --lr 0.05

To test the network:

    python test.py -a wht_resnet50 -b 128 -b10 32

The test code contains a 10-fold test, and the 10-fold test batch size is 32. We reduce this size to avoid the memory issue. 
