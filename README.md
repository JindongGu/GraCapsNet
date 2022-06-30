
# Implementation of <a href="https://arxiv.org/pdf/2012.01674.pdf">GraCapsNet</a>: Interpretable Graph Capsule Networks for Object Recognition 

GraCapsNets: In CapsNets, the primary capsules represent object parts, e.g., eyes and nose of a cat. In our GraCapsNets, we explicitly model the relationship between the primary capsules (i.e., part-part relationship) with graphs. Then, the followed graph pooling operations pool relevant object parts from the graphs to make a classification vote. 

![Overview](imgs/overview.png)

Explanation: Since the graph pooling operation reveals which input features (i.e.object parts) are pooled as relevant ones, explanations can be easily created with pooling attention to explain the classification decisions.


## Training GraCapsNet
```
python main.py
```
The training setting: loss=Marg, constant learning rate=0.001, Adam optimizer

## Creating Explainations
The attention of multi-head graph pooling operation is visualized on the original input. The color bars right indicate the importance of the input features, where blue corresponds to little
relevance, dark red to high relevance. The wings are relevant for the recognition of the class
Plane, and the heads (especially the noses) to Dog.

![Exp](imgs/cifar10_exp.png)

Contact: jindong.gu@outlook.com

If this repo is helpful for you, please cite the paper.

