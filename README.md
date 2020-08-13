[toc]

Nerual Collaborative Filtering模型详解
==

## 背景

> 推荐系统在对用户和物品特征之间的交互建模时，大多采用MF，对user和item的隐特征使用内积计算，这是一种线性方式，**并不足以捕捉到用户交互数据中的复杂结构信息**。
>
> 通过将内积运算替换为可以从数据中学习任意函数的神经体系结构，本文提出了一个名为NCF（Neural network based Collaborative Filtering）的通用框架。 NCF是通用的，可以在其框架下表示和推广矩阵分解。 为了使NCF建模具有非线性效果，提出利用多层感知器来学习用户与物品的交互函数。 两个数据集上的实验表明，与现有方法相比，NCF框架有了显著的改进，使用更深层次的神经网络可以提供更好的推荐性能。

## NeuralCF模型

> 首先提出通用的NCF框架，详细的解释了使用概率模型强调隐式数据的二值属性的NCF如何进行学习。然后证明了在NCF下MF可以被表达和推广。
>
> 为了探索深度神经网络的协同过滤，提出了NCF的实例化，使用多层感知器(MLP)学习用户-项目的交互函数。
>
> 最后，提出了一个新型的神经矩阵分解模型，在NCF框架下结合了MF和MLP；在对用户-物品潜在结构的建模过程中它综合了MF的线性优点和MLP的非线性优点，让模型的表达能力更强。

### 通用框架

![image](https://github.com/ShaoQiBNU/Nerual_Collaborative_Filtering/blob/master/img/1.jpg)

> 如图所示，框架底层是由两个特征向量<a href="https://www.codecogs.com/eqnedit.php?latex=v_{u}^{U}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?v_{u}^{U}" title="v_{u}^{U}" /></a>和<a href="https://www.codecogs.com/eqnedit.php?latex=v_{i}^{I}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?v_{i}^{I}" title="v_{i}^{I}" /></a>组成，分别描述user和item的特征，在输入层之上为Embedding层，将稀疏表示投影到稠密向量embedding上。
>
> 然后将user的Embedding和item的Embedding分别输入多层神经网络结构，称之为神经协同过滤层，将隐向量映射到预测分数。可以自定义NeuralCF中的每一层，以发现用户-物品交互的某些潜在结构。
>
> 最后一个隐含层X的维数决定了模型的能力，输出层是预测的分数<a href="https://www.codecogs.com/eqnedit.php?latex=\hat{y_{ui}}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\hat{y_{ui}}" title="\hat{y_{ui}}" /></a> ,而训练是通过最小化<a href="https://www.codecogs.com/eqnedit.php?latex=\hat{y_{ui}}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\hat{y_{ui}}" title="\hat{y_{ui}}" /></a>和<a href="https://www.codecogs.com/eqnedit.php?latex=y_{ui}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?y_{ui}" title="y_{ui}" /></a>之间的pointwise loss来完成的，常采用平方损失的回归，如下：

![image](https://github.com/ShaoQiBNU/Nerual_Collaborative_Filtering/blob/master/img/2.jpg)

### GMF

> MF可以看做是NCF的一个特例，如下：

![image](https://github.com/ShaoQiBNU/Nerual_Collaborative_Filtering/blob/master/img/3.jpg)

### MLP 

> 由于NCF采用两条路径来对user和item进行建模，所以将这两种路径的特征串联起来是很直观的。然而，简单的向量连接并不能解释user和item的潜在特征之间的任何交互，这对于协同过滤进行建模效果是不够的。为了解决这个问题，文章在连接的向量上添加隐藏层，使用一个标准的MLP来学习用户和物品潜在特征之间的交互，从而可以灵活地学习<a href="https://www.codecogs.com/eqnedit.php?latex=p_{u}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?p_{u}" title="p_{u}" /></a>和<a href="https://www.codecogs.com/eqnedit.php?latex=q_{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?q_{i}" title="q_{i}" /></a>之间的交互。NCF框架下的MLP模型定义为：

![image](https://github.com/ShaoQiBNU/Nerual_Collaborative_Filtering/blob/master/img/4.jpg)

### GMF和MLP融合

> GMF采用线性核函数对潜在特征交互进行建模，MLP采用非线性核函数从数据中学习交互函数，接下来的问题是：如何在NCF框架下融合GMF和MLP，使得它们可以相互增强，从而更好地对复杂的用户-物品矩阵迭代交互进行建模？
>
> 一个简单的解决方案是让GMF和MLP共享相同的Embedding层，然后合并它们的交互函数的输出。具体来说，是将GMF与单层结构相结合的模型MLP表示为：

![image](https://github.com/ShaoQiBNU/Nerual_Collaborative_Filtering/blob/master/img/5.jpg)

> 然而，共享GMF和MLP的Embedding可能会限制融合模型的性能。例如，它意味着GMF和MLP必须使用相同大小的Embedding；对于两种模型的最优嵌入尺寸变化较大的数据集，该方案可能无法获得最优的集成。
>
> 为了给融合模型提供更大的灵活性，GMF和MLP分别学习独立的Embedding，并通过连接它们的最后一个隐藏层来组合这两个模型。如图所示：

![image](https://github.com/ShaoQiBNU/Nerual_Collaborative_Filtering/blob/master/img/6.jpg)

#### 预训练

> 由于NeuMF目标函数的非凸性，基于梯度的优化方法只能找到局部最优解，初始化对深度学习模型的收敛性和性能起着重要的作用。由于NeuMF是一个集成GMF和MLP的模型，可以使用GMF和MLP的预训练模型初始化NeuMF。首先用随机初始化训练GMF和MLP，直到收敛。然后，使用它们的模型参数作为NeuMF参数的相应部分的初始化。唯一的调整是在输出层，将两个模型的权值连接在一起，如下：

![image](https://github.com/ShaoQiBNU/Nerual_Collaborative_Filtering/blob/master/img/7.jpg)

## 实验

> 文章在两个公开数据集Movielens和Pinterest上进行了实验，主要回答了下面三个问题：
>
> 问题1：本文提出的NCF方法是否优于目前最先进的隐式协同过滤方法?
>
> 问题2：提出的优化框架(采用负采样的log损失)如何工作于推荐任务?
>
> 问题3：更深层次的隐藏单元是否有助于从用户-项目交互数据中学习?



#### 问题1

> NeuMF在两个数据集上都取得了最好的性能，显著地超出了最先进的eALS和BPR方法(平均而言，相对于eALS和BPR的相对改进是分别是4.5%和4.9%)。对于Pinterest，即使预测因子为8，NeuMF的表现也明显优于eALS和BPR，后者的预测因子为64，这表明了通过融合线性MF和非线性MLP模型的NeuMF的高表达性。其次，其他两种NCF方法——GMF和MLP——也表现出了相当强的性能。其中，MLP的表现略逊于GMF。请注意，MLP可以通过添加更多的隐藏层来进一步改进(参见4.4节)，这里只展示了三层的性能。对于小的预测因子，GMF在两个数据集上都优于eALS;虽然GMF存在大因子过拟合的问题，但其最佳性能优于eALS。最后，GMF相对于BPR表现出一致的改进，承认了分类感知log损失对于推荐任务的有效性，因为GMF和BPR学习相同的MF模型，但目标函数不同。

![image](https://github.com/ShaoQiBNU/Nerual_Collaborative_Filtering/blob/master/img/8.jpg)

> 下图显示了Top-K推荐列表的性能，其中排名位置K的范围为1到10。

![image](https://github.com/ShaoQiBNU/Nerual_Collaborative_Filtering/blob/master/img/9.jpg)

> 为了证明预训练对NeuMF的效用，文章比较了两个版本的NeuMF的性能——有预训练和没有预训练。对于未经训练的NeuMF，使用Adam通过随机初始化来学习它。
>
> 经过预训练的NeuMF在大多数情况下都能获得更好的性能，只有预测因子为8的MovieLens，预训练方法的表现稍差。与预训练相比，NeuMF的相对改善为2.2%和MovieLens和Pinterest分别为1.1%。这个结果证明了在初始化NeuMF的训练前方法的有效性。

![image](https://github.com/ShaoQiBNU/Nerual_Collaborative_Filtering/blob/master/img/10.jpg)

#### 问题2

> 为了处理隐式反馈的一类性质，文章将推荐转换为一个二分类任务。通过将NCF看作一个概率模型，采用log损失对它进行了优化。下图显示了MovieLens上每次迭代的NCF方法的训练损失(在所有实例上的平均值)和推荐性能。在Pinterest上的结果显示了同样的趋势，由于空间限制而被省略。
>
> 首先可以看到随着迭代次数的增加，NCF模型的训练损耗逐渐减小，推荐性能得到了改善。最有效的更新发生在前10次迭代中，并且更多的迭代可能会过度拟合一个模型(例如，尽管经过10次迭代后，NeuMF的训练损失不断减少，但它的推荐性能实际上在下降)。
>
> 其次，在三种NCF方法中，NeuMF的训练损耗最低，其次是MLP, GMF次之。推荐性能也表现出与了相同的趋势NeuMF > MLP > GMF。

![image](https://github.com/ShaoQiBNU/Nerual_Collaborative_Filtering/blob/master/img/11.jpg)

> 为了说明负采样对NCF方法的影响，文章展示了不同负采样比率的NCF方法的性能。可以清楚地看到，每个正实例仅一个负样本不足以获得最佳性能，而对更多的负实例进行采样是有益的。
>
> 将GMF与BPR进行比较，可以看出采样率为1的GMF的性能与BPR相当，而GMF明显优于BPR。对于这两个数据集，最佳采样率大约是3-6。在Pinterest上发现当采样率大于7时，NCF方法的性能开始下降。结果表明，过于激进地设置采样率可能会对性能造成负面影响。

![image](https://github.com/ShaoQiBNU/Nerual_Collaborative_Filtering/blob/master/img/12.jpg)

![image](https://github.com/ShaoQiBNU/Nerual_Collaborative_Filtering/blob/master/img/13.jpg)

#### 问题3

> 由于使用神经网络学习用户-项目交互功能的研究工作很少，因此使用深度网络结构是否有助于推荐任务的研究就显得很有趣。文章进行了进一步的调查MLP具有不同数量的隐藏层，结果如下：

![image](https://github.com/ShaoQiBNU/Nerual_Collaborative_Filtering/blob/master/img/14.jpg)

![image](https://github.com/ShaoQiBNU/Nerual_Collaborative_Filtering/blob/master/img/15.jpg)

> MLP-3表示MLP方法有三个隐藏层(除了嵌入层之外)，其他方法也有类似的表示法。即使对于具有相同功能的模型，堆叠更多的层也有利于性能。这个结果是非常令人鼓舞的，表明了使用深度模型进行协同推荐的有效性，把这种改善归因于叠加更多非线性层所带来的高非线性。为了验证这一点，文章进一步尝试叠加线性层，使用一个恒等函数作为激活函数。性能要比使用ReLU差的多。
>
> 对于没有隐藏层的MLP-0，嵌入层直接投影到预测，性能非常弱，并不比非个性化的ItemPop更好。这说明：简单地连接用户和项目潜在向量不足以对它们的特征交互进行建模，因此有必要使用隐藏层对其进行转换。


## 代码
https://github.com/hexiangnan/neural_collaborative_filtering
