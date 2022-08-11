# 数据集:
https://archive.ics.uci.edu/ml/datasets/mushroom

# 数据集信息：
该数据集包括与姬松茸和 Lepiota 家族中 23 种带鳃蘑菇相对应的假设样本的描述（第 500-525 页）。每个物种都被确定为绝对可食用、绝对有毒或可食用性未知且不推荐。后一类与有毒的一类结合在一起。该指南明确指出，确定蘑菇的可食用性没有简单的规则；对于毒橡树和常春藤来说，没有像“传单三，顺其自然”这样的规则。

![bushroom](image\bushroom.jpg)

# 属性信息
数据集中每一条数据包含如下特征，特征包括对蘑菇形状，质地，色彩等特征的描述，我们需要以此判断害是否有毒(p)或者可以吃(e).为两个类别的分类问题.
1.cap-shape(帽形):bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
2.cap-surface(帽面): fibrous=f,grooves=g,scaly=y,smooth=s
3.cap-color:brown=n,buff=b,cinnamon=c,gray=g.green=r,pink=p,purple=u,red=e,white=w,yellow=y
4.bruises?: bruises=t,no=f
5.odor(气味): almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=pspicy=s
6.gill-attachment:attached=adescending=dfree=fnotched=n
7.gill-spacing:close=c,crowded=wdistant=d
8.qill-size:broad=bnarrow=n
9.aill-color:black=kbrown=nbuff=b.chocolate=hgray=ggreenrorange=opink=ppurple=ured=ewhite=wyellow=y 
10.stalk-shape:enlarging=etapering=t
11.stalk-root:bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?
12.stalk-surface-above-ring:fibrous=f,scaly=y,silky=k,smooth=s
13.stalk-surface-below-ring:fibrous=fscaly=y,silky=k,smooth=s
14.stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
15.stalk-color-below-ring:brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
16.veil-type(面纱型):partial=p,universal=u
17.veil-color(面纱颜色):brown=n,orange=o,white=w,yellow=y18.ring-number:none=n,one=o,two=t
19.ring-type:cobwebby=c,evanescent=e,flaring=f,large=,none=n,pendant=p,sheathing=s,zone=z
20.spore-print-color(孢子色):black=k,brown=n,buff=bchocolate=h,green=rorange=o,purple=u,white=w,yellow=y
21.population(种群): abundant=a,clustered=c,numerous=n,scattered=s,several=vsolitary=y
22.habitat(栖息地):grasses=g,leaves=1,meadows=m,paths=purban=uwaste=wwoods=d