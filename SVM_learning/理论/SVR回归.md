# SVR


## 不考虑异常值
![SVM回归](image\SVM回归.jpg)
回归要求点要在两个虚线中间，分类是在两边

## 加入松弛变量SVR（考虑异常值）
![SVR加入松弛变量](image\SVR加入松弛变量.jpg)


## 带松弛变量和核函数的SVR

![带松弛变量和核函数的SVR](image\带松弛变量和核函数的SVR.jpg)

先将数据映射到高维空间在做点积，这样做的好处是假设数据是这样弯曲的曲线才能拟合好的。转化后，用两条虚直线就可围起来。

## SVR的数学推导
### 问题描述
- $给定{(x_1,y_1),...,(x_l,y_l)}\subset \mathcal{X} \times \mathbb{R}\\
\mathcal{X} = \mathbb{R}^d$
- $求f(x)=<w,x>+b with w\in\mathcal{X},b\in\mathbb{R}$
- $SVR模型\\
minimize\quad\frac{1}{2}||w||^2\\
subject\quad to \begin{cases}
y_i-<w,x_i>-b\leq\varepsilon \\
<w,x_i>+b-y_i\leq\varepsilon  
\end{cases}
$
$
||w||^2=<w,w>
$

### 带松弛变量的SVR
- $给定{(x_1,y_1),...,(x_l,y_l)}\subset \mathcal{X} \times \mathbb{R}\\
\mathcal{X} = \mathbb{R}^d$
- $求f(x)=<w,x>+b with w\in\mathcal{X},b\in\mathbb{R}$
- $SVR模型\quad C>0\\
minimize\quad\frac{1}{2}||w||^2+C\sum^l_{i=1}(\xi_i+\xi_i^*)\\
subject\quad to \begin{cases}
y_i-<w,x_i>-b\leq\varepsilon + \xi_i\\
<w,x_i>+b-y_i\leq\varepsilon +\xi_i\\
\xi_i,\xi_i^*\geq0
\end{cases}
$

### 损失函数 $\varepsilon$-insensitive loss function
$|\xi|_\varepsilon:=\begin{cases}
0 & if|\xi|\leq\varepsilon\\
|\xi|-\varepsilon & otherwise
\end{cases}
$

![insensitivelossfunction](image\insensitivelossfunction.jpg)

$
minimize\quad\frac{1}{2}||w||^2+C\sum^l_{i=1}(\xi_i+\xi_i^*)\\
subject\quad to \begin{cases}
y_i-<w,x_i>-b\leq\varepsilon + \xi_i\\
<w,x_i>+b-y_i\leq\varepsilon +\xi_i\\
\xi_i,\xi_i^*\geq0
\end{cases}
$

利用拉格朗日的分解方式变成对偶问题

引入$\alpha_i^{(*)},\eta_i^{(*)}\geq0$
$$
\begin{aligned}
L:= \frac{1}{2}||w||^2+C\sum^l_{i=1}(\xi_i+\xi_i^*)&-\sum^l_{i=1}(\eta_i\xi_i+\eta_i^*\xi_i^*)\\
&-\sum^l_{i=1}\alpha_i(\varepsilon+\xi_i-y_i+<w,x_i>+b)\\
&-\sum^l_{i=1}\alpha_i(\varepsilon+\xi_i+y_i-<w,x_i>-b)
\end{aligned}
$$
为了使得L取得极值，对primal参数求导，值对应为0
$$
\begin{aligned}
\partial_b L&=\sum_{i=1}^l(\alpha_i^*-\alpha_i)=0\\
\partial_w L&=w-\sum_{i=1}^l(\alpha_i-\alpha_i^*)x_i=0\\
\partial_{\xi_i^{(*)}} L&=C-\alpha_i^{(*)}-\eta_i^{(*)}=0
\end{aligned}
$$
带入L
$$
maximize\begin{cases}
-\frac{1}{2}\sum_{j=1}^l(\alpha_i-\alpha_i^*)(\alpha_j-\alpha_j^*)<x_i,x_j>\\
-\varepsilon\sum_{i,j=1}^l(\alpha_i+\alpha_i^*)+\sum_{i=1}^l(\alpha_i-\alpha_i^*)
\end{cases}\\
subjct \quad to \quad \sum_{i =1}^l(\alpha_i-\alpha_i^*)=0\quad and \quad \alpha_i,\alpha_i^*\in[0,C]$$ 
又w:

$w=\sum_{i=1}^l(\alpha_i-\alpha_i^*)\phi(x_i)$

做预测：

$f(x)=\sum_{i=1}^l(\alpha_i-\alpha_i^*)<x_i,x>+b$    

计算b：
使用KKT条件：在最优解的点，对偶变量和约束的乘积为0
$$
\alpha_i(\varepsilon+\xi_i-y_i+<w,x_i>+b)=0\\
\alpha_i^*(\varepsilon+\xi_i^*+y_i-<w,x_i>-b)=0\\
(C-\alpha_i)\xi_i=0\\
(C-\alpha_i^*)\xi_i^*=0
$$

1.对应$\alpha_i^{(*)}=C$(不为0)的那
些点在边界之外

2.不对存在一定对偶变量同时为非0情况$\alpha_i\alpha_i^{(*)}=0$

只有在边界之外的数据点，带入如下公式才成立。这些点被称为支持向量，支持向量唯一决定了回归曲线、面。
$$|f(x_i)-y_i|\geq\varepsilon$$

带有松弛变量和核函数的SVR的dual对偶问题：
$$
maximize\begin{cases}
-\frac{1}{2}\sum_{j=1}^l(\alpha_i-\alpha_i^*)(\alpha_j-\alpha_j^*)<x_i,x_j>\\
-\varepsilon\sum_{i,j=1}^l(\alpha_i+\alpha_i^*)+\sum_{i=1}^l(\alpha_i-\alpha_i^*)
\end{cases}\\
subjct \quad to \quad \sum_{i =1}^l(\alpha_i-\alpha_i^*)=0\quad and \quad \alpha_i,\alpha_i^*\in[0,C]$$

$w=\sum_{i=1}^l(\alpha_i-\alpha_i^*)\phi(x_i)$

做预测：

$f(x)=\sum_{i=1}^l(\alpha_i-\alpha_i^*)<x_i,x>+b$   

## SVM Primal v.s. Dual
![SVM Primal v.s. Dual](image\SVMPrimalv.s.Dual.jpg)

## 使用Loss理解SVR
![Loss理解SVR](image\Loss理解SVR.jpg)













