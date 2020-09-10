---
title: "Test"
categories: 
  - Blogging
last_modified_at: 2020-09-10
tags:
  - Test
use_math: true
toc: true
---
#### Transformer-layer distillation

* Loss for attention mechanism

  $$l_{attn} = \frac 1 h sum^h_{i = 1} \text{MSE}(\textbf A_i^S, \textbf A_i^T )$$
  $$
  \lim_{x\to 0}{\frac{e^x-1}{2x}}
  \overset{\left[\frac{0}{0}\right]}{\underset{\mathrm{H}}{=}}
  \lim_{x\to 0}{\frac{e^x}{2}}={\frac{1}{2}}
  $$

* 특이한 점은 softmax를 타기 전의 matrix를 loss function의 input으로 넣는 것인데, 이게 수렴이 더 빠르게 되었다고 한다.
