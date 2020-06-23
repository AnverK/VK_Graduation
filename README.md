# VK_Graduation

This repository contains sources for my bachelor's thesis in [ITMO University](https://en.itmo.ru/) (in international laboratory [Computer technologies](http://ctlab.ifmo.ru/en/)). I write it in collaboration with the CoreML team in [VK](vk.com) which is one of the most popular social networks in Eastern Europe. My thesis is dedicated to causal inference (CI) between different metrics of social networks which are usually obtained by A/B testings. So in this repository I published different approaches for these purposes.

In the root package [src](https://github.com/AnverK/VK_Graduation/tree/master/src) you can find some basic approaches: [proxy-metric](https://github.com/AnverK/VK_Graduation/blob/master/src/MetricLinearRegression.py), [instrumental variable](https://github.com/AnverK/VK_Graduation/blob/master/src/InstrumentalVariable.py) with l0-reguralization. Also there is one more complicated method with random experiments in this [notebook](https://github.com/AnverK/VK_Graduation/blob/master/src/MetricStability.ipynb). It is intended for choosing metrics with the most stable sign in a lot of random experiments. It can help to overcome difficulties caused by bias of usual regression.

Let's examine [causality](https://github.com/AnverK/VK_Graduation/tree/master/src/causality) package. I've started with examining of some [greedy methods](https://github.com/AnverK/VK_Graduation/tree/master/src/causality/greedyBuilder). Althouhg optimal tree can be built in polynomial time, optimal graph inference is NP-hard problem. That's why these algorithms are time-consuming. However, there is important idea of maximizing likelihood of graphs, and its usage you can find below.

The main part of my thesis is in package [pc](https://github.com/AnverK/VK_Graduation/tree/master/src/causality/pc). There I've started with classic algorithms of causality: PC and FCI. I used an excellent library: [pcalg](http://pcalg.r-forge.r-project.org/). Unfortunately, I didn't want to migrate all the project to R from Python. Instead I used [rpy2](https://pypi.org/project/rpy2/) to call pcalg from R. So you can find class for usage PC, FCI and FCI+ algorithms [here](https://github.com/AnverK/VK_Graduation/blob/master/src/causality/pc/CausalGraphBuilder.py) and explained [examples](https://github.com/AnverK/VK_Graduation/blob/master/src/causality/pc/UsageExample.py) of usage. I also provided parsing of logs which are generated from pcalg in [LogParser](https://github.com/AnverK/VK_Graduation/blob/master/src/causality/pc/LogParser.py).

For FCI algorithms output is a bit tricky. It's not a classic oriented graph, but _inducing path graph_. More than that, as causality algorithms are only able to infer equivalence class, FCI output is an equivalence class of inducing path graph or __PAG__ (partial ancestral graph). Because of different semantic I've written [pag](https://github.com/AnverK/VK_Graduation/tree/master/src/causality/pc/pag) package to work with them.

All causality algorithms rely on the independence tests. I've tried to examine some of them in package [independence](https://github.com/AnverK/VK_Graduation/tree/master/src/causality/pc/independence): [FCIT](https://arxiv.org/pdf/1804.02747.pdf) and [KCIT](https://arxiv.org/ftp/arxiv/papers/1202/1202.3775.pdf). Unfortunately, they are not suit for my purposes as they depend on number of samples. More than that, pcalg library has natively implemented gaussian conditional independence test based on partial correlation. You can find my implementation on Python [here](https://github.com/AnverK/VK_Graduation/blob/master/src/causality/pc/independence/GaussIndependenceTest.py). Of course, I didn't use it when calling pcalg, but I used it for my new method of edge orientation which I will describe below.

Also I've tried to reproduce idea of many random experiments. Previously I used it with linear regression, now with graphs. I supposed that the most stable edges could be determined via these experiments. Specifically, I considered edges between short metrics and one chosen long metric (for each long metric). I can't say anything good or bad about this idea, as it's hard to evaluate it somehow. About difficulties of comparison you can read below.

As I said previously, I've come up with a new method of edge orientation. Basically, it was caused by enourmous amount of edges in both PC and FCI algorithms. I've recognised that the most of them were oriented as _colliders_(or _v-structures_). It's a structure like __a -> b <- c__. In classic algorithms these edges are set if b is not in a separating set of a and c(S_ac). In my implementation I allow this to vertex __b__, until it doesn't decrease probability of conditional indepence of __a__ and __c__. The code is [here](https://github.com/AnverK/VK_Graduation/blob/master/src/causality/pc/EdgeOrientation.py).

Comparing of algorithms in causality is an extremely non-trivial task. There is a great [paper](https://arxiv.org/pdf/1910.05387.pdf) about it. Usually, researcher has a choise: test approaches unconvincingly on the real data (visual inspection, usage of prior knowledge) or test them convincingly on the (semi-)synthetic data. As one can guess, both approaches are questionable. So in package [methodsVerification](https://github.com/AnverK/VK_Graduation/tree/master/src/causality/methodsVerification) I've tried to examine both approaches. I estimated likelihood of graphs which were inferred from real data, and I estimated classification metrics on semisynthetic data (it is needed for obtaining ground truth). I also tried to use explicit A\B-testing and prior knowledge, but for some reasons it's very laborious approach, and I can't affect on it right now.

I believe that ASAP (before the end of June 2020) I will publish an article on Medium about causal inference. There I will describe most of difficulties which you can face with during working with causal inference. Also there I will explain a lot of basic things about causal inference in general with tons of examples! 

Stay tuned!