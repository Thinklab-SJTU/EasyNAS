# EasyML

**We will update the codes in our new [repo](https://github.com/Thinklab-SJTU/Opt4Sci)**

EasyML is an AutoML tool based on PyTorch platform, supporting various AutoML tasks including Hyper-parameter Optimization (HPO) and Neural Architecture Search (NAS). EasyML is developed based on the following three aspects of AutoML:

- **Flexible Definition of Search Spaces** that can suit various tasks and applications, supporting continuous and discrete spaces and nested spaces.
- **Multiple Search Algorithms:** that can be freely chosen by customers for specific applications.
- **Scalable Evaluation Mode:** that can easily utilize or integrate evaluation codes.


## Introduction to EasyML

AutoML aims to automatically search optimal configurations of machine learning algorithms for specific application tasks, consisting of three components: search space, search algorithm, and evaluation strategy. EasyML aims to ease AutoML applications and free the developers from the burden of pipeline creation. 

Customers can:
- easily define search spaces for arbitrary tasks based on YAML configuration files;
- easily choose a search algorithm or integrate new algorithms for arbitrary scenarios;
- easily evaluate candidates in a parallel way, so that the rewards can be automatically returned to the search algorithm.

**Supported Search Space**
- [x] Discrete Space
- [x] Continuous Space
- [x] Nested Space

**Supported Search Algorithm**
- [x] Random Search
  - [x] HyperBand
- [ ] Heuristic Algorithm
  - [x] Evolutionary Algorithm, NSGA-II
  - [x] Particle Swarm Optimization
  - [ ] Ant Colony Optimization
- [x] Gradient-based Search Algorithm
  - [ ] First-order-optimization-based Search Algorithm
  - [x] Zero-order-optimization-based Search Algorithm
- [ ] Bayesian Optimization

**Supported Evaluation Strategy**
- [ ] Benchmark Indexing
  - [x] PyCUTEst Benchmark
  - [ ] NAS-Bench 
- [x] Neural Network Traning
- [ ] Evaluation of Edge Devices
  - [x] RK3588
  - [ ] RZ 
- [x] Optimization Solver
  - [x] SCIP


## Get Started
### Define a Search Space
The configuration settings can be defined in the YAML style for arbitrary tasks. Customers can set search spaces for any variables (hyper-parameters) in the YAML configuration file `conf.yaml`:
```python
search_space: &search_space 
  config: &config
    a: !search_space 0.1:0.3 # Continuous Space  
    b: !search_space 0.2:0.8:0.1 # Discrete Space
    c: !search_space [1,2,3,4] # Discrete Space
    d: 20 # Fixed Value
    e: 'abc' # Fixed Value
    f: !search_space # Nested Space
      space:
        f1: f1
        f2: !search_space 
          space: [f21, f22, f23]
          label: f2
        f3: !search_space 
          space: [f31, f32, f33]
          label: f2 # 'f3' has the same sampling property as 'f2'
```
More Details can be found in `cfg/search/test.yaml`

### Choose a Search Algorithm
Customers can choose a search algorithm and set the configurations in the same YAML configuration file `conf.yaml`:
```python
searcher: &searcher
  submodule_name: src.searcher.NSGA2
  args:
    num_epoch: 5
    num_survive: 10
    num_crossover: 4
    num_mutation: 4
    num_population: 10
    prob_mutation: 0.1
```

### Confirm Evaluation Strategies
Customers can set the evaluation strategy in the same YAML configuration file `conf.yaml`:
```python
eval_engine1: &eval_engine
    submodule_name: <CustomEvalEngine1>
    args:
      config: null # To Be Searched. It will be fed by each candidate automatically by our platform during the search procedure
    run_args:
      max_iter: 500
  
eval_engine2: &eval_engine2
    submodule_name: <CustomEvalEngine1>
    args:
      config: null # To Be Searched. It will be fed by each candidate automatically by our platform during the search procedure
    run_args:
      max_iter: 1000

contractor: &contractor
  submodule\_name: src.evaluater.Contractor
  args:
    eval_engines: [*eval_engine1, *eval_engine2]
    num_workers: 3 # parallel workers to evaluate candidates
```

### Begin the Search 
Set the search engine in the same YAML configuration file `conf.yaml`:
```python
search_engine:
  submodule_name: engines.SearchEngine
  args:
    search_space: *search_space
    searcher: *searcher
    contractor: *contractor
```
Then run the script: `python app/run_engine --cfg conf.yaml`

### Examples
- [x] NAS on classification task
  - [x] DARTS: `cfg/cifar10_darts.yaml`
  - [x] ZARTS: `cfg/cifar10_zarts.yaml`
- [x] NAS on detection task
  - [x] EAutoDet: `cfg/coco_EAutoDet.yaml`
  - [x] EAutoDet + EA: `cfg/EAutoDet/`
- [ ] Benchmark
  - [x] optimization on PyCUTEst: `cfg/benchmark/pycutest_*.yaml`
  - [ ] NAS-Bench-201
- [x] HPO for Optimization Solver
  - [x] HPO for SCIP pre-resolver: `cfg/mip/HB_MIK_scip_presolve.yaml`
