# Verification Guided Control Learning with [POLAR](https://github.com/ChaoHuang2018/POLAR)


## Installation

#### System Requirements
Ubuntu 18.04, MATLAB 2016a or later

#### Up-to-date Installation for POLAR
- Install dependencies through apt-get install
```
sudo apt-get install m4 libgmp3-dev libmpfr-dev libmpfr-doc libgsl-dev gsl-bin bison flex gnuplot-x11 libglpk-dev gcc-8 g++-8 libopenmpi-dev
```
#### Addtionally, install the cnpy library
- See install details in [cnpy](https://github.com/rogersce/cnpy)

#### Compile POLAR

```

./compile.sh # under the root directory ./
make # under the root directory to compile the running examples. 

```


## Running examples

### Oscillator
```python
python verifylearn0.py 0(or other flag number you would like, could run simultaneously by different flag numbers)
```

### 3D numerical example
```python
python verifylearn5.py 0(or other flag number you would like, could run simultaneously by different flag numbers)
```

### Tora example
```python
python verifylearn6.py 0(or other flag number you would like, could run simultaneously by different flag numbers)
```

## Contributors
[Yixuan Wang](https://wangyixu14.github.io/), [Chao Huang](https://chaohuang2018.github.io/main/).

## References
[1] Y.Wang, C.Huang, Z.Wang, Z.Wang and Q.Zhu.
[Verification in the Loop: Correct-by-Construction Control Learning with Reach-avoid Guarantees](https://arxiv.org/abs/2106.03245)

[2] C.Huang, J.Fan, W.Li, X.Chen, and Q.Zhu.
[POLAR: A Polynomial Arithmetic Framework for Verifying Neural-Network Controlled Systems]()




