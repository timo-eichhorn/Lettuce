# Lettuce

## Description

This repository contains code for carrying out multi-threaded CPU simulations of 4-dimensional SU(3) gauge theory. The primary motivation behind this code is to test different algorithms aimed at alleviating the problem of topological freezing. Some of the results from simulations performed using this code can be found in the following two papers:

- [[2307.04742]](https://arxiv.org/abs/2307.04742) - *Parallel Tempered Metadynamics: Overcoming potential barriers without surfing or tunneling*
- [[2210.11453]](https://arxiv.org/abs/2210.11453) - *Topology changing update algorithms for SU(3) gauge theory*

Note that **the code is currently undergoing a complete rewrite** (to support simulations with dynamical fermions, GPU offloading, and other), and the main branch may soon be designated as a legacy branch. The current version lacks tests and contains several areas under active development. Still, if you have any questions, feel free to reach out to me.

## Requirements

This project requires [Eigen](https://gitlab.com/libeigen/eigen) and the [PCG PRNG](https://github.com/imneme/pcg-cpp). Both header-only libraries have been included in the repository, so there should be no need to manually install these libraries. Other than that, you need a compiler supporting C++20 (while the current version only rarely uses C++20 features, the upcoming rewrite will heavily rely on concepts).

## Building

Since all dependencies are already included in the repository as header files, you can simply compile the code via:
```
make
```
Optionally, the flag "FIXED_SEED" may be set during compilation to seed the inbuilt PRNG class (used for the update algorithms) with a fixed seed:
```
make FLGS=-DFIXED_SEED
```

## Citing Lettuce

If you use this code, please cite the following:

```
@mastersthesis{EichhornThesis,
  author  = "Eichhorn, Timo",
  title   = "{Slowing Down Critical Slowing Down}",
  school  = "University of Wuppertal",
  year    = "2023"
}
```

## License

This project is licensed under the Mozilla Public License Version 2.0.