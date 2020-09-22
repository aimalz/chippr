# chippr

Cosmological Hierarchical Inference with Probabilistic Photometric Redshifts

## Motivation

This repository is the home of `chippr`, a Python package for estimating quantities of cosmological interest from surveys of photometric redshift posterior probability distributions.  
It is a refactoring of previous [work](https://github.com/aimalz/prob-z) on using probabilistic photometric redshifts to infer the redshift distribution.

## Examples

You can browse the demo notebook here:

* [Basic  Demo for Python 2.7](http://htmlpreview.github.io/?https://github.com/aimalz/chippr/blob/master/docs/notebooks/demo2.html)

## Documentation

Documentation can be found on [ReadTheDocs](http://chippr.readthedocs.io/en/master/).  
The draft of the paper documenting the details of the method can be found [here](https://github.com/aimalz/chippr/blob/master/research/paper/draft.pdf).

## Disclaimer

As can be seen from the git history and Python version, this code is stale and should be understood to be a prototype, originally scoped out for applicability to SDSS DR10-era data of low dimensionality.
As a disclaimer, it will need a major upgrade for flexibility and computational scaling before it can run on data sets like those of modern and future galaxy surveys.

## People

* [Alex I. Malz](https://github.com/aimalz) (German Centre for Cosmological Lensing)

## License, Contributing etc

The code in this repo is available for re-use under the MIT license, which means that you can do whatever you like with it, just don't blame me.
If you end up using any of the code or ideas you find here in your academic research, please cite me as `Malz & Hogg, 2020 \footnote{[arXiv:2007.12178](https://arxiv.org/abs/2007.12178)\texttt{https://github.com/aimalz/chippr}}`.
If you are interested in this project, please do drop me a line via the hyperlinked contact name above, or by [writing me an issue](https://github.com/aimalz/chippr/issues/new).
To get started contributing to the `chippr` project, just fork the repo -- pull requests are always welcome!
