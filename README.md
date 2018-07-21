# lipo-python
Implementing Global optimization of Lipschitz functions

## Scripts

The scripts that work (at least at time of writing) can be run with the following commands (from the root directory of the project).

This one runs the sequential optimizers (the `--synthetic` flag indicates that we 
don't want to optimize the real world cros validation based functions):

```
python src/optimize.py results/syn-test --synthetic
```

This one produces parts of Figure 5 from the original paper:

```
python src/figure_five.py results/syn-test.pkl results/syn-test
```

These commands are generating results for the __synthetic__ objective functions. The functionality to generate results corresponding to the 'real world' objective functions is currently being developed.

## Using Docker

Currently the Docker container is designed simply to allow execution of the scripts on a host that doesn't
have dependencies (eg, python or certain packages) available locally.

The container can be used interactively at this point by cloning this repo, building the image from the
Dockerfile, and running it interactively:

```
git clone https://github.com/UBC-CS/lipo-python.git
cd lipo-python
docker build . -t lipo-python
docker run -v <path-to-cloned-repo>/:/home/ -it lipo-python
```
