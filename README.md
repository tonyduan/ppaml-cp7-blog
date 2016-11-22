# CP7 Submission for BLOG

University of California, Berkeley

Submission for BLOG by Prof. Stuart Russell's group.

## Instructions

To install BLOG, Python3, and the C++ Armadillo library, run the setup script.

    ./setup.sh

Make sure that the directory hierarchy looks like:

    /swift
    /ppaml-blog-cp7
      readme.md
      ...

That is, the `swift` repository is one level up from this readme file.

Then run the inference code. Example usage:

    ./run.sh config/solution-smoketest input/Small results/Small results/Small/log.txt

## Solutions

We ran our code for the Small and Middle datasets and the corresponding solutions can be found in the `solns` folder.

## Details

Using our config, runs the Metropolis-Hastings algorithm.

Our setup script works on Amazon EC2 Ubuntu 14.04.

Configuration is:

- Small dataset: 82 counties -- 16896 random variables, 10M samples
- Middle dataset: 277 counties -- 57066 random variables, 40M samples
- Full dataset: 3059 counties -- 630158 random variables, 100M samples

Requirements are approximately:

- Small dataset: 2 GB RAM.
- Middle dataset: 4 GB RAM.
- Full dataset: 8 GB RAM.

## Miscellaneous Notes

- the `makefile` submitted was used for internal testing purposes.
- compiled BLOG code will be written to the `bin` folder.
- due to the way our preprocessing code works, need to ensure the input folder is not `data/Full`.