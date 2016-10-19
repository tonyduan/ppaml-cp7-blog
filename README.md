# CP7 Submission for BLOG

University of California, Berkeley

Submission for BLOG by Prof. Stuart Russell's group.

## Instructions

To install BLOG, Python3, and the C++ Armadillo library, run the setup script.

    ./setup.sh

Make sure that the directory hierarchy looks something like:

    /swift
    /ppaml-blog-cp7
      readme.md
      ...

That is, the `swift` directory is one level up from this readme file.

Then run the inference code for the dataset of your choice; pick one of:

    make run_small        # writes to out/Small/CountyWeeklyILI.json
    make run_middle       # writes to out/Middle/CountyWeeklyILI.json
    make run_full         # writes to out/Full/CountyWeeklyILI.json

Total loss (as calculated by our evaluation script) will be printed as well.

## Details

By default, runs the Metropolis-Hastings algorithm.

Our setup script works on Amazon EC2 Ubuntu 14.04.

Requirements are approximately:

- Small dataset: 2 GB RAM. Runs 5 million samples.
- Middle dataset: 8 GB RAM. Runs 50 million samples.
- Full dataset: 80 GB RAM