# CP7 Submission for BLOG

University of California, Berkeley
Submission for BLOG by Prof. Stuart Russell's group.

## Instructions

To install BLOG, Python3, and the C++ Armadillo library, run the setup script.

    ./setup.sh

Make sure that the directory hierarchy looks something like:

    /swift
    /ppaml-blog-cp7
      /bin
      /out
      /data

That is, the `swift` directory is one level up from this readme file.

Then run the inference code for the dataset of your choice; pick one of:

    make run_small
    make run_middle
    make run_full

Output will be in corresponding file:

    out/Small/CountyWeeklyILI.json
    out/Middle/CountyWeeklyILI.json
    out/Full/CountyWeeklyILI.json

Total loss (as calculated by my evaluation script) will be printed as well.

## Details

By default, runs the Metropolis-Hastings algorithm with 10 million samples.

Our setup script works on Amazon EC2 Ubuntu 14.04.
