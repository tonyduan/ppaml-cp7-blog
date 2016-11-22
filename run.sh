#!/bin/bash

# arguments: param_file input_dir output_dir log_path

if (( $# != 4 )); then
printf "Usage: %s param_file input_dir output_dir log_path\n" "$0" >&2
exit 1
fi

# load parameter -- n_samples, burn in will be (n_samples - 50)

source $1

# make output directory

mkdir -p "$3"
mkdir -p "data"
mkdir -p "out/Full"
mkdir -p "log"
mkdir -p "data_processed"

# copy input to proper directory for preprocessing

cp -r $2 data/Full

# preprocessing

printf "Preprocessing BLOG code.\n"

python3 preprocessing.py Full > "$4"

cat flu_spread_header.blog \
      flu_spread_model.blog \
      flu_spread_footer.blog > bin/flu_spread_compiled.blog;

printf "Compiling BLOG code to C++.\n"

../swift/swift \
       -i bin/flu_spread_compiled.blog \
       -o bin/flu_spread_compiled.cpp \
       -e GibbsSampler \
       -n $n_samples \
			 --burn-in $(( $n_samples - 50 )) > "$4"

printf "Compiling C++ code.\n"

g++ -I bin -Ofast -std=c++11 \
	bin/flu_spread_compiled.cpp \
	../swift/src/random/*.cpp \
	-o flu_spread_compiled -larmadillo > "$4"

printf "Running inference.\n"

./flu_spread_compiled > out/output_full.txt

printf "Post-processing output.\n"

python3 postprocessing.py Full > "$4"
cp out/Full/CountyWeeklyILI.json "$3"

printf "Output written to "$3"/CountyWeeklyILI.json.\n"

# clean up

rm -rf "out"
rm -rf "data"
rm -rf "log"
rm -rf "data_processed"
rm "flu_spread_footer.blog"
rm "flu_spread_header.blog"
rm "flu_spread_compiled"
