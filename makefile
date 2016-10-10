default: compile

preprocess_small:
	python preprocessing.py Small

preprocess_middle:
	python preprocessing.py Middle

preprocess_full:
	python preprocessing.py Full

compile:
	cat flu_spread_model.blog \
        flu_spread_region_rate.blog \
        flu_spread_obs.blog \
        flu_spread_queries.blog > bin/flu_spread_compiled.blog;

	/Users/tony/Projects/swift/swift \
         -i bin/flu_spread_compiled.blog \
         -o bin/flu_spread_compiled.cpp \
         -e MHSampler \
         -n 1000000

	g++ -I bin -Ofast -std=c++11 \
		bin/flu_spread_compiled.cpp \
		/Users/tony/Projects/swift/src/random/*.cpp \
		-o flu_spread_compiled -larmadillo;

run:
	./flu_spread_compiled > out/output.txt