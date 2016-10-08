default: all

all:
	cat flu_spread_model.blog \
        flu_spread_obs.blog \
        flu_spread_queries.blog > bin/flu_spread_compiled.blog;

	/Users/tony/Projects/swift/swift \
         -i bin/flu_spread_compiled.blog \
         -o bin/flu_spread_compiled.cpp \
         -e MHSampler

    # -o bin/flu_spread_compiled_liuwest.cpp \
    # awk '/perturb/ {$$0="//"$$0}1' bin/flu_spread_compiled_liuwest.cpp \
        > bin/flu_spread_tmp.cpp;
    # awk '/int _cur_loop/ {$$0="//"$$0}1' bin/flu_spread_tmp.cpp \
		> bin/flu_spread_compiled.cpp;
    # rm bin/flu_spread_tmp.cpp

	g++ -I bin -Ofast -std=c++11 \
		bin/flu_spread_compiled.cpp \
		/Users/tony/Projects/swift/src/random/*.cpp \
		-o flu_spread_compiled -larmadillo;

run:
	./flu_spread_compiled > out/output_small.txt
