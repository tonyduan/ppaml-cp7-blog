default: all

all:
	cat flu_spread_model.blog flu_spread_obs.blog flu_spread_queries.blog > \
		bin/flu_spread_compiled.blog;
	/Users/tony/Projects/swift/swift \
		-i bin/flu_spread_compiled.blog \
		-o bin/flu_spread_compiled_liuwest.cpp \
		-e ParticleFilter --particle 10000;
	awk '/perturb/ {$$0="//"$$0}1' bin/flu_spread_compiled_liuwest.cpp \
        > bin/flu_spread_compiled.cpp;
	g++ -I bin -Ofast -std=c++11 \
		bin/flu_spread_compiled.cpp \
		/Users/tony/Projects/swift/src/random/*.cpp \
		-o flu_spread_compiled -larmadillo;
