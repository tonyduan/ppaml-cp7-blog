default: compile

processors:
	jupyter nbconvert --to=script preprocessing.ipynb
	jupyter nbconvert --to=script postprocessing.ipynb

run_small:
	python3 preprocessing.py Small
	make compile
	./flu_spread_compiled > out/output_small.txt
	python3 postprocessing.py Small

run_middle:
	python3 preprocessing.py Middle
	make compile
	./flu_spread_compiled > out/output_middle.txt
	python3 postprocessing.py Middle

run_full:
	python3 preprocessing.py Full
	make compile
	./flu_spread_compiled > out/output_full.txt
	python3 postprocessing.py Full

compile:
	cat flu_spread_header.blog \
        flu_spread_model.blog \
        flu_spread_footer.blog > bin/flu_spread_compiled.blog;

	../swift/swift \
         -i bin/flu_spread_compiled.blog \
         -o bin/flu_spread_compiled.cpp \
         -e GibbsSampler \
         -n 70000000 \
         --burn-in 69999995

	sed -i 's/accu(__fixed_county_map\[r\]\*/dot(__fixed_county_map.row(r),/g' bin/flu_spread_compiled.cpp

	g++ -I bin -Ofast -std=c++11 \
		bin/flu_spread_compiled.cpp \
		../swift/src/random/*.cpp \
		-o flu_spread_compiled -larmadillo;
