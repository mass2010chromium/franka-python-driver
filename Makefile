.PHONY: all
all: python freedrive

python:
	(cd c++; python3 setup.py build)
	(cd c++; sudo python3 setup.py install)

freedrive: bin
	c++ ./c++/franka_freedrive.cpp /usr/local/lib/libfranka.so -I~/libfranka/include -lpthread -o ./bin/freedrive

bin:
	mkdir -p bin

.PHONY: clean
clean:
	rm -f ./bin/franka_driver
	rm -rf c++/build/*
