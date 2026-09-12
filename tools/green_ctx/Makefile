CUDA_HOME ?= /usr/local/cuda
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra
LIB := libgreen_ctx_bootstrap.so

$(LIB): green_ctx_bootstrap.cpp
	$(CXX) $(CXXFLAGS) -shared -fPIC -I$(CUDA_HOME)/include -o $@ $< -lcuda -ldl -lpthread

clean:
	rm -f $(LIB)

.PHONY: clean
