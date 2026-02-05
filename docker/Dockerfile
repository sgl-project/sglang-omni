FROM lmsysorg/sglang:dev AS base


# install UCX
RUN git clone https://github.com/openucx/ucx.git \
    && cd ucx \
    && git checkout v1.20.x \
    && ./autogen.sh \
    && ./contrib/configure-release-mt  \
    --enable-shared                    \
    --disable-static                   \
    --disable-doxygen-doc              \
    --enable-optimizations             \
    --enable-cma                       \
    --enable-devel-headers             \
    --with-cuda=/usr/local/cuda        \
    --with-verbs                       \
    --with-dm                          \
    --prefix=/usr/local                \
    && make -j \
    && make -j install-strip \
    && ldconfig

