CUDA_PATH ?= /usr/local/cuda
NVCC      ?= $(CUDA_PATH)/bin/nvcc

ARCH      ?= -gencode arch=compute_61,code=sm_61 \
             -gencode arch=compute_70,code=sm_70 \
             -gencode arch=compute_75,code=sm_75 \
             -gencode arch=compute_80,code=sm_80 \
             -gencode arch=compute_86,code=sm_86

CXXFLAGS  := -O3 -std=c++17 -lineinfo --use_fast_math --ptxas-options=-v
CXXFLAGS_DEBUG := -O0 -g -G -std=c++17 -lineinfo --ptxas-options=-v -Xcompiler -fno-omit-frame-pointer -DDEBUG

INCLUDES  := -Iinclude -Isrc -I/usr/include/eigen3 

CU_SRC = \
  src/main.cu \
  src/utils.cu \
  src/conv_kernels.cu

CPP_SRC = \
  src/cpu_reference.cpp \
  src/var4help.cpp

CU_OBJ  = $(CU_SRC:.cu=.o)
CPP_OBJ = $(CPP_SRC:.cpp=.o)

OBJ = $(CU_OBJ) $(CPP_OBJ)

BIN = gpu_conv
BIN_DEBUG = gpu_conv_debug

all: $(BIN)

%.o: %.cu
	$(NVCC) $(CXXFLAGS) $(ARCH) $(INCLUDES) -c $< -o $@

%.o: %.cpp
	$(NVCC) $(CXXFLAGS) $(ARCH) $(INCLUDES) -c $< -o $@

$(BIN): $(OBJ)
	$(NVCC) $(CXXFLAGS) $(ARCH) $(INCLUDES) -o $@ $^

debug: $(BIN_DEBUG)

$(BIN_DEBUG): $(OBJ)
	$(NVCC) $(CXXFLAGS_DEBUG) $(ARCH) $(INCLUDES) -o $@ $^

clean:
	rm -f $(BIN) $(BIN_DEBUG) $(OBJ) *.ncu-rep *.nsys-rep

.PHONY: all debug clean









# CUDA_PATH ?= /usr/local/cuda
# NVCC      ?= $(CUDA_PATH)/bin/nvcc

# ARCH      ?= -gencode arch=compute_61,code=sm_61 \
#              -gencode arch=compute_70,code=sm_70 \
#              -gencode arch=compute_75,code=sm_75 \
#              -gencode arch=compute_80,code=sm_80 \
#              -gencode arch=compute_86,code=sm_86

# CXXFLAGS  := -O3 -std=c++17 -lineinfo --use_fast_math --ptxas-options=-v
# CXXFLAGS_DEBUG := -O0 -g -G -std=c++17 -lineinfo --ptxas-options=-v -Xcompiler -fno-omit-frame-pointer -DDEBUG
# LDFLAGS   :=
# INCLUDES  := -Iinclude -Isrc -I/usr/include/eigen3 

# SRC = \
#   src/main.cu \
#   src/utils.cu \
#   src/cpu_reference.cpp \
#   src/conv_kernels.cu \
#   src/var4help.cpp

# BIN = gpu_conv
# BIN_DEBUG = gpu_conv_debug

# all: $(BIN)

# $(BIN): $(SRC)
# 	$(NVCC) $(CXXFLAGS) $(ARCH) $(INCLUDES) -o $@ $^

# debug: $(BIN_DEBUG)

# $(BIN_DEBUG): $(SRC)
# 	$(NVCC) $(CXXFLAGS_DEBUG) $(ARCH) $(INCLUDES) -o $@ $^

# clean:
# 	rm -f $(BIN) $(BIN_DEBUG) *.o *.obj *.exp *.lib *.pdb *.ncu-rep *.nsys-rep

# .PHONY: all debug clean