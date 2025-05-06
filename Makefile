MAIN_NAME=$(notdir $(shell pwd))

# CUDA and OpenCV paths
CUDA_INCLUDE=/opt/cuda/targets/x86_64-linux/include
CUDA_LIB=/opt/cuda/targets/x86_64-linux/lib
OPENCV_INCLUDE=/usr/include/opencv4
OPENCV_LIB=/usr/lib/x86_64-linux-gnu

CPPFLAGS+=-g -I$(OPENCV_INCLUDE) -I$(CUDA_INCLUDE)
LDLIBS=-lcudart -lcuda

# detect opencv lib
OPENCVLIB=$(shell pkgconf --list-package-names | grep opencv)

ifeq ($(OPENCVLIB),)
all:
	@echo OpenCV lib not found!
	@exit 1
else
CPPFLAGS+=$(shell pkgconf --cflags $(OPENCVLIB))
LDFLAGS+=$(shell pkgconf --libs-only-L $(OPENCVLIB))
LDLIBS+=$(shell pkgconf --libs-only-l $(OPENCVLIB))

all: $(MAIN_NAME)

endif

CUDA_OBJS=$(addsuffix .o, $(basename $(wildcard *.cu)))

%.o: %.cu $(wildcard *.h)
	nvcc -c $(CPPFLAGS) -I$(CUDA_INCLUDE) $<

$(MAIN_NAME): $(wildcard *.cpp) $(CUDA_OBJS) $(wildcard *.h)
	g++ $(CPPFLAGS) $(LDFLAGS) $(filter %.cpp %.o, $^) $(LDLIBS) -L$(CUDA_LIB) -o $@

clean:
	rm -f *.o $(MAIN_NAME)
