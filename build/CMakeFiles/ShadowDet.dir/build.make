# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/davide/Scrivania/ShadowDet

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/davide/Scrivania/ShadowDet/build

# Include any dependencies generated for this target.
include CMakeFiles/ShadowDet.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/ShadowDet.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ShadowDet.dir/flags.make

CMakeFiles/ShadowDet.dir/src/Main.cpp.o: CMakeFiles/ShadowDet.dir/flags.make
CMakeFiles/ShadowDet.dir/src/Main.cpp.o: ../src/Main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/davide/Scrivania/ShadowDet/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ShadowDet.dir/src/Main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ShadowDet.dir/src/Main.cpp.o -c /home/davide/Scrivania/ShadowDet/src/Main.cpp

CMakeFiles/ShadowDet.dir/src/Main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ShadowDet.dir/src/Main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/davide/Scrivania/ShadowDet/src/Main.cpp > CMakeFiles/ShadowDet.dir/src/Main.cpp.i

CMakeFiles/ShadowDet.dir/src/Main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ShadowDet.dir/src/Main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/davide/Scrivania/ShadowDet/src/Main.cpp -o CMakeFiles/ShadowDet.dir/src/Main.cpp.s

CMakeFiles/ShadowDet.dir/src/Main.cpp.o.requires:

.PHONY : CMakeFiles/ShadowDet.dir/src/Main.cpp.o.requires

CMakeFiles/ShadowDet.dir/src/Main.cpp.o.provides: CMakeFiles/ShadowDet.dir/src/Main.cpp.o.requires
	$(MAKE) -f CMakeFiles/ShadowDet.dir/build.make CMakeFiles/ShadowDet.dir/src/Main.cpp.o.provides.build
.PHONY : CMakeFiles/ShadowDet.dir/src/Main.cpp.o.provides

CMakeFiles/ShadowDet.dir/src/Main.cpp.o.provides.build: CMakeFiles/ShadowDet.dir/src/Main.cpp.o


# Object files for target ShadowDet
ShadowDet_OBJECTS = \
"CMakeFiles/ShadowDet.dir/src/Main.cpp.o"

# External object files for target ShadowDet
ShadowDet_EXTERNAL_OBJECTS =

ShadowDet: CMakeFiles/ShadowDet.dir/src/Main.cpp.o
ShadowDet: CMakeFiles/ShadowDet.dir/build.make
ShadowDet: /usr/local/lib/libopencv_stitching.so.3.4.1
ShadowDet: /usr/local/lib/libopencv_superres.so.3.4.1
ShadowDet: /usr/local/lib/libopencv_videostab.so.3.4.1
ShadowDet: /usr/local/lib/libopencv_aruco.so.3.4.1
ShadowDet: /usr/local/lib/libopencv_bgsegm.so.3.4.1
ShadowDet: /usr/local/lib/libopencv_bioinspired.so.3.4.1
ShadowDet: /usr/local/lib/libopencv_ccalib.so.3.4.1
ShadowDet: /usr/local/lib/libopencv_dnn_objdetect.so.3.4.1
ShadowDet: /usr/local/lib/libopencv_dpm.so.3.4.1
ShadowDet: /usr/local/lib/libopencv_face.so.3.4.1
ShadowDet: /usr/local/lib/libopencv_freetype.so.3.4.1
ShadowDet: /usr/local/lib/libopencv_fuzzy.so.3.4.1
ShadowDet: /usr/local/lib/libopencv_hfs.so.3.4.1
ShadowDet: /usr/local/lib/libopencv_img_hash.so.3.4.1
ShadowDet: /usr/local/lib/libopencv_line_descriptor.so.3.4.1
ShadowDet: /usr/local/lib/libopencv_optflow.so.3.4.1
ShadowDet: /usr/local/lib/libopencv_reg.so.3.4.1
ShadowDet: /usr/local/lib/libopencv_rgbd.so.3.4.1
ShadowDet: /usr/local/lib/libopencv_saliency.so.3.4.1
ShadowDet: /usr/local/lib/libopencv_stereo.so.3.4.1
ShadowDet: /usr/local/lib/libopencv_structured_light.so.3.4.1
ShadowDet: /usr/local/lib/libopencv_surface_matching.so.3.4.1
ShadowDet: /usr/local/lib/libopencv_tracking.so.3.4.1
ShadowDet: /usr/local/lib/libopencv_xfeatures2d.so.3.4.1
ShadowDet: /usr/local/lib/libopencv_ximgproc.so.3.4.1
ShadowDet: /usr/local/lib/libopencv_xobjdetect.so.3.4.1
ShadowDet: /usr/local/lib/libopencv_xphoto.so.3.4.1
ShadowDet: /usr/local/lib/libopencv_shape.so.3.4.1
ShadowDet: /usr/local/lib/libopencv_photo.so.3.4.1
ShadowDet: /usr/local/lib/libopencv_datasets.so.3.4.1
ShadowDet: /usr/local/lib/libopencv_plot.so.3.4.1
ShadowDet: /usr/local/lib/libopencv_text.so.3.4.1
ShadowDet: /usr/local/lib/libopencv_dnn.so.3.4.1
ShadowDet: /usr/local/lib/libopencv_ml.so.3.4.1
ShadowDet: /usr/local/lib/libopencv_video.so.3.4.1
ShadowDet: /usr/local/lib/libopencv_calib3d.so.3.4.1
ShadowDet: /usr/local/lib/libopencv_features2d.so.3.4.1
ShadowDet: /usr/local/lib/libopencv_highgui.so.3.4.1
ShadowDet: /usr/local/lib/libopencv_videoio.so.3.4.1
ShadowDet: /usr/local/lib/libopencv_phase_unwrapping.so.3.4.1
ShadowDet: /usr/local/lib/libopencv_flann.so.3.4.1
ShadowDet: /usr/local/lib/libopencv_imgcodecs.so.3.4.1
ShadowDet: /usr/local/lib/libopencv_objdetect.so.3.4.1
ShadowDet: /usr/local/lib/libopencv_imgproc.so.3.4.1
ShadowDet: /usr/local/lib/libopencv_core.so.3.4.1
ShadowDet: CMakeFiles/ShadowDet.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/davide/Scrivania/ShadowDet/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ShadowDet"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ShadowDet.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ShadowDet.dir/build: ShadowDet

.PHONY : CMakeFiles/ShadowDet.dir/build

CMakeFiles/ShadowDet.dir/requires: CMakeFiles/ShadowDet.dir/src/Main.cpp.o.requires

.PHONY : CMakeFiles/ShadowDet.dir/requires

CMakeFiles/ShadowDet.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ShadowDet.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ShadowDet.dir/clean

CMakeFiles/ShadowDet.dir/depend:
	cd /home/davide/Scrivania/ShadowDet/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/davide/Scrivania/ShadowDet /home/davide/Scrivania/ShadowDet /home/davide/Scrivania/ShadowDet/build /home/davide/Scrivania/ShadowDet/build /home/davide/Scrivania/ShadowDet/build/CMakeFiles/ShadowDet.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ShadowDet.dir/depend
