# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Default target executed when no arguments are given to make.
default_target: all
.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/alunos/tei/2024/tei26703/AVALIACAO_CED/Avaliacao_CED

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/alunos/tei/2024/tei26703/AVALIACAO_CED/Avaliacao_CED

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "No interactive CMake dialog available..."
	/usr/bin/cmake -E echo No\ interactive\ CMake\ dialog\ available.
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake --regenerate-during-build -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/alunos/tei/2024/tei26703/AVALIACAO_CED/Avaliacao_CED/CMakeFiles /home/alunos/tei/2024/tei26703/AVALIACAO_CED/Avaliacao_CED//CMakeFiles/progress.marks
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/alunos/tei/2024/tei26703/AVALIACAO_CED/Avaliacao_CED/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean
.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named sequential_imp

# Build rule for target.
sequential_imp: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 sequential_imp
.PHONY : sequential_imp

# fast build rule for target.
sequential_imp/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/sequential_imp.dir/build.make CMakeFiles/sequential_imp.dir/build
.PHONY : sequential_imp/fast

#=============================================================================
# Target rules for targets named OpenMp

# Build rule for target.
OpenMp: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 OpenMp
.PHONY : OpenMp

# fast build rule for target.
OpenMp/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/OpenMp.dir/build.make CMakeFiles/OpenMp.dir/build
.PHONY : OpenMp/fast

#=============================================================================
# Target rules for targets named GPU_openmp

# Build rule for target.
GPU_openmp: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 GPU_openmp
.PHONY : GPU_openmp

# fast build rule for target.
GPU_openmp/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/GPU_openmp.dir/build.make CMakeFiles/GPU_openmp.dir/build
.PHONY : GPU_openmp/fast

GPU_openmp/main.o: GPU_openmp/main.cpp.o
.PHONY : GPU_openmp/main.o

# target to build an object file
GPU_openmp/main.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/GPU_openmp.dir/build.make CMakeFiles/GPU_openmp.dir/GPU_openmp/main.cpp.o
.PHONY : GPU_openmp/main.cpp.o

GPU_openmp/main.i: GPU_openmp/main.cpp.i
.PHONY : GPU_openmp/main.i

# target to preprocess a source file
GPU_openmp/main.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/GPU_openmp.dir/build.make CMakeFiles/GPU_openmp.dir/GPU_openmp/main.cpp.i
.PHONY : GPU_openmp/main.cpp.i

GPU_openmp/main.s: GPU_openmp/main.cpp.s
.PHONY : GPU_openmp/main.s

# target to generate assembly for a file
GPU_openmp/main.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/GPU_openmp.dir/build.make CMakeFiles/GPU_openmp.dir/GPU_openmp/main.cpp.s
.PHONY : GPU_openmp/main.cpp.s

GPU_openmp/nsc_solver.o: GPU_openmp/nsc_solver.cpp.o
.PHONY : GPU_openmp/nsc_solver.o

# target to build an object file
GPU_openmp/nsc_solver.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/GPU_openmp.dir/build.make CMakeFiles/GPU_openmp.dir/GPU_openmp/nsc_solver.cpp.o
.PHONY : GPU_openmp/nsc_solver.cpp.o

GPU_openmp/nsc_solver.i: GPU_openmp/nsc_solver.cpp.i
.PHONY : GPU_openmp/nsc_solver.i

# target to preprocess a source file
GPU_openmp/nsc_solver.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/GPU_openmp.dir/build.make CMakeFiles/GPU_openmp.dir/GPU_openmp/nsc_solver.cpp.i
.PHONY : GPU_openmp/nsc_solver.cpp.i

GPU_openmp/nsc_solver.s: GPU_openmp/nsc_solver.cpp.s
.PHONY : GPU_openmp/nsc_solver.s

# target to generate assembly for a file
GPU_openmp/nsc_solver.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/GPU_openmp.dir/build.make CMakeFiles/GPU_openmp.dir/GPU_openmp/nsc_solver.cpp.s
.PHONY : GPU_openmp/nsc_solver.cpp.s

OpenMp/main.o: OpenMp/main.cpp.o
.PHONY : OpenMp/main.o

# target to build an object file
OpenMp/main.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/OpenMp.dir/build.make CMakeFiles/OpenMp.dir/OpenMp/main.cpp.o
.PHONY : OpenMp/main.cpp.o

OpenMp/main.i: OpenMp/main.cpp.i
.PHONY : OpenMp/main.i

# target to preprocess a source file
OpenMp/main.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/OpenMp.dir/build.make CMakeFiles/OpenMp.dir/OpenMp/main.cpp.i
.PHONY : OpenMp/main.cpp.i

OpenMp/main.s: OpenMp/main.cpp.s
.PHONY : OpenMp/main.s

# target to generate assembly for a file
OpenMp/main.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/OpenMp.dir/build.make CMakeFiles/OpenMp.dir/OpenMp/main.cpp.s
.PHONY : OpenMp/main.cpp.s

OpenMp/nsc_solver.o: OpenMp/nsc_solver.cpp.o
.PHONY : OpenMp/nsc_solver.o

# target to build an object file
OpenMp/nsc_solver.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/OpenMp.dir/build.make CMakeFiles/OpenMp.dir/OpenMp/nsc_solver.cpp.o
.PHONY : OpenMp/nsc_solver.cpp.o

OpenMp/nsc_solver.i: OpenMp/nsc_solver.cpp.i
.PHONY : OpenMp/nsc_solver.i

# target to preprocess a source file
OpenMp/nsc_solver.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/OpenMp.dir/build.make CMakeFiles/OpenMp.dir/OpenMp/nsc_solver.cpp.i
.PHONY : OpenMp/nsc_solver.cpp.i

OpenMp/nsc_solver.s: OpenMp/nsc_solver.cpp.s
.PHONY : OpenMp/nsc_solver.s

# target to generate assembly for a file
OpenMp/nsc_solver.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/OpenMp.dir/build.make CMakeFiles/OpenMp.dir/OpenMp/nsc_solver.cpp.s
.PHONY : OpenMp/nsc_solver.cpp.s

sequential_imp/main.o: sequential_imp/main.cpp.o
.PHONY : sequential_imp/main.o

# target to build an object file
sequential_imp/main.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/sequential_imp.dir/build.make CMakeFiles/sequential_imp.dir/sequential_imp/main.cpp.o
.PHONY : sequential_imp/main.cpp.o

sequential_imp/main.i: sequential_imp/main.cpp.i
.PHONY : sequential_imp/main.i

# target to preprocess a source file
sequential_imp/main.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/sequential_imp.dir/build.make CMakeFiles/sequential_imp.dir/sequential_imp/main.cpp.i
.PHONY : sequential_imp/main.cpp.i

sequential_imp/main.s: sequential_imp/main.cpp.s
.PHONY : sequential_imp/main.s

# target to generate assembly for a file
sequential_imp/main.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/sequential_imp.dir/build.make CMakeFiles/sequential_imp.dir/sequential_imp/main.cpp.s
.PHONY : sequential_imp/main.cpp.s

sequential_imp/sequential.o: sequential_imp/sequential.cpp.o
.PHONY : sequential_imp/sequential.o

# target to build an object file
sequential_imp/sequential.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/sequential_imp.dir/build.make CMakeFiles/sequential_imp.dir/sequential_imp/sequential.cpp.o
.PHONY : sequential_imp/sequential.cpp.o

sequential_imp/sequential.i: sequential_imp/sequential.cpp.i
.PHONY : sequential_imp/sequential.i

# target to preprocess a source file
sequential_imp/sequential.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/sequential_imp.dir/build.make CMakeFiles/sequential_imp.dir/sequential_imp/sequential.cpp.i
.PHONY : sequential_imp/sequential.cpp.i

sequential_imp/sequential.s: sequential_imp/sequential.cpp.s
.PHONY : sequential_imp/sequential.s

# target to generate assembly for a file
sequential_imp/sequential.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/sequential_imp.dir/build.make CMakeFiles/sequential_imp.dir/sequential_imp/sequential.cpp.s
.PHONY : sequential_imp/sequential.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... edit_cache"
	@echo "... rebuild_cache"
	@echo "... GPU_openmp"
	@echo "... OpenMp"
	@echo "... sequential_imp"
	@echo "... GPU_openmp/main.o"
	@echo "... GPU_openmp/main.i"
	@echo "... GPU_openmp/main.s"
	@echo "... GPU_openmp/nsc_solver.o"
	@echo "... GPU_openmp/nsc_solver.i"
	@echo "... GPU_openmp/nsc_solver.s"
	@echo "... OpenMp/main.o"
	@echo "... OpenMp/main.i"
	@echo "... OpenMp/main.s"
	@echo "... OpenMp/nsc_solver.o"
	@echo "... OpenMp/nsc_solver.i"
	@echo "... OpenMp/nsc_solver.s"
	@echo "... sequential_imp/main.o"
	@echo "... sequential_imp/main.i"
	@echo "... sequential_imp/main.s"
	@echo "... sequential_imp/sequential.o"
	@echo "... sequential_imp/sequential.i"
	@echo "... sequential_imp/sequential.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

