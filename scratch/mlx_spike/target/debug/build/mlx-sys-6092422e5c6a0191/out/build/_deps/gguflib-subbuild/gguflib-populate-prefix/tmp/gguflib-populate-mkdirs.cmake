# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/Users/alejandro/Documents/WORK/DATOOLS/OpenJEV/scratch/mlx_spike/target/debug/build/mlx-sys-6092422e5c6a0191/out/build/_deps/gguflib-src")
  file(MAKE_DIRECTORY "/Users/alejandro/Documents/WORK/DATOOLS/OpenJEV/scratch/mlx_spike/target/debug/build/mlx-sys-6092422e5c6a0191/out/build/_deps/gguflib-src")
endif()
file(MAKE_DIRECTORY
  "/Users/alejandro/Documents/WORK/DATOOLS/OpenJEV/scratch/mlx_spike/target/debug/build/mlx-sys-6092422e5c6a0191/out/build/_deps/gguflib-build"
  "/Users/alejandro/Documents/WORK/DATOOLS/OpenJEV/scratch/mlx_spike/target/debug/build/mlx-sys-6092422e5c6a0191/out/build/_deps/gguflib-subbuild/gguflib-populate-prefix"
  "/Users/alejandro/Documents/WORK/DATOOLS/OpenJEV/scratch/mlx_spike/target/debug/build/mlx-sys-6092422e5c6a0191/out/build/_deps/gguflib-subbuild/gguflib-populate-prefix/tmp"
  "/Users/alejandro/Documents/WORK/DATOOLS/OpenJEV/scratch/mlx_spike/target/debug/build/mlx-sys-6092422e5c6a0191/out/build/_deps/gguflib-subbuild/gguflib-populate-prefix/src/gguflib-populate-stamp"
  "/Users/alejandro/Documents/WORK/DATOOLS/OpenJEV/scratch/mlx_spike/target/debug/build/mlx-sys-6092422e5c6a0191/out/build/_deps/gguflib-subbuild/gguflib-populate-prefix/src"
  "/Users/alejandro/Documents/WORK/DATOOLS/OpenJEV/scratch/mlx_spike/target/debug/build/mlx-sys-6092422e5c6a0191/out/build/_deps/gguflib-subbuild/gguflib-populate-prefix/src/gguflib-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/Users/alejandro/Documents/WORK/DATOOLS/OpenJEV/scratch/mlx_spike/target/debug/build/mlx-sys-6092422e5c6a0191/out/build/_deps/gguflib-subbuild/gguflib-populate-prefix/src/gguflib-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/Users/alejandro/Documents/WORK/DATOOLS/OpenJEV/scratch/mlx_spike/target/debug/build/mlx-sys-6092422e5c6a0191/out/build/_deps/gguflib-subbuild/gguflib-populate-prefix/src/gguflib-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
