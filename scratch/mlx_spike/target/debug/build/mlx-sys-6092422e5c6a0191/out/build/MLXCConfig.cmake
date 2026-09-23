# Find MLX C
#
# Defines the following variables:
#
#   MLX_C_FOUND            : True if MLX C is found
#   MLX_C_INCLUDE_DIRS     : Include directory
#   MLX_C_LIBRARIES        : Libraries to link against
#   MLX_C_CXX_FLAGS        : Additional compiler flags


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was mlx-c.pc.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

####################################################################################

include(${PACKAGE_PREFIX_DIR}/share/cmake/MLXC/MLXCTargets.cmake)

set_and_check(MLX_C_LIBRARY_DIRS ${PACKAGE_PREFIX_DIR}/lib)
set_and_check(MLX_C_INCLUDE_DIRS ${PACKAGE_PREFIX_DIR}/include)
set(MLX_C_LIBRARIES mlxc)

find_library(MLX_C_LIBRARY mlxc PATHS ${MLX_C_LIBRARY_DIRS})

# set_target_properties(mlxc PROPERTIES
#    CXX_STANDARD 17
#    INTERFACE_COMPILE_OPTIONS "${MLX_C_CXX_FLAGS}"
# )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MLX_C DEFAULT_MSG MLX_C_LIBRARY MLX_C_INCLUDE_DIRS)
