cmake_minimum_required(VERSION 3.18)
project(utopic_package_manager LANGUAGES C CXX)

include(FetchContent)

set(UTOPIC_NATIVE_GIT_REPOSITORY
    "https://github.com/adavyas/Utopic.git"
    CACHE STRING "Git repository containing the Utopic native runtime")
set(UTOPIC_NATIVE_GIT_TAG
    "6943cab5a80ac165bd6c4a14962c6d4b64cb6226"
    CACHE STRING "Pinned Utopic native runtime revision")
set(UTOPIC_NATIVE_SOURCE_DIR
    ""
    CACHE PATH "Optional local checkout of the Utopic native runtime")

if(UTOPIC_NATIVE_SOURCE_DIR)
  add_subdirectory("${UTOPIC_NATIVE_SOURCE_DIR}/native" "${CMAKE_BINARY_DIR}/utopic-native")
else()
  FetchContent_Declare(
    utopic_native
    GIT_REPOSITORY "${UTOPIC_NATIVE_GIT_REPOSITORY}"
    GIT_TAG        "${UTOPIC_NATIVE_GIT_TAG}"
    SOURCE_SUBDIR  native
  )
  FetchContent_MakeAvailable(utopic_native)
endif()

install(TARGETS utopic utopic_server utopic_mcp utopic_acp
  RUNTIME DESTINATION utopic/bin)
