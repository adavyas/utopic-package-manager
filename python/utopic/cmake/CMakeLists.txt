# Package-managed build of Utopic against a package-managed llama.cpp build.
# The package owns the vendored native source snapshot and CMake project, then
# passes UTOPIC_NATIVE_SOURCE_DIR plus UTOPIC_LLAMACPP_DIR during setup.
cmake_minimum_required(VERSION 3.18)
project(utopic CXX C)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(UTOPIC_BACKEND_NAME "auto" CACHE STRING "Backend selected by package-managed setup")

if(DEFINED UTOPIC_NATIVE_SOURCE_DIR)
  get_filename_component(UTOPIC_NATIVE_SOURCE_DIR "${UTOPIC_NATIVE_SOURCE_DIR}" ABSOLUTE)
else()
  message(FATAL_ERROR "UTOPIC_NATIVE_SOURCE_DIR is required. Run `utopic setup` for the package-managed build, or pass -DUTOPIC_NATIVE_SOURCE_DIR=/path/to/package/native when developing the package installer.")
endif()
if(NOT EXISTS "${UTOPIC_NATIVE_SOURCE_DIR}/main.cpp")
  message(FATAL_ERROR "UTOPIC_NATIVE_SOURCE_DIR does not look like a Utopic native source directory: ${UTOPIC_NATIVE_SOURCE_DIR}")
endif()

if(DEFINED UTOPIC_LLAMACPP_DIR)
  set(LLAMA_DIR "${UTOPIC_LLAMACPP_DIR}")
else()
  message(FATAL_ERROR "UTOPIC_LLAMACPP_DIR is required. Run `utopic setup` for the package-managed build, or pass -DUTOPIC_LLAMACPP_DIR=/path/to/llama.cpp when developing the package installer.")
endif()
set(LLAMA_BIN "${LLAMA_DIR}/build/bin")
message(STATUS "linking llama.cpp from: ${LLAMA_DIR}")
message(STATUS "building Utopic native source from: ${UTOPIC_NATIVE_SOURCE_DIR}")

file(READ "${LLAMA_DIR}/include/llama.h" LLAMA_PUBLIC_HEADER)
if(LLAMA_PUBLIC_HEADER MATCHES "llama_diffusion_set_block_decode")
  set(UTOPIC_HAS_BLOCK_DECODE ON)
else()
  set(UTOPIC_HAS_BLOCK_DECODE OFF)
  message(STATUS "llama.cpp block-decode API not found; disabling optional dual-cache path")
endif()

find_package(Threads REQUIRED)

set(UTOPIC_INCLUDES
  "${UTOPIC_NATIVE_SOURCE_DIR}"
  "${LLAMA_DIR}/include"
  "${LLAMA_DIR}/common"
  "${LLAMA_DIR}/ggml/include"
  "${LLAMA_DIR}/vendor")   # vendored cpp-httplib + nlohmann/json (same headers llama.cpp's server uses)

set(UTOPIC_LLAMA_DEFINES
  GGML_BACKEND_SHARED
  GGML_SHARED
  GGML_USE_CPU
  LLAMA_SHARED
  UTOPIC_BACKEND_NAME="${UTOPIC_BACKEND_NAME}")
if(LLAMA_PUBLIC_HEADER MATCHES "int32_t[ \t\r\n]+off")
  list(APPEND UTOPIC_LLAMA_DEFINES UTOPIC_LLAMA_PHASE_HAS_OFFSET)
endif()
if(EXISTS "${LLAMA_BIN}/libggml-cuda.so" OR EXISTS "${LLAMA_BIN}/libggml-cuda.dylib")
  list(APPEND UTOPIC_LLAMA_DEFINES GGML_USE_CUDA)
endif()

# CLI: one-shot generation
add_executable(utopic
  "${UTOPIC_NATIVE_SOURCE_DIR}/main.cpp"
  "${UTOPIC_NATIVE_SOURCE_DIR}/diffusion_driver.cpp")
target_include_directories(utopic PRIVATE ${UTOPIC_INCLUDES})
target_compile_definitions(utopic PRIVATE ${UTOPIC_LLAMA_DEFINES})
target_link_directories(utopic PRIVATE "${LLAMA_BIN}")
target_link_libraries(utopic PRIVATE llama ggml ggml-base)
if(NOT UTOPIC_HAS_BLOCK_DECODE)
  target_compile_definitions(utopic PRIVATE DG_NO_BLOCK_DECODE)
endif()
set_target_properties(utopic PROPERTIES
  BUILD_RPATH "${LLAMA_BIN}" INSTALL_RPATH "${LLAMA_BIN}")

function(add_utopic_json_runner target_name)
  add_executable(${target_name}
    "${UTOPIC_NATIVE_SOURCE_DIR}/runner.cpp"
    "${UTOPIC_NATIVE_SOURCE_DIR}/diffusion_driver.cpp")
  target_include_directories(${target_name} PRIVATE ${UTOPIC_INCLUDES})
  target_compile_definitions(${target_name} PRIVATE ${UTOPIC_LLAMA_DEFINES})
  target_link_directories(${target_name} PRIVATE "${LLAMA_BIN}")
  target_link_libraries(${target_name} PRIVATE llama ggml ggml-base)
  if(NOT UTOPIC_HAS_BLOCK_DECODE)
    target_compile_definitions(${target_name} PRIVATE DG_NO_BLOCK_DECODE)
  endif()
  set_target_properties(${target_name} PROPERTIES
    BUILD_RPATH "${LLAMA_BIN}" INSTALL_RPATH "${LLAMA_BIN}")
endfunction()

# JSON runner: one-shot native execution contract used by the Python MCP/CLI.
add_utopic_json_runner(utopic_runner)

# Server: resident model, OpenAI-compatible /v1/chat/completions
# cpp-httplib is split-mode (.h + .cpp); compile its impl TU in (same as llama.cpp's server lib).
add_executable(utopic_server
  "${UTOPIC_NATIVE_SOURCE_DIR}/server.cpp"
  "${UTOPIC_NATIVE_SOURCE_DIR}/diffusion_driver.cpp"
  "${LLAMA_DIR}/vendor/cpp-httplib/httplib.cpp")
target_include_directories(utopic_server PRIVATE ${UTOPIC_INCLUDES})
target_compile_definitions(utopic_server PRIVATE ${UTOPIC_LLAMA_DEFINES})
target_link_directories(utopic_server PRIVATE "${LLAMA_BIN}")
target_link_libraries(utopic_server PRIVATE llama ggml ggml-base Threads::Threads)
if(NOT UTOPIC_HAS_BLOCK_DECODE)
  target_compile_definitions(utopic_server PRIVATE DG_NO_BLOCK_DECODE)
endif()
set_target_properties(utopic_server PROPERTIES
  BUILD_RPATH "${LLAMA_BIN}" INSTALL_RPATH "${LLAMA_BIN}")

# MCP server: resident model, stdio JSON-RPC tool for Claude Code / Codex
add_executable(utopic_mcp
  "${UTOPIC_NATIVE_SOURCE_DIR}/mcp_server.cpp"
  "${UTOPIC_NATIVE_SOURCE_DIR}/diffusion_driver.cpp")
target_include_directories(utopic_mcp PRIVATE ${UTOPIC_INCLUDES})
target_compile_definitions(utopic_mcp PRIVATE ${UTOPIC_LLAMA_DEFINES})
target_link_directories(utopic_mcp PRIVATE "${LLAMA_BIN}")
target_link_libraries(utopic_mcp PRIVATE llama ggml ggml-base Threads::Threads)
if(NOT UTOPIC_HAS_BLOCK_DECODE)
  target_compile_definitions(utopic_mcp PRIVATE DG_NO_BLOCK_DECODE)
endif()
set_target_properties(utopic_mcp PROPERTIES
  BUILD_RPATH "${LLAMA_BIN}" INSTALL_RPATH "${LLAMA_BIN}")

# ACP agent: resident model, stdio JSON-RPC agent for ACP editors (e.g. Zed)
add_executable(utopic_acp
  "${UTOPIC_NATIVE_SOURCE_DIR}/acp_server.cpp"
  "${UTOPIC_NATIVE_SOURCE_DIR}/diffusion_driver.cpp")
target_include_directories(utopic_acp PRIVATE ${UTOPIC_INCLUDES})
target_compile_definitions(utopic_acp PRIVATE ${UTOPIC_LLAMA_DEFINES})
target_link_directories(utopic_acp PRIVATE "${LLAMA_BIN}")
target_link_libraries(utopic_acp PRIVATE llama ggml ggml-base Threads::Threads)
if(NOT UTOPIC_HAS_BLOCK_DECODE)
  target_compile_definitions(utopic_acp PRIVATE DG_NO_BLOCK_DECODE)
endif()
set_target_properties(utopic_acp PROPERTIES
  BUILD_RPATH "${LLAMA_BIN}" INSTALL_RPATH "${LLAMA_BIN}")
