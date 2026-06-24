# Standalone build of Utopic against a package-managed llama.cpp build.
# We link libllama/ggml as a LIBRARY (rent the forward/kernels) and own the loop.
# utopic-package-manager passes UTOPIC_LLAMACPP_DIR during setup.
cmake_minimum_required(VERSION 3.18)
project(utopic CXX C)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(DEFINED UTOPIC_LLAMACPP_DIR)
  set(LLAMA_DIR "${UTOPIC_LLAMACPP_DIR}")
else()
  message(FATAL_ERROR "UTOPIC_LLAMACPP_DIR is required. Run `utopic setup` for the package-managed build, or pass -DUTOPIC_LLAMACPP_DIR=/path/to/llama.cpp when developing the native runtime directly.")
endif()
set(LLAMA_BIN "${LLAMA_DIR}/build/bin")
message(STATUS "linking llama.cpp from: ${LLAMA_DIR}")

file(READ "${LLAMA_DIR}/include/llama.h" LLAMA_PUBLIC_HEADER)
if(LLAMA_PUBLIC_HEADER MATCHES "llama_diffusion_set_block_decode")
  set(UTOPIC_HAS_BLOCK_DECODE ON)
else()
  set(UTOPIC_HAS_BLOCK_DECODE OFF)
  message(STATUS "llama.cpp block-decode API not found; disabling optional dual-cache path")
endif()

find_package(Threads REQUIRED)

set(UTOPIC_INCLUDES
  "${LLAMA_DIR}/include"
  "${LLAMA_DIR}/common"
  "${LLAMA_DIR}/ggml/include"
  "${LLAMA_DIR}/vendor")   # vendored cpp-httplib + nlohmann/json (same headers llama.cpp's server uses)

set(UTOPIC_LLAMA_DEFINES
  GGML_BACKEND_SHARED
  GGML_SHARED
  GGML_USE_CPU
  LLAMA_SHARED)
if(LLAMA_PUBLIC_HEADER MATCHES "int32_t[ \t\r\n]+off")
  list(APPEND UTOPIC_LLAMA_DEFINES UTOPIC_LLAMA_PHASE_HAS_OFFSET)
endif()
if(EXISTS "${LLAMA_BIN}/libggml-cuda.so" OR EXISTS "${LLAMA_BIN}/libggml-cuda.dylib")
  list(APPEND UTOPIC_LLAMA_DEFINES GGML_USE_CUDA)
endif()

# CLI: one-shot generation
add_executable(utopic main.cpp diffusion_driver.cpp)
target_include_directories(utopic PRIVATE ${UTOPIC_INCLUDES})
target_compile_definitions(utopic PRIVATE ${UTOPIC_LLAMA_DEFINES})
target_link_directories(utopic PRIVATE "${LLAMA_BIN}")
target_link_libraries(utopic PRIVATE llama ggml ggml-base)
if(NOT UTOPIC_HAS_BLOCK_DECODE)
  target_compile_definitions(utopic PRIVATE DG_NO_BLOCK_DECODE)
endif()
set_target_properties(utopic PROPERTIES
  BUILD_RPATH "${LLAMA_BIN}" INSTALL_RPATH "${LLAMA_BIN}")

# Server: resident model, OpenAI-compatible /v1/chat/completions
# cpp-httplib is split-mode (.h + .cpp); compile its impl TU in (same as llama.cpp's server lib).
add_executable(utopic_server server.cpp diffusion_driver.cpp "${LLAMA_DIR}/vendor/cpp-httplib/httplib.cpp")
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
add_executable(utopic_mcp mcp_server.cpp diffusion_driver.cpp)
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
add_executable(utopic_acp acp_server.cpp diffusion_driver.cpp)
target_include_directories(utopic_acp PRIVATE ${UTOPIC_INCLUDES})
target_compile_definitions(utopic_acp PRIVATE ${UTOPIC_LLAMA_DEFINES})
target_link_directories(utopic_acp PRIVATE "${LLAMA_BIN}")
target_link_libraries(utopic_acp PRIVATE llama ggml ggml-base Threads::Threads)
if(NOT UTOPIC_HAS_BLOCK_DECODE)
  target_compile_definitions(utopic_acp PRIVATE DG_NO_BLOCK_DECODE)
endif()
set_target_properties(utopic_acp PROPERTIES
  BUILD_RPATH "${LLAMA_BIN}" INSTALL_RPATH "${LLAMA_BIN}")
