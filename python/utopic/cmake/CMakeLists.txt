# Package-managed build of Utopic against a package-managed llama.cpp build.
# The Utopic repo owns native source. This package owns the CMake project and
# passes UTOPIC_NATIVE_SOURCE_DIR plus UTOPIC_LLAMACPP_DIR during setup.
cmake_minimum_required(VERSION 3.18)
project(utopic CXX C)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(DEFINED UTOPIC_NATIVE_SOURCE_DIR)
  get_filename_component(UTOPIC_NATIVE_SOURCE_DIR "${UTOPIC_NATIVE_SOURCE_DIR}" ABSOLUTE)
else()
  message(FATAL_ERROR "UTOPIC_NATIVE_SOURCE_DIR is required. Run `utopic setup` for the package-managed build, or pass -DUTOPIC_NATIVE_SOURCE_DIR=/path/to/Utopic/native when developing the package installer.")
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
option(UTOPIC_ENABLE_STABLE_DIFFUSION "Build the native image engine against stable-diffusion.cpp" OFF)
option(UTOPIC_ENABLE_SHERPA_ONNX "Build the native TTS plugin against sherpa-onnx" OFF)

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
  LLAMA_SHARED)
if(LLAMA_PUBLIC_HEADER MATCHES "int32_t[ \t\r\n]+off")
  list(APPEND UTOPIC_LLAMA_DEFINES UTOPIC_LLAMA_PHASE_HAS_OFFSET)
endif()
if(EXISTS "${LLAMA_BIN}/libggml-cuda.so" OR EXISTS "${LLAMA_BIN}/libggml-cuda.dylib")
  list(APPEND UTOPIC_LLAMA_DEFINES GGML_USE_CUDA)
endif()

set(UTOPIC_IMAGE_ENGINE_SOURCE "${CMAKE_CURRENT_LIST_DIR}/image_engine_disabled.cpp")
set(UTOPIC_RUNNER_LIBS llama ggml ggml-base Threads::Threads)
if(UTOPIC_ENABLE_STABLE_DIFFUSION)
  if(NOT DEFINED UTOPIC_STABLE_DIFFUSION_DIR)
    message(FATAL_ERROR "UTOPIC_STABLE_DIFFUSION_DIR is required when UTOPIC_ENABLE_STABLE_DIFFUSION=ON")
  endif()
  get_filename_component(UTOPIC_STABLE_DIFFUSION_DIR "${UTOPIC_STABLE_DIFFUSION_DIR}" ABSOLUTE)
  if(NOT EXISTS "${UTOPIC_STABLE_DIFFUSION_DIR}/include/stable-diffusion.h")
    message(FATAL_ERROR "UTOPIC_STABLE_DIFFUSION_DIR must contain include/stable-diffusion.h: ${UTOPIC_STABLE_DIFFUSION_DIR}")
  endif()

  find_library(UTOPIC_GGML_LIBRARY
    NAMES ggml
    PATHS "${LLAMA_BIN}"
    NO_DEFAULT_PATH
    REQUIRED)
  if(NOT TARGET ggml)
    add_library(ggml UNKNOWN IMPORTED GLOBAL)
    set_target_properties(ggml PROPERTIES
      IMPORTED_LOCATION "${UTOPIC_GGML_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${LLAMA_DIR};${LLAMA_DIR}/ggml/include")
  endif()

  set(SD_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(SD_WEBP OFF CACHE BOOL "" FORCE)
  set(SD_WEBM OFF CACHE BOOL "" FORCE)
  set(SD_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  set(SD_BUILD_SHARED_GGML_LIB OFF CACHE BOOL "" FORCE)
  set(SD_USE_SYSTEM_GGML OFF CACHE BOOL "" FORCE)
  if(EXISTS "${LLAMA_BIN}/libggml-cuda.so" OR EXISTS "${LLAMA_BIN}/libggml-cuda.dylib")
    set(SD_CUDA ON CACHE BOOL "" FORCE)
  endif()
  if(EXISTS "${LLAMA_BIN}/libggml-metal.so" OR EXISTS "${LLAMA_BIN}/libggml-metal.dylib")
    set(SD_METAL ON CACHE BOOL "" FORCE)
  endif()

  add_subdirectory("${UTOPIC_STABLE_DIFFUSION_DIR}" "${CMAKE_BINARY_DIR}/stable-diffusion.cpp")
  list(APPEND UTOPIC_INCLUDES "${UTOPIC_STABLE_DIFFUSION_DIR}/include")
  set(UTOPIC_IMAGE_ENGINE_SOURCE "${UTOPIC_NATIVE_SOURCE_DIR}/image_engine.cpp")
  list(APPEND UTOPIC_RUNNER_LIBS stable-diffusion)
endif()

if(UTOPIC_ENABLE_SHERPA_ONNX)
  if(NOT DEFINED UTOPIC_SHERPA_ONNX_DIR)
    message(FATAL_ERROR "UTOPIC_SHERPA_ONNX_DIR is required when UTOPIC_ENABLE_SHERPA_ONNX=ON")
  endif()
  get_filename_component(UTOPIC_SHERPA_ONNX_DIR "${UTOPIC_SHERPA_ONNX_DIR}" ABSOLUTE)
  if(NOT EXISTS "${UTOPIC_SHERPA_ONNX_DIR}/sherpa-onnx/c-api/c-api.h" AND
     NOT EXISTS "${UTOPIC_SHERPA_ONNX_DIR}/include/sherpa-onnx/c-api/c-api.h")
    message(FATAL_ERROR "UTOPIC_SHERPA_ONNX_DIR must contain sherpa-onnx/c-api/c-api.h or include/sherpa-onnx/c-api/c-api.h: ${UTOPIC_SHERPA_ONNX_DIR}")
  endif()

  find_library(UTOPIC_SHERPA_ONNX_LIBRARY
    NAMES sherpa-onnx-c-api
    PATHS
      "${UTOPIC_SHERPA_ONNX_DIR}/lib"
      "${UTOPIC_SHERPA_ONNX_DIR}/build/lib"
      "${UTOPIC_SHERPA_ONNX_DIR}/build/bin"
    NO_DEFAULT_PATH
    REQUIRED)
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

# JSON runner: stable control-plane contract for chat/image/audio/video tasks.
add_executable(utopic_runner
  "${UTOPIC_NATIVE_SOURCE_DIR}/runner.cpp"
  "${UTOPIC_NATIVE_SOURCE_DIR}/runner_contract.cpp"
  "${UTOPIC_NATIVE_SOURCE_DIR}/runner_tasks.cpp"
  "${UTOPIC_NATIVE_SOURCE_DIR}/runner_plugin.cpp"
  "${UTOPIC_NATIVE_SOURCE_DIR}/audio_engine.cpp"
  "${UTOPIC_IMAGE_ENGINE_SOURCE}"
  "${UTOPIC_NATIVE_SOURCE_DIR}/video_engine.cpp"
  "${UTOPIC_NATIVE_SOURCE_DIR}/diffusion_driver.cpp")
target_include_directories(utopic_runner PRIVATE ${UTOPIC_INCLUDES})
target_compile_definitions(utopic_runner PRIVATE ${UTOPIC_LLAMA_DEFINES})
target_link_directories(utopic_runner PRIVATE "${LLAMA_BIN}")
target_link_libraries(utopic_runner PRIVATE ${UTOPIC_RUNNER_LIBS})
if(NOT UTOPIC_HAS_BLOCK_DECODE)
  target_compile_definitions(utopic_runner PRIVATE DG_NO_BLOCK_DECODE)
endif()
if(NOT UTOPIC_ENABLE_STABLE_DIFFUSION)
  target_compile_definitions(utopic_runner PRIVATE UTOPIC_IMAGE_ENGINE_DISABLED)
endif()
set_target_properties(utopic_runner PROPERTIES
  BUILD_RPATH "${LLAMA_BIN}" INSTALL_RPATH "${LLAMA_BIN}")

if(UTOPIC_ENABLE_SHERPA_ONNX)
  add_library(utopic_sherpa_tts SHARED
    "${UTOPIC_NATIVE_SOURCE_DIR}/sherpa_tts_plugin.cpp"
    "${UTOPIC_NATIVE_SOURCE_DIR}/audio_engine.cpp")
  target_include_directories(utopic_sherpa_tts PRIVATE
    ${UTOPIC_INCLUDES}
    "${UTOPIC_SHERPA_ONNX_DIR}"
    "${UTOPIC_SHERPA_ONNX_DIR}/include")
  target_link_libraries(utopic_sherpa_tts PRIVATE ${UTOPIC_SHERPA_ONNX_LIBRARY})
  set_target_properties(utopic_sherpa_tts PROPERTIES
    PREFIX ""
    BUILD_RPATH "${UTOPIC_SHERPA_ONNX_DIR}/lib;${UTOPIC_SHERPA_ONNX_DIR}/build/lib;${UTOPIC_SHERPA_ONNX_DIR}/build/bin"
    INSTALL_RPATH "${UTOPIC_SHERPA_ONNX_DIR}/lib;${UTOPIC_SHERPA_ONNX_DIR}/build/lib;${UTOPIC_SHERPA_ONNX_DIR}/build/bin")
endif()
