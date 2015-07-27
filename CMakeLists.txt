cmake_minimum_required ( VERSION 2.8)

project(Perishable Inventory Management)

include_directories(BEFORE PROJECT_SOURCE_DIR)
add_subdirectory(src)
set( CMAKE_EXPORT_COMPILE_COMMANDS 1  )

