
find_package(Doxygen)



if (DOXYGEN_FOUND)

    # set input and output files
    get_target_property(PUBLIC_HEADER_DIR_LSLAB lslab INTERFACE_INCLUDE_DIRECTORIES)
    get_target_property(PUBLIC_HEADER_DIR_ALLOC allocator INTERFACE_INCLUDE_DIRECTORIES)

    file(GLOB_RECURSE PUBLIC_HEADERS_ALLOC ${PUBLIC_HEADER_DIR_ALLOC}/*.cuh)
    file(GLOB_RECURSE PUBLIC_HEADERS_LSLAB ${PUBLIC_HEADER_DIR_LSLAB}/*.cuh)

    message(${PROJECT_SOURCE_DIR})
    message(${PUBLIC_HEADER_DIR_LSLAB})
    message(${PUBLIC_HEADER_DIR_ALLOC})
    message(${PUBLIC_HEADERS_LSLAB})

    set(DOXYGEN_INPUT_DIR "${PROJECT_SOURCE_DIR}/Allocator/include ${PROJECT_SOURCE_DIR}/LSlab/include")
    set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/docs/doxygen)
    set(DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIR}/html/index.html)
    set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
    set(DOXYFILE_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)

    file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR}) #Doxygen won't create this for us

    add_custom_command(OUTPUT ${DOXYGEN_INDEX_FILE}
            DEPENDS ${PUBLIC_HEADERS_LSLAB} ${PUBLIC_HEADERS_ALLOC}
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
            MAIN_DEPENDENCY ${DOXYFILE_OUT} ${DOXYFILE_IN}
            COMMENT "Generating docs")

    add_custom_target(Doxygen ALL DEPENDS ${DOXYGEN_INDEX_FILE})

else (DOXYGEN_FOUND)
    message("Doxygen need to be installed to generate the doxygen documentation")
endif (DOXYGEN_FOUND)