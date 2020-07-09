
if(${BUILD_DOCUMENTATION} MATCHES ON)
    set(DOXYGEN_FOUND OFF)
    find_program(DOXYGEN_EXECUTABLE doxygen "")
    
    if(DOXYGEN_EXECUTABLE)
      message(STATUS "Doxygen executable found!")
      set(DOXYGEN_FOUND ON)
    endif(DOXYGEN_EXECUTABLE)
    
    if(NOT DOXYGEN_FOUND)
        message(FATAL_ERROR "Doxygen is needed to build the documentation.")
    endif()
    
    set(DOXYGEN_INDEX_FILE ${CMAKE_CURRENT_BINARY_DIR}/xml/index.xml)

    set(doxyfile_in ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
    set(doxyfile ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    configure_file(${doxyfile_in} ${doxyfile} @ONLY)

    add_custom_target(doc
        COMMAND ${DOXYGEN_EXECUTABLE} ${doxyfile}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM)

    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html DESTINATION share/doc)

    # Sphinx
    find_program (SPHINX_EXECUTABLE sphinx-build "")

    if(SPHINX_EXECUTABLE)
        message(STATUS "Sphinx project created!")
        set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR})
        set(SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/sphinx)
        set(SPHINX_INDEX_FILE ${SPHINX_BUILD}/index.html)

        set(sphinx_conf_in ${CMAKE_CURRENT_SOURCE_DIR}/conf.py.in)
        set(sphinx_conf ${CMAKE_CURRENT_SOURCE_DIR}/conf.py)
        configure_file(${sphinx_conf_in} ${sphinx_conf} @ONLY)

        # Only regenerate Sphinx when:
        #  - Doxygen has rerun 
        #  - Our doc files have been updated
        #  - The Sphinx config has been updated
        add_custom_command(OUTPUT ${SPHINX_INDEX_FILE}
            COMMAND 
                ${SPHINX_EXECUTABLE} -b html
                # Tell Breathe where to find the Doxygen output
                -Dbreathe_projects.xnio=${CMAKE_CURRENT_BINARY_DIR}/xml
                ${SPHINX_SOURCE} ${SPHINX_BUILD}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            DEPENDS 
                # Other docs files you want to track should go here (or in some variable)
                ${CMAKE_CURRENT_SOURCE_DIR}/index.rst
                ${DOXYGEN_INDEX_FILE}
            MAIN_DEPENDENCY ${sphinx_conf}
            COMMENT "Generating documentation with Sphinx")

        # Nice named target so we can run the job easily
        add_custom_target(Sphinx ALL DEPENDS ${SPHINX_INDEX_FILE})    
    endif(SPHINX_EXECUTABLE)
endif(${BUILD_DOCUMENTATION} MATCHES ON)
