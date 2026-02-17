#########################################################
#
# OPENCV TUTORIALS 
#
#########################################################

set(TUTORIAL_TARGETS "")
add_executable( display_image display_image.cpp )
target_link_libraries( display_image ${OpenCV_LIBS} )
list(APPEND TUTORIAL_TARGETS display_image)

add_executable( load_modify_image load_modify_image.cpp )
target_link_libraries( load_modify_image ${OpenCV_LIBS} )
list(APPEND TUTORIAL_TARGETS load_modify_image)

add_executable( mat_the_basic_image_container mat_the_basic_image_container.cpp )
target_link_libraries( mat_the_basic_image_container ${OpenCV_LIBS} )
list(APPEND TUTORIAL_TARGETS mat_the_basic_image_container)

foreach(target_name IN LISTS TUTORIAL_TARGETS)
    target_compile_options(${target_name} PRIVATE ${MY_COMPILE_OPTIONS})
    target_compile_definitions(${target_name} PUBLIC ${MY_COMPILE_DEFINITIONS})
endforeach()


add_custom_target( tutorials DEPENDS load_modify_image display_image mat_the_basic_image_container )
