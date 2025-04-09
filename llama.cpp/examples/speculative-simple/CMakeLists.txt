set(TARGET llama-speculative-simple)
add_executable(${TARGET} speculative-simple.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# 새로운 speculative-eagle 타겟 설정
set(TARGET_EAGLE llama-speculative-eagle) # 새 타겟 이름 설정 (예시)
add_executable(${TARGET_EAGLE} speculative-eagle.cpp) # 새 소스 파일 지정
install(TARGETS ${TARGET_EAGLE} RUNTIME) # 새 타겟 설치 규칙 추가
# 새 타겟에 라이브러리 링크 (기존 타겟과 동일한 라이브러리가 필요하다고 가정)
target_link_libraries(${TARGET_EAGLE} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
# 새 타겟에 컴파일 피처 설정 (기존 타겟과 동일한 C++17 표준 사용 가정)
target_compile_features(${TARGET_EAGLE} PRIVATE cxx_std_17)
