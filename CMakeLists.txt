cmake_minimum_required(VERSION 3.5.0)
project(IPM VERSION 0.1.0 LANGUAGES CXX)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#set paths to coinor-libcoinutils3v5
set(COIN_INC_DIR /usr/include/coin/)
set(COIN_SO_PATH /usr/lib/x86_64-linux-gnu/libCoinUtils.so.3.11.4)

#set(HEADER_FILES ${COIN_INC_DIR}/CoinLpIO.hpp)

add_executable(IPM main.cpp) #${HEADER_FILES})

target_include_directories(IPM PUBLIC ${COIN_INC_DIR})
TARGET_LINK_LIBRARIES(IPM ${COIN_SO_PATH})