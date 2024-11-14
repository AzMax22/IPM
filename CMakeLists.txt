cmake_minimum_required(VERSION 3.5.0)
project(IPM VERSION 0.1.0 LANGUAGES C CXX)

#find_package(coinor-libcoinutils3v5 REQUIRED)

set(COIN_INC_DIR /usr/include/coin/)

#set(HEADER_FILES ${COIN_INC_DIR}/CoinLpIO.hpp)

add_executable(IPM main.cpp) #${HEADER_FILES})

target_include_directories(IPM PUBLIC ${COIN_INC_DIR})
TARGET_LINK_LIBRARIES(IPM /usr/lib/x86_64-linux-gnu/libCoinUtils.so.3.11.4)