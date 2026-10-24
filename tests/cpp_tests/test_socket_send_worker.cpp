/*!
 * Copyright (c) 2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifdef USE_SOCKET

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <memory>
#include <thread>
#include <vector>

#include "../../src/network/linkers.h"

namespace LightGBM {
namespace {

class SocketSendWorkerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    TcpSocket::Startup();
    // Bind port zero and keep the listener open until accept, avoiding a
    // find-free-port / bind race with other tests.
    const SOCKET listener = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    ASSERT_NE(listener, INVALID_SOCKET);
    TcpSocket listener_socket(listener);
    ASSERT_TRUE(listener_socket.Bind(0));
    sockaddr_in address{};
#ifdef _WIN32
    int address_size = sizeof(address);
#else
    socklen_t address_size = sizeof(address);
#endif
    ASSERT_EQ(getsockname(listener, reinterpret_cast<sockaddr*>(&address), &address_size), 0);
    listener_socket.Listen();
    client_fd_ = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    ASSERT_NE(client_fd_, INVALID_SOCKET);
    TcpSocket client(client_fd_);
    ASSERT_TRUE(client.Connect("127.0.0.1", ntohs(address.sin_port)));
    TcpSocket server = listener_socket.Accept();
    listener_socket.Close();
    Config config;
    config.num_machines = 1;
    config.local_listen_port = 0;
    config.machines = "rank=0,127.0.0.1:0";
    left_.reset(new Linkers(config));
    right_.reset(new Linkers(config));
    left_->SetLinker(0, client);
    right_->SetLinker(0, server);
    // SetLinker applies the configured minute-based timeout. Override it on
    // the shared descriptors afterwards to keep failure tests bounded.
    client.SetTimeout(1000);
    server.SetTimeout(1000);
  }

  void TearDown() override {
    left_.reset();
    right_.reset();
    TcpSocket::Finalize();
  }

  template <typename Size>
  void Exchange(Size left_size, Size right_size, int seed) {
    std::vector<char> left_send(std::max<int64_t>(1, left_size));
    std::vector<char> right_send(std::max<int64_t>(1, right_size));
    std::vector<char> left_recv(right_send.size(), 0);
    std::vector<char> right_recv(left_send.size(), 0);
    for (Size i = 0; i < left_size; ++i) {
      left_send[i] = static_cast<char>((i + seed) % 113);
    }
    for (Size i = 0; i < right_size; ++i) {
      right_send[i] = static_cast<char>((i + seed + 1) % 117);
    }
    std::exception_ptr left_error, right_error;
    std::thread peer([&]() {
      try {
        right_->SendRecv(0, right_send.data(), right_size, 0, right_recv.data(), left_size);
      } catch (...) {
        right_error = std::current_exception();
      }
    });
    try {
      left_->SendRecv(0, left_send.data(), left_size, 0, left_recv.data(), right_size);
    } catch (...) {
      left_error = std::current_exception();
    }
    peer.join();
    ASSERT_FALSE(left_error);
    ASSERT_FALSE(right_error);
    EXPECT_EQ(left_recv, right_send);
    EXPECT_EQ(right_recv, left_send);
  }

  SOCKET client_fd_ = INVALID_SOCKET;
  std::unique_ptr<Linkers> left_, right_;
};

TEST_F(SocketSendWorkerTest, RepeatedIntAndInt64Exchanges) {
  const int limit = SocketConfig::kSocketBufferSize;
  const std::vector<int> lengths = {0, 17, limit - 1, limit, limit + 1, 1024 * 1024};
  for (int round = 0; round < 8; ++round) {
    for (size_t i = 0; i < lengths.size(); ++i) {
      const int left_size = lengths[i];
      const int right_size = lengths[(i + round) % lengths.size()];
      Exchange<int>(left_size, right_size, round);
      Exchange<int64_t>(left_size, right_size, round + 1);
    }
  }
}

TEST_F(SocketSendWorkerTest, ReceiveTimeoutWaitsForSenderAndAllowsAnotherExchange) {
  // No peer response: receive times out after the zero-byte send completes.
  char byte = 0;
  EXPECT_THROW(left_->SendRecv(0, &byte, int64_t{0}, 0, &byte, int64_t{1}), std::exception);
  Exchange<int64_t>(4096, 3072, 42);
}

TEST_F(SocketSendWorkerTest, SendErrorIsRethrownOnCaller) {
  // Use a closed socket to fail the send independently of OS buffering.
  TcpSocket closed_socket(client_fd_);
  closed_socket.Close();
  left_->SetLinker(0, closed_socket);
  char byte = 0;
  EXPECT_THROW(left_->SendRecv(0, &byte, int64_t{1}, 0, &byte, int64_t{0}), std::exception);
}

TEST_F(SocketSendWorkerTest, DestroyWithoutStartingSender) {
  Exchange<int>(17, 19, 1);
}

}  // namespace
}  // namespace LightGBM

#endif  // USE_SOCKET
