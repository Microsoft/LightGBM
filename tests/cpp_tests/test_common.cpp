/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <gtest/gtest.h>

#include <limits>
#include <vector>

#include "../include/LightGBM/utils/common.h"
#include "../include/LightGBM/utils/threading.h"


// This is a basic test for floating number parsing.
// Most of the test cases come from:
// https://github.com/dmlc/xgboost/blob/master/tests/cpp/common/test_charconv.cc
// https://github.com/Alexhuszagh/rust-lexical/blob/master/data/test-parse-unittests/strtod_tests.toml
class AtofPreciseTest : public testing::Test {
 public:
  struct AtofTestCase {
    const char* data;
    double expected;
  };

  static double TestAtofPrecise(
      const char* data, double expected, bool test_eq = true) {
    double got = 0;
    const char* end = LightGBM::Common::AtofPrecise(data, &got);
    EXPECT_TRUE(end != data) << "fail to parse: " << data;
    EXPECT_EQ(*end, '\0') << "not parsing to end: " << data;
    if (test_eq) {
      EXPECT_EQ(expected, got) << "parse string: " << data;
    }
    return got;
  }

  static double Int64Bits2Double(uint64_t v) {
    union {
      uint64_t i;
      double d;
    } conv;
    conv.i = v;
    return conv.d;
  }
};

TEST_F(AtofPreciseTest, Basic) {
  AtofTestCase test_cases[] = {
      { "0", 0.0 },
      { "0E0", 0.0 },
      { "-0E0", 0.0 },
      { "-0", -0.0 },
      { "1", 1.0 },
      { "1E0", 1.0 },
      { "-1", -1.0 },
      { "-1E0", -1.0 },
      { "123456.0", 123456.0 },
      { "432E1", 432E1 },
      { "1.2345678", 1.2345678 },
      { "2.4414062E-4", 2.4414062E-4 },
      { "3.0540412E5", 3.0540412E5 },
      { "3.355445E7", 3.355445E7 },
      { "1.1754944E-38", 1.1754944E-38 },
  };

  for (auto const& test : test_cases) {
    TestAtofPrecise(test.data, test.expected);
  }
}

TEST_F(AtofPreciseTest, CornerCases) {
  AtofTestCase test_cases[] = {
      { "1e-400", 0.0 },
      { "2.4703282292062326e-324", 0.0 },
      { "4.9406564584124654e-324", Int64Bits2Double(0x0000000000000001LU) },
      { "8.44291197326099e-309", Int64Bits2Double(0x0006123400000001LU) },
      // FLT_MAX
      { "3.40282346638528859811704183484516925440e38",
        static_cast<double>(std::numeric_limits<float>::max()) },
      // FLT_MIN
      { "1.1754943508222875079687365372222456778186655567720875215087517062784172594547271728515625e-38",
        static_cast<double>(std::numeric_limits<float>::min()) },
      // DBL_MAX (1 + (1 - 2^-52)) * 2^1023 = (2^53 - 1) * 2^971
      { "17976931348623157081452742373170435679807056752584499659891747680315"
        "72607800285387605895586327668781715404589535143824642343213268894641"
        "82768467546703537516986049910576551282076245490090389328944075868508"
        "45513394230458323690322294816580855933212334827479782620414472316873"
        "8177180919299881250404026184124858368", std::numeric_limits<double>::max() },
      { "1.7976931348623158e+308", std::numeric_limits<double>::max() },
      // 2^971 * (2^53 - 1 + 1/2) : the smallest number resolving to inf
      {"179769313486231580793728971405303415079934132710037826936173778980444"
       "968292764750946649017977587207096330286416692887910946555547851940402"
       "630657488671505820681908902000708383676273854845817711531764475730270"
       "069855571366959622842914819860834936475292719074168444365510704342711"
       "559699508093042880177904174497792", std::numeric_limits<double>::infinity() },
      // Near DBL_MIN
      { "2.2250738585072009e-308", Int64Bits2Double(0x000fffffffffffffLU) },
      // DBL_MIN 2^-1022
      { "2.2250738585072012e-308", std::numeric_limits<double>::min() },
      { "2.2250738585072014e-308", std::numeric_limits<double>::min() },
  };

  for (auto const& test : test_cases) {
    TestAtofPrecise(test.data, test.expected);
  }
}

TEST_F(AtofPreciseTest, ErrorInput) {
  double got = 0;
  ASSERT_THROW(LightGBM::Common::AtofPrecise("x1", &got),  std::runtime_error);
}

TEST_F(AtofPreciseTest, NaN) {
  AtofTestCase test_cases[] = {
      { "nan", std::numeric_limits<double>::quiet_NaN() },
      { "NaN", std::numeric_limits<double>::quiet_NaN() },
      { "NAN", std::numeric_limits<double>::quiet_NaN() },
      // The behavior for parsing -nan depends on implementation.
      // Thus we skip binary check for negative nan.
      { "-nan", -std::numeric_limits<double>::quiet_NaN() },
      { "-NaN", -std::numeric_limits<double>::quiet_NaN() },
      { "-NAN", -std::numeric_limits<double>::quiet_NaN() },
  };

  for (auto const& test : test_cases) {
    double got = TestAtofPrecise(test.data, test.expected, false);

    EXPECT_TRUE(std::isnan(got)) << "not parsed as NaN: " << test.data;
    if (got > 0) {
      // See comment in test_cases.
      EXPECT_EQ(memcmp(&got, &test.expected, sizeof(test.expected)), 0)
                << "parsed NaN is not the same for every bit: " << test.data;
    }
  }
}

TEST_F(AtofPreciseTest, Inf) {
  AtofTestCase test_cases[] = {
      { "inf", std::numeric_limits<double>::infinity() },
      { "Inf", std::numeric_limits<double>::infinity() },
      { "INF", std::numeric_limits<double>::infinity() },
      { "-inf", -std::numeric_limits<double>::infinity() },
      { "-Inf", -std::numeric_limits<double>::infinity() },
      { "-INF", -std::numeric_limits<double>::infinity() },
  };

  for (auto const& test : test_cases) {
    double got = TestAtofPrecise(test.data, test.expected, false);

    EXPECT_EQ(LightGBM::Common::Sign(test.expected), LightGBM::Common::Sign(got)) << "sign differs parsing: " << test.data;
    EXPECT_TRUE(std::isinf(got)) << "not parsed as infinite: " << test.data;
    EXPECT_EQ(memcmp(&got, &test.expected, sizeof(test.expected)), 0)
              << "parsed infinite is not the same for every bit: " << test.data;
  }
}

TEST(ParallelPartitionRunnerTest, EmptyInputReturnsZero) {
  LightGBM::ParallelPartitionRunner<LightGBM::data_size_t, false> runner(8, 1);
  bool callback_called = false;
  std::vector<LightGBM::data_size_t> output(1, -1);

  const auto left_count = runner.Run<false>(
      0,
      [&](int, LightGBM::data_size_t, LightGBM::data_size_t,
          LightGBM::data_size_t*, LightGBM::data_size_t*) {
        callback_called = true;
        return 0;
      },
      output.data());

  EXPECT_EQ(left_count, 0);
  EXPECT_FALSE(callback_called);
  EXPECT_EQ(output[0], -1);
}
