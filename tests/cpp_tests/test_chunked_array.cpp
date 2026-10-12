/*!
 * Copyright (c) 2021-2026 Microsoft Corporation. All rights reserved.
 * Copyright (c) 2021-2026 The LightGBM developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 *
 * Author: Alberto Ferreira
 */
#include <gtest/gtest.h>

#include <vector>

#include "../include/LightGBM/utils/chunked_array.hpp"

using LightGBM::ChunkedArray;

/*!
  Helper util to compare two vectors.

  Don't compare floating point vectors this way!
*/
template <typename T>
testing::AssertionResult are_vectors_equal(const std::vector<T> &a, const std::vector<T> &b) {
  if (a.size() != b.size()) {
    return testing::AssertionFailure()
      << "Vectors differ in size: "
      << a.size() << " != " << b.size();
  }

  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i]) {
      return testing::AssertionFailure()
        << "Vectors differ at least at position " << i << ": "
        << a[i] << " != " << b[i];
    }
  }

  return testing::AssertionSuccess();
}


class ChunkedArrayTest : public testing::Test {
 protected:
  void SetUp() override {
  }

  void add_items_to_array(const std::vector<int> &vec, ChunkedArray<int> *ca) {
    for (auto v : vec) {
      ca->add(v);
    }
  }

  /*!
    Ensures that if coalesce_to() is called upon the ChunkedArray,
    it would yield the same contents as vec
  */
  testing::AssertionResult coalesced_output_equals_vec(const ChunkedArray<int> &ca, const std::vector<int> &vec,
                                                       const bool all_addresses = false) {
    std::vector<int> out(vec.size());
    ca.coalesce_to(out.data(), all_addresses);
    return are_vectors_equal(out, vec);
  }

  // Constants
  const std::vector<int> REF_VEC = {1, 5, 2, 4, 9, 8, 7};
  const size_t CHUNK_SIZE = 3;
  const size_t OUT_OF_BOUNDS_OFFSET = 4;

  ChunkedArray<int> ca_ = ChunkedArray<int>(CHUNK_SIZE);  //<! Re-used for many tests.
};


/*! ChunkedArray cannot be built from chunks of size 0. */
TEST_F(ChunkedArrayTest, constructorWithChunkSize0Throws) {
  ASSERT_THROW(ChunkedArray<int> chunked_array(0), std::runtime_error);
}

/*! get_chunk_size() should return the size used in the constructor */
TEST_F(ChunkedArrayTest, constructorWithChunkSize) {
  for (size_t chunk_size = 1; chunk_size < 10; ++chunk_size) {
    ChunkedArray<int> chunked_array(chunk_size);
    ASSERT_EQ(chunked_array.get_chunk_size(), chunk_size);
  }
}

/*!
  get_chunk_size() should return the size used in the constructor
  independently of array manipulations.
*/
TEST_F(ChunkedArrayTest, getChunkSizeIsConstant) {
  for (size_t i = 0; i < 3 * CHUNK_SIZE; ++i) {
    ASSERT_EQ(ca_.get_chunk_size(), CHUNK_SIZE);
    ca_.add(0);
  }
}


/*!
  get_add_count() should return the number of add calls,
  independently of the number of chunks used.
*/
TEST_F(ChunkedArrayTest, getChunksCount) {
  ASSERT_EQ(ca_.get_chunks_count(), 1);  // ChunkedArray always starts with 1 chunk.

  for (size_t i = 0; i < 3 * CHUNK_SIZE; ++i) {
    ca_.add(0);
    int expected_chunks = static_cast<int>(i / CHUNK_SIZE) + 1;
    ASSERT_EQ(ca_.get_chunks_count(), expected_chunks) << "with " << i << " add() call(s) "
                                                       << "and CHUNK_SIZE==" << CHUNK_SIZE << ".";
  }
}

/*!
  get_add_count() should return the number of add calls,
  independently of the number of chunks used.
*/
TEST_F(ChunkedArrayTest, getAddCount) {
  for (size_t i = 0; i < 3 * CHUNK_SIZE; ++i) {
    ASSERT_EQ(ca_.get_add_count(), i);
    ca_.add(0);
  }
}

/*!
  Ensure coalesce_to() works and dumps all the inserted data correctly.

  If the ChunkedArray is created from a sequence of add() calls, coalescing to
  an output array after multiple add operations should yield the same
  exact data at both input and output.
*/
TEST_F(ChunkedArrayTest, coalesceTo) {
  std::vector<int> out(REF_VEC.size());
  add_items_to_array(REF_VEC, &ca_);

  ca_.coalesce_to(out.data());

  ASSERT_TRUE(are_vectors_equal(REF_VEC, out));
}

/*!
  After clear the ChunkedArray() should still be usable.
*/
TEST_F(ChunkedArrayTest, clear) {
  const std::vector<int> ref_vec2 = {1, 2, 5, -1};
  add_items_to_array(REF_VEC, &ca_);
  // Start with some content:
  ASSERT_TRUE(coalesced_output_equals_vec(ca_, REF_VEC));

  // Clear & re-use:
  ca_.clear();
  add_items_to_array(ref_vec2, &ca_);

  // Output should match new content:
  ASSERT_TRUE(coalesced_output_equals_vec(ca_, ref_vec2));
}

/*!
  Ensure ChunkedArray is safe against double-frees.
*/
TEST_F(ChunkedArrayTest, doubleFreeSafe) {
  ca_.release();  // Cannot be used any longer from now on.
  ca_.release();  // Ensure we don't segfault.

  SUCCEED();
}

/*!
  Ensure size computations in the getters are correct.
*/
TEST_F(ChunkedArrayTest, totalArraySizeMatchesLastChunkAddCount) {
  add_items_to_array(REF_VEC, &ca_);

  const size_t first_chunks_add_count = (ca_.get_chunks_count() - 1) * ca_.get_chunk_size();
  const size_t last_chunk_add_count = ca_.get_last_chunk_add_count();

  EXPECT_EQ(first_chunks_add_count, static_cast<int>(REF_VEC.size() / CHUNK_SIZE) * CHUNK_SIZE);
  EXPECT_EQ(last_chunk_add_count, REF_VEC.size() % CHUNK_SIZE);
  EXPECT_EQ(first_chunks_add_count + last_chunk_add_count, ca_.get_add_count());
}

/*!
  Assert all values are correct and at the expected addresses throughout the
  several chunks.

  This uses getitem() to reach each individual address of any of the chunks.

  A sentinel value of -1 is used to check for invalid addresses.
  This would occur if there was an improper data layout with the chunks.
*/
TEST_F(ChunkedArrayTest, dataLayoutTestThroughGetitem) {
  add_items_to_array(REF_VEC, &ca_);

  for (size_t i = 0, chunk = 0, in_chunk_idx = 0; i < REF_VEC.size(); ++i) {
    int value = ca_.getitem(chunk, in_chunk_idx, -1);  // -1 works as sentinel value (bad layout found)

    EXPECT_EQ(value, REF_VEC[i]) << " for address (chunk,in_chunk_idx) = (" << chunk << "," << in_chunk_idx << ")";

    if (++in_chunk_idx == ca_.get_chunk_size()) {
      in_chunk_idx = 0;
      ++chunk;
    }
  }
}

/*!
  Perform an array of setitem & getitem at valid and invalid addresses.
  We use several random addresses and trials to avoid writing much code.

  By testing a random number of addresses many more times than the size of the test space
  we are almost guaranteed to cover all possible search addresses.

  We also gradually add more chunks to the ChunkedArray and re-run more trials
  to ensure the valid/invalid addresses are updated.

  With each valid update we add to a "memory" vector the latest inserted values.
  This is used at the end to ensure all values were stored properly, including after
  value overrides.
*/
TEST_F(ChunkedArrayTest, testDataLayoutWithAdvancedInsertionAPI) {
  const size_t MAX_CHUNKS_SEARCH = 5;
  const size_t MAX_IN_CHUNK_SEARCH_IDX = 2 * CHUNK_SIZE;
  // Number of trials for each new ChunkedArray configuration. Pass 100 times over the search space:
  const size_t N_TRIALS = MAX_CHUNKS_SEARCH * MAX_IN_CHUNK_SEARCH_IDX * 100;
  const int INVALID = -1;  // A negative value signaling the requested value lives in an invalid address.
  const int UNINITIALIZED = -99;  // A negative value to signal this was never updated.
  std::vector<int> ref_values(MAX_CHUNKS_SEARCH * CHUNK_SIZE, UNINITIALIZED);  // Memorize latest inserted values.

  // Each outer loop iteration changes the test by adding +1 chunk. We start with 1 chunk only:
  for (size_t chunks = 1; chunks < MAX_CHUNKS_SEARCH; ++chunks) {
    EXPECT_EQ(ca_.get_chunks_count(), chunks);

    // Sweep valid and invalid addresses with a ChunkedArray with `chunks` chunks:
    for (size_t trial = 0; trial < N_TRIALS; ++trial) {
      // Compute a new trial address & value & if it is a valid address:
      const size_t trial_chunk = std::rand() % MAX_CHUNKS_SEARCH;
      const size_t trial_in_chunk_idx = std::rand() % MAX_IN_CHUNK_SEARCH_IDX;
      const int trial_value = std::rand() % 99999;
      const bool valid_address = (trial_chunk < chunks) & (trial_in_chunk_idx < CHUNK_SIZE);

      // Insert item. If at a valid address, 0 is returned, otherwise, -1 is returned:
      EXPECT_EQ(ca_.setitem(trial_chunk, trial_in_chunk_idx, trial_value),
                valid_address ? 0 : -1);
      // If at valid address, check that the stored value is correct & remember it for the future:
      if (valid_address) {
        // Check the just-stored value with getitem():
        EXPECT_EQ(ca_.getitem(trial_chunk, trial_in_chunk_idx, INVALID), trial_value);

        // Also store the just-stored value for future tracking:
        ref_values[trial_chunk * CHUNK_SIZE + trial_in_chunk_idx] = trial_value;
      }
    }

    ca_.new_chunk();  // Just finished a round of trials. Now add a new chunk. Valid addresses will be expanded.
  }

  // Final check: ensure even with overrides, all valid insertions store the latest value at that address:
  std::vector<int> coalesced_out(MAX_CHUNKS_SEARCH * CHUNK_SIZE, UNINITIALIZED);
  ca_.coalesce_to(coalesced_out.data(), true);  // Export all valid addresses.
  for (size_t i = 0; i < ref_values.size(); ++i) {
    if (ref_values[i] != UNINITIALIZED) {
      // Test in 2 ways that the values are correctly laid out in memory:
      EXPECT_EQ(ca_.getitem(i / CHUNK_SIZE, i % CHUNK_SIZE, INVALID), ref_values[i]);
      EXPECT_EQ(coalesced_out[i], ref_values[i]);
    }
  }
}
