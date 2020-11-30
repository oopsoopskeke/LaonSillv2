/*
 * RNG.h
 *
 *  Created on: Sep 4, 2017
 *      Author: jkim
 */

#ifndef RNG_H_
#define RNG_H_

#include <boost/shared_ptr.hpp>

#include <algorithm>
#include <iterator>

#include "boost/random/mersenne_twister.hpp"
#include "boost/random/uniform_int.hpp"
#include "SysLog.h"
#include "MemoryMgmt.h"

//#include "caffe/common.hpp"

typedef boost::mt19937 rng_t;






class RandomGenerator {
public:
	~RandomGenerator();

	static RandomGenerator& Get();

	class RNG {
	public:
		RNG();
		explicit RNG(unsigned int seed);
		explicit RNG(const RNG&);
		RNG& operator=(const RNG&);
		void* generator();
	private:
		class Generator;
		boost::shared_ptr<Generator> generator_;
	};

	// Getters for boost rng
	inline static RNG& rng_stream() {
		if (!Get().random_generator_) {
			RNG* rng = new RNG();
			SASSUME0(rng != NULL);
			Get().random_generator_.reset(rng);
		}
		return *(Get().random_generator_);
	}

protected:
	boost::shared_ptr<RNG> random_generator_;

private:
	RandomGenerator();
};





inline rng_t* soooa_rng() {
	return static_cast<rng_t*>(RandomGenerator::rng_stream().generator());
}

// Fisher–Yates algorithm
template <class RandomAccessIterator, class RandomGenerator>
inline void shuffle(RandomAccessIterator begin, RandomAccessIterator end,
		RandomGenerator* gen) {
	typedef typename std::iterator_traits<RandomAccessIterator>::difference_type
			difference_type;
	typedef typename boost::uniform_int<difference_type> dist_type;
	difference_type length = std::distance(begin, end);
	if (length <= 0) return;

	for (difference_type i = length - 1; i > 0; --i) {
		dist_type dist(0, i);
		std::iter_swap(begin + i, begin + dist(*gen));
	}
}

template <class RandomAccessIterator>
inline void shuffle(RandomAccessIterator begin, RandomAccessIterator end) {
	shuffle(begin, end, soooa_rng());
}









#endif /* RNG_H_ */
