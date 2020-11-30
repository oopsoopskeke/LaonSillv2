#include <boost/thread.hpp>

#include "RNG.h"
#include "StdOutLog.h"


static boost::thread_specific_ptr<RandomGenerator> thread_instance_;

RandomGenerator& RandomGenerator::Get() {
	if (!thread_instance_.get()) {
		RandomGenerator* randomGenerator = NULL;
		//SNEW(randomGenerator, RandomGenerator);
		randomGenerator = new RandomGenerator();
		SASSUME0(randomGenerator != NULL);
		thread_instance_.reset(randomGenerator);
	}
	return *(thread_instance_.get());
}




// random seeding
int64_t cluster_seedgen(void) {
	int64_t s, seed, pid;
	FILE* f = fopen("/dev/urandom", "rb");
	if (f && fread(&seed, 1, sizeof(seed), f) == sizeof(seed)) {
		fclose(f);
		return seed;
	}

	STDOUT_LOG("System entropy source not available, using fallback algorithm to generate seed instead.");
	if (f) {
		fclose(f);
	}

	pid = getpid();
	s = time(NULL);
	seed = std::abs(((s * 181) * ((pid - 83) * 359)) % 104729);
	return seed;
}


RandomGenerator::RandomGenerator() : random_generator_() {}

RandomGenerator::~RandomGenerator() {}







class RandomGenerator::RNG::Generator {
public:
	Generator() : rng_(new rng_t(cluster_seedgen())) {}
	explicit Generator(unsigned int seed) : rng_(new rng_t(seed)) {}
	rng_t* rng() { return rng_.get(); }
private:
	boost::shared_ptr<rng_t> rng_;
};

RandomGenerator::RNG::RNG() : generator_(new Generator()) {}
RandomGenerator::RNG::RNG(unsigned int seed) : generator_(new Generator(seed)) {}

RandomGenerator::RNG& RandomGenerator::RNG::operator=(const RNG& other) {
	generator_ = other.generator_;
	return *this;
}

void* RandomGenerator::RNG::generator() {
	return static_cast<void*>(generator_->rng());
}
