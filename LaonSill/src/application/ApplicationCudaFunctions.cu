#include "MathFunctions.h"
#include "Cuda.h"

template <typename Dtype>
__global__ void _diff_content_loss(const uint32_t n, const Dtype* f,
    const Dtype* p, Dtype* df) {
  CUDA_KERNEL_LOOP(index, n) {
	  if (f[index] > 0)
		  df[index] = f[index] - p[index];
	  else
		  df[index] = 0;
  }
}

template <typename Dtype>
void diff_content_loss(const uint32_t n, const Dtype* f,
    const Dtype* p, Dtype* df) {
	_diff_content_loss<Dtype><<<SOOOA_GET_BLOCKS(n), SOOOA_CUDA_NUM_THREADS>>>(
		n, f, p, df);
}

template void diff_content_loss<float>(const uint32_t n, const float* f,
		const float* p, float* df);


template <typename Dtype>
__global__ void _diff_style_loss(const uint32_t n, const Dtype* f, Dtype* a) {
	CUDA_KERNEL_LOOP(index, n) {
		if (f[index] < 0)
			a[index] = 0;
	}
}

template <typename Dtype>
void diff_style_loss(const uint32_t n, const Dtype* f, Dtype* a) {
	_diff_style_loss<Dtype><<<SOOOA_GET_BLOCKS(n), SOOOA_CUDA_NUM_THREADS>>>(
			n, f, a);
}

template void diff_style_loss<float>(const uint32_t n, const float* f, float* a);


template <typename Dtype>
__global__ void _fill_channel_mean(const uint32_t n, const uint32_t singleChannelSize,
		const Dtype* mean, Dtype* dst) {
	CUDA_KERNEL_LOOP(index, n) {
		int channel = index / singleChannelSize;
		dst[index] = mean[channel];
	}
}

//ignore_if_le_than_zero<Dtype><<<SOOOA_GET_BLOCKS(n), SOOOA_CUDA_NUM_THREADS>>>(n, f, a);

template <typename Dtype>
void fill_channel_mean(const uint32_t n, const uint32_t singleChannelSize,
		const Dtype* mean, Dtype* dst) {
	_fill_channel_mean<Dtype><<<SOOOA_GET_BLOCKS(n), SOOOA_CUDA_NUM_THREADS>>>(
			n, singleChannelSize, mean, dst);
}

template void fill_channel_mean(const uint32_t n, const uint32_t singleChannelSize,
		const float* mean, float* dst);



template <typename Dtype>
__global__ void _bound_data(const uint32_t n, const uint32_t singleChannelSize,
		const Dtype* dataMin, const Dtype* dataMax, Dtype* data) {
	CUDA_KERNEL_LOOP(index, n) {
		int channel = index / singleChannelSize;
		if (data[index] > dataMax[channel])
			data[index] = dataMax[channel];
		else if (data[index] < dataMin[channel])
			data[index] = dataMin[channel];
	}
}

template <typename Dtype>
void bound_data(const uint32_t n, const uint32_t singleChannelSize, const Dtype* dataMin,
		const Dtype* dataMax, Dtype* data) {
	_bound_data<Dtype><<<SOOOA_GET_BLOCKS(n), SOOOA_CUDA_NUM_THREADS>>>(n, singleChannelSize,
			dataMin, dataMax, data);
}

template void bound_data(const uint32_t n, const uint32_t singleChannelSize,
		const float* dataMin, const float* dataMax, float* data);





template <typename Dtype>
__global__ void _reset_when_condition_le_0(const uint32_t n, const Dtype* condition, Dtype* data) {
	CUDA_KERNEL_LOOP(index, n) {
		if (condition[index] <= 0)
			data[index] = 0;
	}
}

template <typename Dtype>
void reset_when_condition_le_0(const uint32_t n, const Dtype* condition, Dtype* data) {
	_reset_when_condition_le_0<Dtype><<<SOOOA_GET_BLOCKS(n), SOOOA_CUDA_NUM_THREADS>>>(n,
			condition, data);
}

template void reset_when_condition_le_0(const uint32_t n, const float* condition,
		float* data);






template <typename Dtype>
__global__ void _optimize_adagrad(int n, const Dtype* dx, Dtype* cache, Dtype* x,
    const Dtype lr, const Dtype eps) {
	/****
	 * Adagrad Alogorithm
	 *
	 * cache += dx**2
	 * x += -learning_rate * dx / (sqrt(cache) + eps)
	 *
	 */
	CUDA_KERNEL_LOOP(index, n) {
		cache[index] += dx[index] * dx[index];
		x[index] += (-1.0) * lr * dx[index] / (sqrt(cache[index]) + eps);
	}
}

template <typename Dtype>
void optimize_adagrad(const uint32_t n, const Dtype* dx, Dtype* cache, Dtype* x,
		const Dtype lr, const Dtype eps) {
	_optimize_adagrad<Dtype><<<SOOOA_GET_BLOCKS(n), SOOOA_CUDA_NUM_THREADS>>>(n, dx, cache,
			x, lr, eps);
}

template void optimize_adagrad(const uint32_t n, const float* dx, float* cache, float* x,
		const float lr, const float eps);



template <typename Dtype>
__global__ void _optimize_rmsprop(int n, const Dtype* dx, Dtype* cache, Dtype* x,
    const Dtype lr, const Dtype eps, const Dtype dr) {
    /****
     * RMSprop
     *
     * cache = decay_rate * cache + (1 - decay_rate) * dx**2
     * x += - learning_rate * dx / (sqrt(cache) + eps)
     *
     */
	CUDA_KERNEL_LOOP(index, n) {
		cache[index] = dr * cache[index] + (1.0 - dr) * dx[index] * dx[index];
		x[index] += (-1.0) * lr * dx[index] / (sqrt(cache[index]) + eps);
	}
}

template <typename Dtype>
void optimize_rmsprop(const uint32_t n, const Dtype* dx, Dtype* cache, Dtype* x,
	    const Dtype lr, const Dtype eps, const Dtype dr) {
	_optimize_rmsprop<Dtype><<<SOOOA_GET_BLOCKS(n), SOOOA_CUDA_NUM_THREADS>>>(n, dx, cache,
			x, lr, eps, dr);
}

template void optimize_rmsprop(const uint32_t n, const float* dx, float* cache, float* x,
	    const float lr, const float eps, const float dr);







template <typename Dtype>
__global__ void _optimize_adam(int n, const Dtype* dx, Dtype* m, Dtype* v, Dtype* x,
    const Dtype lr, const Dtype eps, const Dtype beta1, const Dtype beta2) {
    /****
     * Adam
     *
     * m = beta1 * m + (1 - beta1) * dx
     * v = beta2 * v + (1 - beta2) * (dx**2)
     * x += -learning_rate * m / (sqrt(v) + eps)
     *
     */
	CUDA_KERNEL_LOOP(index, n) {
		m[index] = beta1 * m[index] + (1.0 - beta1) * dx[index];
		v[index] = beta2 * v[index] + (1.0 - beta2) * dx[index] * dx[index];
		x[index] += (-1.0) * lr * m[index] / (sqrt(v[index]) + eps);
	}
}


template <typename Dtype>
void optimize_adam(const uint32_t n, const Dtype* dx, Dtype* m, Dtype* v, Dtype* x,
	    const Dtype lr, const Dtype eps, const Dtype beta1, const Dtype beta2) {
	_optimize_adam<Dtype><<<SOOOA_GET_BLOCKS(n), SOOOA_CUDA_NUM_THREADS>>>(n, dx, m, v, x,
			lr, eps, beta1, beta2);
}

template void optimize_adam(const uint32_t n, const float* dx, float* m, float* v, float* x,
	    const float lr, const float eps, const float beta1, const float beta2);




































