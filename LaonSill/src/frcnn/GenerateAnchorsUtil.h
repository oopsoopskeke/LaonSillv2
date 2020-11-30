/*
 * GenerateAnchorsUtil.h
 *
 *  Created on: Nov 18, 2016
 *      Author: jkim
 */

#ifndef GENERATEANCHORSUTIL_H_
#define GENERATEANCHORSUTIL_H_

#include <vector>

#include "frcnn_common.h"

class GenerateAnchorsUtil {
public:
	static void generateAnchors(std::vector<std::vector<float>>& anchors,
			const std::vector<uint32_t>& scales,
			const uint32_t baseSize=16,
			const std::vector<float>& ratios = {0.5f, 1.0f, 2.0f}) {
		// Generate anchor (reference) windows by enumerating aspect ratios X
		// scales wrt a reference (0, 0, 15, 15) window.

		const std::vector<uint32_t> baseAnchor = {0, 0, baseSize-1, baseSize-1};
		std::vector<std::vector<float>> ratioAnchors;
		_ratioEnum(baseAnchor, ratios, ratioAnchors);
		_scaleEnum(scales, ratioAnchors, anchors);
		//_printAnchors(anchors);
	}


private:
	static void _ratioEnum(const std::vector<uint32_t>& anchor,
			const std::vector<float>& ratios, std::vector<std::vector<float>>& anchors) {
		// Enumerate a set of anchors for each aspect ratio wrt an anchor

		uint32_t w, h;
		float xCtr, yCtr;
		_whctrs(anchor, w, h, xCtr, yCtr);

		uint32_t size = w * h;
		std::vector<float> sizeRatios;
		np_scalar_divided_by_array(float(size), ratios, sizeRatios);

		std::vector<float> sqSizeRatios;
		np_sqrt(sizeRatios, sqSizeRatios);
		std::vector<uint32_t> ws;
		np_round(sqSizeRatios, ws);

		std::vector<float> wsRatios;
		np_array_elementwise_mul(ws, ratios, wsRatios);
		std::vector<uint32_t> hs;
		np_round(wsRatios, hs);

		_mkanchors(ws, hs, xCtr, yCtr, anchors);
	}

	template <typename Dtype>
	static void _whctrs(const std::vector<Dtype>& anchor,
			uint32_t& w, uint32_t& h, float& xCtr, float& yCtr) {
		// Return width, height, x center, and y center for an anchor (window).

		w = anchor[2] - anchor[0] + 1;
		h = anchor[3] - anchor[1] + 1;
		xCtr = anchor[0] + 0.5 * (w - 1);
		yCtr = anchor[1] + 0.5 * (h - 1);
	}

	static void _mkanchors(const std::vector<uint32_t>& ws, const std::vector<uint32_t>& hs,
			const float xCtr, const float yCtr,
			std::vector<std::vector<float>>& anchors) {
		// Given a vector of widths (ws) and heights (hs) around a center
		// (xCtr, yCtr), output a set of anchors (windows).

		const uint32_t numAnchors = ws.size();
		anchors.resize(numAnchors);

		for (uint32_t i = 0; i < numAnchors; i++) {
			anchors[i].resize(4);

			anchors[i][0] = xCtr - 0.5f * (ws[i] - 1);
			anchors[i][1] = yCtr - 0.5f * (hs[i] - 1);
			anchors[i][2] = xCtr + 0.5f * (ws[i] - 1);
			anchors[i][3] = yCtr + 0.5f * (hs[i] - 1);
		}
	}

	static void _scaleEnum(const std::vector<uint32_t>& scales,
			std::vector<std::vector<float>>& baseAnchors,
			std::vector<std::vector<float>>& anchors) {
		// Enumerate a set of anchors for each scale wrt an anchor.

		uint32_t w, h;
		float xCtr, yCtr;

		std::vector<std::vector<float>> partialAnchors;
		const uint32_t numAnchors = baseAnchors.size();
		for (uint32_t i = 0; i < numAnchors; i++) {
			std::vector<float>& anchor = baseAnchors[i];
			_whctrs(anchor, w, h, xCtr, yCtr);

			std::vector<uint32_t> ws;
			np_scalar_multiplied_by_array(w, scales, ws);
			std::vector<uint32_t> hs;
			np_scalar_multiplied_by_array(h, scales, hs);

			_mkanchors(ws, hs, xCtr, yCtr, partialAnchors);
			anchors.insert(anchors.end(), partialAnchors.begin(), partialAnchors.end());
		}
	}

	/*
	static void _printAnchors(std::vector<std::vector<float>>& anchors) {
		const uint32_t numAnchors = anchors.size();
		const uint32_t anchorDim = anchors[0].size();

		std::cout << "Anchors: " << std::endl;
		for (uint32_t i = 0; i < numAnchors; i++) {
			std::cout << "[ ";
			for (uint32_t j = 0; j < anchorDim; j++) {
				std::cout << anchors[i][j] << ", ";
			}
			std::cout << "]," << std::endl;
		}
	}
	*/

};

#endif /* GENERATEANCHORSUTIL_H_ */
