/*
 * Sampler.h
 *
 *  Created on: Sep 6, 2017
 *      Author: jkim
 */

#ifndef SAMPLER_H_
#define SAMPLER_H_

#include <vector>

#include "Datum.h"
#include "LayerPropParam.h"


// Find all annotated NormalizedBBox.
void GroupObjectBBoxes(const AnnotatedDatum& annoDatum,
                       std::vector<NormalizedBBox>* objectBBoxes);

// Check if a sampled bbox satisfy the constraints with all object bboxes.
bool SatisfySampleConstraint(const NormalizedBBox& sampledBBox,
                             const std::vector<NormalizedBBox>& objectBBoxes,
                             const SampleConstraint& sampleConstraint);

// Sample a NormalizedBBox given the specifictions.
void SampleBBox(const Sampler& sampler, NormalizedBBox* sampledBBox);

// Generate samples from NormalizedBBox using the BatchSampler.
void GenerateSamples(const NormalizedBBox& sourceBBox,
                     const std::vector<NormalizedBBox>& objectBBoxes,
                     const BatchSampler& batchSampler,
                     std::vector<NormalizedBBox>* sampledBBoxes);

// Generate samples from AnnotatedDatum using the BatchSampler.
// All sampled bboxes which satisfy the constraints defined in BatchSampler
// is stored in sampled_bboxes.
void GenerateBatchSamples(const AnnotatedDatum& annoDatum,
                          const std::vector<BatchSampler>& batchSamplers,
                          std::vector<NormalizedBBox>* sampledBBoxes);



#endif /* SAMPLER_H_ */
