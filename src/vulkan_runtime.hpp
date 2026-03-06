#pragma once

#include <cstdint>
#include <string>
#include <vector>

enum class RuntimeOutputTarget {
	Info,
	FinalNir,
	Asm,
};

enum class RuntimeBindingModel : uint8_t {
	Classic,
	PushDescriptor,
	DescriptorBuffer,
	DescriptorHeap,
};

enum class SpirvTargetVersion {
	V10 = 10,
	V11 = 11,
	V12 = 12,
	V13 = 13,
	V14 = 14,
	V15 = 15,
	V16 = 16,
};

enum class RuntimeFailure {
	None = 0,
	UnhandledSpirvExtensions,
	UnhandledSpirvCapabilities,
	VkCreateInstanceFailed,
	VkEnumeratePhysicalDevicesFailed,
	NoPhysicalDevice,
	NoComputeQueue,
	MissingExtensionPushDescriptor,
	MissingExtensionDescriptorBuffer,
	MissingExtensionDescriptorHeap,
	MissingExtensionMaintenance5,
	MissingExtensionShaderUntypedPointers,
	MissingExtensionShaderBfloat16,
	MissingExtensionShaderIntegerDotProduct,
	MissingExtensionRayQuery,
	MissingExtensionAccelerationStructure,
	MissingExtensionDeferredHostOperations,
	MissingExtensionBufferDeviceAddress,
	MissingExtensionShaderClock,
	MissingExtensionCooperativeMatrix,
	MissingExtensionShaderFloatControls2,
	MissingExtensionShaderExpectAssume,
	MissingExtensionShaderSubgroupRotate,
	MissingFeatureStorageBufferArrayNonUniformIndexing,
	MissingFeaturePushDescriptor,
	MissingFeatureShaderUntypedPointers,
	MissingFeatureShaderInt64,
	MissingFeatureShaderInt16,
	MissingFeatureShaderFloat64,
	MissingFeatureShaderInt8,
	MissingFeatureShaderFloat16,
	MissingFeatureShaderFloatControls2,
	MissingFeatureShaderExpectAssume,
	MissingFeatureShaderSubgroupRotate,
	MissingFeatureShaderSubgroupRotateClustered,
	MissingFeatureStorageBuffer8BitAccess,
	MissingFeatureUniformAndStorageBuffer8BitAccess,
	MissingFeatureStoragePushConstant8,
	MissingFeatureStorageBuffer16BitAccess,
	MissingFeatureUniformAndStorageBuffer16BitAccess,
	MissingFeatureStoragePushConstant16,
	MissingFeatureRuntimeDescriptorArray,
	MissingFeatureVulkanMemoryModel,
	MissingFeatureVulkanMemoryModelDeviceScope,
	MissingFeatureShaderUniformBufferArrayNonUniformIndexing,
	MissingFeatureShaderSampledImageArrayNonUniformIndexing,
	MissingFeatureShaderStorageBufferArrayNonUniformIndexing,
	MissingFeatureShaderStorageImageArrayNonUniformIndexing,
	MissingFeatureShaderInputAttachmentArrayNonUniformIndexing,
	MissingFeatureBufferDeviceAddress,
	MissingFeatureSubgroupArithmetic,
	MissingFeatureSubgroupOpsMask,
	MissingExtensionSubgroupSizeControl,
	MissingFeatureSubgroupSizeControl,
	MissingFeatureComputeFullSubgroups,
	RequestedSubgroupSizeOutOfRange,
	RequestedSubgroupSizeStageUnsupported,
	MissingFeatureShaderDeviceClock,
	MissingFeatureShaderSubgroupClock,
	MissingFeatureCooperativeMatrix,
	MissingFeatureShaderBFloat16Type,
	MissingFeatureShaderBFloat16DotProduct,
	MissingFeatureShaderBFloat16CooperativeMatrix,
	MissingFeatureShaderIntegerDotProduct,
	MissingFeatureAccelerationStructure,
	MissingFeatureRayQuery,
	VkCreateDeviceFailed,
	VkCreateShaderModuleFailed,
	VkCreateDescriptorSetLayoutFailed,
	VkCreatePipelineLayoutFailed,
	VkCreateComputePipelineFailed,
	PipelineExecutableUnsupported,
	MissingPipelineExecutableFunctions,
	NoTextInternalRepresentation,
	FailedToSelectFinalNirRepresentation,
	FailedToSelectAsmRepresentation,
	UnexpectedExecutableCount,
};

struct RuntimeDumpOptions {
	RuntimeOutputTarget target = RuntimeOutputTarget::Asm;
};

struct RuntimeConfig {
	RuntimeBindingModel binding_model		   = RuntimeBindingModel::Classic;
	bool				enable_bda			   = false;
	uint32_t			required_subgroup_size = 0;
	bool				require_full_subgroups = false;
	std::string			gpu_key;
};

struct RuntimeResult {
	RuntimeFailure failure = RuntimeFailure::None;
	std::string	   detail;

	[[nodiscard]] bool ok() const { return failure == RuntimeFailure::None; }
};

RuntimeResult run_pipeline_dump(const std::vector<uint32_t> &spirv, const RuntimeDumpOptions &options,
								const RuntimeConfig &config, std::string &out_text);
RuntimeResult run_device_info_dump(const RuntimeConfig &config, std::string &out_text);
RuntimeResult query_max_supported_spirv_version(const RuntimeConfig &config, SpirvTargetVersion &out_version);
