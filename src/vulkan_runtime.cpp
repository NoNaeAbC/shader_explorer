#include "vulkan_runtime.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <format>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <spirv/unified1/spirv.h>
#include <vulkan/vulkan.h>

/*
 *
 * Warning: All of the feature detection and selection code is AI generated and minimaly reviewed.
 *
 */

namespace {
	RuntimeResult runtime_ok() { return RuntimeResult{}; }

	RuntimeResult runtime_fail(RuntimeFailure failure, std::string detail = {}) {
		RuntimeResult out;
		out.failure = failure;
		out.detail	= std::move(detail);
		return out;
	}

	enum class BindingModel : uint8_t {
		Classic,
		PushDescriptor,
		DescriptorBuffer,
		DescriptorHeap,
	};

	struct ResolvedRuntimeConfig {
		BindingModel binding_model			= BindingModel::Classic;
		bool		 enable_bda				= false;
		uint32_t	 required_subgroup_size = 0;
		bool		 require_full_subgroups = false;
		std::string	 gpu_key;
	};

	struct InstanceScope {
		VkInstance instance = VK_NULL_HANDLE;
		~InstanceScope() {
			if (instance != VK_NULL_HANDLE) { vkDestroyInstance(instance, nullptr); }
		}
	};

	ResolvedRuntimeConfig resolve_runtime_config(const RuntimeConfig &config) {
		ResolvedRuntimeConfig cfg = {
				.binding_model			= BindingModel::Classic,
				.enable_bda				= config.enable_bda,
				.required_subgroup_size = config.required_subgroup_size,
				.require_full_subgroups = config.require_full_subgroups,
				.gpu_key				= config.gpu_key,
		};
		switch (config.binding_model) {
			case RuntimeBindingModel::Classic:
				cfg.binding_model = BindingModel::Classic;
				break;
			case RuntimeBindingModel::PushDescriptor:
				cfg.binding_model = BindingModel::PushDescriptor;
				break;
			case RuntimeBindingModel::DescriptorBuffer:
				cfg.binding_model = BindingModel::DescriptorBuffer;
				break;
			case RuntimeBindingModel::DescriptorHeap:
				cfg.binding_model = BindingModel::DescriptorHeap;
				break;
		}
		return cfg;
	}

	int select_compute_queue(VkPhysicalDevice device) {
		uint32_t family_count = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &family_count, nullptr);
		if (family_count == 0) { return -1; }
		std::vector<VkQueueFamilyProperties> families(family_count);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &family_count, families.data());
		for (uint32_t i = 0; i < family_count; ++i) {
			if ((families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0U) { return static_cast<int>(i); }
		}
		return -1;
	}

	bool device_supports_extension(VkPhysicalDevice device, const char *extension_name) {
		uint32_t extension_count = 0;
		if (vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, nullptr) != VK_SUCCESS) {
			return false;
		}

		std::vector<VkExtensionProperties> extensions(extension_count);
		if (vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, extensions.data()) != VK_SUCCESS) {
			return false;
		}

		for (const auto &extension: extensions) {
			if (std::strcmp(extension.extensionName, extension_name) == 0) { return true; }
		}
		return false;
	}

	struct ReflectedBinding {
		uint32_t		 set	 = 0;
		uint32_t		 binding = 0;
		VkDescriptorType type	 = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		uint32_t		 count	 = 1;
	};

	struct TypeInfo {
		enum class Kind : uint8_t {
			Unknown,
			Pointer,
			Struct,
			Array,
			RuntimeArray,
			Sampler,
			Image,
			SampledImage,
			AccelStruct,
		};
		Kind	 kind		   = Kind::Unknown;
		uint32_t elem_type	   = 0;
		uint32_t storage_class = 0;
		uint32_t sampled	   = 0;
		uint32_t length_id	   = 0;
	};

	struct ShaderRequirements {
		bool				  needs_int64							  = false;
		bool				  needs_int16							  = false;
		bool				  needs_int8							  = false;
		bool				  needs_float16							  = false;
		bool				  needs_float64							  = false;
		bool				  needs_shader_float_controls2			  = false;
		bool				  needs_shader_expect_assume			  = false;
		bool				  needs_storage_buffer_8bit				  = false;
		bool				  needs_uniform_storage_buffer_8bit		  = false;
		bool				  needs_storage_push_constant_8bit		  = false;
		bool				  needs_storage_buffer_16bit			  = false;
		bool				  needs_uniform_storage_buffer_16bit	  = false;
		bool				  needs_storage_push_constant_16bit		  = false;
		bool				  needs_runtime_descriptor_array		  = false;
		bool				  needs_uniform_buffer_array_nonuniform	  = false;
		bool				  needs_sampled_image_array_nonuniform	  = false;
		bool				  needs_storage_buffer_array_nonuniform	  = false;
		bool				  needs_storage_image_array_nonuniform	  = false;
		bool				  needs_input_attachment_array_nonuniform = false;
		bool				  needs_physical_storage_buffer_addresses = false;
		bool				  needs_shader_untyped_pointers			  = false;
		bool				  needs_descriptor_heap					  = false;
		bool				  needs_integer_dot_product				  = false;
		bool				  needs_vulkan_memory_model				  = false;
		bool				  needs_vulkan_memory_model_device_scope  = false;
		bool				  needs_bfloat16_type					  = false;
		bool				  needs_bfloat16_dot_product			  = false;
		bool				  needs_bfloat16_cooperative_matrix		  = false;
		bool				  needs_ray_query						  = false;
		bool				  uses_shader_clock						  = false;
		bool				  needs_shader_device_clock				  = false;
		bool				  needs_shader_subgroup_clock			  = false;
		bool				  uses_cooperative_matrix				  = false;
		uint32_t			  required_subgroup_ops					  = 0;
		std::set<std::string> unhandled_extensions;
		std::set<uint32_t>	  unhandled_capabilities;
	};

	std::string decode_spirv_string(const uint32_t *words, uint16_t word_count) {
		std::string out;
		for (uint16_t i = 0; i < word_count; ++i) {
			uint32_t const w = words[i];
			for (int byte = 0; byte < 4; ++byte) {
				char const c = static_cast<char>((w >> (byte * 8)) & 0xFFU);
				if (c == '\0') { return out; }
				out.push_back(c);
			}
		}
		return out;
	}

	std::string_view capability_name(uint32_t cap) {
		using namespace std::string_view_literals;

		switch (cap) {
			case SpvCapabilityMatrix:
				return "Matrix"sv;
			case SpvCapabilityShader:
				return "Shader"sv;
			case SpvCapabilityInt64:
				return "Int64"sv;
			case SpvCapabilityInt16:
				return "Int16"sv;
			case SpvCapabilityInt8:
				return "Int8"sv;
			case SpvCapabilityFloat16:
				return "Float16"sv;
			case SpvCapabilityFloat64:
				return "Float64"sv;
			case SpvCapabilityStorageBuffer8BitAccess:
				return "StorageBuffer8BitAccess"sv;
			case SpvCapabilityUniformAndStorageBuffer8BitAccess:
				return "UniformAndStorageBuffer8BitAccess"sv;
			case SpvCapabilityStoragePushConstant8:
				return "StoragePushConstant8"sv;
			case SpvCapabilityStorageBuffer16BitAccess:
				return "StorageBuffer16BitAccess"sv;
			case SpvCapabilityUniformAndStorageBuffer16BitAccess:
				return "UniformAndStorageBuffer16BitAccess"sv;
			case SpvCapabilityStoragePushConstant16:
				return "StoragePushConstant16"sv;
			case SpvCapabilityRuntimeDescriptorArray:
				return "RuntimeDescriptorArray"sv;
			case SpvCapabilityUniformBufferArrayNonUniformIndexing:
				return "UniformBufferArrayNonUniformIndexing"sv;
			case SpvCapabilitySampledImageArrayNonUniformIndexing:
				return "SampledImageArrayNonUniformIndexing"sv;
			case SpvCapabilityStorageBufferArrayNonUniformIndexing:
				return "StorageBufferArrayNonUniformIndexing"sv;
			case SpvCapabilityStorageImageArrayNonUniformIndexing:
				return "StorageImageArrayNonUniformIndexing"sv;
			case SpvCapabilityInputAttachmentArrayNonUniformIndexing:
				return "InputAttachmentArrayNonUniformIndexing"sv;
			case SpvCapabilityPhysicalStorageBufferAddresses:
				return "PhysicalStorageBufferAddresses"sv;
			case SpvCapabilityUntypedPointersKHR:
				return "UntypedPointersKHR"sv;
			case SpvCapabilityDescriptorHeapEXT:
				return "DescriptorHeapEXT"sv;
			case SpvCapabilityExpectAssumeKHR:
				return "ExpectAssumeKHR"sv;
			case SpvCapabilityShaderNonUniform:
				return "ShaderNonUniform"sv;
			case SpvCapabilityShaderClockKHR:
				return "ShaderClockKHR"sv;
			case SpvCapabilityFloatControls2:
				return "FloatControls2"sv;
			case SpvCapabilityCooperativeMatrixKHR:
				return "CooperativeMatrixKHR"sv;
			case SpvCapabilityBFloat16TypeKHR:
				return "BFloat16TypeKHR"sv;
			case SpvCapabilityBFloat16DotProductKHR:
				return "BFloat16DotProductKHR"sv;
			case SpvCapabilityBFloat16CooperativeMatrixKHR:
				return "BFloat16CooperativeMatrixKHR"sv;
			case SpvCapabilityRayQueryProvisionalKHR:
				return "RayQueryProvisionalKHR"sv;
			case SpvCapabilityRayQueryKHR:
				return "RayQueryKHR"sv;
			case SpvCapabilityRayCullMaskKHR:
				return "RayCullMaskKHR"sv;
			case SpvCapabilityDotProductInputAll:
				return "DotProductInputAll"sv;
			case SpvCapabilityDotProductInput4x8Bit:
				return "DotProductInput4x8Bit"sv;
			case SpvCapabilityDotProductInput4x8BitPacked:
				return "DotProductInput4x8BitPacked"sv;
			case SpvCapabilityDotProduct:
				return "DotProduct"sv;
			case SpvCapabilityVulkanMemoryModel:
				return "VulkanMemoryModel"sv;
			case SpvCapabilityVulkanMemoryModelDeviceScope:
				return "VulkanMemoryModelDeviceScope"sv;
			case SpvCapabilityGroupNonUniform:
				return "GroupNonUniform"sv;
			case SpvCapabilityGroupNonUniformVote:
				return "GroupNonUniformVote"sv;
			case SpvCapabilityGroupNonUniformArithmetic:
				return "GroupNonUniformArithmetic"sv;
			case SpvCapabilityGroupNonUniformBallot:
				return "GroupNonUniformBallot"sv;
			case SpvCapabilityGroupNonUniformShuffle:
				return "GroupNonUniformShuffle"sv;
			case SpvCapabilityGroupNonUniformShuffleRelative:
				return "GroupNonUniformShuffleRelative"sv;
			case SpvCapabilityGroupNonUniformClustered:
				return "GroupNonUniformClustered"sv;
			case SpvCapabilityGroupNonUniformQuad:
				return "GroupNonUniformQuad"sv;
			case SpvCapabilityGroupNonUniformPartitionedEXT:
				return "GroupNonUniformPartitionedEXT"sv;
			case SpvCapabilityGroupNonUniformRotateKHR:
				return "GroupNonUniformRotateKHR"sv;
			default:
				return "UnknownCapability"sv;
		}
	}

	std::string format_unhandled_capabilities(const std::set<uint32_t> &caps) {
		if (caps.empty()) { return {}; }
		std::string out;
		bool		first = true;
		for (uint32_t const cap: caps) {
			if (!first) { out += ", "; }
			first = false;
			out += std::format("{}({})", capability_name(cap), cap);
		}
		return out;
	}

	std::string format_unhandled_extensions(const std::set<std::string> &exts) {
		if (exts.empty()) { return {}; }
		std::string out;
		bool		first = true;
		for (const std::string &ext: exts) {
			if (!first) { out += ", "; }
			first = false;
			out += ext;
		}
		return out;
	}

	ShaderRequirements analyze_spirv_requirements(const std::vector<uint32_t> &spirv) {
		ShaderRequirements req;
		if (spirv.size() < 5) { return req; }

		std::unordered_map<uint32_t, uint32_t> constants;
		std::vector<uint32_t>				   read_clock_scope_ids;

		size_t idx = 5;
		while (idx < spirv.size()) {
			uint32_t const word0 = spirv[idx];
			auto const	   wc	 = static_cast<uint16_t>(word0 >> 16);
			auto const	   op	 = static_cast<uint16_t>(word0 & 0xFFFFU);
			if (wc == 0 || idx + wc > spirv.size()) { break; }
			const uint32_t *w = &spirv[idx];

			if (op == SpvOpCapability && wc >= 2) {
				switch (w[1]) {
					case SpvCapabilityMatrix:
					case SpvCapabilityShader:
						break;
					case SpvCapabilityGroupNonUniform:
						req.required_subgroup_ops |= VK_SUBGROUP_FEATURE_BASIC_BIT;
						break;
					case SpvCapabilityInt64:
						req.needs_int64 = true;
						break;
					case SpvCapabilityInt16:
						req.needs_int16 = true;
						break;
					case SpvCapabilityInt8:
						req.needs_int8 = true;
						break;
					case SpvCapabilityFloat16:
						req.needs_float16 = true;
						break;
					case SpvCapabilityFloat64:
						req.needs_float64 = true;
						break;
					case SpvCapabilityFloatControls2:
						req.needs_shader_float_controls2 = true;
						break;
					case SpvCapabilityStorageBuffer8BitAccess:
						req.needs_storage_buffer_8bit = true;
						break;
					case SpvCapabilityUniformAndStorageBuffer8BitAccess:
						req.needs_uniform_storage_buffer_8bit = true;
						break;
					case SpvCapabilityStoragePushConstant8:
						req.needs_storage_push_constant_8bit = true;
						break;
					case SpvCapabilityStorageBuffer16BitAccess:
						req.needs_storage_buffer_16bit = true;
						break;
					case SpvCapabilityUniformAndStorageBuffer16BitAccess:
						req.needs_uniform_storage_buffer_16bit = true;
						break;
					case SpvCapabilityStoragePushConstant16:
						req.needs_storage_push_constant_16bit = true;
						break;
					case SpvCapabilityRuntimeDescriptorArray:
						req.needs_runtime_descriptor_array = true;
						break;
					case SpvCapabilityShaderNonUniform:
						// Marker capability for non-uniform indexing model; specific indexed
						// capabilities below drive concrete feature requirements.
						break;
					case SpvCapabilityUniformBufferArrayNonUniformIndexing:
						req.needs_uniform_buffer_array_nonuniform = true;
						break;
					case SpvCapabilitySampledImageArrayNonUniformIndexing:
						req.needs_sampled_image_array_nonuniform = true;
						break;
					case SpvCapabilityStorageBufferArrayNonUniformIndexing:
						req.needs_storage_buffer_array_nonuniform = true;
						break;
					case SpvCapabilityStorageImageArrayNonUniformIndexing:
						req.needs_storage_image_array_nonuniform = true;
						break;
					case SpvCapabilityInputAttachmentArrayNonUniformIndexing:
						req.needs_input_attachment_array_nonuniform = true;
						break;
					case SpvCapabilityPhysicalStorageBufferAddresses:
						req.needs_physical_storage_buffer_addresses = true;
						break;
					case SpvCapabilityUntypedPointersKHR:
						req.needs_shader_untyped_pointers = true;
						break;
					case SpvCapabilityDescriptorHeapEXT:
						req.needs_descriptor_heap		  = true;
						req.needs_shader_untyped_pointers = true;
						break;
					case SpvCapabilityExpectAssumeKHR:
						req.needs_shader_expect_assume = true;
						break;
					case SpvCapabilityDotProductInputAll:
					case SpvCapabilityDotProductInput4x8Bit:
					case SpvCapabilityDotProductInput4x8BitPacked:
					case SpvCapabilityDotProduct:
						req.needs_integer_dot_product = true;
						break;
					case SpvCapabilityVulkanMemoryModel:
						req.needs_vulkan_memory_model = true;
						break;
					case SpvCapabilityVulkanMemoryModelDeviceScope:
						req.needs_vulkan_memory_model			   = true;
						req.needs_vulkan_memory_model_device_scope = true;
						break;
					case SpvCapabilityBFloat16TypeKHR:
						req.needs_bfloat16_type = true;
						break;
					case SpvCapabilityBFloat16DotProductKHR:
						req.needs_bfloat16_dot_product = true;
						break;
					case SpvCapabilityBFloat16CooperativeMatrixKHR:
						req.needs_bfloat16_cooperative_matrix = true;
						req.uses_cooperative_matrix			  = true;
						break;
					case SpvCapabilityRayQueryProvisionalKHR:
					case SpvCapabilityRayQueryKHR:
					case SpvCapabilityRayCullMaskKHR:
						req.needs_ray_query = true;
						break;
					case SpvCapabilityCooperativeMatrixKHR:
						req.uses_cooperative_matrix = true;
						break;
					case SpvCapabilityGroupNonUniformVote:
						req.required_subgroup_ops |= VK_SUBGROUP_FEATURE_BASIC_BIT;
						req.required_subgroup_ops |= VK_SUBGROUP_FEATURE_VOTE_BIT;
						break;
					case SpvCapabilityGroupNonUniformArithmetic:
						req.required_subgroup_ops |= VK_SUBGROUP_FEATURE_BASIC_BIT;
						req.required_subgroup_ops |= VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;
						break;
					case SpvCapabilityGroupNonUniformBallot:
						req.required_subgroup_ops |= VK_SUBGROUP_FEATURE_BASIC_BIT;
						req.required_subgroup_ops |= VK_SUBGROUP_FEATURE_BALLOT_BIT;
						break;
					case SpvCapabilityGroupNonUniformShuffle:
						req.required_subgroup_ops |= VK_SUBGROUP_FEATURE_BASIC_BIT;
						req.required_subgroup_ops |= VK_SUBGROUP_FEATURE_SHUFFLE_BIT;
						break;
					case SpvCapabilityGroupNonUniformShuffleRelative:
						req.required_subgroup_ops |= VK_SUBGROUP_FEATURE_BASIC_BIT;
						req.required_subgroup_ops |= VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT;
						break;
					case SpvCapabilityGroupNonUniformClustered:
						req.required_subgroup_ops |= VK_SUBGROUP_FEATURE_BASIC_BIT;
						req.required_subgroup_ops |= VK_SUBGROUP_FEATURE_CLUSTERED_BIT;
						break;
					case SpvCapabilityGroupNonUniformQuad:
						req.required_subgroup_ops |= VK_SUBGROUP_FEATURE_BASIC_BIT;
						req.required_subgroup_ops |= VK_SUBGROUP_FEATURE_QUAD_BIT;
						break;
					case SpvCapabilityGroupNonUniformPartitionedEXT:
						req.required_subgroup_ops |= VK_SUBGROUP_FEATURE_BASIC_BIT;
						req.required_subgroup_ops |= VK_SUBGROUP_FEATURE_PARTITIONED_BIT_EXT;
						break;
					case SpvCapabilityGroupNonUniformRotateKHR:
						req.required_subgroup_ops |= VK_SUBGROUP_FEATURE_BASIC_BIT;
						req.required_subgroup_ops |= VK_SUBGROUP_FEATURE_ROTATE_BIT;
						break;
					default:
						req.unhandled_capabilities.insert(w[1]);
						break;
				}
			} else if (op == SpvOpGroupNonUniformRotateKHR) {
				req.required_subgroup_ops |= VK_SUBGROUP_FEATURE_BASIC_BIT;
				req.required_subgroup_ops |= VK_SUBGROUP_FEATURE_ROTATE_BIT;
				if (wc >= 7) { req.required_subgroup_ops |= VK_SUBGROUP_FEATURE_ROTATE_CLUSTERED_BIT; }
			} else if (op == SpvOpExtension && wc >= 2) {
				std::string const ext = decode_spirv_string(&w[1], static_cast<uint16_t>(wc - 1));
				if (ext == "SPV_KHR_shader_clock") {
					req.uses_shader_clock = true;
				} else if (ext == "SPV_KHR_cooperative_matrix") {
					req.uses_cooperative_matrix = true;
				} else if (ext == "SPV_KHR_integer_dot_product") {
					req.needs_integer_dot_product = true;
				} else if (ext == "SPV_KHR_bfloat16") {
					// Fine to accept extension here; capability parsing determines exact feature bits.
				} else if (ext == "SPV_KHR_storage_buffer_storage_class") {
					// Legacy storage class extension; no extra runtime feature gate needed here.
				} else if (ext == "SPV_KHR_ray_query") {
					req.needs_ray_query = true;
				} else if (ext == "SPV_KHR_untyped_pointers") {
					req.needs_shader_untyped_pointers = true;
				} else if (ext == "SPV_EXT_descriptor_heap") {
					req.needs_descriptor_heap				  = true;
					req.needs_storage_buffer_array_nonuniform = true;
					req.needs_shader_untyped_pointers		  = true;
				} else if (ext == "SPV_KHR_expect_assume") {
					req.needs_shader_expect_assume = true;
				} else if (ext == "SPV_KHR_float_controls2") {
					req.needs_shader_float_controls2 = true;
				} else {
					req.unhandled_extensions.insert(ext);
				}
			} else if (op == SpvOpConstant && wc >= 4) {
				constants[w[2]] = w[3];
			} else if (op == SpvOpReadClockKHR && wc >= 4) {
				read_clock_scope_ids.push_back(w[3]);
				req.uses_shader_clock = true;
			}
			idx += wc;
		}

		for (uint32_t const scope_id: read_clock_scope_ids) {
			auto it = constants.find(scope_id);
			if (it == constants.end()) { continue; }
			if (it->second == SpvScopeDevice) {
				req.needs_shader_device_clock = true;
			} else if (it->second == SpvScopeSubgroup) {
				req.needs_shader_subgroup_clock = true;
			}
		}

		if (req.uses_shader_clock && !req.needs_shader_device_clock && !req.needs_shader_subgroup_clock) {
			req.needs_shader_device_clock	= true;
			req.needs_shader_subgroup_clock = true;
		}

		return req;
	}

	uint32_t resolve_descriptor_count(const std::unordered_map<uint32_t, TypeInfo> &types,
									  const std::unordered_map<uint32_t, uint64_t> &constants, uint32_t &type_id) {
		uint32_t count = 1;
		while (true) {
			auto it = types.find(type_id);
			if (it == types.end()) { break; }
			const TypeInfo &t = it->second;
			if (t.kind == TypeInfo::Kind::Array) {
				auto c_it = constants.find(t.length_id);
				count	  = c_it == constants.end() ? 1 : static_cast<uint32_t>(c_it->second);
				type_id	  = t.elem_type;
				continue;
			}
			if (t.kind == TypeInfo::Kind::RuntimeArray) {
				count	= 1;
				type_id = t.elem_type;
				continue;
			}
			break;
		}
		return count == 0 ? 1 : count;
	}

	std::vector<ReflectedBinding> reflect_descriptor_bindings(const std::vector<uint32_t> &spirv) {
		if (spirv.size() < 5) { return {}; }


		std::unordered_map<uint32_t, TypeInfo> types;
		std::unordered_map<uint32_t, uint64_t> constants;
		std::unordered_map<uint32_t, uint32_t> binding_decoration;
		std::unordered_map<uint32_t, uint32_t> set_decoration;
		std::set<uint32_t>					   block_structs;
		std::set<uint32_t>					   buffer_block_structs;
		struct VarInfo {
			uint32_t pointer_type  = 0;
			uint32_t storage_class = 0;
		};
		std::unordered_map<uint32_t, VarInfo> variables;

		size_t idx = 5;
		while (idx < spirv.size()) {
			uint32_t const word0 = spirv[idx];
			auto const	   wc	 = uint16_t(word0 >> 16);
			auto const	   op	 = uint16_t(word0 & 0xFFFFU);
			if (wc == 0 || idx + wc > spirv.size()) { break; }
			const uint32_t *w = &spirv[idx];
			switch (op) {
				case SpvOpDecorate:
					if (wc >= 3) {
						uint32_t const target = w[1];
						uint32_t const deco	  = w[2];
						if (deco == SpvDecorationBinding && wc >= 4) {
							binding_decoration[target] = w[3];
						} else if (deco == SpvDecorationDescriptorSet && wc >= 4) {
							set_decoration[target] = w[3];
						} else if (deco == SpvDecorationBlock) {
							block_structs.insert(target);
						} else if (deco == SpvDecorationBufferBlock) {
							buffer_block_structs.insert(target);
						}
					}
					break;
				case SpvOpTypePointer:
					if (wc >= 4) {
						TypeInfo t;
						t.kind			= TypeInfo::Kind::Pointer;
						t.storage_class = w[2];
						t.elem_type		= w[3];
						types[w[1]]		= t;
					}
					break;
				case SpvOpTypeStruct:
					if (wc >= 2) {
						TypeInfo t;
						t.kind		= TypeInfo::Kind::Struct;
						types[w[1]] = t;
					}
					break;
				case SpvOpTypeArray:
					if (wc >= 4) {
						TypeInfo t;
						t.kind		= TypeInfo::Kind::Array;
						t.elem_type = w[2];
						t.length_id = w[3];
						types[w[1]] = t;
					}
					break;
				case SpvOpTypeRuntimeArray:
					if (wc >= 3) {
						TypeInfo t;
						t.kind		= TypeInfo::Kind::RuntimeArray;
						t.elem_type = w[2];
						types[w[1]] = t;
					}
					break;
				case SpvOpTypeSampler:
					if (wc >= 2) {
						TypeInfo t;
						t.kind		= TypeInfo::Kind::Sampler;
						types[w[1]] = t;
					}
					break;
				case SpvOpTypeImage:
					if (wc >= 8) {
						TypeInfo t;
						t.kind		= TypeInfo::Kind::Image;
						t.sampled	= w[7];
						types[w[1]] = t;
					}
					break;
				case SpvOpTypeSampledImage:
					if (wc >= 3) {
						TypeInfo t;
						t.kind		= TypeInfo::Kind::SampledImage;
						t.elem_type = w[2];
						types[w[1]] = t;
					}
					break;
				case SpvOpTypeAccelerationStructureKHR:
					if (wc >= 2) {
						TypeInfo t;
						t.kind		= TypeInfo::Kind::AccelStruct;
						types[w[1]] = t;
					}
					break;
				case SpvOpConstant:
					if (wc >= 4) { constants[w[2]] = w[3]; }
					break;
				case SpvOpVariable:
					if (wc >= 4) { variables[w[2]] = VarInfo{.pointer_type = w[1], .storage_class = w[3]}; }
					break;
				default:
					break;
			}
			idx += wc;
		}

		std::vector<ReflectedBinding> out;
		for (const auto &[var_id, var]: variables) {
			auto b_it = binding_decoration.find(var_id);
			auto s_it = set_decoration.find(var_id);
			if (b_it == binding_decoration.end() || s_it == set_decoration.end()) { continue; }
			auto p_it = types.find(var.pointer_type);
			if (p_it == types.end() || p_it->second.kind != TypeInfo::Kind::Pointer) { continue; }

			uint32_t	   base_type		= p_it->second.elem_type;
			uint32_t const descriptor_count = resolve_descriptor_count(types, constants, base_type);
			auto		   t_it				= types.find(base_type);
			if (t_it == types.end()) { continue; }

			VkDescriptorType desc_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			if (var.storage_class == SpvStorageClassStorageBuffer) {
				desc_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			} else if (var.storage_class == SpvStorageClassUniform) {
				if (buffer_block_structs.contains(base_type)) {
					desc_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
				} else if (block_structs.contains(base_type)) {
					desc_type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
				} else {
					continue;
				}
			} else if (var.storage_class == SpvStorageClassUniformConstant) {
				if (t_it->second.kind == TypeInfo::Kind::Sampler) {
					desc_type = VK_DESCRIPTOR_TYPE_SAMPLER;
				} else if (t_it->second.kind == TypeInfo::Kind::SampledImage) {
					desc_type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
				} else if (t_it->second.kind == TypeInfo::Kind::Image) {
					desc_type = t_it->second.sampled == 2 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
														  : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
				} else if (t_it->second.kind == TypeInfo::Kind::AccelStruct) {
					desc_type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
				} else {
					continue;
				}
			} else {
				continue;
			}

			out.push_back(ReflectedBinding{
					.set	 = s_it->second,
					.binding = b_it->second,
					.type	 = desc_type,
					.count	 = descriptor_count,
			});
		}

		return out;
	}

} // namespace

RuntimeResult run_device_info_dump(const RuntimeConfig &config, std::string &out_text) {
	std::string out;

	VkApplicationInfo const app_info = {
			.sType				= VK_STRUCTURE_TYPE_APPLICATION_INFO,
			.pNext				= nullptr,
			.pApplicationName	= "shader_explorer",
			.applicationVersion = 1,
			.pEngineName		= "shader_explorer engine",
			.engineVersion		= 1,
			.apiVersion			= VK_API_VERSION_1_4,
	};

	InstanceScope			   scope;
	VkInstanceCreateInfo const instance_ci = {
			.sType					 = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
			.pNext					 = nullptr,
			.flags					 = 0,
			.pApplicationInfo		 = &app_info,
			.enabledLayerCount		 = 0,
			.ppEnabledLayerNames	 = nullptr,
			.enabledExtensionCount	 = 0,
			.ppEnabledExtensionNames = nullptr,
	};
	if (vkCreateInstance(&instance_ci, nullptr, &scope.instance) != VK_SUCCESS) {
		return runtime_fail(RuntimeFailure::VkCreateInstanceFailed);
	}

	uint32_t	   device_count		= 0;
	VkResult const enumerate_result = vkEnumeratePhysicalDevices(scope.instance, &device_count, nullptr);
	if (enumerate_result != VK_SUCCESS) { return runtime_fail(RuntimeFailure::VkEnumeratePhysicalDevicesFailed); }
	if (device_count == 0) {
		std::string detail = "device_count=0";
		if (!config.gpu_key.empty()) {
			detail += ", gpu_preset=";
			detail += config.gpu_key;
		}
		return runtime_fail(RuntimeFailure::NoPhysicalDevice, detail);
	}

	std::vector<VkPhysicalDevice> devices(device_count);
	vkEnumeratePhysicalDevices(scope.instance, &device_count, devices.data());
	VkPhysicalDevice physical_device = devices[0];

	VkPhysicalDeviceSubgroupProperties subgroup_props{};
	subgroup_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
	subgroup_props.pNext = nullptr;
	VkPhysicalDeviceProperties2 props2{};
	props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
	props2.pNext = &subgroup_props;
	vkGetPhysicalDeviceProperties2(physical_device, &props2);

	const VkPhysicalDeviceProperties &props = props2.properties;

	out += std::format("device.name={}\n", props.deviceName);
	out += std::format("device.vendor_id={}\n", props.vendorID);
	out += std::format("device.device_id={}\n", props.deviceID);
	out += std::format("device.api_version={}.{}.{}\n", VK_API_VERSION_MAJOR(props.apiVersion),
					   VK_API_VERSION_MINOR(props.apiVersion), VK_API_VERSION_PATCH(props.apiVersion));
	out += std::format("limits.max_compute_shared_memory_size={}\n", props.limits.maxComputeSharedMemorySize);
	out += std::format("limits.max_compute_work_group_invocations={}\n", props.limits.maxComputeWorkGroupInvocations);
	out += std::format("limits.max_compute_work_group_size={},{},{}\n", props.limits.maxComputeWorkGroupSize[0],
					   props.limits.maxComputeWorkGroupSize[1], props.limits.maxComputeWorkGroupSize[2]);
	out += std::format("subgroup.size={}\n", subgroup_props.subgroupSize);
	out += std::format("subgroup.supported_stages=0x{:x}\n", subgroup_props.supportedStages);
	out += std::format("subgroup.supported_operations=0x{:x}\n", subgroup_props.supportedOperations);
	out += std::format("subgroup.supports_arithmetic={}\n",
					   ((subgroup_props.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0U) ? "true"
																										 : "false");
	out += std::format("subgroup.quad_operations_in_all_stages={}\n",
					   subgroup_props.quadOperationsInAllStages == VK_TRUE ? "true" : "false");

	VkPhysicalDeviceRayQueryFeaturesKHR ray_query_features = {
			.sType	  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR,
			.pNext	  = nullptr,
			.rayQuery = VK_FALSE,
	};
	VkPhysicalDeviceAccelerationStructureFeaturesKHR accel_struct_features = {
			.sType								= VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
			.pNext								= &ray_query_features,
			.accelerationStructure				= VK_FALSE,
			.accelerationStructureCaptureReplay = VK_FALSE,
			.accelerationStructureIndirectBuild = VK_FALSE,
			.accelerationStructureHostCommands	= VK_FALSE,
			.descriptorBindingAccelerationStructureUpdateAfterBind = VK_FALSE,
	};
	VkPhysicalDeviceShaderIntegerDotProductFeatures integer_dot_product_features = {
			.sType					 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES,
			.pNext					 = &accel_struct_features,
			.shaderIntegerDotProduct = VK_FALSE,
	};
	VkPhysicalDeviceShaderBfloat16FeaturesKHR bfloat16_features = {
			.sType							 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR,
			.pNext							 = &integer_dot_product_features,
			.shaderBFloat16Type				 = VK_FALSE,
			.shaderBFloat16DotProduct		 = VK_FALSE,
			.shaderBFloat16CooperativeMatrix = VK_FALSE,
	};
	VkPhysicalDeviceCooperativeMatrixFeaturesKHR cooperative_matrix_features = {
			.sType								 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
			.pNext								 = &bfloat16_features,
			.cooperativeMatrix					 = VK_FALSE,
			.cooperativeMatrixRobustBufferAccess = VK_FALSE,
	};
	VkPhysicalDeviceShaderClockFeaturesKHR shader_clock_features = {
			.sType				 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR,
			.pNext				 = &cooperative_matrix_features,
			.shaderSubgroupClock = VK_FALSE,
			.shaderDeviceClock	 = VK_FALSE,
	};
	VkPhysicalDeviceShaderUntypedPointersFeaturesKHR untyped_ptr_features = {
			.sType				   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_UNTYPED_POINTERS_FEATURES_KHR,
			.pNext				   = &shader_clock_features,
			.shaderUntypedPointers = VK_FALSE,
	};
	VkPhysicalDeviceDescriptorHeapFeaturesEXT descriptor_heap_features{};
	descriptor_heap_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT;
	descriptor_heap_features.pNext = &untyped_ptr_features;
	VkPhysicalDeviceDescriptorBufferFeaturesEXT descriptor_buffer_features = {
			.sType								= VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT,
			.pNext								= &descriptor_heap_features,
			.descriptorBuffer					= VK_FALSE,
			.descriptorBufferCaptureReplay		= VK_FALSE,
			.descriptorBufferImageLayoutIgnored = VK_FALSE,
			.descriptorBufferPushDescriptors	= VK_FALSE,
	};
	VkPhysicalDeviceSubgroupSizeControlFeatures subgroup_size_control = {
			.sType				  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES,
			.pNext				  = &descriptor_buffer_features,
			.subgroupSizeControl  = VK_FALSE,
			.computeFullSubgroups = VK_FALSE,
	};
	VkPhysicalDeviceShaderSubgroupRotateFeatures subgroup_rotate_features = {
			.sType						   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_ROTATE_FEATURES,
			.pNext						   = &subgroup_size_control,
			.shaderSubgroupRotate		   = VK_FALSE,
			.shaderSubgroupRotateClustered = VK_FALSE,
	};
	VkPhysicalDeviceShaderFloatControls2Features shader_float_controls2_features = {
			.sType				  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT_CONTROLS_2_FEATURES,
			.pNext				  = &subgroup_rotate_features,
			.shaderFloatControls2 = VK_FALSE,
	};
	VkPhysicalDeviceShaderExpectAssumeFeatures shader_expect_assume_features = {
			.sType				= VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_EXPECT_ASSUME_FEATURES,
			.pNext				= &shader_float_controls2_features,
			.shaderExpectAssume = VK_FALSE,
	};
	VkPhysicalDeviceVulkan14Features vk14 = {
			.sType									= VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES,
			.pNext									= &shader_expect_assume_features,
			.globalPriorityQuery					= VK_FALSE,
			.shaderSubgroupRotate					= VK_FALSE,
			.shaderSubgroupRotateClustered			= VK_FALSE,
			.shaderFloatControls2					= VK_FALSE,
			.shaderExpectAssume						= VK_FALSE,
			.rectangularLines						= VK_FALSE,
			.bresenhamLines							= VK_FALSE,
			.smoothLines							= VK_FALSE,
			.stippledRectangularLines				= VK_FALSE,
			.stippledBresenhamLines					= VK_FALSE,
			.stippledSmoothLines					= VK_FALSE,
			.vertexAttributeInstanceRateDivisor		= VK_FALSE,
			.vertexAttributeInstanceRateZeroDivisor = VK_FALSE,
			.indexTypeUint8							= VK_FALSE,
			.dynamicRenderingLocalRead				= VK_FALSE,
			.maintenance5							= VK_FALSE,
			.maintenance6							= VK_FALSE,
			.pipelineProtectedAccess				= VK_FALSE,
			.pipelineRobustness						= VK_FALSE,
			.hostImageCopy							= VK_FALSE,
			.pushDescriptor							= VK_FALSE,
	};
	VkPhysicalDeviceVulkan13Features vk13 = {
			.sType												= VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
			.pNext												= &vk14,
			.robustImageAccess									= VK_FALSE,
			.inlineUniformBlock									= VK_FALSE,
			.descriptorBindingInlineUniformBlockUpdateAfterBind = VK_FALSE,
			.pipelineCreationCacheControl						= VK_FALSE,
			.privateData										= VK_FALSE,
			.shaderDemoteToHelperInvocation						= VK_FALSE,
			.shaderTerminateInvocation							= VK_FALSE,
			.subgroupSizeControl								= VK_FALSE,
			.computeFullSubgroups								= VK_FALSE,
			.synchronization2									= VK_FALSE,
			.textureCompressionASTC_HDR							= VK_FALSE,
			.shaderZeroInitializeWorkgroupMemory				= VK_FALSE,
			.dynamicRendering									= VK_FALSE,
			.shaderIntegerDotProduct							= VK_FALSE,
			.maintenance4										= VK_FALSE,
	};
	VkPhysicalDeviceVulkan12Features vk12{};
	vk12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
	vk12.pNext = &vk13;
	VkPhysicalDeviceVulkan11Features vk11{};
	vk11.sType						= VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
	vk11.pNext						= &vk12;
	VkPhysicalDeviceFeatures2 feat2 = {
			.sType	  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
			.pNext	  = &vk11,
			.features = {},
	};
	vkGetPhysicalDeviceFeatures2(physical_device, &feat2);
	bool const has_vulkan_1_4 = props.apiVersion >= VK_API_VERSION_1_4;

	VkPhysicalDeviceSubgroupSizeControlProperties subgroup_size_props{};
	subgroup_size_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES;
	subgroup_size_props.pNext = nullptr;
	VkPhysicalDeviceProperties2 props_ext2{};
	props_ext2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
	props_ext2.pNext = &subgroup_size_props;
	vkGetPhysicalDeviceProperties2(physical_device, &props_ext2);

	out += std::format("feature.shader_int64={}\n", feat2.features.shaderInt64 == VK_TRUE ? "true" : "false");
	out += std::format("feature.shader_int16={}\n", feat2.features.shaderInt16 == VK_TRUE ? "true" : "false");
	out += std::format("feature.shader_float64={}\n", feat2.features.shaderFloat64 == VK_TRUE ? "true" : "false");
	out += std::format("feature.storage_buffer_16bit_access={}\n",
					   vk11.storageBuffer16BitAccess == VK_TRUE ? "true" : "false");
	out += std::format("feature.uniform_and_storage_buffer_16bit_access={}\n",
					   vk11.uniformAndStorageBuffer16BitAccess == VK_TRUE ? "true" : "false");
	out += std::format("feature.storage_push_constant_16={}\n",
					   vk11.storagePushConstant16 == VK_TRUE ? "true" : "false");
	out += std::format("feature.shader_int8={}\n", vk12.shaderInt8 == VK_TRUE ? "true" : "false");
	out += std::format("feature.shader_float16={}\n", vk12.shaderFloat16 == VK_TRUE ? "true" : "false");
	out += std::format("feature.shader_float_controls2={}\n",
					   (has_vulkan_1_4 ? vk14.shaderFloatControls2
									   : shader_float_controls2_features.shaderFloatControls2) == VK_TRUE
							   ? "true"
							   : "false");
	out += std::format("feature.shader_expect_assume={}\n",
					   (has_vulkan_1_4 ? vk14.shaderExpectAssume : shader_expect_assume_features.shaderExpectAssume) ==
									   VK_TRUE
							   ? "true"
							   : "false");
	out += std::format("feature.storage_buffer_8bit_access={}\n",
					   vk12.storageBuffer8BitAccess == VK_TRUE ? "true" : "false");
	out += std::format("feature.uniform_and_storage_buffer_8bit_access={}\n",
					   vk12.uniformAndStorageBuffer8BitAccess == VK_TRUE ? "true" : "false");
	out += std::format("feature.storage_push_constant_8={}\n", vk12.storagePushConstant8 == VK_TRUE ? "true" : "false");
	out += std::format("feature.runtime_descriptor_array={}\n",
					   vk12.runtimeDescriptorArray == VK_TRUE ? "true" : "false");
	out += std::format("feature.vulkan_memory_model={}\n", vk12.vulkanMemoryModel == VK_TRUE ? "true" : "false");
	out += std::format("feature.vulkan_memory_model_device_scope={}\n",
					   vk12.vulkanMemoryModelDeviceScope == VK_TRUE ? "true" : "false");
	out += std::format("feature.shader_uniform_buffer_array_nonuniform_indexing={}\n",
					   vk12.shaderUniformBufferArrayNonUniformIndexing == VK_TRUE ? "true" : "false");
	out += std::format("feature.shader_sampled_image_array_nonuniform_indexing={}\n",
					   vk12.shaderSampledImageArrayNonUniformIndexing == VK_TRUE ? "true" : "false");
	out += std::format("feature.shader_storage_buffer_array_nonuniform_indexing={}\n",
					   vk12.shaderStorageBufferArrayNonUniformIndexing == VK_TRUE ? "true" : "false");
	out += std::format("feature.shader_storage_image_array_nonuniform_indexing={}\n",
					   vk12.shaderStorageImageArrayNonUniformIndexing == VK_TRUE ? "true" : "false");
	out += std::format("feature.shader_input_attachment_array_nonuniform_indexing={}\n",
					   vk12.shaderInputAttachmentArrayNonUniformIndexing == VK_TRUE ? "true" : "false");
	out += std::format("feature.buffer_device_address={}\n", vk12.bufferDeviceAddress == VK_TRUE ? "true" : "false");
	out += std::format("feature.shader_untyped_pointers={}\n",
					   untyped_ptr_features.shaderUntypedPointers == VK_TRUE ? "true" : "false");
	out += std::format("feature.shader_device_clock={}\n",
					   shader_clock_features.shaderDeviceClock == VK_TRUE ? "true" : "false");
	out += std::format("feature.shader_subgroup_clock={}\n",
					   shader_clock_features.shaderSubgroupClock == VK_TRUE ? "true" : "false");
	out += std::format("feature.shader_subgroup_rotate={}\n",
					   (has_vulkan_1_4 ? vk14.shaderSubgroupRotate : subgroup_rotate_features.shaderSubgroupRotate) ==
									   VK_TRUE
							   ? "true"
							   : "false");
	out += std::format("feature.shader_subgroup_rotate_clustered={}\n",
					   (has_vulkan_1_4 ? vk14.shaderSubgroupRotateClustered
									   : subgroup_rotate_features.shaderSubgroupRotateClustered) == VK_TRUE
							   ? "true"
							   : "false");
	out += std::format("feature.cooperative_matrix={}\n",
					   cooperative_matrix_features.cooperativeMatrix == VK_TRUE ? "true" : "false");
	out += std::format("feature.shader_bfloat16_type={}\n",
					   bfloat16_features.shaderBFloat16Type == VK_TRUE ? "true" : "false");
	out += std::format("feature.shader_bfloat16_dot_product={}\n",
					   bfloat16_features.shaderBFloat16DotProduct == VK_TRUE ? "true" : "false");
	out += std::format("feature.shader_bfloat16_cooperative_matrix={}\n",
					   bfloat16_features.shaderBFloat16CooperativeMatrix == VK_TRUE ? "true" : "false");
	out += std::format("feature.shader_integer_dot_product={}\n",
					   integer_dot_product_features.shaderIntegerDotProduct == VK_TRUE ? "true" : "false");
	out += std::format("feature.acceleration_structure={}\n",
					   accel_struct_features.accelerationStructure == VK_TRUE ? "true" : "false");
	out += std::format("feature.ray_query={}\n", ray_query_features.rayQuery == VK_TRUE ? "true" : "false");
	out += std::format("feature.descriptor_buffer={}\n",
					   descriptor_buffer_features.descriptorBuffer == VK_TRUE ? "true" : "false");
	out += std::format("feature.descriptor_heap={}\n",
					   descriptor_heap_features.descriptorHeap == VK_TRUE ? "true" : "false");
	bool const push_descriptor_supported =
			has_vulkan_1_4 || device_supports_extension(physical_device, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
	out += std::format("feature.push_descriptor={}\n", push_descriptor_supported ? "true" : "false");
	out += std::format("feature.shader_subgroup_extended_types={}\n",
					   vk12.shaderSubgroupExtendedTypes == VK_TRUE ? "true" : "false");
	out += std::format("feature.subgroup_size_control={}\n",
					   subgroup_size_control.subgroupSizeControl == VK_TRUE ? "true" : "false");
	out += std::format("feature.compute_full_subgroups={}\n",
					   subgroup_size_control.computeFullSubgroups == VK_TRUE ? "true" : "false");
	out += std::format("subgroup_size_control.min={}\n", subgroup_size_props.minSubgroupSize);
	out += std::format("subgroup_size_control.max={}\n", subgroup_size_props.maxSubgroupSize);
	out += std::format("subgroup_size_control.required_stages=0x{:x}\n",
					   subgroup_size_props.requiredSubgroupSizeStages);

	out += std::format("extension.VK_KHR_push_descriptor={}\n",
					   device_supports_extension(physical_device, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME) ? "true"
																										 : "false");
	out += std::format("extension.VK_EXT_descriptor_buffer={}\n",
					   device_supports_extension(physical_device, VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME) ? "true"
																										   : "false");
	out += std::format("extension.VK_EXT_descriptor_heap={}\n",
					   device_supports_extension(physical_device, VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME) ? "true"
																										 : "false");
	out += std::format("extension.VK_KHR_maintenance5={}\n",
					   device_supports_extension(physical_device, VK_KHR_MAINTENANCE_5_EXTENSION_NAME) ? "true"
																									   : "false");
	out += std::format("extension.VK_KHR_shader_untyped_pointers={}\n",
					   device_supports_extension(physical_device, VK_KHR_SHADER_UNTYPED_POINTERS_EXTENSION_NAME)
							   ? "true"
							   : "false");
	out += std::format("extension.VK_KHR_shader_bfloat16={}\n",
					   device_supports_extension(physical_device, VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME) ? "true"
																										 : "false");
	out += std::format("extension.VK_KHR_shader_integer_dot_product={}\n",
					   device_supports_extension(physical_device, VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME)
							   ? "true"
							   : "false");
	out += std::format("extension.VK_KHR_ray_query={}\n",
					   device_supports_extension(physical_device, VK_KHR_RAY_QUERY_EXTENSION_NAME) ? "true" : "false");
	out += std::format("extension.VK_KHR_acceleration_structure={}\n",
					   device_supports_extension(physical_device, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)
							   ? "true"
							   : "false");
	out += std::format("extension.VK_KHR_deferred_host_operations={}\n",
					   device_supports_extension(physical_device, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME)
							   ? "true"
							   : "false");
	out += std::format(
			"extension.VK_KHR_buffer_device_address={}\n",
			device_supports_extension(physical_device, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME) ? "true" : "false");
	out += std::format("extension.VK_KHR_shader_clock={}\n",
					   device_supports_extension(physical_device, VK_KHR_SHADER_CLOCK_EXTENSION_NAME) ? "true"
																									  : "false");
	out += std::format(
			"extension.VK_EXT_subgroup_size_control={}\n",
			device_supports_extension(physical_device, VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME) ? "true" : "false");
	bool const has_khr_coop = device_supports_extension(physical_device, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
	out += std::format("extension.VK_KHR_cooperative_matrix={}\n", has_khr_coop ? "true" : "false");
	if (has_khr_coop) {
		auto get_coop_props = reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
				vkGetInstanceProcAddr(scope.instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
		if (get_coop_props != nullptr) {
			uint32_t coop_count = 0;
			if (get_coop_props(physical_device, &coop_count, nullptr) == VK_SUCCESS && coop_count > 0) {
				std::vector<VkCooperativeMatrixPropertiesKHR> coop_props(coop_count);
				for (auto &p: coop_props) {
					p		= {};
					p.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
					p.pNext = nullptr;
				}
				if (get_coop_props(physical_device, &coop_count, coop_props.data()) == VK_SUCCESS) {
					out += std::format("coop_matrix.property_count={}\n", coop_count);
					for (uint32_t i = 0; i < coop_count; ++i) {
						const auto &p = coop_props[i];

						auto type_to_string = [](VkComponentTypeKHR component_type) {
							using namespace std::string_view_literals;
							switch (component_type) {
#define CASE(X)                                                                                                        \
	case X:                                                                                                            \
		return #X##sv;
								CASE(VK_COMPONENT_TYPE_FLOAT16_KHR)
								CASE(VK_COMPONENT_TYPE_FLOAT32_KHR)
								CASE(VK_COMPONENT_TYPE_FLOAT64_KHR)
								CASE(VK_COMPONENT_TYPE_SINT8_KHR)
								CASE(VK_COMPONENT_TYPE_SINT16_KHR)
								CASE(VK_COMPONENT_TYPE_SINT32_KHR)
								CASE(VK_COMPONENT_TYPE_SINT64_KHR)
								CASE(VK_COMPONENT_TYPE_UINT8_KHR)
								CASE(VK_COMPONENT_TYPE_UINT16_KHR)
								CASE(VK_COMPONENT_TYPE_UINT32_KHR)
								CASE(VK_COMPONENT_TYPE_UINT64_KHR)
								CASE(VK_COMPONENT_TYPE_BFLOAT16_KHR)
								CASE(VK_COMPONENT_TYPE_SINT8_PACKED_NV)
								CASE(VK_COMPONENT_TYPE_UINT8_PACKED_NV)
								CASE(VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT)
								CASE(VK_COMPONENT_TYPE_FLOAT8_E5M2_EXT)
#undef CASE
								default:
									return "Unknown Type Name"sv;
							}
						};

						out += std::format(R"(
  {{
    .m_size                  = {},
    .n_size                  = {},
    .k_size                  = {},
    .a_type                  = {},
    .b_type                  = {},
    .c_type                  = {},
    .result_type             = {},
    .saturating_accumulation = {},
  }}
)",
										   p.MSize, p.NSize, p.KSize, type_to_string(p.AType), type_to_string(p.BType),
										   type_to_string(p.CType), type_to_string(p.ResultType),
										   p.saturatingAccumulation == VK_TRUE ? "true" : "false");
					}
				}
			} else {
				out += "coop_matrix.property_count=0\n";
			}
		} else {
			out += "coop_matrix.query=unavailable\n";
		}
	}

	out_text = out;
	return runtime_ok();
}

RuntimeResult query_max_supported_spirv_version(const RuntimeConfig &config, SpirvTargetVersion &out_version) {
	(void) config;
	VkApplicationInfo const app_info = {
			.sType				= VK_STRUCTURE_TYPE_APPLICATION_INFO,
			.pNext				= nullptr,
			.pApplicationName	= "shader_explorer",
			.applicationVersion = VK_MAKE_VERSION(1, 0, 0),
			.pEngineName		= "shader_explorer engine",
			.engineVersion		= VK_MAKE_VERSION(1, 0, 0),
			.apiVersion			= VK_API_VERSION_1_4,
	};

	InstanceScope			   scope;
	VkInstanceCreateInfo const instance_ci = {
			.sType					 = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
			.pNext					 = nullptr,
			.flags					 = 0,
			.pApplicationInfo		 = &app_info,
			.enabledLayerCount		 = 0,
			.ppEnabledLayerNames	 = nullptr,
			.enabledExtensionCount	 = 0,
			.ppEnabledExtensionNames = nullptr,
	};
	if (vkCreateInstance(&instance_ci, nullptr, &scope.instance) != VK_SUCCESS) {
		return runtime_fail(RuntimeFailure::VkCreateInstanceFailed);
	}

	uint32_t	   device_count		= 0;
	VkResult const enumerate_result = vkEnumeratePhysicalDevices(scope.instance, &device_count, nullptr);
	if (enumerate_result != VK_SUCCESS) { return runtime_fail(RuntimeFailure::VkEnumeratePhysicalDevicesFailed); }
	if (device_count == 0) { return runtime_fail(RuntimeFailure::NoPhysicalDevice); }

	std::vector<VkPhysicalDevice> devices(device_count);
	vkEnumeratePhysicalDevices(scope.instance, &device_count, devices.data());
	VkPhysicalDevice physical_device = devices[0];

	VkPhysicalDeviceProperties props = {
			.apiVersion		   = 0,
			.driverVersion	   = 0,
			.vendorID		   = 0,
			.deviceID		   = 0,
			.deviceType		   = VK_PHYSICAL_DEVICE_TYPE_OTHER,
			.deviceName		   = "",
			.pipelineCacheUUID = {},
			.limits			   = {},
			.sparseProperties  = {},
	};
	vkGetPhysicalDeviceProperties(physical_device, &props);
	const uint32_t api				 = props.apiVersion;
	const bool	   has_spirv_1_4_ext = device_supports_extension(physical_device, VK_KHR_SPIRV_1_4_EXTENSION_NAME);

	if (api >= VK_API_VERSION_1_3) {
		out_version = SpirvTargetVersion::V16;
	} else if (api >= VK_API_VERSION_1_2) {
		out_version = SpirvTargetVersion::V15;
	} else if (api >= VK_API_VERSION_1_1) {
		out_version = has_spirv_1_4_ext ? SpirvTargetVersion::V14 : SpirvTargetVersion::V13;
	} else {
		out_version = SpirvTargetVersion::V10;
	}

	return runtime_ok();
}

RuntimeResult run_pipeline_dump(const std::vector<uint32_t> &spirv, const RuntimeDumpOptions &options,
								const RuntimeConfig &config, std::string &out_text) {
	ResolvedRuntimeConfig	 cfg = resolve_runtime_config(config);
	ShaderRequirements const req = analyze_spirv_requirements(spirv);
	if (req.needs_descriptor_heap) { cfg.binding_model = BindingModel::DescriptorHeap; }
	if (!req.unhandled_extensions.empty()) {
		return runtime_fail(RuntimeFailure::UnhandledSpirvExtensions,
							format_unhandled_extensions(req.unhandled_extensions));
	}
	if (!req.unhandled_capabilities.empty()) {
		return runtime_fail(RuntimeFailure::UnhandledSpirvCapabilities,
							format_unhandled_capabilities(req.unhandled_capabilities));
	}

	VkApplicationInfo const app_info = {
			.sType				= VK_STRUCTURE_TYPE_APPLICATION_INFO,
			.pNext				= nullptr,
			.pApplicationName	= "shader_explorer",
			.applicationVersion = VK_MAKE_VERSION(1, 0, 0),
			.pEngineName		= "shader_explorer engine",
			.engineVersion		= VK_MAKE_VERSION(1, 0, 0),
			.apiVersion			= VK_API_VERSION_1_4,
	};

	VkInstance				   instance	   = VK_NULL_HANDLE;
	VkInstanceCreateInfo const instance_ci = {
			.sType					 = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
			.pNext					 = nullptr,
			.flags					 = 0,
			.pApplicationInfo		 = &app_info,
			.enabledLayerCount		 = 0,
			.ppEnabledLayerNames	 = nullptr,
			.enabledExtensionCount	 = 0,
			.ppEnabledExtensionNames = nullptr,
	};
	if (vkCreateInstance(&instance_ci, nullptr, &instance) != VK_SUCCESS) {
		return runtime_fail(RuntimeFailure::VkCreateInstanceFailed);
	}

	uint32_t	   device_count		= 0;
	VkResult const enumerate_result = vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
	if (enumerate_result != VK_SUCCESS) {
		vkDestroyInstance(instance, nullptr);
		return runtime_fail(RuntimeFailure::VkEnumeratePhysicalDevicesFailed);
	}
	if (device_count == 0) {
		vkDestroyInstance(instance, nullptr);
		return runtime_fail(RuntimeFailure::NoPhysicalDevice);
	}

	std::vector<VkPhysicalDevice> devices(device_count);
	vkEnumeratePhysicalDevices(instance, &device_count, devices.data());

	VkPhysicalDevice		   physical_device		 = devices[0];
	VkPhysicalDeviceProperties physical_device_props = {
			.apiVersion		   = 0,
			.driverVersion	   = 0,
			.vendorID		   = 0,
			.deviceID		   = 0,
			.deviceType		   = VK_PHYSICAL_DEVICE_TYPE_OTHER,
			.deviceName		   = "",
			.pipelineCacheUUID = {},
			.limits			   = {},
			.sparseProperties  = {},
	};
	vkGetPhysicalDeviceProperties(physical_device, &physical_device_props);
	bool const has_vulkan_1_3 = physical_device_props.apiVersion >= VK_API_VERSION_1_3;
	bool const has_vulkan_1_4 = physical_device_props.apiVersion >= VK_API_VERSION_1_4;
	// We target Vulkan 1.0 as baseline behavior. For features promoted to core, we accept
	// either path:
	// - core version is high enough, or
	// - extension is present on lower core versions.
	// This keeps older platforms viable while still enabling newer features when available.
	bool const needs_subgroup_size_control = cfg.required_subgroup_size > 0 || cfg.require_full_subgroups;
	bool const needs_subgroup_rotate	   = (req.required_subgroup_ops & VK_SUBGROUP_FEATURE_ROTATE_BIT) != 0U;
	bool const needs_subgroup_rotate_clustered =
			(req.required_subgroup_ops & VK_SUBGROUP_FEATURE_ROTATE_CLUSTERED_BIT) != 0U;
	int const queue_family = select_compute_queue(physical_device);
	if (queue_family < 0) {
		vkDestroyInstance(instance, nullptr);
		return runtime_fail(RuntimeFailure::NoComputeQueue);
	}

	std::vector<const char *> requested_extensions;
	auto					  request_extension = [&](const char *ext) {
		 for (const char *existing: requested_extensions) {
			 if (std::strcmp(existing, ext) == 0) { return; }
		 }
		 requested_extensions.push_back(ext);
	};
	bool const supports_pipeline_exec =
			device_supports_extension(physical_device, VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);

	if (!supports_pipeline_exec) {
		vkDestroyInstance(instance, nullptr);
		return runtime_fail(RuntimeFailure::PipelineExecutableUnsupported);
	}

	if (supports_pipeline_exec) { request_extension(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME); }
	if (needs_subgroup_size_control && !has_vulkan_1_3) {
		if (!device_supports_extension(physical_device, VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME)) {
			vkDestroyInstance(instance, nullptr);
			return runtime_fail(RuntimeFailure::MissingExtensionSubgroupSizeControl);
		}
		request_extension(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
	}
	if ((needs_subgroup_rotate || needs_subgroup_rotate_clustered) && !has_vulkan_1_4) {
		if (!device_supports_extension(physical_device, VK_KHR_SHADER_SUBGROUP_ROTATE_EXTENSION_NAME)) {
			vkDestroyInstance(instance, nullptr);
			return runtime_fail(RuntimeFailure::MissingExtensionShaderSubgroupRotate);
		}
		request_extension(VK_KHR_SHADER_SUBGROUP_ROTATE_EXTENSION_NAME);
	}
	if (req.needs_shader_float_controls2 && !has_vulkan_1_4) {
		if (!device_supports_extension(physical_device, VK_KHR_SHADER_FLOAT_CONTROLS_2_EXTENSION_NAME)) {
			vkDestroyInstance(instance, nullptr);
			return runtime_fail(RuntimeFailure::MissingExtensionShaderFloatControls2);
		}
		request_extension(VK_KHR_SHADER_FLOAT_CONTROLS_2_EXTENSION_NAME);
	}
	if (req.needs_shader_expect_assume && !has_vulkan_1_4) {
		if (!device_supports_extension(physical_device, VK_KHR_SHADER_EXPECT_ASSUME_EXTENSION_NAME)) {
			vkDestroyInstance(instance, nullptr);
			return runtime_fail(RuntimeFailure::MissingExtensionShaderExpectAssume);
		}
		request_extension(VK_KHR_SHADER_EXPECT_ASSUME_EXTENSION_NAME);
	}

	if (cfg.binding_model == BindingModel::PushDescriptor) {
		if (!has_vulkan_1_4) {
			if (!device_supports_extension(physical_device, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME)) {
				vkDestroyInstance(instance, nullptr);
				return runtime_fail(RuntimeFailure::MissingExtensionPushDescriptor);
			}
			request_extension(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
		}
	}

	if (cfg.binding_model == BindingModel::DescriptorBuffer) {
		if (!device_supports_extension(physical_device, VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME)) {
			vkDestroyInstance(instance, nullptr);
			return runtime_fail(RuntimeFailure::MissingExtensionDescriptorBuffer);
		}
		request_extension(VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME);
	}

	if (cfg.binding_model == BindingModel::DescriptorHeap) {
		if (!device_supports_extension(physical_device, VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME)) {
			vkDestroyInstance(instance, nullptr);
			return runtime_fail(RuntimeFailure::MissingExtensionDescriptorHeap);
		}
		if (!device_supports_extension(physical_device, VK_KHR_MAINTENANCE_5_EXTENSION_NAME)) {
			vkDestroyInstance(instance, nullptr);
			return runtime_fail(RuntimeFailure::MissingExtensionMaintenance5);
		}
		if (!device_supports_extension(physical_device, VK_KHR_SHADER_UNTYPED_POINTERS_EXTENSION_NAME)) {
			vkDestroyInstance(instance, nullptr);
			return runtime_fail(RuntimeFailure::MissingExtensionShaderUntypedPointers);
		}
		request_extension(VK_KHR_MAINTENANCE_5_EXTENSION_NAME);
		request_extension(VK_KHR_SHADER_UNTYPED_POINTERS_EXTENSION_NAME);
		request_extension(VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME);
	}

	bool const needs_untyped_pointers =
			req.needs_shader_untyped_pointers || cfg.binding_model == BindingModel::DescriptorHeap;
	if (needs_untyped_pointers && cfg.binding_model != BindingModel::DescriptorHeap) {
		if (!device_supports_extension(physical_device, VK_KHR_SHADER_UNTYPED_POINTERS_EXTENSION_NAME)) {
			vkDestroyInstance(instance, nullptr);
			return runtime_fail(RuntimeFailure::MissingExtensionShaderUntypedPointers);
		}
		request_extension(VK_KHR_SHADER_UNTYPED_POINTERS_EXTENSION_NAME);
	}
	bool const needs_bfloat16 =
			req.needs_bfloat16_type || req.needs_bfloat16_dot_product || req.needs_bfloat16_cooperative_matrix;
	if (needs_bfloat16) {
		if (!device_supports_extension(physical_device, VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME)) {
			vkDestroyInstance(instance, nullptr);
			return runtime_fail(RuntimeFailure::MissingExtensionShaderBfloat16);
		}
		request_extension(VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME);
	}
	if (req.needs_integer_dot_product && !has_vulkan_1_3) {
		if (!device_supports_extension(physical_device, VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME)) {
			vkDestroyInstance(instance, nullptr);
			return runtime_fail(RuntimeFailure::MissingExtensionShaderIntegerDotProduct);
		}
		request_extension(VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME);
	}
	if (req.needs_ray_query) {
		if (!device_supports_extension(physical_device, VK_KHR_RAY_QUERY_EXTENSION_NAME)) {
			vkDestroyInstance(instance, nullptr);
			return runtime_fail(RuntimeFailure::MissingExtensionRayQuery);
		}
		if (!device_supports_extension(physical_device, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)) {
			vkDestroyInstance(instance, nullptr);
			return runtime_fail(RuntimeFailure::MissingExtensionAccelerationStructure);
		}
		if (!device_supports_extension(physical_device, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME)) {
			vkDestroyInstance(instance, nullptr);
			return runtime_fail(RuntimeFailure::MissingExtensionDeferredHostOperations);
		}
		request_extension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
		request_extension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
		request_extension(VK_KHR_RAY_QUERY_EXTENSION_NAME);
	}

	bool const needs_bda = cfg.enable_bda || req.needs_physical_storage_buffer_addresses || req.needs_ray_query;
	if (needs_bda) {
		if (!device_supports_extension(physical_device, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME)) {
			vkDestroyInstance(instance, nullptr);
			return runtime_fail(RuntimeFailure::MissingExtensionBufferDeviceAddress);
		}
		request_extension(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
	}

	if (req.uses_shader_clock) {
		if (!device_supports_extension(physical_device, VK_KHR_SHADER_CLOCK_EXTENSION_NAME)) {
			vkDestroyInstance(instance, nullptr);
			return runtime_fail(RuntimeFailure::MissingExtensionShaderClock);
		}
		request_extension(VK_KHR_SHADER_CLOCK_EXTENSION_NAME);
	}

	if (req.uses_cooperative_matrix) {
		if (!device_supports_extension(physical_device, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME)) {
			vkDestroyInstance(instance, nullptr);
			return runtime_fail(RuntimeFailure::MissingExtensionCooperativeMatrix);
		}
		request_extension(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
	}

	constexpr float				  priority = 1.0F;
	VkDeviceQueueCreateInfo const queue_ci = {
			.sType			  = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
			.pNext			  = nullptr,
			.flags			  = 0,
			.queueFamilyIndex = uint32_t(queue_family),
			.queueCount		  = 1,
			.pQueuePriorities = &priority,
	};

	VkDeviceCreateInfo device_ci = {
			.sType					 = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
			.pNext					 = nullptr,
			.flags					 = 0,
			.queueCreateInfoCount	 = 1,
			.pQueueCreateInfos		 = &queue_ci,
			.enabledLayerCount		 = 0,
			.ppEnabledLayerNames	 = nullptr,
			.enabledExtensionCount	 = uint32_t(requested_extensions.size()),
			.ppEnabledExtensionNames = requested_extensions.data(),
			.pEnabledFeatures		 = nullptr,
	};

	VkPhysicalDeviceSubgroupSizeControlProperties subgroup_size_props = {
			.sType						  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES,
			.pNext						  = nullptr,
			.minSubgroupSize			  = 0,
			.maxSubgroupSize			  = 0,
			.maxComputeWorkgroupSubgroups = 0,
			.requiredSubgroupSizeStages	  = 0,
	};
	VkPhysicalDeviceSubgroupProperties subgroup_props = {
			.sType					   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
			.pNext					   = &subgroup_size_props,
			.subgroupSize			   = 0,
			.supportedStages		   = 0,
			.supportedOperations	   = 0,
			.quadOperationsInAllStages = VK_FALSE,
	};
	VkPhysicalDeviceProperties2 subgroup_props2 = {
			.sType		= VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
			.pNext		= &subgroup_props,
			.properties = {},
	};
	vkGetPhysicalDeviceProperties2(physical_device, &subgroup_props2);

	VkPhysicalDeviceDescriptorHeapFeaturesEXT descriptor_heap_features{};
	descriptor_heap_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT;
	descriptor_heap_features.pNext = nullptr;
	VkPhysicalDeviceDescriptorBufferFeaturesEXT descriptor_buffer_features = {
			.sType								= VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT,
			.pNext								= &descriptor_heap_features,
			.descriptorBuffer					= VK_FALSE,
			.descriptorBufferCaptureReplay		= VK_FALSE,
			.descriptorBufferImageLayoutIgnored = VK_FALSE,
			.descriptorBufferPushDescriptors	= VK_FALSE,
	};
	VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR pipeline_exec_features = {
			.sType					= VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR,
			.pNext					= &descriptor_buffer_features,
			.pipelineExecutableInfo = VK_TRUE,
	};
	VkPhysicalDeviceSubgroupSizeControlFeatures subgroup_size_control_features = {
			.sType				  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES,
			.pNext				  = &pipeline_exec_features,
			.subgroupSizeControl  = VK_FALSE,
			.computeFullSubgroups = VK_FALSE,
	};
	VkPhysicalDeviceShaderSubgroupRotateFeatures subgroup_rotate_features = {
			.sType						   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_ROTATE_FEATURES,
			.pNext						   = &subgroup_size_control_features,
			.shaderSubgroupRotate		   = VK_FALSE,
			.shaderSubgroupRotateClustered = VK_FALSE,
	};
	VkPhysicalDeviceRayQueryFeaturesKHR ray_query_features = {
			.sType	  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR,
			.pNext	  = &subgroup_rotate_features,
			.rayQuery = VK_FALSE,
	};
	VkPhysicalDeviceAccelerationStructureFeaturesKHR accel_struct_features = {
			.sType								= VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
			.pNext								= &ray_query_features,
			.accelerationStructure				= VK_FALSE,
			.accelerationStructureCaptureReplay = VK_FALSE,
			.accelerationStructureIndirectBuild = VK_FALSE,
			.accelerationStructureHostCommands	= VK_FALSE,
			.descriptorBindingAccelerationStructureUpdateAfterBind = VK_FALSE,
	};
	VkPhysicalDeviceShaderIntegerDotProductFeatures integer_dot_product_features = {
			.sType					 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES,
			.pNext					 = &accel_struct_features,
			.shaderIntegerDotProduct = VK_FALSE,
	};
	VkPhysicalDeviceShaderBfloat16FeaturesKHR bfloat16_features = {
			.sType							 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR,
			.pNext							 = &integer_dot_product_features,
			.shaderBFloat16Type				 = VK_FALSE,
			.shaderBFloat16DotProduct		 = VK_FALSE,
			.shaderBFloat16CooperativeMatrix = VK_FALSE,
	};
	VkPhysicalDeviceCooperativeMatrixFeaturesKHR cooperative_matrix_features = {
			.sType								 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
			.pNext								 = &bfloat16_features,
			.cooperativeMatrix					 = VK_FALSE,
			.cooperativeMatrixRobustBufferAccess = VK_FALSE,
	};
	VkPhysicalDeviceShaderClockFeaturesKHR shader_clock_features = {
			.sType				 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR,
			.pNext				 = &cooperative_matrix_features,
			.shaderSubgroupClock = VK_FALSE,
			.shaderDeviceClock	 = VK_FALSE,
	};
	VkPhysicalDeviceShaderUntypedPointersFeaturesKHR untyped_ptr_features = {
			.sType				   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_UNTYPED_POINTERS_FEATURES_KHR,
			.pNext				   = &shader_clock_features,
			.shaderUntypedPointers = VK_FALSE,
	};
	VkPhysicalDeviceShaderFloatControls2Features shader_float_controls2_features = {
			.sType				  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT_CONTROLS_2_FEATURES,
			.pNext				  = &untyped_ptr_features,
			.shaderFloatControls2 = VK_FALSE,
	};
	VkPhysicalDeviceShaderExpectAssumeFeatures shader_expect_assume_features = {
			.sType				= VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_EXPECT_ASSUME_FEATURES,
			.pNext				= &shader_float_controls2_features,
			.shaderExpectAssume = VK_FALSE,
	};
	VkPhysicalDevice16BitStorageFeatures storage16_features = {
			.sType								= VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
			.pNext								= &shader_expect_assume_features,
			.storageBuffer16BitAccess			= VK_FALSE,
			.uniformAndStorageBuffer16BitAccess = VK_FALSE,
			.storagePushConstant16				= VK_FALSE,
			.storageInputOutput16				= VK_FALSE,
	};
	VkPhysicalDeviceVulkan14Features vk14_features = {
			.sType									= VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES,
			.pNext									= &storage16_features,
			.globalPriorityQuery					= VK_FALSE,
			.shaderSubgroupRotate					= VK_FALSE,
			.shaderSubgroupRotateClustered			= VK_FALSE,
			.shaderFloatControls2					= VK_FALSE,
			.shaderExpectAssume						= VK_FALSE,
			.rectangularLines						= VK_FALSE,
			.bresenhamLines							= VK_FALSE,
			.smoothLines							= VK_FALSE,
			.stippledRectangularLines				= VK_FALSE,
			.stippledBresenhamLines					= VK_FALSE,
			.stippledSmoothLines					= VK_FALSE,
			.vertexAttributeInstanceRateDivisor		= VK_FALSE,
			.vertexAttributeInstanceRateZeroDivisor = VK_FALSE,
			.indexTypeUint8							= VK_FALSE,
			.dynamicRenderingLocalRead				= VK_FALSE,
			.maintenance5							= VK_FALSE,
			.maintenance6							= VK_FALSE,
			.pipelineProtectedAccess				= VK_FALSE,
			.pipelineRobustness						= VK_FALSE,
			.hostImageCopy							= VK_FALSE,
			.pushDescriptor							= VK_FALSE,
	};
	VkPhysicalDeviceVulkan13Features vk13_features{};
	vk13_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
	vk13_features.pNext = &vk14_features;
	VkPhysicalDeviceVulkan12Features vk12_features{};
	vk12_features.sType							  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
	vk12_features.pNext							  = &vk13_features;
	VkPhysicalDeviceFeatures2 supported_features2 = {
			.sType	  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
			.pNext	  = &vk12_features,
			.features = {},
	};
	// Query all supported features from a single pNext chain.
	vkGetPhysicalDeviceFeatures2(physical_device, &supported_features2);
	bool const push_descriptor_supported = has_vulkan_1_4 ? (vk14_features.pushDescriptor == VK_TRUE) : true;
	bool const shader_float_controls2_supported =
			has_vulkan_1_4 ? (vk14_features.shaderFloatControls2 == VK_TRUE)
						   : (shader_float_controls2_features.shaderFloatControls2 == VK_TRUE);
	bool const shader_expect_assume_supported = has_vulkan_1_4
														? (vk14_features.shaderExpectAssume == VK_TRUE)
														: (shader_expect_assume_features.shaderExpectAssume == VK_TRUE);
	bool const subgroup_rotate_supported	  = has_vulkan_1_4 ? (vk14_features.shaderSubgroupRotate == VK_TRUE)
															   : (subgroup_rotate_features.shaderSubgroupRotate == VK_TRUE);
	bool const subgroup_rotate_clustered_supported =
			has_vulkan_1_4 ? (vk14_features.shaderSubgroupRotateClustered == VK_TRUE)
						   : (subgroup_rotate_features.shaderSubgroupRotateClustered == VK_TRUE);

	if (cfg.binding_model == BindingModel::PushDescriptor && !push_descriptor_supported) {
		vkDestroyInstance(instance, nullptr);
		return runtime_fail(RuntimeFailure::MissingFeaturePushDescriptor);
	}

	if (needs_untyped_pointers && untyped_ptr_features.shaderUntypedPointers != VK_TRUE) {
		vkDestroyInstance(instance, nullptr);
		return runtime_fail(RuntimeFailure::MissingFeatureShaderUntypedPointers);
	}

	if (req.needs_int64 && supported_features2.features.shaderInt64 != VK_TRUE) {
		vkDestroyInstance(instance, nullptr);
		return runtime_fail(RuntimeFailure::MissingFeatureShaderInt64);
	}
	if (req.needs_int16 && supported_features2.features.shaderInt16 != VK_TRUE) {
		vkDestroyInstance(instance, nullptr);
		return runtime_fail(RuntimeFailure::MissingFeatureShaderInt16);
	}
	if (req.needs_float64 && supported_features2.features.shaderFloat64 != VK_TRUE) {
		vkDestroyInstance(instance, nullptr);
		return runtime_fail(RuntimeFailure::MissingFeatureShaderFloat64);
	}
	if (req.needs_int8 && vk12_features.shaderInt8 != VK_TRUE) {
		vkDestroyInstance(instance, nullptr);
		return runtime_fail(RuntimeFailure::MissingFeatureShaderInt8);
	}
	if (req.needs_float16 && vk12_features.shaderFloat16 != VK_TRUE) {
		vkDestroyInstance(instance, nullptr);
		return runtime_fail(RuntimeFailure::MissingFeatureShaderFloat16);
	}
	if (req.needs_shader_float_controls2 && !shader_float_controls2_supported) {
		vkDestroyInstance(instance, nullptr);
		return runtime_fail(RuntimeFailure::MissingFeatureShaderFloatControls2);
	}
	if (req.needs_shader_expect_assume && !shader_expect_assume_supported) {
		vkDestroyInstance(instance, nullptr);
		return runtime_fail(RuntimeFailure::MissingFeatureShaderExpectAssume);
	}
	if (needs_subgroup_rotate && !subgroup_rotate_supported) {
		vkDestroyInstance(instance, nullptr);
		return runtime_fail(RuntimeFailure::MissingFeatureShaderSubgroupRotate);
	}
	if (needs_subgroup_rotate_clustered && !subgroup_rotate_clustered_supported) {
		vkDestroyInstance(instance, nullptr);
		return runtime_fail(RuntimeFailure::MissingFeatureShaderSubgroupRotateClustered);
	}
	if (req.needs_storage_buffer_8bit && vk12_features.storageBuffer8BitAccess != VK_TRUE) {
		vkDestroyInstance(instance, nullptr);
		return runtime_fail(RuntimeFailure::MissingFeatureStorageBuffer8BitAccess);
	}
	if (req.needs_uniform_storage_buffer_8bit && vk12_features.uniformAndStorageBuffer8BitAccess != VK_TRUE) {
		vkDestroyInstance(instance, nullptr);
		return runtime_fail(RuntimeFailure::MissingFeatureUniformAndStorageBuffer8BitAccess);
	}
	if (req.needs_storage_push_constant_8bit && vk12_features.storagePushConstant8 != VK_TRUE) {
		vkDestroyInstance(instance, nullptr);
		return runtime_fail(RuntimeFailure::MissingFeatureStoragePushConstant8);
	}
	if (req.needs_storage_buffer_16bit && storage16_features.storageBuffer16BitAccess != VK_TRUE) {
		vkDestroyInstance(instance, nullptr);
		return runtime_fail(RuntimeFailure::MissingFeatureStorageBuffer16BitAccess);
	}
	if (req.needs_uniform_storage_buffer_16bit && storage16_features.uniformAndStorageBuffer16BitAccess != VK_TRUE) {
		vkDestroyInstance(instance, nullptr);
		return runtime_fail(RuntimeFailure::MissingFeatureUniformAndStorageBuffer16BitAccess);
	}
	if (req.needs_storage_push_constant_16bit && storage16_features.storagePushConstant16 != VK_TRUE) {
		vkDestroyInstance(instance, nullptr);
		return runtime_fail(RuntimeFailure::MissingFeatureStoragePushConstant16);
	}
	if (req.needs_runtime_descriptor_array && vk12_features.runtimeDescriptorArray != VK_TRUE) {
		vkDestroyInstance(instance, nullptr);
		return runtime_fail(RuntimeFailure::MissingFeatureRuntimeDescriptorArray);
	}
	if (req.needs_vulkan_memory_model && vk12_features.vulkanMemoryModel != VK_TRUE) {
		vkDestroyInstance(instance, nullptr);
		return runtime_fail(RuntimeFailure::MissingFeatureVulkanMemoryModel);
	}
	if (req.needs_vulkan_memory_model_device_scope && vk12_features.vulkanMemoryModelDeviceScope != VK_TRUE) {
		vkDestroyInstance(instance, nullptr);
		return runtime_fail(RuntimeFailure::MissingFeatureVulkanMemoryModelDeviceScope);
	}
	if (req.needs_uniform_buffer_array_nonuniform &&
		vk12_features.shaderUniformBufferArrayNonUniformIndexing != VK_TRUE) {
		vkDestroyInstance(instance, nullptr);
		return runtime_fail(RuntimeFailure::MissingFeatureShaderUniformBufferArrayNonUniformIndexing);
	}
	if (req.needs_sampled_image_array_nonuniform &&
		vk12_features.shaderSampledImageArrayNonUniformIndexing != VK_TRUE) {
		vkDestroyInstance(instance, nullptr);
		return runtime_fail(RuntimeFailure::MissingFeatureShaderSampledImageArrayNonUniformIndexing);
	}
	if (req.needs_storage_buffer_array_nonuniform &&
		vk12_features.shaderStorageBufferArrayNonUniformIndexing != VK_TRUE) {
		vkDestroyInstance(instance, nullptr);
		return runtime_fail(RuntimeFailure::MissingFeatureShaderStorageBufferArrayNonUniformIndexing);
	}
	if (req.needs_storage_image_array_nonuniform &&
		vk12_features.shaderStorageImageArrayNonUniformIndexing != VK_TRUE) {
		vkDestroyInstance(instance, nullptr);
		return runtime_fail(RuntimeFailure::MissingFeatureShaderStorageImageArrayNonUniformIndexing);
	}
	if (req.needs_input_attachment_array_nonuniform &&
		vk12_features.shaderInputAttachmentArrayNonUniformIndexing != VK_TRUE) {
		vkDestroyInstance(instance, nullptr);
		return runtime_fail(RuntimeFailure::MissingFeatureShaderInputAttachmentArrayNonUniformIndexing);
	}
	if (req.needs_physical_storage_buffer_addresses && vk12_features.bufferDeviceAddress != VK_TRUE) {
		vkDestroyInstance(instance, nullptr);
		return runtime_fail(RuntimeFailure::MissingFeatureBufferDeviceAddress);
	}

	if ((subgroup_props.supportedOperations & req.required_subgroup_ops) != req.required_subgroup_ops) {
		uint32_t const	  missing = req.required_subgroup_ops & ~subgroup_props.supportedOperations;
		std::string const detail  = std::format("required=0x{:x}, supported=0x{:x}, missing=0x{:x}",
												req.required_subgroup_ops, subgroup_props.supportedOperations, missing);
		vkDestroyInstance(instance, nullptr);
		return runtime_fail(RuntimeFailure::MissingFeatureSubgroupOpsMask, detail);
	}
	if (needs_subgroup_size_control && subgroup_size_control_features.subgroupSizeControl != VK_TRUE) {
		vkDestroyInstance(instance, nullptr);
		return runtime_fail(RuntimeFailure::MissingFeatureSubgroupSizeControl);
	}
	if (cfg.require_full_subgroups && subgroup_size_control_features.computeFullSubgroups != VK_TRUE) {
		vkDestroyInstance(instance, nullptr);
		return runtime_fail(RuntimeFailure::MissingFeatureComputeFullSubgroups);
	}
	if (cfg.required_subgroup_size > 0) {
		if (cfg.required_subgroup_size < subgroup_size_props.minSubgroupSize ||
			cfg.required_subgroup_size > subgroup_size_props.maxSubgroupSize) {
			vkDestroyInstance(instance, nullptr);
			return runtime_fail(RuntimeFailure::RequestedSubgroupSizeOutOfRange);
		}
		if ((subgroup_size_props.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) == 0) {
			vkDestroyInstance(instance, nullptr);
			return runtime_fail(RuntimeFailure::RequestedSubgroupSizeStageUnsupported);
		}
	}

	if (req.uses_shader_clock) {
		if (req.needs_shader_device_clock && shader_clock_features.shaderDeviceClock != VK_TRUE) {
			vkDestroyInstance(instance, nullptr);
			return runtime_fail(RuntimeFailure::MissingFeatureShaderDeviceClock);
		}
		if (req.needs_shader_subgroup_clock && shader_clock_features.shaderSubgroupClock != VK_TRUE) {
			vkDestroyInstance(instance, nullptr);
			return runtime_fail(RuntimeFailure::MissingFeatureShaderSubgroupClock);
		}
	}

	if (req.uses_cooperative_matrix && cooperative_matrix_features.cooperativeMatrix != VK_TRUE) {
		vkDestroyInstance(instance, nullptr);
		return runtime_fail(RuntimeFailure::MissingFeatureCooperativeMatrix);
	}
	if (req.needs_bfloat16_type && bfloat16_features.shaderBFloat16Type != VK_TRUE) {
		vkDestroyInstance(instance, nullptr);
		return runtime_fail(RuntimeFailure::MissingFeatureShaderBFloat16Type);
	}
	if (req.needs_bfloat16_dot_product && bfloat16_features.shaderBFloat16DotProduct != VK_TRUE) {
		vkDestroyInstance(instance, nullptr);
		return runtime_fail(RuntimeFailure::MissingFeatureShaderBFloat16DotProduct);
	}
	if (req.needs_bfloat16_cooperative_matrix && bfloat16_features.shaderBFloat16CooperativeMatrix != VK_TRUE) {
		vkDestroyInstance(instance, nullptr);
		return runtime_fail(RuntimeFailure::MissingFeatureShaderBFloat16CooperativeMatrix);
	}
	if (req.needs_integer_dot_product && integer_dot_product_features.shaderIntegerDotProduct != VK_TRUE) {
		vkDestroyInstance(instance, nullptr);
		return runtime_fail(RuntimeFailure::MissingFeatureShaderIntegerDotProduct);
	}
	if (req.needs_ray_query && accel_struct_features.accelerationStructure != VK_TRUE) {
		vkDestroyInstance(instance, nullptr);
		return runtime_fail(RuntimeFailure::MissingFeatureAccelerationStructure);
	}
	if (req.needs_ray_query && ray_query_features.rayQuery != VK_TRUE) {
		vkDestroyInstance(instance, nullptr);
		return runtime_fail(RuntimeFailure::MissingFeatureRayQuery);
	}

	VkPhysicalDeviceVulkan12Features vk12_enable = {
			.sType										  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
			.pNext										  = nullptr,
			.samplerMirrorClampToEdge					  = VK_FALSE,
			.drawIndirectCount							  = VK_FALSE,
			.storageBuffer8BitAccess					  = req.needs_storage_buffer_8bit ? VK_TRUE : VK_FALSE,
			.uniformAndStorageBuffer8BitAccess			  = req.needs_uniform_storage_buffer_8bit ? VK_TRUE : VK_FALSE,
			.storagePushConstant8						  = req.needs_storage_push_constant_8bit ? VK_TRUE : VK_FALSE,
			.shaderBufferInt64Atomics					  = VK_FALSE,
			.shaderSharedInt64Atomics					  = VK_FALSE,
			.shaderFloat16								  = req.needs_float16 ? VK_TRUE : VK_FALSE,
			.shaderInt8									  = req.needs_int8 ? VK_TRUE : VK_FALSE,
			.descriptorIndexing							  = VK_FALSE,
			.shaderInputAttachmentArrayDynamicIndexing	  = VK_FALSE,
			.shaderUniformTexelBufferArrayDynamicIndexing = VK_FALSE,
			.shaderStorageTexelBufferArrayDynamicIndexing = VK_FALSE,
			.shaderUniformBufferArrayNonUniformIndexing =
					req.needs_uniform_buffer_array_nonuniform ? VK_TRUE : VK_FALSE,
			.shaderSampledImageArrayNonUniformIndexing = req.needs_sampled_image_array_nonuniform ? VK_TRUE : VK_FALSE,
			.shaderStorageBufferArrayNonUniformIndexing =
					req.needs_storage_buffer_array_nonuniform ? VK_TRUE : VK_FALSE,
			.shaderStorageImageArrayNonUniformIndexing = req.needs_storage_image_array_nonuniform ? VK_TRUE : VK_FALSE,
			.shaderInputAttachmentArrayNonUniformIndexing =
					req.needs_input_attachment_array_nonuniform ? VK_TRUE : VK_FALSE,
			.shaderUniformTexelBufferArrayNonUniformIndexing	= VK_FALSE,
			.shaderStorageTexelBufferArrayNonUniformIndexing	= VK_FALSE,
			.descriptorBindingUniformBufferUpdateAfterBind		= VK_FALSE,
			.descriptorBindingSampledImageUpdateAfterBind		= VK_FALSE,
			.descriptorBindingStorageImageUpdateAfterBind		= VK_FALSE,
			.descriptorBindingStorageBufferUpdateAfterBind		= VK_FALSE,
			.descriptorBindingUniformTexelBufferUpdateAfterBind = VK_FALSE,
			.descriptorBindingStorageTexelBufferUpdateAfterBind = VK_FALSE,
			.descriptorBindingUpdateUnusedWhilePending			= VK_FALSE,
			.descriptorBindingPartiallyBound					= VK_FALSE,
			.descriptorBindingVariableDescriptorCount			= VK_FALSE,
			.runtimeDescriptorArray			  = req.needs_runtime_descriptor_array ? VK_TRUE : VK_FALSE,
			.samplerFilterMinmax			  = VK_FALSE,
			.scalarBlockLayout				  = VK_FALSE,
			.imagelessFramebuffer			  = VK_FALSE,
			.uniformBufferStandardLayout	  = VK_FALSE,
			.shaderSubgroupExtendedTypes	  = VK_FALSE,
			.separateDepthStencilLayouts	  = VK_FALSE,
			.hostQueryReset					  = VK_FALSE,
			.timelineSemaphore				  = VK_FALSE,
			.bufferDeviceAddress			  = needs_bda ? VK_TRUE : VK_FALSE,
			.bufferDeviceAddressCaptureReplay = VK_FALSE,
			.bufferDeviceAddressMultiDevice	  = VK_FALSE,
			.vulkanMemoryModel				  = req.needs_vulkan_memory_model ? VK_TRUE : VK_FALSE,
			.vulkanMemoryModelDeviceScope	  = req.needs_vulkan_memory_model_device_scope ? VK_TRUE : VK_FALSE,
			.vulkanMemoryModelAvailabilityVisibilityChains = VK_FALSE,
			.shaderOutputViewportIndex					   = VK_FALSE,
			.shaderOutputLayer							   = VK_FALSE,
			.subgroupBroadcastDynamicId					   = VK_FALSE,
	};

	VkPhysicalDeviceShaderUntypedPointersFeaturesKHR untyped_ptr_enable = {
			.sType				   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_UNTYPED_POINTERS_FEATURES_KHR,
			.pNext				   = nullptr,
			.shaderUntypedPointers = needs_untyped_pointers ? VK_TRUE : VK_FALSE,
	};
	VkPhysicalDevice16BitStorageFeatures storage16_enable = {
			.sType								= VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
			.pNext								= nullptr,
			.storageBuffer16BitAccess			= req.needs_storage_buffer_16bit ? VK_TRUE : VK_FALSE,
			.uniformAndStorageBuffer16BitAccess = req.needs_uniform_storage_buffer_16bit ? VK_TRUE : VK_FALSE,
			.storagePushConstant16				= req.needs_storage_push_constant_16bit ? VK_TRUE : VK_FALSE,
			.storageInputOutput16				= VK_FALSE,
	};

	VkPhysicalDeviceShaderClockFeaturesKHR shader_clock_enable = {
			.sType				 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR,
			.pNext				 = nullptr,
			.shaderSubgroupClock = req.needs_shader_subgroup_clock ? VK_TRUE : VK_FALSE,
			.shaderDeviceClock	 = req.needs_shader_device_clock ? VK_TRUE : VK_FALSE,
	};

	VkPhysicalDeviceCooperativeMatrixFeaturesKHR cooperative_matrix_enable = {
			.sType								 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
			.pNext								 = nullptr,
			.cooperativeMatrix					 = req.uses_cooperative_matrix ? VK_TRUE : VK_FALSE,
			.cooperativeMatrixRobustBufferAccess = VK_FALSE,
	};
	VkPhysicalDeviceShaderBfloat16FeaturesKHR bfloat16_enable = {
			.sType							 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR,
			.pNext							 = nullptr,
			.shaderBFloat16Type				 = req.needs_bfloat16_type ? VK_TRUE : VK_FALSE,
			.shaderBFloat16DotProduct		 = req.needs_bfloat16_dot_product ? VK_TRUE : VK_FALSE,
			.shaderBFloat16CooperativeMatrix = req.needs_bfloat16_cooperative_matrix ? VK_TRUE : VK_FALSE,
	};
	VkPhysicalDeviceShaderIntegerDotProductFeatures integer_dot_product_enable = {
			.sType					 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES,
			.pNext					 = nullptr,
			.shaderIntegerDotProduct = req.needs_integer_dot_product ? VK_TRUE : VK_FALSE,
	};
	VkPhysicalDeviceAccelerationStructureFeaturesKHR accel_struct_enable = {
			.sType								= VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
			.pNext								= nullptr,
			.accelerationStructure				= req.needs_ray_query ? VK_TRUE : VK_FALSE,
			.accelerationStructureCaptureReplay = VK_FALSE,
			.accelerationStructureIndirectBuild = VK_FALSE,
			.accelerationStructureHostCommands	= VK_FALSE,
			.descriptorBindingAccelerationStructureUpdateAfterBind = VK_FALSE,
	};
	VkPhysicalDeviceRayQueryFeaturesKHR ray_query_enable = {
			.sType	  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR,
			.pNext	  = nullptr,
			.rayQuery = req.needs_ray_query ? VK_TRUE : VK_FALSE,
	};
	VkPhysicalDeviceSubgroupSizeControlFeatures subgroup_size_control_enable = {
			.sType				  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES,
			.pNext				  = nullptr,
			.subgroupSizeControl  = needs_subgroup_size_control ? VK_TRUE : VK_FALSE,
			.computeFullSubgroups = cfg.require_full_subgroups ? VK_TRUE : VK_FALSE,
	};
	VkPhysicalDeviceShaderSubgroupRotateFeatures subgroup_rotate_enable = {
			.sType						   = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_ROTATE_FEATURES,
			.pNext						   = nullptr,
			.shaderSubgroupRotate		   = needs_subgroup_rotate ? VK_TRUE : VK_FALSE,
			.shaderSubgroupRotateClustered = needs_subgroup_rotate_clustered ? VK_TRUE : VK_FALSE,
	};
	VkPhysicalDeviceShaderFloatControls2Features shader_float_controls2_enable = {
			.sType				  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT_CONTROLS_2_FEATURES,
			.pNext				  = nullptr,
			.shaderFloatControls2 = req.needs_shader_float_controls2 ? VK_TRUE : VK_FALSE,
	};
	VkPhysicalDeviceShaderExpectAssumeFeatures shader_expect_assume_enable = {
			.sType				= VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_EXPECT_ASSUME_FEATURES,
			.pNext				= nullptr,
			.shaderExpectAssume = req.needs_shader_expect_assume ? VK_TRUE : VK_FALSE,
	};
	VkPhysicalDeviceVulkan14Features vk14_enable = {
			.sType									= VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES,
			.pNext									= nullptr,
			.globalPriorityQuery					= VK_FALSE,
			.shaderSubgroupRotate					= needs_subgroup_rotate ? VK_TRUE : VK_FALSE,
			.shaderSubgroupRotateClustered			= needs_subgroup_rotate_clustered ? VK_TRUE : VK_FALSE,
			.shaderFloatControls2					= req.needs_shader_float_controls2 ? VK_TRUE : VK_FALSE,
			.shaderExpectAssume						= req.needs_shader_expect_assume ? VK_TRUE : VK_FALSE,
			.rectangularLines						= VK_FALSE,
			.bresenhamLines							= VK_FALSE,
			.smoothLines							= VK_FALSE,
			.stippledRectangularLines				= VK_FALSE,
			.stippledBresenhamLines					= VK_FALSE,
			.stippledSmoothLines					= VK_FALSE,
			.vertexAttributeInstanceRateDivisor		= VK_FALSE,
			.vertexAttributeInstanceRateZeroDivisor = VK_FALSE,
			.indexTypeUint8							= VK_FALSE,
			.dynamicRenderingLocalRead				= VK_FALSE,
			.maintenance5							= VK_FALSE,
			.maintenance6							= VK_FALSE,
			.pipelineProtectedAccess				= VK_FALSE,
			.pipelineRobustness						= VK_FALSE,
			.hostImageCopy							= VK_FALSE,
			.pushDescriptor = cfg.binding_model == BindingModel::PushDescriptor ? VK_TRUE : VK_FALSE,
	};

	VkPhysicalDeviceFeatures const enabled_features = {
			.robustBufferAccess						 = VK_FALSE,
			.fullDrawIndexUint32					 = VK_FALSE,
			.imageCubeArray							 = VK_FALSE,
			.independentBlend						 = VK_FALSE,
			.geometryShader							 = VK_FALSE,
			.tessellationShader						 = VK_FALSE,
			.sampleRateShading						 = VK_FALSE,
			.dualSrcBlend							 = VK_FALSE,
			.logicOp								 = VK_FALSE,
			.multiDrawIndirect						 = VK_FALSE,
			.drawIndirectFirstInstance				 = VK_FALSE,
			.depthClamp								 = VK_FALSE,
			.depthBiasClamp							 = VK_FALSE,
			.fillModeNonSolid						 = VK_FALSE,
			.depthBounds							 = VK_FALSE,
			.wideLines								 = VK_FALSE,
			.largePoints							 = VK_FALSE,
			.alphaToOne								 = VK_FALSE,
			.multiViewport							 = VK_FALSE,
			.samplerAnisotropy						 = VK_FALSE,
			.textureCompressionETC2					 = VK_FALSE,
			.textureCompressionASTC_LDR				 = VK_FALSE,
			.textureCompressionBC					 = VK_FALSE,
			.occlusionQueryPrecise					 = VK_FALSE,
			.pipelineStatisticsQuery				 = VK_FALSE,
			.vertexPipelineStoresAndAtomics			 = VK_FALSE,
			.fragmentStoresAndAtomics				 = VK_FALSE,
			.shaderTessellationAndGeometryPointSize	 = VK_FALSE,
			.shaderImageGatherExtended				 = VK_FALSE,
			.shaderStorageImageExtendedFormats		 = VK_FALSE,
			.shaderStorageImageMultisample			 = VK_FALSE,
			.shaderStorageImageReadWithoutFormat	 = VK_FALSE,
			.shaderStorageImageWriteWithoutFormat	 = VK_FALSE,
			.shaderUniformBufferArrayDynamicIndexing = VK_FALSE,
			.shaderSampledImageArrayDynamicIndexing	 = VK_FALSE,
			.shaderStorageBufferArrayDynamicIndexing = VK_FALSE,
			.shaderStorageImageArrayDynamicIndexing	 = VK_FALSE,
			.shaderClipDistance						 = VK_FALSE,
			.shaderCullDistance						 = VK_FALSE,
			.shaderFloat64							 = req.needs_float64 ? VK_TRUE : VK_FALSE,
			.shaderInt64							 = req.needs_int64 ? VK_TRUE : VK_FALSE,
			.shaderInt16							 = req.needs_int16 ? VK_TRUE : VK_FALSE,
			.shaderResourceResidency				 = VK_FALSE,
			.shaderResourceMinLod					 = VK_FALSE,
			.sparseBinding							 = VK_FALSE,
			.sparseResidencyBuffer					 = VK_FALSE,
			.sparseResidencyImage2D					 = VK_FALSE,
			.sparseResidencyImage3D					 = VK_FALSE,
			.sparseResidency2Samples				 = VK_FALSE,
			.sparseResidency4Samples				 = VK_FALSE,
			.sparseResidency8Samples				 = VK_FALSE,
			.sparseResidency16Samples				 = VK_FALSE,
			.sparseResidencyAliased					 = VK_FALSE,
			.variableMultisampleRate				 = VK_FALSE,
			.inheritedQueries						 = VK_FALSE,
	};
	device_ci.pEnabledFeatures = &enabled_features;

	VkPhysicalDeviceBufferDeviceAddressFeatures bda_features = {
			.sType							  = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,
			.pNext							  = nullptr,
			.bufferDeviceAddress			  = needs_bda ? VK_TRUE : VK_FALSE,
			.bufferDeviceAddressCaptureReplay = VK_FALSE,
			.bufferDeviceAddressMultiDevice	  = VK_FALSE,
	};

	void *feature_chain = nullptr;
	auto  push_feature	= [&](auto *feature_struct) {
		  auto *base	= reinterpret_cast<VkBaseOutStructure *>(feature_struct);
		  base->pNext	= static_cast<VkBaseOutStructure *>(feature_chain);
		  feature_chain = base;
	};

	if (supports_pipeline_exec) { push_feature(&pipeline_exec_features); }
	if (cfg.binding_model == BindingModel::DescriptorBuffer) { push_feature(&descriptor_buffer_features); }
	bool use_vk12_enable = vk12_enable.shaderInt8 == VK_TRUE || vk12_enable.shaderFloat16 == VK_TRUE ||
						   vk12_enable.storageBuffer8BitAccess == VK_TRUE ||
						   vk12_enable.uniformAndStorageBuffer8BitAccess == VK_TRUE ||
						   vk12_enable.storagePushConstant8 == VK_TRUE ||
						   vk12_enable.runtimeDescriptorArray == VK_TRUE || vk12_enable.vulkanMemoryModel == VK_TRUE ||
						   vk12_enable.vulkanMemoryModelDeviceScope == VK_TRUE ||
						   vk12_enable.shaderUniformBufferArrayNonUniformIndexing == VK_TRUE ||
						   vk12_enable.shaderSampledImageArrayNonUniformIndexing == VK_TRUE ||
						   vk12_enable.shaderStorageBufferArrayNonUniformIndexing == VK_TRUE ||
						   vk12_enable.shaderStorageImageArrayNonUniformIndexing == VK_TRUE ||
						   vk12_enable.shaderInputAttachmentArrayNonUniformIndexing == VK_TRUE ||
						   vk12_enable.bufferDeviceAddress == VK_TRUE;
	bool const use_storage16_enable = storage16_enable.storageBuffer16BitAccess == VK_TRUE ||
									  storage16_enable.uniformAndStorageBuffer16BitAccess == VK_TRUE ||
									  storage16_enable.storagePushConstant16 == VK_TRUE;
	bool const use_vk14_enable = vk14_enable.shaderSubgroupRotate == VK_TRUE ||
								 vk14_enable.shaderSubgroupRotateClustered == VK_TRUE ||
								 vk14_enable.shaderFloatControls2 == VK_TRUE ||
								 vk14_enable.shaderExpectAssume == VK_TRUE || vk14_enable.pushDescriptor == VK_TRUE;
	bool const use_shader_expect_assume_enable	 = shader_expect_assume_enable.shaderExpectAssume == VK_TRUE;
	bool const use_shader_float_controls2_enable = shader_float_controls2_enable.shaderFloatControls2 == VK_TRUE;
	bool const use_subgroup_rotate_enable		 = subgroup_rotate_enable.shaderSubgroupRotate == VK_TRUE ||
											subgroup_rotate_enable.shaderSubgroupRotateClustered == VK_TRUE;
	if (cfg.binding_model == BindingModel::DescriptorHeap) {
		push_feature(&descriptor_heap_features);
		push_feature(&untyped_ptr_enable);
		use_vk12_enable = true;
	} else if (needs_untyped_pointers) {
		push_feature(&untyped_ptr_enable);
	}
	if (has_vulkan_1_4 && use_vk14_enable) { push_feature(&vk14_enable); }
	if (!has_vulkan_1_4 && use_shader_expect_assume_enable) { push_feature(&shader_expect_assume_enable); }
	if (!has_vulkan_1_4 && use_shader_float_controls2_enable) { push_feature(&shader_float_controls2_enable); }
	if (!has_vulkan_1_4 && use_subgroup_rotate_enable) { push_feature(&subgroup_rotate_enable); }
	if (use_vk12_enable) { push_feature(&vk12_enable); }
	if (use_storage16_enable) { push_feature(&storage16_enable); }
	if (req.uses_shader_clock) { push_feature(&shader_clock_enable); }
	if (req.uses_cooperative_matrix) { push_feature(&cooperative_matrix_enable); }
	if (needs_bfloat16) { push_feature(&bfloat16_enable); }
	if (req.needs_integer_dot_product) { push_feature(&integer_dot_product_enable); }
	if (req.needs_ray_query) {
		push_feature(&accel_struct_enable);
		push_feature(&ray_query_enable);
	}
	if (needs_subgroup_size_control) { push_feature(&subgroup_size_control_enable); }
	if (needs_bda) { push_feature(&bda_features); }
	device_ci.pNext = feature_chain;

	VkDevice device = VK_NULL_HANDLE;
	if (vkCreateDevice(physical_device, &device_ci, nullptr, &device) != VK_SUCCESS) {
		vkDestroyInstance(instance, nullptr);
		return runtime_fail(RuntimeFailure::VkCreateDeviceFailed);
	}

	VkShaderModuleCreateInfo const module_ci = {
			.sType	  = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
			.pNext	  = nullptr,
			.flags	  = 0,
			.codeSize = spirv.size() * sizeof(uint32_t),
			.pCode	  = spirv.data(),
	};

	VkShaderModule shader_module = VK_NULL_HANDLE;
	if (vkCreateShaderModule(device, &module_ci, nullptr, &shader_module) != VK_SUCCESS) {
		vkDestroyDevice(device, nullptr);
		vkDestroyInstance(instance, nullptr);
		return runtime_fail(RuntimeFailure::VkCreateShaderModuleFailed);
	}

	auto reflected = reflect_descriptor_bindings(spirv);

	std::map<uint32_t, std::vector<VkDescriptorSetLayoutBinding>> set_bindings;
	for (const auto &b: reflected) {
		auto &vec = set_bindings[b.set];
		vec.push_back(VkDescriptorSetLayoutBinding{
				.binding			= b.binding,
				.descriptorType		= b.type,
				.descriptorCount	= b.count,
				.stageFlags			= VK_SHADER_STAGE_COMPUTE_BIT,
				.pImmutableSamplers = nullptr,
		});
	}

	std::vector<VkDescriptorSetLayout> descriptor_set_layouts;
	std::vector<VkDescriptorSetLayout> pipeline_set_layouts;
	if (!set_bindings.empty()) {
		uint32_t const max_set = set_bindings.rbegin()->first;
		pipeline_set_layouts.assign(max_set + 1, VK_NULL_HANDLE);
		descriptor_set_layouts.reserve(set_bindings.size());
		for (auto &[set_idx, bindings]: set_bindings) {
			VkDescriptorSetLayoutCreateInfo const ds_layout_ci = {
					.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
					.pNext = nullptr,
					.flags = (cfg.binding_model == BindingModel::PushDescriptor
									  ? static_cast<VkDescriptorSetLayoutCreateFlags>(
												VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR)
									  : VkDescriptorSetLayoutCreateFlags{0}) |
							 (cfg.binding_model == BindingModel::DescriptorBuffer
									  ? static_cast<VkDescriptorSetLayoutCreateFlags>(
												VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT)
									  : VkDescriptorSetLayoutCreateFlags{0}),
					.bindingCount = uint32_t(bindings.size()),
					.pBindings	  = bindings.data(),
			};

			VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
			if (vkCreateDescriptorSetLayout(device, &ds_layout_ci, nullptr, &set_layout) != VK_SUCCESS) {
				for (VkDescriptorSetLayout created: descriptor_set_layouts) {
					vkDestroyDescriptorSetLayout(device, created, nullptr);
				}
				vkDestroyShaderModule(device, shader_module, nullptr);
				vkDestroyDevice(device, nullptr);
				vkDestroyInstance(instance, nullptr);
				return runtime_fail(RuntimeFailure::VkCreateDescriptorSetLayoutFailed, std::to_string(set_idx));
			}
			descriptor_set_layouts.push_back(set_layout);
			pipeline_set_layouts[set_idx] = set_layout;
		}
	}

	VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
	if (cfg.binding_model != BindingModel::DescriptorHeap) {
		uint32_t const push_constant_size = physical_device_props.limits.maxPushConstantsSize;
		if (push_constant_size == 0) {
			vkDestroyShaderModule(device, shader_module, nullptr);
			vkDestroyDevice(device, nullptr);
			vkDestroyInstance(instance, nullptr);
			return runtime_fail(RuntimeFailure::VkCreatePipelineLayoutFailed, "device reports maxPushConstantsSize=0");
		}
		VkPushConstantRange const push_range = {
				.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
				.offset		= 0,
				.size		= push_constant_size,
		};

		VkPipelineLayoutCreateInfo const layout_ci = {
				.sType					= VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
				.pNext					= nullptr,
				.flags					= 0,
				.setLayoutCount			= uint32_t(pipeline_set_layouts.size()),
				.pSetLayouts			= pipeline_set_layouts.empty() ? nullptr : pipeline_set_layouts.data(),
				.pushConstantRangeCount = 1,
				.pPushConstantRanges	= &push_range,
		};

		if (vkCreatePipelineLayout(device, &layout_ci, nullptr, &pipeline_layout) != VK_SUCCESS) {
			for (VkDescriptorSetLayout set_layout: descriptor_set_layouts) {
				vkDestroyDescriptorSetLayout(device, set_layout, nullptr);
			}
			vkDestroyShaderModule(device, shader_module, nullptr);
			vkDestroyDevice(device, nullptr);
			vkDestroyInstance(instance, nullptr);
			return runtime_fail(
					RuntimeFailure::VkCreatePipelineLayoutFailed,
					std::format("vkCreatePipelineLayout failed (maxPushConstantsSize={})", push_constant_size));
		}
	}

	VkPipeline pipeline = VK_NULL_HANDLE;

	VkPipelineShaderStageRequiredSubgroupSizeCreateInfo required_subgroup_size_create_info = {
			.sType				  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
			.pNext				  = nullptr,
			.requiredSubgroupSize = cfg.required_subgroup_size,
	};

	VkPipelineShaderStageCreateInfo const stage_create_info = {
			.sType				 = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.pNext				 = (cfg.required_subgroup_size > 0 ? &required_subgroup_size_create_info : nullptr),
			.flags				 = (cfg.require_full_subgroups ? static_cast<VkPipelineShaderStageCreateFlags>(
															 VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT)
															   : VkPipelineShaderStageCreateFlags{0}),
			.stage				 = VK_SHADER_STAGE_COMPUTE_BIT,
			.module				 = shader_module,
			.pName				 = "main",
			.pSpecializationInfo = nullptr,
	};

	VkPipelineCreateFlags2CreateInfoKHR pipeline_flags2_ci = {
			.sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO_KHR,
			.pNext = nullptr,
			.flags = VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR |
					 (cfg.binding_model == BindingModel::DescriptorBuffer
							  ? VK_PIPELINE_CREATE_2_DESCRIPTOR_BUFFER_BIT_EXT
							  : 0) |
					 (cfg.binding_model == BindingModel::DescriptorHeap ? VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT
																		: 0),
	};
	VkComputePipelineCreateInfo const pipeline_ci = {
			.sType	= VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
			.pNext	= (pipeline_flags2_ci.flags != 0 ? &pipeline_flags2_ci : nullptr),
			.flags	= VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR,
			.stage	= stage_create_info,
			.layout = (cfg.binding_model == BindingModel::DescriptorHeap ? VK_NULL_HANDLE : pipeline_layout),
			.basePipelineHandle = VK_NULL_HANDLE,
			.basePipelineIndex	= 0,
	};

	if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_ci, nullptr, &pipeline) != VK_SUCCESS) {
		if (pipeline_layout != VK_NULL_HANDLE) { vkDestroyPipelineLayout(device, pipeline_layout, nullptr); }
		for (VkDescriptorSetLayout set_layout: descriptor_set_layouts) {
			vkDestroyDescriptorSetLayout(device, set_layout, nullptr);
		}
		vkDestroyShaderModule(device, shader_module, nullptr);
		vkDestroyDevice(device, nullptr);
		vkDestroyInstance(instance, nullptr);
		return runtime_fail(RuntimeFailure::VkCreateComputePipelineFailed);
	}

	auto vkGetPipelineExecutablePropertiesKHR = reinterpret_cast<PFN_vkGetPipelineExecutablePropertiesKHR>(
			vkGetDeviceProcAddr(device, "vkGetPipelineExecutablePropertiesKHR"));
	auto vkGetPipelineExecutableInternalRepresentationsKHR =
			reinterpret_cast<PFN_vkGetPipelineExecutableInternalRepresentationsKHR>(
					vkGetDeviceProcAddr(device, "vkGetPipelineExecutableInternalRepresentationsKHR"));

	if ((vkGetPipelineExecutablePropertiesKHR == nullptr) ||
		(vkGetPipelineExecutableInternalRepresentationsKHR == nullptr)) {
		return runtime_fail(RuntimeFailure::MissingPipelineExecutableFunctions);
	}
	VkPipelineInfoKHR const	 pipeline_info = {
			 .sType	   = VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR,
			 .pNext	   = nullptr,
			 .pipeline = pipeline,
	 };

	uint32_t exe_count = 0;
	vkGetPipelineExecutablePropertiesKHR(device, &pipeline_info, &exe_count, nullptr);

	if (exe_count != 1) { return runtime_fail(RuntimeFailure::UnexpectedExecutableCount); }

	VkPipelineExecutablePropertiesKHR executable = {
			.sType		  = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR,
			.pNext		  = nullptr,
			.stages		  = 0,
			.name		  = "",
			.description  = "",
			.subgroupSize = 0,
	};
	vkGetPipelineExecutablePropertiesKHR(device, &pipeline_info, &exe_count, &executable);

	VkPipelineExecutableInfoKHR const exe_info = {
			.sType			 = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR,
			.pNext			 = nullptr,
			.pipeline		 = pipeline,
			.executableIndex = 0,
	};

	uint32_t repr_count = 0;

	vkGetPipelineExecutableInternalRepresentationsKHR(device, &exe_info, &repr_count, nullptr);

	std::vector<std::optional<std::string>>			   text_reprs(repr_count);
	std::vector<VkPipelineExecutableInternalRepresentationKHR> representations(repr_count);
	std::vector<std::vector<char>>							   buffers(repr_count);
	for (uint32_t repr_index = 0; repr_index < repr_count; ++repr_index) {
		representations[repr_index] = {
				.sType		 = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION_KHR,
				.pNext		 = nullptr,
				.name		 = "",
				.description = "",
				.isText		 = VK_FALSE,
				.dataSize	 = 0,
				.pData		 = nullptr,
		};
	}

	(void) vkGetPipelineExecutableInternalRepresentationsKHR(device, &exe_info, &repr_count, representations.data());
	for (uint32_t repr_index = 0; repr_index < repr_count; ++repr_index) {
		buffers[repr_index].resize(representations[repr_index].dataSize);

		representations[repr_index].pData	 = buffers[repr_index].data();
		representations[repr_index].dataSize = buffers[repr_index].size();
	}
	VkResult const repr_result =
			vkGetPipelineExecutableInternalRepresentationsKHR(device, &exe_info, &repr_count, representations.data());
	if (repr_result != VK_SUCCESS && repr_result != VK_INCOMPLETE) {
		return runtime_fail(RuntimeFailure::FailedToSelectAsmRepresentation);
	}

	for (uint32_t repr_index = 0; repr_index < repr_count; ++repr_index) {
		if (representations[repr_index].isText == 0U) { continue; }

		size_t repr_size = buffers[repr_index].size();
		if (repr_size > 0 && buffers[repr_index][repr_size - 1] == '\0') { --repr_size; }

		text_reprs[repr_index].emplace(buffers[repr_index].data(), repr_size);
	}


	if ((options.target == RuntimeOutputTarget::FinalNir || options.target == RuntimeOutputTarget::Asm) &&
		text_reprs.empty()) {
		vkDestroyPipeline(device, pipeline, nullptr);
		if (pipeline_layout != VK_NULL_HANDLE) { vkDestroyPipelineLayout(device, pipeline_layout, nullptr); }
		for (VkDescriptorSetLayout set_layout: descriptor_set_layouts) {
			vkDestroyDescriptorSetLayout(device, set_layout, nullptr);
		}
		vkDestroyShaderModule(device, shader_module, nullptr);
		vkDestroyDevice(device, nullptr);
		vkDestroyInstance(instance, nullptr);
		return runtime_fail(RuntimeFailure::NoTextInternalRepresentation,
							cfg.gpu_key + ":" + std::to_string(exe_count));
	}
	using namespace std::string_view_literals;
	const std::vector nir_names{
			"Final NIR"sv,
			"NIR shader"sv,
			"NIR Shader(s)"sv,
	};
	const std::vector asm_names{
			"Assembly"sv,
			"GEN Assembly"sv,
			"NAK assembly"sv,
			"IR3 Assembly"sv,
	};

	const auto &names_references =
			(options.target == RuntimeOutputTarget::FinalNir ? nir_names : /* RuntimeOutputTarget::Asm */ asm_names);


	for (uint32_t repr_index = 0; repr_index < repr_count; ++repr_index) {
		if (std::ranges::contains(names_references, representations[repr_index].name)) {
			if (!text_reprs[repr_index].has_value()) { continue; }
			out_text = *text_reprs[repr_index];
			goto done;
		}
	}

	vkDestroyPipeline(device, pipeline, nullptr);
	if (pipeline_layout != VK_NULL_HANDLE) { vkDestroyPipelineLayout(device, pipeline_layout, nullptr); }
	for (VkDescriptorSetLayout set_layout: descriptor_set_layouts) {
		vkDestroyDescriptorSetLayout(device, set_layout, nullptr);
	}
	vkDestroyShaderModule(device, shader_module, nullptr);
	vkDestroyDevice(device, nullptr);
	vkDestroyInstance(instance, nullptr);
	return runtime_fail(RuntimeFailure::FailedToSelectFinalNirRepresentation);

done:
	vkDestroyPipeline(device, pipeline, nullptr);
	if (pipeline_layout != VK_NULL_HANDLE) { vkDestroyPipelineLayout(device, pipeline_layout, nullptr); }
	for (VkDescriptorSetLayout set_layout: descriptor_set_layouts) {
		vkDestroyDescriptorSetLayout(device, set_layout, nullptr);
	}
	vkDestroyShaderModule(device, shader_module, nullptr);
	vkDestroyDevice(device, nullptr);
	vkDestroyInstance(instance, nullptr);

	return runtime_ok();
}
