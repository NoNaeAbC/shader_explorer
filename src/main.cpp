#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <print>
#include <sstream>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

#define SLANG_STATIC 1
#include <SPIRV/GlslangToSpv.h>
#include <StandAlone/DirStackFileIncluder.h>
#include <glslang/Public/ResourceLimits.h>
#include <glslang/Public/ShaderLang.h>
#include <slang-com-ptr.h>
#include <slang.h>
#include <spirv-tools/libspirv.hpp>
#include <spirv-tools/optimizer.hpp>

#include "vulkan_runtime.hpp"

namespace fs = std::filesystem;
namespace {

	struct device_config {
		std::string						   key;
		std::string						   vendor;
		std::string						   description;
		std::string						   vulkan_icd_file;
		std::string						   shim_file;
		std::map<std::string, std::string> env;
	};

	enum class OutputTarget : std::uint8_t {
		Info,
		Spirv,
		FinalNir,
		Asm,
	};

	enum class SourceLanguage : std::uint8_t {
		Auto,
		Slang,
		Glsl,
	};

	enum class SpirvTargetMode : std::uint8_t {
		Max,
		Explicit,
	};

	enum class InternalMode : std::uint8_t {
		None,
		Info,
		Pipeline,
		SpirvMax,
	};

	enum class InternalFailure : std::uint8_t {
		None = 0,
		MissingResponseFd,
		InvalidResponseFd,
		ChildReadSpirvFailed,
		ChildRuntimeFailed,
		ChildResponseWriteFailed,
		PipeCreateFailed,
		ForkFailed,
		ChildExecFailed,
		ParentReadFailed,
		ParentWaitFailed,
		InvalidResponseHeader,
		MissingResponsePayload,
		ChildExitedNonZero,
		ChildSignaled,
	};

	struct ChildResponseHeader {
		uint32_t magic			  = 0x53455852U; // "SEXR"
		uint32_t version		  = 1;
		uint32_t internal_failure = 0;
		uint32_t runtime_failure  = 0;
		int32_t	 sys_errno		  = 0;
		uint32_t detail_size	  = 0;
		uint32_t output_size	  = 0;
	};

	struct ChildResponse {
		InternalFailure internal_failure = InternalFailure::None;
		RuntimeFailure	runtime_failure	 = RuntimeFailure::None;
		int				sys_errno		 = 0;
		std::string		detail;
		std::string		output;
	};

	struct CliOptions {
		std::string				gpu_key;
		bool					list_gpus				= false;
		OutputTarget			output_target			= OutputTarget::Asm;
		SourceLanguage			source_lang				= SourceLanguage::Auto;
		SpirvTargetMode			spirv_target_mode		= SpirvTargetMode::Max;
		SpirvTargetVersion		requested_spirv_target	= SpirvTargetVersion::V10;
		std::string				binding_model			= "classic";
		size_t					requested_subgroup_size = 0;
		bool					require_full_subgroups	= false;
		bool					no_color				= false;
		std::string				output_path				= "-";
		std::optional<fs::path> shader_path;
		InternalMode			internal_mode		 = InternalMode::None;
		bool					internal_shim_ready	 = false;
		int						internal_spirv_fd	 = -1;
		size_t					internal_spirv_bytes = 0;
		int						internal_response_fd = -1;
	};


	bool parse_source_language(std::string_view text, SourceLanguage &out_lang) {
		if (text == "auto") {
			out_lang = SourceLanguage::Auto;
			return true;
		}
		if (text == "slang") {
			out_lang = SourceLanguage::Slang;
			return true;
		}
		if (text == "glsl") {
			out_lang = SourceLanguage::Glsl;
			return true;
		}
		return false;
	}

	const char *spirv_target_to_string(SpirvTargetVersion v) {
		switch (v) {
			case SpirvTargetVersion::V10:
				return "1.0";
			case SpirvTargetVersion::V11:
				return "1.1";
			case SpirvTargetVersion::V12:
				return "1.2";
			case SpirvTargetVersion::V13:
				return "1.3";
			case SpirvTargetVersion::V14:
				return "1.4";
			case SpirvTargetVersion::V15:
				return "1.5";
			case SpirvTargetVersion::V16:
				return "1.6";
		}
		return "unknown";
	}

	bool parse_spirv_target_arg(std::string_view text, SpirvTargetMode &out_mode, SpirvTargetVersion &out_version) {
		if (text == "max") {
			out_mode = SpirvTargetMode::Max;
			return true;
		}
		out_mode = SpirvTargetMode::Explicit;
		if (text == "1.0") {
			out_version = SpirvTargetVersion::V10;
			return true;
		}
		if (text == "1.1") {
			out_version = SpirvTargetVersion::V11;
			return true;
		}
		if (text == "1.2") {
			out_version = SpirvTargetVersion::V12;
			return true;
		}
		if (text == "1.3") {
			out_version = SpirvTargetVersion::V13;
			return true;
		}
		if (text == "1.4") {
			out_version = SpirvTargetVersion::V14;
			return true;
		}
		if (text == "1.5") {
			out_version = SpirvTargetVersion::V15;
			return true;
		}
		if (text == "1.6") {
			out_version = SpirvTargetVersion::V16;
			return true;
		}
		return false;
	}

	std::string_view trim(std::string_view text) {
		const char *start = text.data();
		const char *end	  = text.data() + text.size();

		while (start < end && isspace(*start)) { ++start; }
		while (start < end && isspace(*end)) { --end; }

		return std::string_view(start, end - start);
	}

	bool parse_spirv_version_text(const std::string_view text, SpirvTargetVersion &out_version) {
		const std::string_view trimmed_tex = trim(text);
		SpirvTargetMode		   mode		   = SpirvTargetMode::Explicit;
		return parse_spirv_target_arg(trimmed_tex, mode, out_version) && mode == SpirvTargetMode::Explicit;
	}

#ifndef SHADER_EXPLORER_DEFAULT_MESA_LIBDIR
#define SHADER_EXPLORER_DEFAULT_MESA_LIBDIR ""
#endif
#ifndef SHADER_EXPLORER_DEFAULT_VULKAN_ICD_DIR
#define SHADER_EXPLORER_DEFAULT_VULKAN_ICD_DIR ""
#endif

	struct SpirvSharedMemory {
		int	   fd	 = -1;
		size_t bytes = 0;
	};

	bool create_spirv_shared_memory(const std::vector<uint32_t> &spirv, SpirvSharedMemory &out_mem) {
		out_mem.fd	  = -1;
		out_mem.bytes = spirv.size() * sizeof(uint32_t);
		if (out_mem.bytes == 0) { return false; }

		int const fd = memfd_create("shader_explorer_spirv", 0);
		if (fd < 0) { return false; }
		if (ftruncate(fd, static_cast<off_t>(out_mem.bytes)) != 0) {
			close(fd);
			return false;
		}

		void *mapping = mmap(nullptr, out_mem.bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
		if (mapping == MAP_FAILED) {
			close(fd);
			return false;
		}
		std::memcpy(mapping, spirv.data(), out_mem.bytes);
		munmap(mapping, out_mem.bytes);

		out_mem.fd = fd;
		return true;
	}

	bool read_spirv_from_shared_memory(int fd, size_t bytes, std::vector<uint32_t> &out_spirv) {
		if (fd < 0) { return false; }
		if (bytes == 0 || (bytes % sizeof(uint32_t)) != 0) { return false; }

		void *mapping = mmap(nullptr, bytes, PROT_READ, MAP_SHARED, fd, 0);
		if (mapping == MAP_FAILED) { return false; }

		out_spirv.resize(bytes / sizeof(uint32_t));
		std::memcpy(out_spirv.data(), mapping, bytes);
		munmap(mapping, bytes);
		return true;
	}

	std::string read_text_file(const fs::path &path) {
		int const fd = open(path.c_str(), O_RDONLY);
		if (fd < 0) { return ""; }
		struct stat st;
		if (fstat(fd, &st) != 0 || st.st_size <= 0) {
			close(fd);
			return "";
		}

		auto const size	   = static_cast<size_t>(st.st_size);
		void	  *mapping = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
		close(fd);
		if (mapping == MAP_FAILED) { return ""; }

		std::string out(static_cast<const char *>(mapping), size);
		munmap(mapping, size);
		return out;
	}

	void maybe_add_device(std::map<std::string, device_config> &catalog, const device_config &cfg) {
		if (!catalog.contains(cfg.key)) { catalog.emplace(cfg.key, cfg); }
	}

#include "generated_gpu_catalog.inc"

	std::map<std::string, device_config> build_device_catalog() {
		std::map<std::string, device_config> catalog;
		append_generated_device_catalog(catalog);
		return catalog;
	}

	bool parse_size_arg(const char *text, size_t &out_value) {
		if ((text == nullptr) || (text[0] == 0)) { return false; }
		char *end						= nullptr;
		errno							= 0;
		unsigned long long const parsed = std::strtoull(text, &end, 10);
		if (errno != 0 || (end == nullptr) || *end != '\0') { return false; }
		out_value = static_cast<size_t>(parsed);
		return true;
	}

	bool parse_output_target(std::string_view text, OutputTarget &out_target) {
		if (text == "info") {
			out_target = OutputTarget::Info;
			return true;
		}
		if (text == "spirv") {
			out_target = OutputTarget::Spirv;
			return true;
		}
		if (text == "final_nir") {
			out_target = OutputTarget::FinalNir;
			return true;
		}
		if (text == "asm") {
			out_target = OutputTarget::Asm;
			return true;
		}
		return false;
	}

	bool parse_binding_model_arg(std::string_view text, std::string &out_model) {
		if (text == "classic" || text == "push_descriptor" || text == "descriptor_buffer") {
			out_model = std::string(text);
			return true;
		}
		return false;
	}

	bool write_text_output(std::string_view output_path, const std::string &text) {
		if (output_path.empty() || output_path == "-") {
			std::print("{}", text);
			if (!text.empty() && text.back() != '\n') { std::println(""); }
			return true;
		}

		fs::path const path(output_path);
		std::ofstream  out(path, std::ios::binary);
		if (!out) {
			std::println(stderr, "failed to open output file: {}", path.string());
			return false;
		}
		out << text;
		if (!text.empty() && text.back() != '\n') { out << "\n"; }
		if (!out) {
			std::println(stderr, "failed to write output file: {}", path.string());
			return false;
		}
		return true;
	}

	const char *runtime_failure_message(RuntimeFailure failure) {
		switch (failure) {
			case RuntimeFailure::None:
				return "no error";
			case RuntimeFailure::UnhandledSpirvExtensions:
				return "shader uses unmapped SPIR-V extensions";
			case RuntimeFailure::UnhandledSpirvCapabilities:
				return "shader uses unmapped SPIR-V capabilities";
			case RuntimeFailure::VkCreateInstanceFailed:
				return "failed to create Vulkan instance";
			case RuntimeFailure::VkEnumeratePhysicalDevicesFailed:
				return "failed to enumerate Vulkan physical devices";
			case RuntimeFailure::NoPhysicalDevice:
				return "no Vulkan physical device found";
			case RuntimeFailure::NoComputeQueue:
				return "no compute queue available on selected device";
			case RuntimeFailure::MissingExtensionPushDescriptor:
				return "VK_KHR_push_descriptor is not supported";
			case RuntimeFailure::MissingExtensionDescriptorBuffer:
				return "VK_EXT_descriptor_buffer is not supported";
			case RuntimeFailure::MissingExtensionDescriptorHeap:
				return "VK_EXT_descriptor_heap is not supported";
			case RuntimeFailure::MissingExtensionMaintenance5:
				return "VK_KHR_maintenance5 is required but unsupported";
			case RuntimeFailure::MissingExtensionShaderUntypedPointers:
				return "VK_KHR_shader_untyped_pointers is required but unsupported";
			case RuntimeFailure::MissingExtensionShaderBfloat16:
				return "VK_KHR_shader_bfloat16 is required but unsupported";
			case RuntimeFailure::MissingExtensionShaderIntegerDotProduct:
				return "VK_KHR_shader_integer_dot_product is required but unsupported";
			case RuntimeFailure::MissingExtensionRayQuery:
				return "VK_KHR_ray_query is required but unsupported";
			case RuntimeFailure::MissingExtensionAccelerationStructure:
				return "VK_KHR_acceleration_structure is required but unsupported";
			case RuntimeFailure::MissingExtensionDeferredHostOperations:
				return "VK_KHR_deferred_host_operations is required but unsupported";
			case RuntimeFailure::MissingExtensionBufferDeviceAddress:
				return "VK_KHR_buffer_device_address is required but unsupported";
			case RuntimeFailure::MissingExtensionShaderClock:
				return "VK_KHR_shader_clock is required but unsupported";
			case RuntimeFailure::MissingExtensionCooperativeMatrix:
				return "VK_KHR_cooperative_matrix is required but unsupported";
			case RuntimeFailure::MissingExtensionShaderFloatControls2:
				return "VK_KHR_shader_float_controls2 is required but unsupported";
			case RuntimeFailure::MissingExtensionShaderExpectAssume:
				return "VK_KHR_shader_expect_assume is required but unsupported";
			case RuntimeFailure::MissingExtensionShaderSubgroupRotate:
				return "VK_KHR_shader_subgroup_rotate is required but unsupported";
			case RuntimeFailure::MissingFeatureStorageBufferArrayNonUniformIndexing:
				return "feature shaderStorageBufferArrayNonUniformIndexing is unsupported";
			case RuntimeFailure::MissingFeaturePushDescriptor:
				return "feature pushDescriptor is unsupported";
			case RuntimeFailure::MissingFeatureShaderUntypedPointers:
				return "feature shaderUntypedPointers is unsupported";
			case RuntimeFailure::MissingFeatureShaderInt64:
				return "feature shaderInt64 is unsupported";
			case RuntimeFailure::MissingFeatureShaderInt16:
				return "feature shaderInt16 is unsupported";
			case RuntimeFailure::MissingFeatureShaderFloat64:
				return "feature shaderFloat64 is unsupported";
			case RuntimeFailure::MissingFeatureShaderInt8:
				return "feature shaderInt8 is unsupported";
			case RuntimeFailure::MissingFeatureShaderFloat16:
				return "feature shaderFloat16 is unsupported";
			case RuntimeFailure::MissingFeatureShaderFloatControls2:
				return "feature shaderFloatControls2 is unsupported";
			case RuntimeFailure::MissingFeatureShaderExpectAssume:
				return "feature shaderExpectAssume is unsupported";
			case RuntimeFailure::MissingFeatureShaderSubgroupRotate:
				return "feature shaderSubgroupRotate is unsupported";
			case RuntimeFailure::MissingFeatureShaderSubgroupRotateClustered:
				return "feature shaderSubgroupRotateClustered is unsupported";
			case RuntimeFailure::MissingFeatureStorageBuffer8BitAccess:
				return "feature storageBuffer8BitAccess is unsupported";
			case RuntimeFailure::MissingFeatureUniformAndStorageBuffer8BitAccess:
				return "feature uniformAndStorageBuffer8BitAccess is unsupported";
			case RuntimeFailure::MissingFeatureStoragePushConstant8:
				return "feature storagePushConstant8 is unsupported";
			case RuntimeFailure::MissingFeatureStorageBuffer16BitAccess:
				return "feature storageBuffer16BitAccess is unsupported";
			case RuntimeFailure::MissingFeatureUniformAndStorageBuffer16BitAccess:
				return "feature uniformAndStorageBuffer16BitAccess is unsupported";
			case RuntimeFailure::MissingFeatureStoragePushConstant16:
				return "feature storagePushConstant16 is unsupported";
			case RuntimeFailure::MissingFeatureRuntimeDescriptorArray:
				return "feature runtimeDescriptorArray is unsupported";
			case RuntimeFailure::MissingFeatureVulkanMemoryModel:
				return "feature vulkanMemoryModel is unsupported";
			case RuntimeFailure::MissingFeatureVulkanMemoryModelDeviceScope:
				return "feature vulkanMemoryModelDeviceScope is unsupported";
			case RuntimeFailure::MissingFeatureShaderUniformBufferArrayNonUniformIndexing:
				return "feature shaderUniformBufferArrayNonUniformIndexing is unsupported";
			case RuntimeFailure::MissingFeatureShaderSampledImageArrayNonUniformIndexing:
				return "feature shaderSampledImageArrayNonUniformIndexing is unsupported";
			case RuntimeFailure::MissingFeatureShaderStorageBufferArrayNonUniformIndexing:
				return "feature shaderStorageBufferArrayNonUniformIndexing is unsupported";
			case RuntimeFailure::MissingFeatureShaderStorageImageArrayNonUniformIndexing:
				return "feature shaderStorageImageArrayNonUniformIndexing is unsupported";
			case RuntimeFailure::MissingFeatureShaderInputAttachmentArrayNonUniformIndexing:
				return "feature shaderInputAttachmentArrayNonUniformIndexing is unsupported";
			case RuntimeFailure::MissingFeatureBufferDeviceAddress:
				return "feature bufferDeviceAddress is unsupported";
			case RuntimeFailure::MissingFeatureSubgroupArithmetic:
				return "required subgroup arithmetic support is unavailable";
			case RuntimeFailure::MissingFeatureSubgroupOpsMask:
				return "required subgroup operations mask is unavailable";
			case RuntimeFailure::MissingExtensionSubgroupSizeControl:
				return "VK_EXT_subgroup_size_control is required but unsupported";
			case RuntimeFailure::MissingFeatureSubgroupSizeControl:
				return "feature subgroupSizeControl is unsupported";
			case RuntimeFailure::MissingFeatureComputeFullSubgroups:
				return "feature computeFullSubgroups is unsupported";
			case RuntimeFailure::RequestedSubgroupSizeOutOfRange:
				return "requested subgroup size is out of supported range";
			case RuntimeFailure::RequestedSubgroupSizeStageUnsupported:
				return "required subgroup size is not supported for compute stage";
			case RuntimeFailure::MissingFeatureShaderDeviceClock:
				return "feature shaderDeviceClock is unsupported";
			case RuntimeFailure::MissingFeatureShaderSubgroupClock:
				return "feature shaderSubgroupClock is unsupported";
			case RuntimeFailure::MissingFeatureCooperativeMatrix:
				return "feature cooperativeMatrix is unsupported";
			case RuntimeFailure::MissingFeatureShaderBFloat16Type:
				return "feature shaderBFloat16Type is unsupported";
			case RuntimeFailure::MissingFeatureShaderBFloat16DotProduct:
				return "feature shaderBFloat16DotProduct is unsupported";
			case RuntimeFailure::MissingFeatureShaderBFloat16CooperativeMatrix:
				return "feature shaderBFloat16CooperativeMatrix is unsupported";
			case RuntimeFailure::MissingFeatureShaderIntegerDotProduct:
				return "feature shaderIntegerDotProduct is unsupported";
			case RuntimeFailure::MissingFeatureAccelerationStructure:
				return "feature accelerationStructure is unsupported";
			case RuntimeFailure::MissingFeatureRayQuery:
				return "feature rayQuery is unsupported";
			case RuntimeFailure::VkCreateDeviceFailed:
				return "vkCreateDevice failed";
			case RuntimeFailure::VkCreateShaderModuleFailed:
				return "vkCreateShaderModule failed";
			case RuntimeFailure::VkCreateDescriptorSetLayoutFailed:
				return "vkCreateDescriptorSetLayout failed";
			case RuntimeFailure::VkCreatePipelineLayoutFailed:
				return "vkCreatePipelineLayout failed";
			case RuntimeFailure::VkCreateComputePipelineFailed:
				return "vkCreateComputePipelines failed";
			case RuntimeFailure::PipelineExecutableUnsupported:
				return "pipeline executable properties are unsupported for this target";
			case RuntimeFailure::MissingPipelineExecutableFunctions:
				return "required pipeline executable query functions are missing";
			case RuntimeFailure::NoTextInternalRepresentation:
				return "driver exposed no text internal representations";
			case RuntimeFailure::FailedToSelectFinalNirRepresentation:
				return "failed to select final NIR representation";
			case RuntimeFailure::FailedToSelectAsmRepresentation:
				return "failed to select assembly representation";
			case RuntimeFailure::UnexpectedExecutableCount:
				return "The current implementation makes the assumption that one Vulkan compute shader maps internally "
					   "to a single hardware compute shader, this assumption was broken. I will only start thinking "
					   "about thinking how to represent multiple hardware shader when I actually observe a real case "
					   "of it happening.";
		}
		return "unknown runtime failure";
	}

	const char *internal_failure_message(InternalFailure failure) {
		switch (failure) {
			case InternalFailure::None:
				return "no internal error";
			case InternalFailure::MissingResponseFd:
				return "missing child response fd";
			case InternalFailure::InvalidResponseFd:
				return "invalid child response fd";
			case InternalFailure::ChildReadSpirvFailed:
				return "child failed to read shared SPIR-V payload";
			case InternalFailure::ChildRuntimeFailed:
				return "child runtime execution failed";
			case InternalFailure::ChildResponseWriteFailed:
				return "child failed to write response payload";
			case InternalFailure::PipeCreateFailed:
				return "failed to create IPC pipe";
			case InternalFailure::ForkFailed:
				return "failed to fork child process";
			case InternalFailure::ChildExecFailed:
				return "child failed to exec runtime binary";
			case InternalFailure::ParentReadFailed:
				return "parent failed to read IPC payload";
			case InternalFailure::ParentWaitFailed:
				return "parent failed to wait for child";
			case InternalFailure::InvalidResponseHeader:
				return "child response header is invalid";
			case InternalFailure::MissingResponsePayload:
				return "child returned no IPC payload";
			case InternalFailure::ChildExitedNonZero:
				return "child exited with non-zero status";
			case InternalFailure::ChildSignaled:
				return "child terminated by signal";
		}
		return "unknown internal failure";
	}

	bool write_all(int fd, const void *data, size_t size) {
		const char *cursor	  = static_cast<const char *>(data);
		size_t		remaining = size;
		while (remaining > 0) {
			ssize_t const wrote = write(fd, cursor, remaining);
			if (wrote < 0) {
				if (errno == EINTR) { continue; }
				return false;
			}
			cursor += static_cast<size_t>(wrote);
			remaining -= static_cast<size_t>(wrote);
		}
		return true;
	}

	bool read_all_to_end(int fd, std::vector<char> &out) {
		std::array<char, 4096> buffer{};
		for (;;) {
			ssize_t const got = read(fd, buffer.data(), buffer.size());
			if (got == 0) { return true; }
			if (got < 0) {
				if (errno == EINTR) { continue; }
				return false;
			}
			out.insert(out.end(), buffer.data(), buffer.data() + got);
		}
	}

	bool write_child_response(int fd, const ChildResponse &response) {
		ChildResponseHeader header;
		header.internal_failure = static_cast<uint32_t>(response.internal_failure);
		header.runtime_failure	= static_cast<uint32_t>(response.runtime_failure);
		header.sys_errno		= response.sys_errno;
		header.detail_size		= static_cast<uint32_t>(response.detail.size());
		header.output_size		= static_cast<uint32_t>(response.output.size());
		if (!write_all(fd, &header, sizeof(header))) { return false; }
		if (header.detail_size > 0 && !write_all(fd, response.detail.data(), response.detail.size())) { return false; }
		if (header.output_size > 0 && !write_all(fd, response.output.data(), response.output.size())) { return false; }
		return true;
	}

	bool parse_child_response(const std::vector<char> &payload, ChildResponse &out_response) {
		if (payload.size() < sizeof(ChildResponseHeader)) { return false; }
		ChildResponseHeader header;
		std::memcpy(&header, payload.data(), sizeof(header));
		if (header.magic != 0x53455852U || header.version != 1) { return false; }
		size_t const needed = sizeof(ChildResponseHeader) + static_cast<size_t>(header.detail_size) +
							  static_cast<size_t>(header.output_size);
		if (payload.size() != needed) { return false; }
		out_response.internal_failure = static_cast<InternalFailure>(header.internal_failure);
		out_response.runtime_failure  = static_cast<RuntimeFailure>(header.runtime_failure);
		out_response.sys_errno		  = header.sys_errno;
		const char *cursor			  = payload.data() + sizeof(ChildResponseHeader);
		out_response.detail.assign(cursor, cursor + header.detail_size);
		cursor += header.detail_size;
		out_response.output.assign(cursor, cursor + header.output_size);
		return true;
	}

	ChildResponse run_internal_child_with_shim(const char *argv0, const fs::path &shim_path,
											   const std::vector<std::string> &extra_args) {
		ChildResponse response;

		std::array<int, 2> pipefd{{-1, -1}};
		if (pipe(pipefd.data()) != 0) {
			response.internal_failure = InternalFailure::PipeCreateFailed;
			response.sys_errno		  = errno;
			return response;
		}

		pid_t const pid = fork();
		if (pid < 0) {
			response.internal_failure = InternalFailure::ForkFailed;
			response.sys_errno		  = errno;
			close(pipefd[0]);
			close(pipefd[1]);
			return response;
		}

		if (pid == 0) {
			close(pipefd[0]);
			int devnull = open("/dev/null", O_WRONLY);
			if (devnull >= 0) {
				(void) dup2(devnull, STDOUT_FILENO);
				(void) dup2(devnull, STDERR_FILENO);
				if (devnull > STDERR_FILENO) { close(devnull); }
			}

			std::string preload = shim_path.string();
			if (const char *existing = std::getenv("LD_PRELOAD")) {
				if (existing[0] != 0) {
					preload += ":";
					preload += existing;
				}
			}
			setenv("LD_PRELOAD", preload.c_str(), 1);
			setenv("MESA_SHADER_CACHE_DISABLE", "true", 1);

			std::vector<std::string> args;
			args.emplace_back(argv0);
			args.insert(args.end(), extra_args.begin(), extra_args.end());
			args.emplace_back("--internal-response-fd");
			args.emplace_back(std::to_string(pipefd[1]));

			std::vector<char *> c_argv;
			c_argv.reserve(args.size() + 1);
			for (std::string &arg: args) { c_argv.push_back(arg.data()); }
			c_argv.push_back(nullptr);

			execv(argv0, c_argv.data());

			ChildResponse exec_fail;
			exec_fail.internal_failure = InternalFailure::ChildExecFailed;
			exec_fail.sys_errno		   = errno;
			(void) write_child_response(pipefd[1], exec_fail);
			close(pipefd[1]);
			_exit(127);
		}

		close(pipefd[1]);
		std::vector<char> payload;
		if (!read_all_to_end(pipefd[0], payload)) {
			response.internal_failure = InternalFailure::ParentReadFailed;
			response.sys_errno		  = errno;
		}
		close(pipefd[0]);

		int status = 0;
		if (waitpid(pid, &status, 0) < 0) {
			response.internal_failure = InternalFailure::ParentWaitFailed;
			response.sys_errno		  = errno;
			return response;
		}

		if (!payload.empty()) {
			if (!parse_child_response(payload, response)) {
				response.internal_failure = InternalFailure::InvalidResponseHeader;
			}
		} else if (response.internal_failure == InternalFailure::None) {
			if (WIFSIGNALED(status)) {
				response.internal_failure = InternalFailure::ChildSignaled;
				response.detail			  = std::to_string(WTERMSIG(status));
			} else if (WIFEXITED(status) && WEXITSTATUS(status) != 0) {
				response.internal_failure = InternalFailure::ChildExitedNonZero;
				response.detail			  = std::to_string(WEXITSTATUS(status));
			} else {
				response.internal_failure = InternalFailure::MissingResponsePayload;
			}
		}

		return response;
	}

	fs::path default_mesa_libdir() {
		if constexpr (!std::string_view(SHADER_EXPLORER_DEFAULT_MESA_LIBDIR).empty()) {
			return {SHADER_EXPLORER_DEFAULT_MESA_LIBDIR};
		}
		return {};
	}

	fs::path default_vulkan_icd_dir() {
		if constexpr (!std::string_view(SHADER_EXPLORER_DEFAULT_VULKAN_ICD_DIR).empty()) {
			return {SHADER_EXPLORER_DEFAULT_VULKAN_ICD_DIR};
		}
		return {};
	}

	fs::path resolve_installed_library_path(const fs::path &installed_libdir, std::string_view library_file) {
		return installed_libdir / library_file;
	}

	fs::path resolve_installed_icd_path(const fs::path &installed_icd_dir, std::string_view icd_file) {
		return installed_icd_dir / icd_file;
	}


	bool mesa_artifacts_ready(const device_config &config, std::string &reason) {
		fs::path const installed_libdir = default_mesa_libdir();
		fs::path const installed_icd_dir = default_vulkan_icd_dir();
		fs::path const icd_path = resolve_installed_icd_path(installed_icd_dir, config.vulkan_icd_file);
		if (!fs::exists(icd_path)) {
			reason = "missing ICD JSON: " + icd_path.string();
			return false;
		}
		if (!config.shim_file.empty()) {
			fs::path const shim_path = resolve_installed_library_path(installed_libdir, config.shim_file);
			if (!fs::exists(shim_path)) {
				reason = "missing DRM shim: " + shim_path.string();
				return false;
			}
		}
		return true;
	}

	bool write_icd_files_for_drivers(const std::vector<fs::path> &icd_paths, const std::string &gpu_key) {
		if (icd_paths.empty()) {
			std::println(stderr, "no Vulkan ICD drivers provided for gpu preset '{}'", gpu_key);
			return false;
		}

		std::string vk_driver_files;

		for (const fs::path &icd_path: icd_paths) {
			if (!fs::exists(icd_path)) {
				std::println(stderr, "missing Mesa ICD JSON: {}", icd_path.string());
				return false;
			}
			if (!vk_driver_files.empty()) { vk_driver_files += ":"; }
			vk_driver_files += icd_path.string();
		}

		setenv("VK_DRIVER_FILES", vk_driver_files.c_str(), 1);
		setenv("VK_ICD_FILENAMES", vk_driver_files.c_str(), 1);
		return true;
	}

	bool configure_gpu_environment(const device_config &config, bool load_shim_now) {
		std::string reason;
		if (!mesa_artifacts_ready(config, reason)) {
			std::println(stderr, "Mesa artifacts not ready for gpu preset '{}': {}", config.key, reason);
			return false;
		}

		fs::path const installed_libdir = default_mesa_libdir();
		fs::path const installed_icd_dir = default_vulkan_icd_dir();
		std::vector<fs::path> icd_paths = {resolve_installed_icd_path(installed_icd_dir, config.vulkan_icd_file)};

		if (!write_icd_files_for_drivers(icd_paths, config.key)) { return false; }

		for (const auto &entry: config.env) { setenv(entry.first.c_str(), entry.second.c_str(), 1); }

		if (load_shim_now && !config.shim_file.empty()) {
			fs::path const shim_path = resolve_installed_library_path(installed_libdir, config.shim_file);
			void const	  *shim_handle = dlopen(shim_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
			if (shim_handle == nullptr) {
				const char *const dl_error = dlerror();
				std::println(stderr, "failed to load DRM shim via dlopen: {}\ndlerror: {}", shim_path.string(),
							 (dl_error != nullptr) ? dl_error : "unknown");
				return false;
			}
		}

		return true;
	}

	bool maybe_print_slang_blob(const char *label, slang::IBlob *blob) {
		if (blob == nullptr) { return false; }
		const char	*text = static_cast<const char *>(blob->getBufferPointer());
		size_t const size = blob->getBufferSize();
		if ((text == nullptr) || size == 0) { return false; }
		std::println(stderr, "{}:\n{}", label, std::string(text, size));
		return true;
	}

	const char *slang_profile_for_spirv_target(SpirvTargetVersion target) {
		switch (target) {
			case SpirvTargetVersion::V10:
				return "spirv_1_0";
			case SpirvTargetVersion::V11:
				return "spirv_1_1";
			case SpirvTargetVersion::V12:
				return "spirv_1_2";
			case SpirvTargetVersion::V13:
				return "spirv_1_3";
			case SpirvTargetVersion::V14:
				return "spirv_1_4";
			case SpirvTargetVersion::V15:
				return "spirv_1_5";
			case SpirvTargetVersion::V16:
				return "spirv_1_6";
		}
		return "spirv_1_0";
	}

	glslang::EShTargetLanguageVersion glslang_target_spv(SpirvTargetVersion target) {
		switch (target) {
			case SpirvTargetVersion::V10:
				return glslang::EShTargetSpv_1_0;
			case SpirvTargetVersion::V11:
				return glslang::EShTargetSpv_1_1;
			case SpirvTargetVersion::V12:
				return glslang::EShTargetSpv_1_2;
			case SpirvTargetVersion::V13:
				return glslang::EShTargetSpv_1_3;
			case SpirvTargetVersion::V14:
				return glslang::EShTargetSpv_1_4;
			case SpirvTargetVersion::V15:
				return glslang::EShTargetSpv_1_5;
			case SpirvTargetVersion::V16:
				return glslang::EShTargetSpv_1_6;
		}
		return glslang::EShTargetSpv_1_0;
	}

	glslang::EShTargetClientVersion glslang_target_vulkan_client(SpirvTargetVersion target) {
		switch (target) {
			case SpirvTargetVersion::V10:
				return glslang::EShTargetVulkan_1_0;
			case SpirvTargetVersion::V11:
			case SpirvTargetVersion::V12:
			case SpirvTargetVersion::V13:
			case SpirvTargetVersion::V14:
				return glslang::EShTargetVulkan_1_1;
			case SpirvTargetVersion::V15:
				return glslang::EShTargetVulkan_1_2;
			case SpirvTargetVersion::V16:
				return glslang::EShTargetVulkan_1_3;
		}
		return glslang::EShTargetVulkan_1_0;
	}

	spv_target_env spirv_tools_target_env(SpirvTargetVersion target) {
		switch (target) {
			case SpirvTargetVersion::V10:
				return SPV_ENV_VULKAN_1_0;
			case SpirvTargetVersion::V11:
			case SpirvTargetVersion::V12:
			case SpirvTargetVersion::V13:
				return SPV_ENV_VULKAN_1_1;
			case SpirvTargetVersion::V14:
				return SPV_ENV_VULKAN_1_1_SPIRV_1_4;
			case SpirvTargetVersion::V15:
				return SPV_ENV_VULKAN_1_2;
			case SpirvTargetVersion::V16:
				return SPV_ENV_VULKAN_1_3;
		}
		return SPV_ENV_VULKAN_1_0;
	}

	// warning AI code with minimal review
	bool compile_slang(const fs::path &shader_path, SpirvTargetVersion spirv_target, std::vector<uint32_t> &out_spirv) {
		Slang::ComPtr<slang::IGlobalSession> global_session;
		if (SLANG_FAILED(slang_createGlobalSession(SLANG_API_VERSION, global_session.writeRef()))) {
			std::println(stderr, "failed to create Slang global session");
			return false;
		}

		slang::TargetDesc const target_desc = {
				.format	 = SLANG_SPIRV,
				.profile = global_session->findProfile(slang_profile_for_spirv_target(spirv_target)),
				.flags	 = SLANG_TARGET_FLAG_GENERATE_SPIRV_DIRECTLY,
				.forceGLSLScalarBufferLayout = false,
		};
		if (target_desc.profile == SLANG_PROFILE_UNKNOWN) {
			std::println(stderr, "slang profile not available for SPIR-V target {}",
						 spirv_target_to_string(spirv_target));
			return false;
		}

		fs::path const	shader_dir_path = shader_path.has_parent_path() ? shader_path.parent_path() : fs::path(".");
		std::error_code ec;
		fs::path		shader_dir_abs = fs::absolute(shader_dir_path, ec);
		if (ec || shader_dir_abs.empty()) {
			ec.clear();
			shader_dir_abs = fs::current_path(ec);
		}
		if (ec || shader_dir_abs.empty()) {
			std::println(stderr, "failed to resolve shader directory for Slang search path: {}",
						 shader_dir_path.string());
			return false;
		}
		std::string const				  shader_dir = shader_dir_abs.string();
		std::array<const char *, 1> const search_paths{{shader_dir.c_str()}};

		slang::SessionDesc const session_desc = {
				.targets		 = &target_desc,
				.targetCount	 = 1,
				.searchPaths	 = search_paths.data(),
				.searchPathCount = 1,
		};

		Slang::ComPtr<slang::ISession> session;
		if (SLANG_FAILED(global_session->createSession(session_desc, session.writeRef()))) {
			std::println(stderr, "failed to create Slang session");
			return false;
		}

		std::string const source_text = read_text_file(shader_path);
		if (source_text.empty()) {
			std::println(stderr, "failed to read shader source: {}", shader_path.string());
			return false;
		}

		std::string const			module_name = shader_path.stem().string();
		std::string const			shader_file = shader_path.string();
		Slang::ComPtr<slang::IBlob> diagnostics;
		slang::IModule *module = session->loadModuleFromSourceString(module_name.c_str(), shader_file.c_str(),
																	 source_text.c_str(), diagnostics.writeRef());
		if (module == nullptr) { maybe_print_slang_blob("slang diagnostics", diagnostics.get()); }
		if (module == nullptr) {
			std::println(stderr, "failed to load Slang module from source");
			return false;
		}

		Slang::ComPtr<slang::IEntryPoint> entry_point;
		SlangResult						  entry_result = module->findEntryPointByName("main", entry_point.writeRef());
		if (SLANG_FAILED(entry_result)) {
			Slang::ComPtr<slang::IBlob> entry_diag;
			entry_result = module->findAndCheckEntryPoint("main", SLANG_STAGE_COMPUTE, entry_point.writeRef(),
														  entry_diag.writeRef());
			maybe_print_slang_blob("slang entry-point diagnostics", entry_diag.get());
		}
		if (SLANG_FAILED(entry_result) || (entry_point == nullptr)) {
			std::println(stderr, "failed to resolve compute entry point `main`");
			return false;
		}

		std::array<slang::IComponentType *, 2> components{{module, entry_point.get()}};
		Slang::ComPtr<slang::IComponentType>   composite;
		Slang::ComPtr<slang::IBlob>			   composite_diag;
		SlangResult const					   composite_result = session->createCompositeComponentType(
				 components.data(), components.size(), composite.writeRef(), composite_diag.writeRef());
		if (SLANG_FAILED(composite_result)) {
			maybe_print_slang_blob("slang composite diagnostics", composite_diag.get());
		}
		if (SLANG_FAILED(composite_result) || (composite == nullptr)) {
			std::println(stderr, "failed to create Slang composite component");
			return false;
		}

		Slang::ComPtr<slang::IComponentType> linked_program;
		Slang::ComPtr<slang::IBlob>			 link_diag;
		SlangResult const link_result = composite->link(linked_program.writeRef(), link_diag.writeRef());
		if (SLANG_FAILED(link_result)) { maybe_print_slang_blob("slang link diagnostics", link_diag.get()); }
		if (SLANG_FAILED(link_result) || (linked_program == nullptr)) {
			std::println(stderr, "failed to link Slang program");
			return false;
		}

		Slang::ComPtr<slang::IBlob> spirv_blob;
		Slang::ComPtr<slang::IBlob> code_diag;
		SlangResult const			code_result =
				linked_program->getEntryPointCode(0, 0, spirv_blob.writeRef(), code_diag.writeRef());
		if (SLANG_FAILED(code_result)) { maybe_print_slang_blob("slang codegen diagnostics", code_diag.get()); }
		if (SLANG_FAILED(code_result) || (spirv_blob == nullptr)) {
			std::println(stderr, "failed to generate SPIR-V via Slang API");
			return false;
		}

		size_t const size = spirv_blob->getBufferSize();
		if (size == 0 || (size % sizeof(uint32_t)) != 0) {
			std::println(stderr, "invalid SPIR-V blob returned by Slang");
			return false;
		}

		out_spirv.resize(size / sizeof(uint32_t));
		std::memcpy(out_spirv.data(), spirv_blob->getBufferPointer(), size);
		return true;
	}

	std::string shader_extension_lower(const fs::path &shader_path) {
		std::string ext = shader_path.extension().string();
		for (char &c: ext) { c = static_cast<char>(std::tolower(static_cast<unsigned char>(c))); }
		return ext;
	}

	bool ensure_glslang_process_initialized() {
		static bool const initialized = glslang::InitializeProcess();
		return initialized;
	}

	// warning AI code with minimal review
	bool compile_glsl_compute(const fs::path &shader_path, SpirvTargetVersion spirv_target,
							  std::vector<uint32_t> &out_spirv) {
		if (!ensure_glslang_process_initialized()) {
			std::println(stderr, "failed to initialize in-process glslang compiler");
			return false;
		}

		std::string const source_text = read_text_file(shader_path);
		if (source_text.empty()) {
			std::println(stderr, "failed to read GLSL source: {}", shader_path.string());
			return false;
		}

		std::string const source_name	  = shader_path.string();
		const char		 *source		  = source_text.c_str();
		const char		 *source_name_ptr = source_name.c_str();

		constexpr EShMessages messages = EShMsgSpvRules;

		glslang::TShader shader(EShLangCompute);
		shader.setStringsWithLengthsAndNames(&source, nullptr, &source_name_ptr, 1);
		shader.setEntryPoint("main");
		shader.setSourceEntryPoint("main");
		shader.setEnvInput(glslang::EShSourceGlsl, EShLangCompute, glslang::EShClientVulkan, 450);
		shader.setEnvClient(glslang::EShClientVulkan, glslang_target_vulkan_client(spirv_target));
		shader.setEnvTarget(glslang::EShTargetSpv, glslang_target_spv(spirv_target));

		const TBuiltInResource *resources = GetDefaultResources();
		if (resources == nullptr) {
			std::println(stderr, "failed to get glslang resource limits");
			return false;
		}
		DirStackFileIncluder includer;
		fs::path const		 shader_dir = shader_path.has_parent_path() ? shader_path.parent_path() : fs::path(".");
		includer.pushExternalLocalDirectory(shader_dir.string());
		if (!shader.parse(resources, 450, false, messages, includer)) {
			std::println(stderr, "glslang parse failed for {}", shader_path.string());
			const char *info_log = shader.getInfoLog();
			if ((info_log != nullptr) && (info_log[0] != 0)) { std::println(stderr, "{}", info_log); }
			const char *debug_log = shader.getInfoDebugLog();
			if ((debug_log != nullptr) && (debug_log[0] != 0)) { std::println(stderr, "{}", debug_log); }
			return false;
		}

		glslang::TProgram program;
		program.addShader(&shader);
		if (!program.link(messages)) {
			std::println(stderr, "glslang link failed for {}", shader_path.string());
			const char *info_log = program.getInfoLog();
			if ((info_log != nullptr) && (info_log[0] != 0)) { std::println(stderr, "{}", info_log); }
			const char *debug_log = program.getInfoDebugLog();
			if ((debug_log != nullptr) && (debug_log[0] != 0)) { std::println(stderr, "{}", debug_log); }
			return false;
		}

		glslang::TIntermediate const *intermediate = program.getIntermediate(EShLangCompute);
		if (intermediate == nullptr) {
			std::println(stderr, "glslang produced no compute intermediate for {}", shader_path.string());
			return false;
		}

		glslang::SpvOptions options = {
				.disableOptimizer = true,
		};
		glslang::GlslangToSpv(*intermediate, out_spirv, &options);
		if (out_spirv.empty()) {
			std::println(stderr, "glslang produced empty SPIR-V for {}", shader_path.string());
			return false;
		}

		return true;
	}

	bool run_spirv_opt(const std::vector<uint32_t> &input_spirv, SpirvTargetVersion spirv_target,
					   std::vector<uint32_t> &optimized_spirv) {
		optimized_spirv = input_spirv;
		spvtools::Optimizer optimizer(spirv_tools_target_env(spirv_target));
		optimizer.SetMessageConsumer(
				[](spv_message_level_t level, const char *, const spv_position_t &pos, const char *message) {
					if (level <= SPV_MSG_WARNING) {
						std::println(stderr, "spirv-opt: line {}: {}", pos.line, (message != nullptr) ? message : "");
					}
				});
		optimizer.RegisterPerformancePasses();

		std::vector<uint32_t> out;
		if (!optimizer.Run(input_spirv.data(), input_spirv.size(), &out)) {
			std::println(stderr, "in-process spirv-opt failed");
			return false;
		}
		if (out.empty()) {
			std::println(stderr, "in-process spirv-opt produced empty output");
			return false;
		}
		optimized_spirv = std::move(out);
		return true;
	}

	bool run_spirv_dis(const std::vector<uint32_t> &spirv, bool use_color, std::string &disassembly) {
		spvtools::SpirvTools tools(SPV_ENV_UNIVERSAL_1_6);
		tools.SetMessageConsumer(
				[](spv_message_level_t level, const char *, const spv_position_t &pos, const char *message) {
					if (level <= SPV_MSG_WARNING) {
						std::println(stderr, "spirv-dis: line {}: {}", pos.line, (message != nullptr) ? message : "");
					}
				});

		uint32_t options = SPV_BINARY_TO_TEXT_OPTION_INDENT | SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES |
						   SPV_BINARY_TO_TEXT_OPTION_COMMENT;
		if (use_color) { options |= SPV_BINARY_TO_TEXT_OPTION_COLOR; }
		if (!tools.Disassemble(spirv, &disassembly, options)) {
			std::println(stderr, "in-process spirv-dis failed");
			return false;
		}
		return true;
	}

	void print_usage(const std::map<std::string, device_config> &catalog) {
		std::println(stderr,
					 "usage: shader_explorer [--version] [--gpu <key>] [--lang <auto|slang|glsl>] [--target "
					 "<info|spirv|final_nir|asm>] "
					 "[--binding-model <classic|push_descriptor|descriptor_buffer>] "
					 "[--spirv-target <max|1.0|1.1|1.2|1.3|1.4|1.5|1.6>] "
					 "[--subgroup-size <n>] [--require-full-subgroups] [--no-color] "
					 "[--output <-|file>] [--list-gpus] "
					 "<shader.slang|shader.glsl>\n"
					 "\n"
					 "options:\n"
					 "  --version        print program version and exit\n"
					 "  --list-gpus      list available GPU presets\n"
					 "  --gpu <key>      select a GPU preset\n"
					 "  --lang <l>       source language: auto, slang, glsl (default: auto)\n"
					 "  --target <t>     output target: info, spirv, final_nir, asm (default: asm)\n"
					 "  --binding-model <m>  runtime descriptor model: classic, push_descriptor, descriptor_buffer\n"
					 "                       descriptor_heap is selected automatically when required by shader SPIR-V\n"
					 "  --spirv-target <v>  SPIR-V target version: max or explicit 1.0..1.6 (default: max)\n"
					 "  --subgroup-size <n> request VK_EXT_subgroup_size_control required subgroup size for compute "
					 "stage\n"
					 "  --require-full-subgroups  request full-subgroup dispatch behavior for compute stage\n"
					 "  --no-color       disable ANSI color in text output\n"
					 "  --output <path>  output destination: - (stdout) or file path (default: -)\n"
					 "\n"
					 "available GPUs:");
		for (const auto &[key, cfg]: catalog) { std::println(stderr, "  {} - {}", key, cfg.description); }
	}

	void print_runtime_failure_with_detail(RuntimeFailure failure, std::string_view detail) {
		std::println(stderr, "runtime failure: {}", runtime_failure_message(failure));
		if (!detail.empty()) { std::println(stderr, "detail: {}", detail); }
	}

	int report_runtime_result_failure(const RuntimeResult &result) {
		if (!result.ok()) {
			print_runtime_failure_with_detail(result.failure, result.detail);
			return 1;
		}
		return 0;
	}

	int report_child_failure(const char *context, const ChildResponse &child) {
		if (child.internal_failure == InternalFailure::ChildRuntimeFailed &&
			child.runtime_failure != RuntimeFailure::None) {
			print_runtime_failure_with_detail(child.runtime_failure, child.detail);
			return 1;
		}
		if (child.internal_failure != InternalFailure::None) {
			std::string detail_suffix;
			if (!child.detail.empty()) { detail_suffix = std::format(" ({})", child.detail); }
			std::string errno_suffix;
			if (child.sys_errno != 0) { errno_suffix = std::format(" errno={}", child.sys_errno); }
			std::println(stderr, "internal child failure while {}: {}{}{}", context,
						 internal_failure_message(child.internal_failure), detail_suffix, errno_suffix);
			return 1;
		}
		if (child.runtime_failure != RuntimeFailure::None) {
			print_runtime_failure_with_detail(child.runtime_failure, child.detail);
			return 1;
		}
		return 0;
	}

	std::optional<int> parse_cli_options(int argc, char **argv, const std::map<std::string, device_config> &catalog,
										 CliOptions &options) {
		for (int i = 1; i < argc; ++i) {
			std::string_view const arg = argv[i];
			if (arg == "--help" || arg == "-h") {
				print_usage(catalog);
				return 0;
			}
			if (arg == "--version") {
				std::println("{}", SHADER_EXPLORER_VERSION);
				return 0;
			}
			if (arg == "--internal-mode") {
				if (i + 1 >= argc) {
					std::println(stderr, "--internal-mode requires a value");
					return 1;
				}
				std::string_view const mode = argv[++i];
				if (mode == "info") {
					options.internal_mode = InternalMode::Info;
				} else if (mode == "pipeline") {
					options.internal_mode = InternalMode::Pipeline;
				} else if (mode == "spirv-max") {
					options.internal_mode = InternalMode::SpirvMax;
				} else {
					std::println(stderr, "invalid --internal-mode value: {}", mode);
					return 1;
				}
				continue;
			}
			if (arg == "--internal-shim-ready") {
				options.internal_shim_ready = true;
				continue;
			}
			if (arg == "--internal-spirv-fd") {
				if (i + 1 >= argc) {
					std::println(stderr, "--internal-spirv-fd requires a value");
					return 1;
				}
				options.internal_spirv_fd = std::atoi(argv[++i]);
				continue;
			}
			if (arg == "--internal-spirv-bytes") {
				if (i + 1 >= argc) {
					std::println(stderr, "--internal-spirv-bytes requires a value");
					return 1;
				}
				if (!parse_size_arg(argv[++i], options.internal_spirv_bytes)) {
					std::println(stderr, "invalid --internal-spirv-bytes value");
					return 1;
				}
				continue;
			}
			if (arg == "--internal-response-fd") {
				if (i + 1 >= argc) { return 1; }
				options.internal_response_fd = std::atoi(argv[++i]);
				continue;
			}
			if (arg == "--list-gpus") {
				options.list_gpus = true;
				continue;
			}
			if (arg == "--setup-mesa" || arg == "--auto-build-mesa") {
				std::println(stderr,
							 "{} is no longer supported in the runtime binary.\n"
							 "Use Meson configuration/build to provide Mesa artifacts.",
							 arg);
				return 1;
			}
			if (arg == "--gpu") {
				if (i + 1 >= argc) {
					std::println(stderr, "--gpu requires a value");
					return 1;
				}
				options.gpu_key = argv[++i];
				continue;
			}
			if (arg == "--target") {
				if (i + 1 >= argc) {
					std::println(stderr, "--target requires a value");
					return 1;
				}
				if (!parse_output_target(argv[++i], options.output_target)) {
					std::println(stderr, "invalid --target value, expected one of: info, spirv, final_nir, asm");
					return 1;
				}
				continue;
			}
			if (arg == "--binding-model") {
				if (i + 1 >= argc) {
					std::println(stderr, "--binding-model requires a value");
					return 1;
				}
				if (!parse_binding_model_arg(argv[++i], options.binding_model)) {
					std::println(stderr,
								 "invalid --binding-model value, expected one of: classic, push_descriptor, "
								 "descriptor_buffer");
					return 1;
				}
				continue;
			}
			if (arg == "--spirv-target") {
				if (i + 1 >= argc) {
					std::println(stderr, "--spirv-target requires a value");
					return 1;
				}
				if (!parse_spirv_target_arg(argv[++i], options.spirv_target_mode, options.requested_spirv_target)) {
					std::println(stderr,
								 "invalid --spirv-target value, expected one of: max, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, "
								 "1.6");
					return 1;
				}
				continue;
			}
			if (arg == "--subgroup-size") {
				if (i + 1 >= argc) {
					std::println(stderr, "--subgroup-size requires a value");
					return 1;
				}
				if (!parse_size_arg(argv[++i], options.requested_subgroup_size) ||
					options.requested_subgroup_size == 0) {
					std::println(stderr, "invalid --subgroup-size value, expected a positive integer");
					return 1;
				}
				continue;
			}
			if (arg == "--require-full-subgroups") {
				options.require_full_subgroups = true;
				continue;
			}
			if (arg == "--no-color") {
				options.no_color = true;
				continue;
			}
			if (arg == "--lang") {
				if (i + 1 >= argc) {
					std::println(stderr, "--lang requires a value");
					return 1;
				}
				if (!parse_source_language(argv[++i], options.source_lang)) {
					std::println(stderr, "invalid --lang value, expected one of: auto, slang, glsl");
					return 1;
				}
				continue;
			}
			if (arg == "--output") {
				if (i + 1 >= argc) {
					std::println(stderr, "--output requires a value");
					return 1;
				}
				options.output_path = argv[++i];
				continue;
			}
			if (!options.shader_path.has_value()) {
				options.shader_path = fs::path(arg);
				continue;
			}
			std::println(stderr, "unexpected extra argument: {}", arg);
			print_usage(catalog);
			return 1;
		}
		return std::nullopt;
	}

	std::optional<int> maybe_handle_internal_mode(const CliOptions &options) {
		if (options.internal_mode == InternalMode::None) { return std::nullopt; }

		ChildResponse response;
		if (options.internal_response_fd < 0) {
			response.internal_failure = InternalFailure::MissingResponseFd;
			return 1;
		}

		if (options.internal_mode == InternalMode::Info) {
			RuntimeConfig const runtime_config = {
					.binding_model			= options.binding_model == "push_descriptor"
													  ? RuntimeBindingModel::PushDescriptor
													  : (options.binding_model == "descriptor_buffer"
																 ? RuntimeBindingModel::DescriptorBuffer
																 : (options.binding_model == "descriptor_heap"
																			? RuntimeBindingModel::DescriptorHeap
																			: RuntimeBindingModel::Classic)),
					.enable_bda				= false,
					.required_subgroup_size = static_cast<uint32_t>(options.requested_subgroup_size),
					.require_full_subgroups = options.require_full_subgroups,
					.gpu_key				= options.gpu_key,
			};
			RuntimeResult const runtime = run_device_info_dump(runtime_config, response.output);
			if (!runtime.ok()) {
				response.internal_failure = InternalFailure::ChildRuntimeFailed;
				response.runtime_failure  = runtime.failure;
				response.detail			  = runtime.detail;
			}
			return write_child_response(options.internal_response_fd, response) ? 0 : 1;
		}

		if (options.internal_mode == InternalMode::Pipeline) {
			std::vector<uint32_t> spirv;
			if (!read_spirv_from_shared_memory(options.internal_spirv_fd, options.internal_spirv_bytes, spirv)) {
				response.internal_failure = InternalFailure::ChildReadSpirvFailed;
				if (options.internal_spirv_fd >= 0) { close(options.internal_spirv_fd); }
				(void) write_child_response(options.internal_response_fd, response);
				return 1;
			}
			close(options.internal_spirv_fd);
			RuntimeDumpOptions const dump_options = {
					.target = (options.output_target == OutputTarget::FinalNir) ? RuntimeOutputTarget::FinalNir
																				: RuntimeOutputTarget::Asm,
			};
			RuntimeConfig const runtime_config = {
					.binding_model			= options.binding_model == "push_descriptor"
													  ? RuntimeBindingModel::PushDescriptor
													  : (options.binding_model == "descriptor_buffer"
																 ? RuntimeBindingModel::DescriptorBuffer
																 : (options.binding_model == "descriptor_heap"
																			? RuntimeBindingModel::DescriptorHeap
																			: RuntimeBindingModel::Classic)),
					.enable_bda				= false,
					.required_subgroup_size = static_cast<uint32_t>(options.requested_subgroup_size),
					.require_full_subgroups = options.require_full_subgroups,
					.gpu_key				= options.gpu_key,
			};
			RuntimeResult const runtime = run_pipeline_dump(spirv, dump_options, runtime_config, response.output);
			if (!runtime.ok()) {
				response.internal_failure = InternalFailure::ChildRuntimeFailed;
				response.runtime_failure  = runtime.failure;
				response.detail			  = runtime.detail;
			}
			return write_child_response(options.internal_response_fd, response) ? 0 : 1;
		}

		SpirvTargetVersion	max_supported  = SpirvTargetVersion::V10;
		RuntimeConfig const runtime_config = {
				.binding_model			= options.binding_model == "push_descriptor"
												  ? RuntimeBindingModel::PushDescriptor
												  : (options.binding_model == "descriptor_buffer"
															 ? RuntimeBindingModel::DescriptorBuffer
															 : (options.binding_model == "descriptor_heap"
																		? RuntimeBindingModel::DescriptorHeap
																		: RuntimeBindingModel::Classic)),
				.enable_bda				= false,
				.required_subgroup_size = static_cast<uint32_t>(options.requested_subgroup_size),
				.require_full_subgroups = options.require_full_subgroups,
				.gpu_key				= options.gpu_key,
		};
		RuntimeResult const runtime = query_max_supported_spirv_version(runtime_config, max_supported);
		if (!runtime.ok()) {
			response.internal_failure = InternalFailure::ChildRuntimeFailed;
			response.runtime_failure  = runtime.failure;
			response.detail			  = runtime.detail;
		} else {
			response.output = spirv_target_to_string(max_supported);
		}
		return write_child_response(options.internal_response_fd, response) ? 0 : 1;
	}

	RuntimeBindingModel runtime_binding_model_from_cli(const CliOptions &options) {
		if (options.binding_model == "push_descriptor") { return RuntimeBindingModel::PushDescriptor; }
		if (options.binding_model == "descriptor_buffer") { return RuntimeBindingModel::DescriptorBuffer; }
		if (options.binding_model == "descriptor_heap") { return RuntimeBindingModel::DescriptorHeap; }
		return RuntimeBindingModel::Classic;
	}

	RuntimeConfig build_runtime_config(const CliOptions &options, const device_config &active_config) {
		return RuntimeConfig{
				.binding_model			= runtime_binding_model_from_cli(options),
				.enable_bda				= false,
				.required_subgroup_size = static_cast<uint32_t>(options.requested_subgroup_size),
				.require_full_subgroups = options.require_full_subgroups,
				.gpu_key				= active_config.key,
		};
	}

	void append_runtime_selection_args(const CliOptions &options, const device_config &active_config,
									   std::vector<std::string> &args) {
		args.emplace_back("--gpu");
		args.emplace_back(active_config.key);
		args.emplace_back("--binding-model");
		args.emplace_back(options.binding_model);
		if (options.requested_subgroup_size > 0) {
			args.emplace_back("--required-subgroup-size");
			args.emplace_back(std::to_string(options.requested_subgroup_size));
		}
		if (options.require_full_subgroups) { args.emplace_back("--require-full-subgroups"); }
	}

	int handle_info_target(const CliOptions &options, const device_config &active_config, const char *argv0) {
		std::string info_text;
		if (!active_config.shim_file.empty() && !options.internal_shim_ready) {
			fs::path const installed_libdir = default_mesa_libdir();
			fs::path const shim_path = resolve_installed_library_path(installed_libdir, active_config.shim_file);
			std::vector<std::string> child_args = {"--internal-mode", "info", "--internal-shim-ready"};
			append_runtime_selection_args(options, active_config, child_args);
			ChildResponse const child = run_internal_child_with_shim(argv0, shim_path, child_args);
			if (report_child_failure("collecting device info", child) != 0) { return 1; }
			info_text = child.output;
		} else {
			RuntimeConfig const runtime_config = build_runtime_config(options, active_config);
			RuntimeResult const rc			   = run_device_info_dump(runtime_config, info_text);
			if (report_runtime_result_failure(rc) != 0) { return 1; }
		}

		return write_text_output(options.output_path, info_text) ? 0 : 1;
	}

	bool query_max_supported_spirv(const CliOptions &options, const device_config &active_config, const char *argv0,
								   SpirvTargetVersion &out_version) {
		out_version = SpirvTargetVersion::V10;
		if (!active_config.shim_file.empty() && !options.internal_shim_ready) {
			fs::path const installed_libdir = default_mesa_libdir();
			fs::path const shim_path = resolve_installed_library_path(installed_libdir, active_config.shim_file);
			std::vector<std::string> child_args = {"--internal-mode", "spirv-max", "--internal-shim-ready"};
			append_runtime_selection_args(options, active_config, child_args);
			ChildResponse const child = run_internal_child_with_shim(argv0, shim_path, child_args);
			if (report_child_failure("querying max SPIR-V target", child) != 0) { return false; }
			if (!parse_spirv_version_text(child.output, out_version)) {
				std::println(stderr, "internal child returned invalid max SPIR-V version payload: '{}'", child.output);
				return false;
			}
			return true;
		}

		RuntimeConfig const runtime_config = build_runtime_config(options, active_config);
		RuntimeResult const rc			   = query_max_supported_spirv_version(runtime_config, out_version);
		return report_runtime_result_failure(rc) == 0;
	}

	bool resolve_active_spirv_target(const CliOptions &options, SpirvTargetVersion max_supported,
									 SpirvTargetVersion &out_active) {
		if (options.spirv_target_mode == SpirvTargetMode::Max) {
			out_active = max_supported;
			return true;
		}
		if (static_cast<int>(options.requested_spirv_target) > static_cast<int>(max_supported)) {
			std::println(stderr,
						 "requested SPIR-V target {} is not supported by the selected driver; maximum supported is {}",
						 spirv_target_to_string(options.requested_spirv_target), spirv_target_to_string(max_supported));
			return false;
		}
		out_active = options.requested_spirv_target;
		return true;
	}

	bool compile_shader_for_target(const CliOptions &options, SpirvTargetVersion active_spirv_target,
								   std::vector<uint32_t> &out_spirv) {
		if (!options.shader_path.has_value()) { return false; }
		if (options.source_lang == SourceLanguage::Slang) {
			return compile_slang(*options.shader_path, active_spirv_target, out_spirv);
		}
		if (options.source_lang == SourceLanguage::Glsl) {
			return compile_glsl_compute(*options.shader_path, active_spirv_target, out_spirv);
		}

		std::string const extension = shader_extension_lower(*options.shader_path);
		if (extension == ".slang") { return compile_slang(*options.shader_path, active_spirv_target, out_spirv); }
		if (extension == ".glsl" || extension == ".comp") {
			return compile_glsl_compute(*options.shader_path, active_spirv_target, out_spirv);
		}
		std::println(stderr,
					 "unsupported shader file extension: {}\n"
					 "supported in auto mode: .slang, .glsl, .comp\n"
					 "or pass --lang slang|glsl to force parser selection",
					 extension);
		return false;
	}

	int handle_pipeline_target(const CliOptions &options, const device_config &active_config, const char *argv0,
							   const std::vector<uint32_t> &spirv) {
		if (!active_config.shim_file.empty() && !options.internal_shim_ready) {
			SpirvSharedMemory spirv_mem;
			if (!create_spirv_shared_memory(spirv, spirv_mem)) {
				std::println(stderr, "failed to prepare shared-memory SPIR-V payload for child runtime");
				return 1;
			}

			std::vector<std::string> const child_args = {
					"--internal-mode",
					"pipeline",
					"--internal-shim-ready",
					"--internal-spirv-fd",
					std::to_string(spirv_mem.fd),
					"--internal-spirv-bytes",
					std::to_string(spirv_mem.bytes),
					"--target",
					options.output_target == OutputTarget::FinalNir ? "final_nir" : "asm",
			};
			std::vector<std::string> child_args_with_config = child_args;
			append_runtime_selection_args(options, active_config, child_args_with_config);

			fs::path const installed_libdir = default_mesa_libdir();
			fs::path const shim_path = resolve_installed_library_path(installed_libdir, active_config.shim_file);
			ChildResponse const child	  = run_internal_child_with_shim(argv0, shim_path, child_args_with_config);
			close(spirv_mem.fd);
			if (report_child_failure("dumping pipeline", child) != 0) { return 1; }
			return write_text_output(options.output_path, child.output) ? 0 : 1;
		}

		RuntimeDumpOptions const dump_options = {
				.target = options.output_target == OutputTarget::FinalNir ? RuntimeOutputTarget::FinalNir
																		  : RuntimeOutputTarget::Asm,
		};
		std::string			out_text;
		RuntimeConfig const runtime_config = build_runtime_config(options, active_config);
		RuntimeResult const rc			   = run_pipeline_dump(spirv, dump_options, runtime_config, out_text);
		if (report_runtime_result_failure(rc) != 0) { return 1; }
		return write_text_output(options.output_path, out_text) ? 0 : 1;
	}

} // namespace

int main(int argc, char **argv) {
	try {
		auto catalog = build_device_catalog();

		CliOptions options;
		if (std::optional<int> parse_exit = parse_cli_options(argc, argv, catalog, options); parse_exit.has_value()) {
			return *parse_exit;
		}
		if (default_mesa_libdir().empty() || default_vulkan_icd_dir().empty()) {
			std::println(stderr, "failed to resolve installed Mesa runtime directories");
			return 1;
		}
		if (std::optional<int> internal_exit = maybe_handle_internal_mode(options); internal_exit.has_value()) {
			return *internal_exit;
		}

		if (options.list_gpus) {
			print_usage(catalog);
			return 0;
		}
		if (options.gpu_key.empty()) { std::println(stderr, "No GPU selected."); }

		auto config_it = catalog.find(options.gpu_key);
		if (config_it == catalog.end()) {
			std::println(stderr, "unknown gpu key: {}", options.gpu_key);
			print_usage(catalog);
			return 1;
		}
		device_config const active_config = config_it->second;
		if (!configure_gpu_environment(active_config, options.internal_shim_ready)) { return 1; }

		if (options.output_target == OutputTarget::Info) {
			return handle_info_target(options, active_config, argv[0]);
		}

		SpirvTargetVersion max_supported_spirv = SpirvTargetVersion::V10;
		if (!query_max_supported_spirv(options, active_config, argv[0], max_supported_spirv)) {
			return 1;
		}
		SpirvTargetVersion active_spirv_target = SpirvTargetVersion::V10;
		if (!resolve_active_spirv_target(options, max_supported_spirv, active_spirv_target)) { return 1; }

		if (!options.shader_path.has_value()) {
			print_usage(catalog);
			return 1;
		}
		if (!fs::exists(*options.shader_path)) {
			std::println(stderr, "shader file not found: {}", options.shader_path->string());
			return 1;
		}

		std::vector<uint32_t> spirv_input;
		if (!compile_shader_for_target(options, active_spirv_target, spirv_input)) { return 1; }
		std::vector<uint32_t> spirv_optimized;
		if (!run_spirv_opt(spirv_input, active_spirv_target, spirv_optimized)) { return 1; }

		if (options.output_target == OutputTarget::Spirv) {
			std::string disassembly;
			bool const use_color = !options.no_color && options.output_path == "-";
			if (!run_spirv_dis(spirv_optimized, use_color, disassembly)) { return 1; }
			return write_text_output(options.output_path, disassembly) ? 0 : 1;
		}

		return handle_pipeline_target(options, active_config, argv[0], spirv_optimized);
	} catch (const std::format_error &e) {
		std::println(stderr, "format error: {}", e.what());
		return 2;
	} catch (...) {
		std::println(stderr, "unexpected fatal error");
		return 1;
	}
}
