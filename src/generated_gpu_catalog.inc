void append_generated_device_catalog(std::map<std::string, device_config> &catalog) {
	maybe_add_device(catalog, {
									  .key				 = "arm-g72",
									  .vendor			 = "arm",
									  .description		 = "Panfrost PanVK (G72)",
									  .vulkan_icd_file = "panfrost_icd.x86_64.json",
									  .shim_file = "libpanfrost_noop_drm_shim.so",
									  .env = {{"PAN_GPU_ID", "6201"}, {"PAN_I_WANT_A_BROKEN_VULKAN_DRIVER", "1"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "arm-g31",
									  .vendor			 = "arm",
									  .description		 = "Panfrost PanVK (G31)",
									  .vulkan_icd_file = "panfrost_icd.x86_64.json",
									  .shim_file = "libpanfrost_noop_drm_shim.so",
									  .env = {{"PAN_GPU_ID", "7003"}, {"PAN_I_WANT_A_BROKEN_VULKAN_DRIVER", "1"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "arm-g51",
									  .vendor			 = "arm",
									  .description		 = "Panfrost PanVK (G51)",
									  .vulkan_icd_file = "panfrost_icd.x86_64.json",
									  .shim_file = "libpanfrost_noop_drm_shim.so",
									  .env = {{"PAN_GPU_ID", "7000"}, {"PAN_I_WANT_A_BROKEN_VULKAN_DRIVER", "1"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "arm-g52",
									  .vendor			 = "arm",
									  .description		 = "Panfrost PanVK (G52)",
									  .vulkan_icd_file = "panfrost_icd.x86_64.json",
									  .shim_file = "libpanfrost_noop_drm_shim.so",
									  .env = {{"PAN_GPU_ID", "7202"}, {"PAN_I_WANT_A_BROKEN_VULKAN_DRIVER", "1"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "arm-g76",
									  .vendor			 = "arm",
									  .description		 = "Panfrost PanVK (G76)",
									  .vulkan_icd_file = "panfrost_icd.x86_64.json",
									  .shim_file = "libpanfrost_noop_drm_shim.so",
									  .env = {{"PAN_GPU_ID", "7201"}, {"PAN_I_WANT_A_BROKEN_VULKAN_DRIVER", "1"}},
							  });

	maybe_add_device(catalog, {
									  .key				 = "arm-g610",
									  .vendor			 = "arm",
									  .description		 = "Panfrost PanVK (G610)",
									  .vulkan_icd_file = "panfrost_icd.x86_64.json",
									  .shim_file = "libpanfrost_noop_drm_shim.so",
									  .env				 = {{"PAN_GPU_ID", "a807"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "arm-g310",
									  .vendor			 = "arm",
									  .description		 = "Panfrost PanVK (G310)",
									  .vulkan_icd_file = "panfrost_icd.x86_64.json",
									  .shim_file = "libpanfrost_noop_drm_shim.so",
									  .env				 = {{"PAN_GPU_ID", "ac04:4"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "arm-g720",
									  .vendor			 = "arm",
									  .description		 = "Panfrost PanVK (G720) variant 4",
									  .vulkan_icd_file = "panfrost_icd.x86_64.json",
									  .shim_file = "libpanfrost_noop_drm_shim.so",
									  .env				 = {{"PAN_GPU_ID", "c800:4"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "arm-g725",
									  .vendor			 = "arm",
									  .description		 = "Panfrost PanVK (G725) variant 4",
									  .vulkan_icd_file = "panfrost_icd.x86_64.json",
									  .shim_file = "libpanfrost_noop_drm_shim.so",
									  .env				 = {{"PAN_GPU_ID", "d800:4"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "amd-renoir",
									  .vendor			 = "amd",
									  .description		 = "AMD RADV (renoir)",
									  .vulkan_icd_file = "radeon_icd.x86_64.json",
									  .shim_file = "libamdgpu_noop_drm_shim.so",
									  .env = {{"AMDGPU_GPU_ID", "renoir"}, {"MESA_LOADER_DRIVER_OVERRIDE", "amdgpu"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "amd-raven",
									  .vendor			 = "amd",
									  .description		 = "AMD RADV (raven)",
									  .vulkan_icd_file = "radeon_icd.x86_64.json",
									  .shim_file = "libamdgpu_noop_drm_shim.so",
									  .env = {{"AMDGPU_GPU_ID", "raven"}, {"MESA_LOADER_DRIVER_OVERRIDE", "amdgpu"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "amd-raven2",
									  .vendor			 = "amd",
									  .description		 = "AMD RADV (raven2)",
									  .vulkan_icd_file = "radeon_icd.x86_64.json",
									  .shim_file = "libamdgpu_noop_drm_shim.so",
									  .env = {{"AMDGPU_GPU_ID", "raven2"}, {"MESA_LOADER_DRIVER_OVERRIDE", "amdgpu"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "amd-stoney",
									  .vendor			 = "amd",
									  .description		 = "AMD RADV (stoney)",
									  .vulkan_icd_file = "radeon_icd.x86_64.json",
									  .shim_file = "libamdgpu_noop_drm_shim.so",
									  .env = {{"AMDGPU_GPU_ID", "stoney"}, {"MESA_LOADER_DRIVER_OVERRIDE", "amdgpu"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "amd-vangogh",
									  .vendor			 = "amd",
									  .description		 = "AMD RADV (vangogh)",
									  .vulkan_icd_file = "radeon_icd.x86_64.json",
									  .shim_file = "libamdgpu_noop_drm_shim.so",
									  .env = {{"AMDGPU_GPU_ID", "vangogh"}, {"MESA_LOADER_DRIVER_OVERRIDE", "amdgpu"}},
							  });
	maybe_add_device(catalog,
					 {
							 .key				= "amd-raphael_mendocino",
							 .vendor			= "amd",
							 .description		= "AMD RADV (raphael_mendocino)",
							 .vulkan_icd_file = "radeon_icd.x86_64.json",
							 .shim_file = "libamdgpu_noop_drm_shim.so",
							 .env = {{"AMDGPU_GPU_ID", "raphael_mendocino"}, {"MESA_LOADER_DRIVER_OVERRIDE", "amdgpu"}},
					 });
	maybe_add_device(catalog,
					 {
							 .key				= "amd-polaris10",
							 .vendor			= "amd",
							 .description		= "AMD RADV (polaris10)",
							 .vulkan_icd_file = "radeon_icd.x86_64.json",
							 .shim_file = "libamdgpu_noop_drm_shim.so",
							 .env = {{"AMDGPU_GPU_ID", "polaris10"}, {"MESA_LOADER_DRIVER_OVERRIDE", "amdgpu"}},
					 });
	maybe_add_device(catalog,
					 {
							 .key				= "amd-polaris12",
							 .vendor			= "amd",
							 .description		= "AMD RADV (polaris12)",
							 .vulkan_icd_file = "radeon_icd.x86_64.json",
							 .shim_file = "libamdgpu_noop_drm_shim.so",
							 .env = {{"AMDGPU_GPU_ID", "polaris12"}, {"MESA_LOADER_DRIVER_OVERRIDE", "amdgpu"}},
					 });
	maybe_add_device(catalog, {
									  .key				 = "amd-vega10",
									  .vendor			 = "amd",
									  .description		 = "AMD RADV (vega10)",
									  .vulkan_icd_file = "radeon_icd.x86_64.json",
									  .shim_file = "libamdgpu_noop_drm_shim.so",
									  .env = {{"AMDGPU_GPU_ID", "vega10"}, {"MESA_LOADER_DRIVER_OVERRIDE", "amdgpu"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "amd-navi10",
									  .vendor			 = "amd",
									  .description		 = "AMD RADV (navi10)",
									  .vulkan_icd_file = "radeon_icd.x86_64.json",
									  .shim_file = "libamdgpu_noop_drm_shim.so",
									  .env = {{"AMDGPU_GPU_ID", "navi10"}, {"MESA_LOADER_DRIVER_OVERRIDE", "amdgpu"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "amd-navi31",
									  .vendor			 = "amd",
									  .description		 = "AMD RADV (navi31)",
									  .vulkan_icd_file = "radeon_icd.x86_64.json",
									  .shim_file = "libamdgpu_noop_drm_shim.so",
									  .env = {{"AMDGPU_GPU_ID", "navi31"}, {"MESA_LOADER_DRIVER_OVERRIDE", "amdgpu"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "amd-navi33",
									  .vendor			 = "amd",
									  .description		 = "AMD RADV (navi33)",
									  .vulkan_icd_file = "radeon_icd.x86_64.json",
									  .shim_file = "libamdgpu_noop_drm_shim.so",
									  .env = {{"AMDGPU_GPU_ID", "navi33"}, {"MESA_LOADER_DRIVER_OVERRIDE", "amdgpu"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "amd-navi21",
									  .vendor			 = "amd",
									  .description		 = "AMD RADV (navi21)",
									  .vulkan_icd_file = "radeon_icd.x86_64.json",
									  .shim_file = "libamdgpu_noop_drm_shim.so",
									  .env = {{"AMDGPU_GPU_ID", "navi21"}, {"MESA_LOADER_DRIVER_OVERRIDE", "amdgpu"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "amd-pitcairn",
									  .vendor			 = "amd",
									  .description		 = "AMD RADV (pitcairn)",
									  .vulkan_icd_file = "radeon_icd.x86_64.json",
									  .shim_file = "libamdgpu_noop_drm_shim.so",
									  .env = {{"AMDGPU_GPU_ID", "pitcairn"}, {"MESA_LOADER_DRIVER_OVERRIDE", "amdgpu"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "amd-bonaire",
									  .vendor			 = "amd",
									  .description		 = "AMD RADV (bonaire)",
									  .vulkan_icd_file = "radeon_icd.x86_64.json",
									  .shim_file = "libamdgpu_noop_drm_shim.so",
									  .env = {{"AMDGPU_GPU_ID", "bonaire"}, {"MESA_LOADER_DRIVER_OVERRIDE", "amdgpu"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "amd-strix1",
									  .vendor			 = "amd",
									  .description		 = "AMD RADV (strix1)",
									  .vulkan_icd_file = "radeon_icd.x86_64.json",
									  .shim_file = "libamdgpu_noop_drm_shim.so",
									  .env = {{"AMDGPU_GPU_ID", "strix1"}, {"MESA_LOADER_DRIVER_OVERRIDE", "amdgpu"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "amd-gfx1201",
									  .vendor			 = "amd",
									  .description		 = "AMD RADV (gfx1201)",
									  .vulkan_icd_file = "radeon_icd.x86_64.json",
									  .shim_file = "libamdgpu_noop_drm_shim.so",
									  .env = {{"AMDGPU_GPU_ID", "gfx1201"}, {"MESA_LOADER_DRIVER_OVERRIDE", "amdgpu"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "amd-vega20",
									  .vendor			 = "amd",
									  .description		 = "AMD RADV (vega20)",
									  .vulkan_icd_file = "radeon_icd.x86_64.json",
									  .shim_file = "libamdgpu_noop_drm_shim.so",
									  .env = {{"AMDGPU_GPU_ID", "vega20"}, {"MESA_LOADER_DRIVER_OVERRIDE", "amdgpu"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "qualcomm-630",
									  .vendor			 = "qualcomm",
									  .description		 = "Freedreno Turnip (A630)",
									  .vulkan_icd_file = "freedreno_icd.x86_64.json",
									  .shim_file = "libfreedreno_noop_drm_shim.so",
									  .env = {{"FD_GPU_ID", "630"}, {"MESA_LOADER_DRIVER_OVERRIDE", "msm"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "qualcomm-618",
									  .vendor			 = "qualcomm",
									  .description		 = "Freedreno Turnip (A618)",
									  .vulkan_icd_file = "freedreno_icd.x86_64.json",
									  .shim_file = "libfreedreno_noop_drm_shim.so",
									  .env = {{"FD_GPU_ID", "618"}, {"MESA_LOADER_DRIVER_OVERRIDE", "msm"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "qualcomm-660",
									  .vendor			 = "qualcomm",
									  .description		 = "Freedreno Turnip (A660)",
									  .vulkan_icd_file = "freedreno_icd.x86_64.json",
									  .shim_file = "libfreedreno_noop_drm_shim.so",
									  .env = {{"FD_GPU_ID", "660"}, {"MESA_LOADER_DRIVER_OVERRIDE", "msm"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "qualcomm-702",
									  .vendor			 = "qualcomm",
									  .description		 = "Freedreno Turnip (A702)",
									  .vulkan_icd_file = "freedreno_icd.x86_64.json",
									  .shim_file = "libfreedreno_noop_drm_shim.so",
									  .env = {{"FD_GPU_ID", "702"}, {"MESA_LOADER_DRIVER_OVERRIDE", "msm"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "qualcomm-730",
									  .vendor			 = "qualcomm",
									  .description		 = "Freedreno Turnip (A730)",
									  .vulkan_icd_file = "freedreno_icd.x86_64.json",
									  .shim_file = "libfreedreno_noop_drm_shim.so",
									  .env = {{"FD_GPU_ID", "730"}, {"MESA_LOADER_DRIVER_OVERRIDE", "msm"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "qualcomm-740",
									  .vendor			 = "qualcomm",
									  .description		 = "Freedreno Turnip (A740)",
									  .vulkan_icd_file = "freedreno_icd.x86_64.json",
									  .shim_file = "libfreedreno_noop_drm_shim.so",
									  .env = {{"FD_GPU_ID", "740"}, {"MESA_LOADER_DRIVER_OVERRIDE", "msm"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "qualcomm-750",
									  .vendor			 = "qualcomm",
									  .description		 = "Freedreno Turnip (A750)",
									  .vulkan_icd_file = "freedreno_icd.x86_64.json",
									  .shim_file = "libfreedreno_noop_drm_shim.so",
									  .env = {{"FD_GPU_ID", "750"}, {"MESA_LOADER_DRIVER_OVERRIDE", "msm"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "qualcomm-830",
									  .vendor			 = "qualcomm",
									  .description		 = "Freedreno Turnip (A830)",
									  .vulkan_icd_file = "freedreno_icd.x86_64.json",
									  .shim_file = "libfreedreno_noop_drm_shim.so",
									  .env = {{"FD_GPU_ID", "830"}, {"MESA_LOADER_DRIVER_OVERRIDE", "msm"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "intel-ivb",
									  .vendor			 = "intel",
									  .description		 = "Intel ANV (ivb)",
									  .vulkan_icd_file = "intel_hasvk_icd.x86_64.json",
									  .shim_file = "libintel_noop_drm_shim.so",
									  .env				 = {{"INTEL_STUB_GPU_PLATFORM", "ivb"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "intel-hsw",
									  .vendor			 = "intel",
									  .description		 = "Intel ANV (hsw)",
									  .vulkan_icd_file = "intel_hasvk_icd.x86_64.json",
									  .shim_file = "libintel_noop_drm_shim.so",
									  .env				 = {{"INTEL_STUB_GPU_PLATFORM", "hsw"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "intel-byt",
									  .vendor			 = "intel",
									  .description		 = "Intel ANV (byt)",
									  .vulkan_icd_file = "intel_hasvk_icd.x86_64.json",
									  .shim_file = "libintel_noop_drm_shim.so",
									  .env				 = {{"INTEL_STUB_GPU_PLATFORM", "byt"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "intel-bdw",
									  .vendor			 = "intel",
									  .description		 = "Intel ANV (bdw)",
									  .vulkan_icd_file = "intel_hasvk_icd.x86_64.json",
									  .shim_file = "libintel_noop_drm_shim.so",
									  .env				 = {{"INTEL_STUB_GPU_PLATFORM", "bdw"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "intel-chv",
									  .vendor			 = "intel",
									  .description		 = "Intel ANV (chv)",
									  .vulkan_icd_file = "intel_hasvk_icd.x86_64.json",
									  .shim_file = "libintel_noop_drm_shim.so",
									  .env				 = {{"INTEL_STUB_GPU_PLATFORM", "chv"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "intel-skl",
									  .vendor			 = "intel",
									  .description		 = "Intel ANV (skl)",
									  .vulkan_icd_file = "intel_icd.x86_64.json",
									  .shim_file = "libintel_noop_drm_shim.so",
									  .env				 = {{"INTEL_STUB_GPU_PLATFORM", "skl"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "intel-bxt",
									  .vendor			 = "intel",
									  .description		 = "Intel ANV (bxt)",
									  .vulkan_icd_file = "intel_icd.x86_64.json",
									  .shim_file = "libintel_noop_drm_shim.so",
									  .env				 = {{"INTEL_STUB_GPU_PLATFORM", "bxt"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "intel-kbl",
									  .vendor			 = "intel",
									  .description		 = "Intel ANV (kbl)",
									  .vulkan_icd_file = "intel_icd.x86_64.json",
									  .shim_file = "libintel_noop_drm_shim.so",
									  .env				 = {{"INTEL_STUB_GPU_PLATFORM", "kbl"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "intel-aml",
									  .vendor			 = "intel",
									  .description		 = "Intel ANV (aml)",
									  .vulkan_icd_file = "intel_icd.x86_64.json",
									  .shim_file = "libintel_noop_drm_shim.so",
									  .env				 = {{"INTEL_STUB_GPU_PLATFORM", "aml"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "intel-glk",
									  .vendor			 = "intel",
									  .description		 = "Intel ANV (glk)",
									  .vulkan_icd_file = "intel_icd.x86_64.json",
									  .shim_file = "libintel_noop_drm_shim.so",
									  .env				 = {{"INTEL_STUB_GPU_PLATFORM", "glk"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "intel-cfl",
									  .vendor			 = "intel",
									  .description		 = "Intel ANV (cfl)",
									  .vulkan_icd_file = "intel_icd.x86_64.json",
									  .shim_file = "libintel_noop_drm_shim.so",
									  .env				 = {{"INTEL_STUB_GPU_PLATFORM", "cfl"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "intel-whl",
									  .vendor			 = "intel",
									  .description		 = "Intel ANV (whl)",
									  .vulkan_icd_file = "intel_icd.x86_64.json",
									  .shim_file = "libintel_noop_drm_shim.so",
									  .env				 = {{"INTEL_STUB_GPU_PLATFORM", "whl"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "intel-cml",
									  .vendor			 = "intel",
									  .description		 = "Intel ANV (cml)",
									  .vulkan_icd_file = "intel_icd.x86_64.json",
									  .shim_file = "libintel_noop_drm_shim.so",
									  .env				 = {{"INTEL_STUB_GPU_PLATFORM", "cml"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "intel-icl",
									  .vendor			 = "intel",
									  .description		 = "Intel ANV (icl)",
									  .vulkan_icd_file = "intel_icd.x86_64.json",
									  .shim_file = "libintel_noop_drm_shim.so",
									  .env				 = {{"INTEL_STUB_GPU_PLATFORM", "icl"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "intel-ehl",
									  .vendor			 = "intel",
									  .description		 = "Intel ANV (ehl)",
									  .vulkan_icd_file = "intel_icd.x86_64.json",
									  .shim_file = "libintel_noop_drm_shim.so",
									  .env				 = {{"INTEL_STUB_GPU_PLATFORM", "ehl"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "intel-jsl",
									  .vendor			 = "intel",
									  .description		 = "Intel ANV (jsl)",
									  .vulkan_icd_file = "intel_icd.x86_64.json",
									  .shim_file = "libintel_noop_drm_shim.so",
									  .env				 = {{"INTEL_STUB_GPU_PLATFORM", "jsl"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "intel-tgl",
									  .vendor			 = "intel",
									  .description		 = "Intel ANV (tgl)",
									  .vulkan_icd_file = "intel_icd.x86_64.json",
									  .shim_file = "libintel_noop_drm_shim.so",
									  .env				 = {{"INTEL_STUB_GPU_PLATFORM", "tgl"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "intel-rkl",
									  .vendor			 = "intel",
									  .description		 = "Intel ANV (rkl)",
									  .vulkan_icd_file = "intel_icd.x86_64.json",
									  .shim_file = "libintel_noop_drm_shim.so",
									  .env				 = {{"INTEL_STUB_GPU_PLATFORM", "rkl"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "intel-dg1",
									  .vendor			 = "intel",
									  .description		 = "Intel ANV (dg1)",
									  .vulkan_icd_file = "intel_icd.x86_64.json",
									  .shim_file = "libintel_noop_drm_shim.so",
									  .env				 = {{"INTEL_STUB_GPU_PLATFORM", "dg1"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "intel-adl",
									  .vendor			 = "intel",
									  .description		 = "Intel ANV (adl)",
									  .vulkan_icd_file = "intel_icd.x86_64.json",
									  .shim_file = "libintel_noop_drm_shim.so",
									  .env				 = {{"INTEL_STUB_GPU_PLATFORM", "adl"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "intel-sg1",
									  .vendor			 = "intel",
									  .description		 = "Intel ANV (sg1)",
									  .vulkan_icd_file = "intel_icd.x86_64.json",
									  .shim_file = "libintel_noop_drm_shim.so",
									  .env				 = {{"INTEL_STUB_GPU_PLATFORM", "sg1"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "intel-rpl",
									  .vendor			 = "intel",
									  .description		 = "Intel ANV (rpl)",
									  .vulkan_icd_file = "intel_icd.x86_64.json",
									  .shim_file = "libintel_noop_drm_shim.so",
									  .env				 = {{"INTEL_STUB_GPU_PLATFORM", "rpl"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "intel-dg2",
									  .vendor			 = "intel",
									  .description		 = "Intel ANV (dg2)",
									  .vulkan_icd_file = "intel_icd.x86_64.json",
									  .shim_file = "libintel_noop_drm_shim.so",
									  .env				 = {{"INTEL_STUB_GPU_PLATFORM", "dg2"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "intel-mtl",
									  .vendor			 = "intel",
									  .description		 = "Intel ANV (mtl)",
									  .vulkan_icd_file = "intel_icd.x86_64.json",
									  .shim_file = "libintel_noop_drm_shim.so",
									  .env				 = {{"INTEL_STUB_GPU_PLATFORM", "mtl"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "intel-arl",
									  .vendor			 = "intel",
									  .description		 = "Intel ANV (arl)",
									  .vulkan_icd_file = "intel_icd.x86_64.json",
									  .shim_file = "libintel_noop_drm_shim.so",
									  .env				 = {{"INTEL_STUB_GPU_PLATFORM", "arl"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "intel-lnl",
									  .vendor			 = "intel",
									  .description		 = "Intel ANV (lnl)",
									  .vulkan_icd_file = "intel_icd.x86_64.json",
									  .shim_file = "libintel_noop_drm_shim.so",
									  .env				 = {{"INTEL_STUB_GPU_PLATFORM", "lnl"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "intel-bmg",
									  .vendor			 = "intel",
									  .description		 = "Intel ANV (bmg)",
									  .vulkan_icd_file = "intel_icd.x86_64.json",
									  .shim_file = "libintel_noop_drm_shim.so",
									  .env				 = {{"INTEL_STUB_GPU_PLATFORM", "bmg"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "intel-ptl",
									  .vendor			 = "intel",
									  .description		 = "Intel ANV (ptl)",
									  .vulkan_icd_file = "intel_icd.x86_64.json",
									  .shim_file = "libintel_noop_drm_shim.so",
									  .env				 = {{"INTEL_STUB_GPU_PLATFORM", "ptl"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "nvidia-gk104",
									  .vendor			 = "nvidia",
									  .description		 = "Nouveau NVK (GK104, GeForce GTX 680, chipset e4)",
									  .vulkan_icd_file = "nouveau_icd.x86_64.json",
									  .shim_file = "libnouveau_noop_drm_shim.so",
									  .env = {{"NOUVEAU_CHIPSET", "e4"}, {"MESA_LOADER_DRIVER_OVERRIDE", "nouveau"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "nvidia-gk110",
									  .vendor			 = "nvidia",
									  .description		 = "Nouveau NVK (GK110, GeForce GTX 780, chipset f0)",
									  .vulkan_icd_file = "nouveau_icd.x86_64.json",
									  .shim_file = "libnouveau_noop_drm_shim.so",
									  .env = {{"NOUVEAU_CHIPSET", "f0"}, {"MESA_LOADER_DRIVER_OVERRIDE", "nouveau"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "nvidia-gm107",
									  .vendor			 = "nvidia",
									  .description		 = "Nouveau NVK (GM107, GeForce GTX 750, chipset 117)",
									  .vulkan_icd_file = "nouveau_icd.x86_64.json",
									  .shim_file = "libnouveau_noop_drm_shim.so",
									  .env = {{"NOUVEAU_CHIPSET", "117"}, {"MESA_LOADER_DRIVER_OVERRIDE", "nouveau"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "nvidia-gm204",
									  .vendor			 = "nvidia",
									  .description		 = "Nouveau NVK (GM204, GeForce GTX 980, chipset 124)",
									  .vulkan_icd_file = "nouveau_icd.x86_64.json",
									  .shim_file = "libnouveau_noop_drm_shim.so",
									  .env = {{"NOUVEAU_CHIPSET", "124"}, {"MESA_LOADER_DRIVER_OVERRIDE", "nouveau"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "nvidia-gp104",
									  .vendor			 = "nvidia",
									  .description		 = "Nouveau NVK (GP104, GeForce GTX 1080, chipset 134)",
									  .vulkan_icd_file = "nouveau_icd.x86_64.json",
									  .shim_file = "libnouveau_noop_drm_shim.so",
									  .env = {{"NOUVEAU_CHIPSET", "134"}, {"MESA_LOADER_DRIVER_OVERRIDE", "nouveau"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "nvidia-gv100",
									  .vendor			 = "nvidia",
									  .description		 = "Nouveau NVK (GV100, TITAN V, chipset 140)",
									  .vulkan_icd_file = "nouveau_icd.x86_64.json",
									  .shim_file = "libnouveau_noop_drm_shim.so",
									  .env = {{"NOUVEAU_CHIPSET", "140"}, {"MESA_LOADER_DRIVER_OVERRIDE", "nouveau"}},
							  });
	maybe_add_device(catalog, {
									  .key				 = "nvidia-tu102",
									  .vendor			 = "nvidia",
									  .description		 = "Nouveau NVK (TU102, GeForce RTX 2080, chipset 162)",
									  .vulkan_icd_file = "nouveau_icd.x86_64.json",
									  .shim_file = "libnouveau_noop_drm_shim.so",
									  .env = {{"NOUVEAU_CHIPSET", "162"}, {"MESA_LOADER_DRIVER_OVERRIDE", "nouveau"}},
							  });
}
