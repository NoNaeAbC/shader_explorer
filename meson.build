# Warning this file is AI generated

project('shader_explorer', 'cpp',
  version : '0.1.0',
  default_options : ['cpp_std=c++23']
)

# -----------------------------------------------------------------------------
# Tooling
# -----------------------------------------------------------------------------
fs       = import('fs')
cmake_mod = import('cmake')
python   = import('python').find_installation(required: true)

mesonp   = find_program('meson', required: true)
ninja    = find_program('ninja', required: true)

cpp      = meson.get_compiler('cpp')
threads  = dependency('threads', required: true)
vulkan   = dependency('vulkan', required: true)
dl       = cpp.find_library('dl', required: true)
m        = cpp.find_library('m', required: true)

strict_cpp_args = []
if not get_option('no_warnings')
  strict_candidate_args = [
    '-ffast-math',
    '-Wall',
    '-Wextra',
  ]
  if not get_option('no_werror')
    strict_candidate_args += ['-Werror']
  endif
  strict_cpp_args = cpp.get_supported_arguments(strict_candidate_args)
  message('shader_explorer strict warning flags enabled: ' + ' '.join(strict_cpp_args))
else
  message('shader_explorer warning flags disabled by -Dno_warnings=true')
endif

source_root = meson.project_source_root()
build_root  = meson.project_build_root()

# -----------------------------------------------------------------------------
# Mesa as a Meson subproject
#
# Assumes you have either:
#   subprojects/mesa.wrap
# or:
#   subprojects/mesa/meson.build
#
# We configure Mesa as a Meson subproject. Do not invoke `ninja` recursively
# from a custom target; that re-enters the same build graph and can corrupt
# Ninja state under parallel builds.
# -----------------------------------------------------------------------------
mesa_sp = subproject('mesa',
  default_options : [
    'vulkan-drivers=amd,broadcom,freedreno,intel,intel_hasvk,panfrost,nouveau',
    'tools=drm-shim,intel',

    'opengl=false',
    'gles1=disabled',
    'gles2=disabled',
    'egl=disabled',
    'glx=disabled',
    'gbm=disabled',
    'llvm=enabled',
    'gallium-drivers=',
  ]
)

# Meson’s conventional build dir for Meson subprojects is:
#   <top-build>/subprojects/<name>
mesa_build_root  = build_root / 'subprojects' / 'mesa'
mesa_source_root = source_root / 'subprojects' / 'mesa'

if not fs.exists(mesa_source_root / 'meson.build')
  error('Mesa source root is missing at: ' + mesa_source_root)
endif

mesa_ninja_targets = [
  'src/amd/vulkan/libvulkan_radeon.so',
  'src/amd/drm-shim/libamdgpu_noop_drm_shim.so',
  'src/broadcom/vulkan/libvulkan_broadcom.so',
  'src/freedreno/vulkan/libvulkan_freedreno.so',
  'src/freedreno/drm-shim/libfreedreno_noop_drm_shim.so',
  'src/intel/vulkan/libvulkan_intel.so',
  'src/intel/vulkan_hasvk/libvulkan_intel_hasvk.so',
  'src/intel/tools/libintel_noop_drm_shim.so',
  'src/panfrost/vulkan/libvulkan_panfrost.so',
  'src/panfrost/drm-shim/libpanfrost_noop_drm_shim.so',
  'src/nouveau/vulkan/libvulkan_nouveau.so',
  'src/nouveau/drm-shim/libnouveau_noop_drm_shim.so',
]

mesa_artifacts = custom_target(
  'mesa_artifacts',
  output : 'mesa_artifacts.stamp',
  command : [
    python, files('tools/run_and_stamp.py'),
    '@OUTPUT@',
    python, '-c', 'pass',
  ],
  build_by_default : true,
)

# -----------------------------------------------------------------------------
# External Slang build (./slang, prepared by build_slang.sh)
# -----------------------------------------------------------------------------
slang_source_root = source_root / 'slang'
slang_build_root_external = slang_source_root / '__CMake_build_external'
slang_include_root = slang_source_root / 'include'
slang_generated_include_root = slang_build_root_external / 'source' / 'slang' / 'slang-version-header'
slang_lib_root = slang_build_root_external / 'Release' / 'lib'

if not fs.exists(slang_include_root / 'slang.h')
  error('Missing external Slang headers at ' + (slang_include_root / 'slang.h') +
    '\nRun ./build_slang.sh before: meson setup build')
endif
if not fs.exists(slang_generated_include_root / 'slang-tag-version.h')
  error('Missing generated Slang headers at ' + (slang_generated_include_root / 'slang-tag-version.h') +
    '\nRun ./build_slang.sh before: meson setup build')
endif
if not fs.exists(slang_lib_root / 'libslang-compiler.so')
  error('Missing external Slang library at ' + (slang_lib_root / 'libslang-compiler.so') +
    '\nRun ./build_slang.sh before: meson setup build')
endif

slang_compiler_lib = cpp.find_library(
  'slang-compiler',
  dirs : [slang_lib_root],
  required : true,
)

slang_dep = declare_dependency(
  compile_args : [
    '-I' + slang_include_root,
    '-I' + slang_generated_include_root,
  ],
  dependencies : [slang_compiler_lib],
  link_args : ['-Wl,-rpath,' + slang_lib_root],
)

# -----------------------------------------------------------------------------
# CMake subproject: SPIRV-Tools (from wrap), static linkage
# -----------------------------------------------------------------------------
spirv_headers_opts = cmake_mod.subproject_options()
spirv_headers_opts.add_cmake_defines({
  'BUILD_TESTING': false,
})
spirv_headers_sp = cmake_mod.subproject('spirv-headers', options: spirv_headers_opts)
if not spirv_headers_sp.found()
  error('spirv-headers subproject not found')
endif

spirv_tools_opts = cmake_mod.subproject_options()
spirv_headers_root = source_root / 'subprojects' / 'spirv-headers'
spirv_headers_include = spirv_headers_root / 'include'
spirv_tools_opts.add_cmake_defines({
  'BUILD_SHARED_LIBS': false,
  'SPIRV_SKIP_TESTS': true,
  'SPIRV_SKIP_EXECUTABLES': true,
  'SPIRV-Headers_SOURCE_DIR': spirv_headers_root,
  'CMAKE_C_FLAGS': '-I' + spirv_headers_include,
  'CMAKE_CXX_FLAGS': '-I' + spirv_headers_include,
})
spirv_tools_sp = cmake_mod.subproject('spirv-tools', options: spirv_tools_opts)

spirv_tools_dep = disabler()
spirv_tools_opt_dep = disabler()

if spirv_tools_sp.found()
  spirv_tools_targets = spirv_tools_sp.target_list()
  message('SPIRV-Tools: CMake targets: ' + ', '.join(spirv_tools_targets))

  spirv_tools_dep = spirv_tools_sp.dependency('SPIRV-Tools-static')
  spirv_tools_opt_dep = spirv_tools_sp.dependency('SPIRV-Tools-opt')
endif

if not spirv_tools_dep.found()
  error('SPIRV-Tools static dependency not found in spirv-tools subproject')
endif
if not spirv_tools_opt_dep.found()
  error('SPIRV-Tools-opt static dependency not found in spirv-tools subproject')
endif

# -----------------------------------------------------------------------------
# CMake subproject: glslang (from wrap)
# -----------------------------------------------------------------------------
glslang_opts = cmake_mod.subproject_options()
glslang_opts.add_cmake_defines({
  'BUILD_SHARED_LIBS': false,
  'CMAKE_BUILD_TYPE': 'Release',
  'ENABLE_OPT': 0,
})
glslang_sp = cmake_mod.subproject('glslang', options: glslang_opts)

# --- glslang ---
glslang_dep = disabler()
glslang_link_with = []

if glslang_sp.found()
  glslang_targets = glslang_sp.target_list()
  message('glslang: CMake targets: ' + ', '.join(glslang_targets))

  if 'glslang' in glslang_targets
    glslang_dep = glslang_sp.dependency('glslang')  # no required: kwarg here
  else
    glslang_target_name = 'glslang'
    if glslang_target_name in glslang_targets
      glslang_link_with += glslang_sp.target(glslang_target_name)
    endif
  endif

  foreach t : ['glslang-default-resource-limits', 'SPIRV', 'OSDependent', 'MachineIndependent', 'GenericCodeGen']
    if t in glslang_targets
      glslang_link_with += glslang_sp.target(t)
    endif
  endforeach
endif

deps = [
  vulkan,
  threads,
  dl,
  m,
  slang_dep,
  spirv_tools_dep,
  spirv_tools_opt_dep,
]

if glslang_dep.found()
  deps += glslang_dep
endif

# -----------------------------------------------------------------------------
# Final executable
# -----------------------------------------------------------------------------
executable(
  'shader_explorer',
  files(
    'src/main.cpp',
    'src/vulkan_runtime.cpp',
  ),
  cpp_args : [
    '-DSHADER_EXPLORER_DEFAULT_MESA_BUILD_ROOT="' + mesa_build_root + '"',
  ] + strict_cpp_args,
  dependencies : deps,
  link_with : glslang_link_with,
  link_depends : [mesa_artifacts],
  build_by_default : true,
  install : true,
  install_dir : get_option('bindir'),
)
