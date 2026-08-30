"""Compatibility exports for the retired OpenGL implementation module.

The implementation lives in the runtime, tensor, factory, dispatch, resource,
metadata, convolution, and operator modules.  This module intentionally has no
active implementation of its own.
"""

from . import backend as _backend
from . import convolution as _convolution
from . import diagnostics as _diagnostics
from . import dispatch as _dispatch
from . import factories as _factories
from . import kernels as _kernels
from . import metadata as _metadata
from . import operation_context as _operation_context
from . import resources as _resources
from . import tensor as _tensor
from . import runtime as _runtime_module
from . import profiling as _profiling
from .ops import activation as _activation
from .ops import arithmetic as _arithmetic
from .ops import concat as _concat
from .ops import matmul as _matmul
from .ops import normalization as _normalization
from .ops import pooling as _pooling
from .ops import resize as _resize
from .ops import softmax as _softmax
from .storage import StorageLayout as _StorageLayout

OpenGLBackend = _backend.OpenGLBackend
Gm45Tensor = _tensor.Gm45Tensor
PRIVATEUSE_DEVICE = _tensor.PRIVATEUSE_DEVICE

init = _backend.init
shutdown = _backend.shutdown
set_trace = _backend.set_trace
debug_enabled = _backend.debug_enabled
profile_enabled = _backend.profile_enabled
profile_report = _backend.profile_report
profile_reset = _backend.profile_reset
reset_unsupported_report = _backend.reset_unsupported_report
unsupported_report = _backend.unsupported_report
tensor = _backend.tensor
randn = _backend.randn
to_gm45 = _backend.to_gm45
is_gm45_tensor = _backend.is_gm45_tensor
install_tensor_method = _backend.install_tensor_method

CONV_PHYSICAL_TILE_LIMIT = _convolution.CONV_PHYSICAL_TILE_LIMIT
_conv_program = _convolution._conv_program
_conv_shader_source = _convolution._conv_shader_source
_conv_tile_shader_source = _convolution._conv_tile_shader_source
_tile_copy_shader_source = _convolution._tile_copy_shader_source
_new_physical_packed_owner = _convolution._new_physical_packed_owner
_render_convolution_tiled = _convolution._render_convolution_tiled
_tile_diagnostic_snapshots = _convolution._tile_diagnostic_snapshots
_last_tile_geometry = _convolution._last_tile_geometry
_is_contiguous_logical = _metadata.is_contiguous_logical
_require_contiguous_logical = _metadata.require_contiguous_logical
_normalize_shape = _metadata.normalize_shape
_metadata_view = _metadata.metadata_view
_metadata_transpose = _metadata.metadata_transpose
_metadata_unsqueeze = _metadata.metadata_unsqueeze
_metadata_squeeze = _metadata.metadata_squeeze
_metadata_expand = _metadata.metadata_expand
_metadata_split = _metadata.metadata_split
_program = _kernels.program
_glsl_float = _kernels.glsl_float
_ParameterCacheEntry = _resources.ParameterCacheEntry
_runtime_required = _runtime_module.runtime_required
_register_privateuse_name = _factories.register_privateuse_name
_install_privateuse1_factory_kernels = _factories.install_privateuse1_factory_kernels
_empty_gm45 = _factories.empty_gm45
_new_zero_element_placeholder = _factories._new_zero_element_placeholder
_arange_program = _factories.arange_program
_arange_shader_source = _factories.arange_shader_source
_arange_length = _factories._arange_length
_render_arange = _factories.render_arange
_arange_default_gm45 = _factories.arange_default
_arange_start_gm45 = _factories.arange_start
_arange_start_step_gm45 = _factories.arange_start_step
_validate_factory_options = _factories._validate_factory_options
_new_empty_packed_texture = _operation_context.output_texture
_new_empty_matrix_texture = _matmul.new_empty_matrix_texture
_create_rgba32f_texture = _resources.create_rgba32f_texture
_acquire_scratch_texture = _resources.acquire_scratch_texture
_release_scratch_texture = _resources.release_scratch_texture
_upload_array_to_texture = _resources.upload_array_to_texture
_upload_raw_packed_array = _resources.upload_raw_packed_array
_parameter_cache_key = _resources.parameter_cache_key
_cached_parameter_texture = _resources.cached_parameter_texture
_texture_from_cpu = _tensor.texture_from_cpu
_validate_cpu_input = _tensor._validate_cpu_input
_read_texture = _tensor.readback_tensor
_DispatchBridge = _dispatch.DispatchBridge

_batchnorm_program = _normalization._batchnorm_program
_batchnorm_shader_source = _normalization._batchnorm_shader_source
_render_batch_norm = _normalization._render_batch_norm
_silu_program = _activation._silu_program
_silu_shader_source = _activation._silu_shader_source
_packed_sigmoid_program = _activation._packed_sigmoid_program
_packed_sigmoid_shader_source = _activation._packed_sigmoid_shader_source
_render_packed_sigmoid = _activation._render_packed_sigmoid
_render_silu_inplace = _activation._render_silu_inplace
_packed_add_program = _arithmetic._packed_add_program
_packed_add_shader_source = _arithmetic._packed_add_shader_source
_packed_sub_program = _arithmetic._packed_sub_program
_packed_sub_shader_source = _arithmetic._packed_sub_shader_source
_render_packed_sub = _arithmetic._render_packed_sub
_packed_strided_add_program = _arithmetic._packed_strided_add_program
_packed_strided_add_shader_source = _arithmetic._packed_strided_add_shader_source
_render_packed_strided_add = _arithmetic._render_packed_strided_add
_packed_scalar_div_program = _arithmetic._packed_scalar_div_program
_packed_scalar_div_shader_source = _arithmetic._packed_scalar_div_shader_source
_render_packed_scalar_div = _arithmetic._render_packed_scalar_div
_packed_broadcast_mul_program = _arithmetic._packed_broadcast_mul_program
_packed_broadcast_mul_shader_source = _arithmetic._packed_broadcast_mul_shader_source
_render_packed_broadcast_mul = _arithmetic._render_packed_broadcast_mul
_scalar_add_program = _arithmetic._scalar_add_program
_scalar_add_shader_source = _arithmetic._scalar_add_shader_source
_render_scalar_add = _arithmetic._render_scalar_add
_render_packed_add = _arithmetic._render_packed_add
_stack_program = _concat._stack_program
_stack_shader_source = _concat._stack_shader_source
_fill_program = _concat._fill_program
_fill_shader_source = _concat._fill_shader_source
_cat_program = _concat._cat_program
_cat_dim0_2d_program = _concat._cat_dim0_2d_program
_cat_dim0_2d_shader_source = _concat._cat_dim0_2d_shader_source
_cat_shader_source = _concat._cat_shader_source
_cat_lastdim_program = _concat._cat_lastdim_program
_cat_lastdim_shader_source = _concat._cat_lastdim_shader_source
_cat_dim1_3d_program = _concat._cat_dim1_3d_program
_cat_dim1_3d_shader_source = _concat._cat_dim1_3d_shader_source
_render_stack = _concat._render_stack
_render_fill_scalar = _concat._render_fill_scalar
_render_cat = _concat._render_cat
_render_cat_dim0_2d = _concat._render_cat_dim0_2d
_render_cat_lastdim_3d = _concat._render_cat_lastdim_3d
_render_cat_dim1_3d = _concat._render_cat_dim1_3d
_maxpool_program = _pooling._maxpool_program
_maxpool_shader_source = _pooling._maxpool_shader_source
_render_max_pool2d_with_indices = _pooling._render_max_pool2d_with_indices
_as_pair = _pooling._as_pair
_upsample_nearest2d_program = _resize._upsample_nearest2d_program
_upsample_nearest2d_shader_source = _resize._upsample_nearest2d_shader_source
_render_upsample_nearest2d = _resize.render_upsample_nearest2d
_softmax_program = _softmax._softmax_program
_softmax_shader_source = _softmax._softmax_shader_source
_render_softmax = _softmax._render_softmax

_is_scalar_operand = _operation_context.is_scalar_operand
_scalar_value = _operation_context.scalar_value
_render_binary = _dispatch._render_binary


def __getattr__(name):
    if name == "_runtime":
        return _runtime_module._runtime
    raise AttributeError(name)
