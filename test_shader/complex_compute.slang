[[vk::binding(0)]] RWStructuredBuffer<uint> input_data;
[[vk::binding(1)]] RWStructuredBuffer<uint> scratch;
[[vk::binding(2)]] RWStructuredBuffer<uint> output_data;

groupshared uint shared_data[64];

[numthreads(64, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID,
          uint3 group_id : SV_GroupID,
          uint3 group_thread_id : SV_GroupThreadID) {
  uint local_id = group_thread_id.x;
  uint global_id = dispatch_id.x;
  uint value = input_data[global_id];

  value = (value ^ (value << 13)) + (value >> 7);
  value = (value * 1664525u) + 1013904223u;

  if ((value & 1u) == 0u) {
    value = value ^ 0x9e3779b9u;
  } else {
    value = value + 0x7f4a7c15u;
  }

  shared_data[local_id] = value;
  GroupMemoryBarrierWithGroupSync();

  for (uint stride = 32u; stride > 0u; stride >>= 1u) {
    if (local_id < stride) {
      shared_data[local_id] += shared_data[local_id + stride];
    }
    GroupMemoryBarrierWithGroupSync();
  }

  if (local_id == 0u) {
    scratch[group_id.x] = shared_data[0];
  }
  GroupMemoryBarrierWithGroupSync();

  if (local_id < 4u) {
    uint mix = scratch[group_id.x];
    mix = (mix << (local_id + 1u)) | (mix >> (31u - local_id));
    uint out_index = group_id.x * 4u + local_id;
    output_data[out_index] = mix;
  }
}
