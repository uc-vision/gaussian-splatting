
import taichi as ti
from taichi.math import mat3, vec4

@ti.func
def quat_to_mat(q:vec4) -> mat3:
  w, x, y, z = q
  x2, y2, z2 = x*x, y*y, z*z

  return ti.math.mat3(
    1 - 2*y2 - 2*z2, 2*x*y - 2*w*z, 2*x*z + 2*w*y,
    2*x*y + 2*w*z, 1 - 2*x2 - 2*z2, 2*y*z - 2*w*x,
    2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x2 - 2*y2
  )

@ti.kernel
def quat_to_mat_kernel(q:ti.types.ndarray(4, dtype=ti.f32), m:ti.types.ndarray(9, dtype=ti.f32)):
  for i in range(0, q.shape[0]):
    m[i] = quat_to_mat(ti.math.normalize(q[i]))

