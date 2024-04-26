/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}

__device__ void print_matrix4x4(glm::mat4 m){

	printf("%f, %f, %f, %f\n%f, %f, %f, %f\n%f, %f, %f, %f\n",
		m[0][0],m[0][1],m[0][2],m[0][3],
		m[1][0],m[1][1],m[1][2],m[1][3],
		m[2][0],m[2][1],m[2][2],m[2][3],
		m[3][0],m[3][1],m[3][2],m[3][3]
		
	);
}
__device__ void print_matrix3x3(glm::mat3x4 m){

	printf("%f, %f, %f\n%f, %f, %f\n%f, %f, %f\n\n",
		m[0][0],m[0][1],m[0][2],
		m[1][0],m[1][1],m[1][2],
		m[2][0],m[2][1],m[2][2]
		
	);
}
__device__ glm::mat3 makeMat3FromMat4x4(glm::mat4 mat4x4){ // 取前三行三列元素
	glm::mat3 mat3x3 = glm::mat3(
		mat4x4[0][0], mat4x4[0][1], mat4x4[0][2],
		mat4x4[1][0], mat4x4[1][1], mat4x4[1][2],
		mat4x4[2][0], mat4x4[2][1], mat4x4[2][2]
	);

	return mat3x3;
}
// __device__ glm::mat3 makeMat3FromMat4(glm::mat4 mat4x4){ // 取前三行三列元素
// 	glm::mat3 mat3x3 = glm::mat3(
// 		mat4x4[0][0], mat4x4[0][1], mat4x4[0][2],
// 		mat4x4[1][0], mat4x4[1][1], mat4x4[1][2],
// 		mat4x4[2][0], mat4x4[2][1], mat4x4[2][2]
// 	);

// 	return mat3x3;
// }


__device__ void compute2DGSBBox(
	const glm::mat4 W2C, // w2c
	const glm::mat3 projmatrix, // intrinsic
	const glm::vec4 quaternion,
	const glm::vec3 scale,
	const float3 p,
	const glm::mat3x3 dL_dKWH,
	glm::vec3* dL_dmean2D,
	glm::vec3* dL_dmeans,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot
){
	// glm::mat4x3 dL_dKWH = glm::transpose(dL_dKWH);
	float fx = projmatrix[0][0];
	float fy = projmatrix[1][1];
	float cx = projmatrix[0][2];
	float cy = projmatrix[1][2];
	// print_matrix3x3(dL_dKWH);
	// print_matrix3x3(projmatrix);
	// printf("%f,%f,%f,%f\n\n",fx,fy,cx,cy);
	// printf("dL_dscale: %f, %f, %f\n", dL_dscale[idx].x,dL_dscale[idx].y,dL_dscale[idx].z);
	glm::mat3 dL_dM(0.0f); 
	// T = K @ M 
	// dL_dM计算正确
	dL_dM[0][0] = fx * dL_dKWH[0][0];
	dL_dM[0][1] = fx * dL_dKWH[0][1];
	dL_dM[0][2] = fx * dL_dKWH[0][2];

	dL_dM[1][0] = fy * dL_dKWH[1][0];
	dL_dM[1][1] = fy * dL_dKWH[1][1];
	dL_dM[1][2] = fy * dL_dKWH[1][2];
	
	dL_dM[2][0] = cx * dL_dKWH[0][0] + cy * dL_dKWH[1][0] + dL_dKWH[2][0];
	dL_dM[2][1] = cx * dL_dKWH[0][1] + cy * dL_dKWH[1][1] + dL_dKWH[2][1];
	dL_dM[2][2] = cx * dL_dKWH[0][2] + cy * dL_dKWH[1][2] + dL_dKWH[2][2];
	// printf("%f,%f",cx,cy);
	// print_matrix3x3(dL_dKWH);// dL_dKWH很准
	// print_matrix3x3(dL_dM);
	// M[2,0:3] = p

	// pv再经过W的变换
	glm::mat3 viewmatrix_R = makeMat3FromMat4x4(W2C); 
	dL_dmeans->x = dL_dM[0][2];
	dL_dmeans->y = dL_dM[1][2];
	dL_dmeans->z = dL_dM[2][2];
	// 此处的三维点梯度dL_dmeans是相机坐标系下的三维点梯度
	// 将质心的梯度投影到像素平面上
	*dL_dmean2D = *dL_dmeans;

	// 注意这里梯度不是齐次向量
	glm::vec4 p_view = glm::vec4(p.x, p.y, p.z, 1.0f); 
	p_view = p_view * W2C; //相机坐标系
	if (dL_dmean2D->z >= 0){
		
		dL_dmean2D->x = dL_dmean2D->x / p_view.z;
		dL_dmean2D->y = dL_dmean2D->y / p_view.z;
		// dL_dmean2D->x = dL_dmean2D->x;
		// dL_dmean2D->y = dL_dmean2D->y;
		
	}
	else{
		dL_dmean2D->x = dL_dmean2D->x / p_view.z;
		dL_dmean2D->y = dL_dmean2D->y / p_view.z;
		// dL_dmean2D->x = dL_dmean2D->x;
		// dL_dmean2D->y = dL_dmean2D->y;

	}

	//世界坐标系三维点梯度计算正确
	*dL_dmeans = viewmatrix_R * (*dL_dmeans);

	// printf("%f, %f, %f\n",dL_dmeans->x,dL_dmeans->y,dL_dmeans->z);
	// Y = uv_view = H @ W
	// dL_dH计算正确
	glm::mat3 dL_dH(0.0f);
	dL_dH[0][0] = dL_dM[0][0]*W2C[0][0] + dL_dM[1][0]*W2C[1][0] + dL_dM[2][0]*W2C[2][0];
	dL_dH[1][0] = dL_dM[0][0]*W2C[0][1] + dL_dM[1][0]*W2C[1][1] + dL_dM[2][0]*W2C[2][1];
	dL_dH[2][0] = dL_dM[0][0]*W2C[0][2] + dL_dM[1][0]*W2C[1][2] + dL_dM[2][0]*W2C[2][2];

	dL_dH[0][1] = dL_dM[0][1]*W2C[0][0] + dL_dM[1][1]*W2C[1][0] + dL_dM[2][1]*W2C[2][0];
	dL_dH[1][1] = dL_dM[0][1]*W2C[0][1] + dL_dM[1][1]*W2C[1][1] + dL_dM[2][1]*W2C[2][1];
	dL_dH[2][1] = dL_dM[0][1]*W2C[0][2] + dL_dM[1][1]*W2C[1][2] + dL_dM[2][1]*W2C[2][2];
	// print_matrix3x3(dL_dH);
	

	// H = (R @ S)^T
	float r = quaternion.x;
	float x = quaternion.y;
	float y = quaternion.z;
	float z = quaternion.w;
	
	glm::mat3 R = glm::mat3(// 每一列是一个向量
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);
	


	
	


	dL_dscale->x = dL_dH[0][0]*R[0][0] + dL_dH[1][0]*R[1][0] + dL_dH[2][0]*R[2][0];
	dL_dscale->y = dL_dH[0][1]*R[0][1] + dL_dH[1][1]*R[1][1] + dL_dH[2][1]*R[2][1];
	dL_dscale->z = 0.0f;
	// q = r, x, y, z
	dL_dH[0][0] *= scale.x;
	dL_dH[1][0] *= scale.x;
	dL_dH[2][0] *= scale.x;

	dL_dH[0][1] *= scale.y;
	dL_dH[1][1] *= scale.y;
	dL_dH[2][1] *= scale.y;

	dL_dH[0][2] *= scale.z;
	dL_dH[1][2] *= scale.z;
	dL_dH[2][2] *= scale.z;
	// 此处变为dL_dR


	dL_drot->x = 2*x*dL_dH[2][1] - 2*y*dL_dH[2][0] + 2*z*(dL_dH[1][0]-dL_dH[0][1]);
	dL_drot->y = 2*r*dL_dH[2][1] - 4*x*dL_dH[1][1] + 2*y*(dL_dH[1][0]+dL_dH[0][1]) + 2*z*dL_dH[2][0];
	dL_drot->z = -2*r*dL_dH[2][0] + 2*x*(dL_dH[1][0]+dL_dH[0][1]) - 4*y*dL_dH[0][0] + 2*z*dL_dH[2][1];
	dL_drot->w = 2*r*(dL_dH[1][0] - dL_dH[0][1]) + 2*x*dL_dH[2][0] + 2*y*dL_dH[2][1] - 4*z*(dL_dH[0][0]+dL_dH[1][1]);
	// printf("%f, %f, %f, %f\n",dL_drot->x,dL_drot->y,dL_drot->z,dL_drot->w);

}

// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const float3* means,
	const float* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const glm::mat4* viewmatrix, // w2c.T
	const glm::mat3* projmatrix, // intrinsic.T
	const glm::vec3* campos,
	const glm::mat3x3* dL_dKWH,
	glm::vec3* dL_dmean2D,
	glm::vec3* dL_dmeans, // 对质心的导数
	float* dL_dcolor,
	// float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{   
	auto idx = cg::this_grid().thread_rank();
	
	float my_radius = sqrt(radii[2 * idx]*radii[2 * idx] + radii[2 * idx + 1]*radii[2 * idx + 1]);
	if (idx >= P || !(my_radius > 0.00001f))
		return;
	// print_matrix3x3(dL_dKWH[idx]);

	// float3 m = means[idx]; // 三维点坐标
	// 先不考虑投影点的梯度(因为没有滤波)
	// 则这里的梯度全部由dL_dKWH提供
	// 后续如果要加的话需要加一条KWH -> point_image_xy -> g的路径
	// 这条路径可以在preprocess中实现（因为路径通过dL_dmean2D）
	compute2DGSBBox(
		*viewmatrix,
		*projmatrix,
		rotations[idx],
		scales[idx],
		means[idx],
		dL_dKWH[idx],
		dL_dmean2D + idx,
		dL_dmeans + idx,
		dL_dscale + idx,
		dL_drot + idx
	);
	
	// Compute gradient updates due to computing colors from SHs
	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3*)means, *campos, shs, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dmeans, (glm::vec3*)dL_dsh);

}

// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const glm::mat3x3* __restrict__ KWH, //!!!
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ colors,
	const float* __restrict__ final_Ts, // \Pi (1-alpha*g) (j=1,...,N)
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels, //dl/dc
	glm::mat3x3* __restrict__ dL_dKWH,
	glm::vec3* __restrict__ dL_dmean2D,
	// float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };
	const float coff = 1 / (sqrt(2) / 2);


	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE]; // 一个tile上一个round中256个高斯的颜色

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };
	float dL_dpixel[C];
	if (inside)
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];

	float last_alpha = 0;
	float last_color[C] = { 0 };

	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W; 
	// 前向过程中，通过J直接把相机坐标投影到像素坐标，这里需要计算相对归一化坐标的梯度，从而计算对投影点的梯度，从而计算对高斯位置的梯度
	// 注意J只是将分布变换到了像素坐标系，投影过程是先投影到归一化坐标系，再变换到像素坐标系
	const float ddely_dy = 0.5 * H;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];// 深度排序的point中从后往前算
			collected_id[block.thread_rank()] = coll_id; // 高斯绝对索引
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];// 列主序到行主序
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			// 如果大于等于，说明提前截止了，此时T=\Pi (1-ag) i=1,...,j-1
			// 如果一开始就小于，说明所有的高斯都参与了渲染，下面的T需要先除以一个(1-alpha)
			// 如果循环中小于，说明当前point_list中一个round中，j+1索引前面的所有高斯都参与了渲染，而第j+1以及后面的没有参与渲染，此时T=\Pi (1-ag) i=1,...,j
			// 同样需要除以一个(1-alpha)
			if (contributor >= last_contributor) 
				continue;

			// Compute blending values, as before.
			// const float2 xy = collected_xy[j]; // 高斯投影点，如果要滤波会用到

			// const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float4 con_o = collected_conic_opacity[j];
			// const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			// if (power > 0.0f)
			// 	continue;
			// const float G = exp(power);
			
			const glm::mat4x3 T_t = KWH[collected_id[j]];//
			const glm::vec3 k = -T_t[0] + pixf.x * T_t[2]; // hu
			const glm::vec3 l = -T_t[1] + pixf.y * T_t[2]; // hv
			const glm::vec3 point = glm::cross(k,l); 
			const float dist3d = (point.x * point.x + point.y * point.y) / (point.z * point.z);
			if (dist3d > 1.0f)
				continue;
			const float G = exp(-0.5 * dist3d);

			const float alpha = min(0.99f, con_o.w * G); // 这里的alpha指ag
			if (alpha < 1.0f / 255.0f)
				continue;

			T = T / (1.f - alpha); // T_j
			const float dchannel_dcolor = alpha * T;  // dc_hat/dc 对每个高斯颜色的梯度

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dalpha = 0.0f;
			const int global_id = collected_id[j];
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j]; // 高斯的颜色c_i
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch]; // j=0时，accum=0
				last_color[ch] = c;

				const float dL_dchannel = dL_dpixel[ch]; // dl/dc_hat，所有高斯的该值都一样
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				// Update the gradients w.r.t. color of the Gaussian. 
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
			}
			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel; // 有背景则有一个固定值

			// T, G, dL_dG, 计算正确
			// Helpful reusable temporary variables
			const float dL_dG = con_o.w * dL_dalpha; // dl/dg_i
			// 需要计算k,l,dd_dp,dL_dd
			// const float dL_dd = -G / 2;
			const float dL_dd = -G / 2 * dL_dG;
			const float p3_p3 = point.z * point.z;
			const glm::vec3 dd_dp = {
				2 * point.x / p3_p3, 
				2 * point.y / p3_p3,
				-2 * (point.x * point.x + point.y * point.y) / (p3_p3 * point.z)	
			};
			float x = pixf.x;
			float y = pixf.y;
			// printf("%f\n",dL_dd);
			// if(point.x/point.z > 4.0f || point.y/point.z > 4.0f)
				// printf("%f, %f\n", pixf.x, pixf.y);
			atomicAdd(&dL_dKWH[global_id][0][0], dL_dd * ( dd_dp.y * l.z - dd_dp.z * l.y));
			atomicAdd(&dL_dKWH[global_id][0][1], dL_dd * (-dd_dp.x * l.z + dd_dp.z * l.x));
			atomicAdd(&dL_dKWH[global_id][0][2], dL_dd * ( dd_dp.x * l.y - dd_dp.y * l.x));

			atomicAdd(&dL_dKWH[global_id][1][0], dL_dd * (-dd_dp.y * k.z + dd_dp.z * k.y));
			atomicAdd(&dL_dKWH[global_id][1][1], dL_dd * ( dd_dp.x * k.z - dd_dp.z * k.x));
			atomicAdd(&dL_dKWH[global_id][1][2], dL_dd * (-dd_dp.x * k.y + dd_dp.y * k.x));

			atomicAdd(&dL_dKWH[global_id][2][0], -dL_dd * (dd_dp.y * ( x * l.z - y * k.z) + dd_dp.z * (-x * l.y + y * k.y)));
			atomicAdd(&dL_dKWH[global_id][2][1], -dL_dd * (dd_dp.x * (-x * l.z + y * k.z) + dd_dp.z * ( x * l.x - y * k.x)));
			atomicAdd(&dL_dKWH[global_id][2][2], -dL_dd * (dd_dp.x * ( x * l.y - y * k.y) + dd_dp.y * (-x * l.x + y * k.x)));


			// const float gdx = G * d.x;
			// const float gdy = G * d.y;
			// const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			// const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

			// // Update gradients w.r.t. 2D mean position of the Gaussian
			// // ddelx_dx = 0.5 * W
			// atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx); 
			// atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);

			// // Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
			// atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG); //a
			// atomicAdd(&dL_dconic2D[global_id].y, -0.5f * gdx * d.y * dL_dG); //b
			// atomicAdd(&dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG); //c

			// Update gradients w.r.t. opacity of the Gaussian
			atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);

		}
	}
}

void BACKWARD::preprocess(
	int P, int D, int M,
	const float3* means3D,
	const float* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	// const float* cov3Ds,
	const glm::mat4* viewmatrix,
	const glm::mat3* projmatrix,
	// const float focal_x, float focal_y,
	// const float tan_fovx, float tan_fovy,
	const glm::vec3* campos,
	const glm::mat3x3* dL_dKWH,
	glm::vec3* dL_dmean2D,
	// const float* dL_dconic,
	glm::vec3* dL_dmean3D,
	float* dL_dcolor,
	// float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{

	// computeLocalGaussianCUDA << <(P + 255) / 256, 256>> >(

	// );


	// Propagate gradients for the path of 2D conic matrix computation. 
	// Somewhat long, thus it is its own kernel rather than being part of 
	// "preprocess". When done, loss gradient w.r.t. 3D means has been
	// modified and gradient w.r.t. 3D covariance matrix has been computed.	
	// computeCov2DCUDA << <(P + 255) / 256, 256 >> > (
	// 	P,
	// 	means3D,
	// 	radii,
	// 	// cov3Ds,
	// 	focal_x,
	// 	focal_y,
	// 	tan_fovx,
	// 	tan_fovy,
	// 	viewmatrix,
	// 	dL_dconic,
	// 	(float3*)dL_dmean3D,
	// 	dL_dcov3D);

	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desireD), propagate 3D covariance
	// matrix gradients to scale and rotation.
	preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P, D, M,
		(float3*)means3D,
		radii,
		shs,
		clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		viewmatrix,
		projmatrix,
		campos,
		dL_dKWH,
		dL_dmean2D,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		// dL_dcov3D,
		dL_dsh,
		dL_dscale,
		dL_drot);
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float* bg_color,
	const float2* means2D,
	const glm::mat3x3* KWH,
	const float4* conic_opacity,
	const float* colors,
	const float* final_Ts, // accum_alpha
	const uint32_t* n_contrib,
	const float* dL_dpixels,
	glm::mat3x3* dL_dKWH,
	glm::vec3* dL_dmean2D,
	// float4* dL_dconic2D,
	float* dL_dopacity,
	float* dL_dcolors)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		bg_color,
		means2D,
		KWH,
		conic_opacity,
		colors,
		final_Ts,
		n_contrib,
		dL_dpixels,
		dL_dKWH, // 较难确认
		dL_dmean2D,
		// dL_dconic2D,
		dL_dopacity, // 正确
		dL_dcolors  // 正确
		);
	
}