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


#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>


// for debug
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream> 
namespace cg = cooperative_groups;

__device__ glm::mat3 makeMat3FromMat4(glm::mat4 mat4x4){ // 取前三行三列元素
	glm::mat3 mat3x3 = glm::mat3(
		mat4x4[0][0], mat4x4[0][1], mat4x4[0][2],
		mat4x4[1][0], mat4x4[1][1], mat4x4[1][2],
		mat4x4[2][0], mat4x4[2][1], mat4x4[2][2]
	);

	return mat3x3;
}

__device__ glm::mat3 computeRotScaFromQua(glm::vec4 quaternion, glm::vec3 scale, float* normal){
	
	glm::mat3 S = glm::mat3(
		scale.x, 0.0f, 0.0f,
		0.0f, scale.y, 0.0f,
		0.0f, 0.0f, scale.z
	);

	float r = quaternion.x;
	float x = quaternion.y;
	float y = quaternion.z;
	float z = quaternion.w;

	glm::mat3 R = glm::mat3(// 每一列是一个向量
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	// glm::mat3 R = glm::mat3(// 每一行是一个向量
	// 	1.f - 2.f * (y * y + z * z), 2.f * (x * y + r * z), 2.f * (x * z - r * y),
	// 	2.f * (x * y - r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z + r * x),
	// 	2.f * (x * z + r * y), 2.f * (y * z - r * x), 1.f - 2.f * (x * x + y * y)
	// );

	glm::mat3 L = S * R;
	// glm::mat3 L = R * S;
	normal[0] = R[0][2];
	normal[0] = R[1][2];
	normal[0] = R[2][2];

	return L;
}
// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}



__device__ float computeLocalGaussian(
	const float sigma,
	const glm::mat3x3 T_t, // KWH
	const float2 center,  // 像素坐标
	const float2 point_image // 高斯投影
	// const float dist3d, // uv平面上的距离
	// float* dist // xy平面上的距离
){
	glm::vec3 k = -T_t[0] + center.x * T_t[2]; // hu
	glm::vec3 l = -T_t[1] + center.y * T_t[2]; // hv
	
	glm::vec3 point = glm::cross(k,l); 

	point /= point.z; // 该像素在uv平面上的投影点

	float dist3d = point.x * point.x + point.y * point.y;

	// float g = exp(- (u * u + v * v) / 2);
	float coff = 1 / (sqrt(2) / 2);

	glm::vec2 offset = glm::vec2(center.x - point_image.x, center.y - point_image.y);
	// float dist2d = coff * glm::length(offset);
	// dist2d *= dist2d;
	// 低通滤波

	// float dist = min(dist2d,dist3d); // 1/2/3 sigma以内保留
	float dist = dist3d;
	if (dist > sigma * sigma) return 0.0f;
	// if (dist > sigma) return 0.0f;


	return exp(-0.5 * dist);

	// float2 point_2d = {point.x,point.y};
	// return point_2d;
}





__device__ void print_matrix3x3(glm::mat3 M){
	printf("%.12f, %.12f, %.12f\n%.12f, %.12f, %.12f\n%.12f, %.12f, %.12f\n\n", 
	M[0][0],M[0][1],M[0][2],
	M[1][0],M[1][1],M[1][2],
	M[2][0],M[2][1],M[2][2]);
}


__device__ void print_matrix3x(glm::mat3x4 m){

	printf("%f, %f, %f, %f\n%f, %f, %f, %f\n%f, %f, %f, %f\n\n",
		m[0][0],m[0][1],m[0][2],m[0][3],
		m[1][0],m[1][1],m[1][2],m[1][3],
		m[2][0],m[2][1],m[2][2],m[2][3]
		
	);
}

__device__ void compute2DGSBBox(
	const float sigma_2,
	const glm::mat4 viewmatrix, // w2c
	const glm::mat3 projmatrix, // K
	const glm::vec4 quaternion,
	const glm::vec3 scale,
	const float* p, // 高斯质心
	glm::vec4* p_view, // 已经算过了
	float* normal,
	float* radii,
	float2* point_image,
	glm::mat3* T_t
){

	glm::mat3 rotation = computeRotScaFromQua(quaternion, scale, normal); // R，尽量少用tanspose
	// normal[0] = rotation[0][2];
	// normal[1] = rotation[1][2];
	// normal[2] = rotation[2][2];
	// printf("%f,%f,%f ",normal[0],normal[1],normal[2]);

	// printf("%f,%f,%f",normal[0],normal[1],normal[2]);
	
	// cout << glm::determinant(rotation) << endl;
	// glm::vec4 means3D = glm::vec4(p[0],p[1],p[2],1.0f);
	// *p_view = viewmatrix * means3D; // 相机坐标系下的高斯点
	// 计算深度用p_view

	glm::mat3 viewmatrix_R = makeMat3FromMat4(viewmatrix); 

	
	// glm::mat3 uv_view = viewmatrix_R * rotation; // 3x3，相机坐标系下，高斯椭球的朝向转置，每个行向量是高斯每根轴的朝向与尺度
	glm::mat3 uv_view = rotation * viewmatrix_R; // 3x3，相机坐标系下，高斯椭球的朝向转置，每个列向量是高斯每根轴的朝向与尺度
	// print_matrix3x3(uv_view); 
	// 计算会有精度误差，即使都是32位浮点数，rotation和viewmatrix_R精确到12位小数都一样，但uv_view依然会有0.001%～1%的误差
	// 通常由于cuda舍入和python不一致所致
	// std::printf("sucess1");

	// glm::mat4 projmatrix = glm::make_mat4(proj);
	// glm::mat3x3 M = glm::mat3x3(
	// 	uv_view[0][0], 	uv_view[0][1], 	uv_view[0][2],
	// 	uv_view[1][0], 	uv_view[1][1], 	uv_view[1][2],
	// 	p_view->x,		p_view->y,		p_view->z
	// );
	glm::mat3x3 M = glm::mat3x3(
		uv_view[0][0], 	uv_view[0][1], 	p_view->x,				
		uv_view[1][0], 	uv_view[1][1], 	p_view->y,
		uv_view[2][0],  uv_view[2][1],  p_view->z
	);

	// glm::mat3x3 T_o = (projmatrix) * M;
	*T_t = M * projmatrix;
	// float* T_o_first = &T_o[0][0];
	// for(int i = 0; i < 12; i++)
	// 	T[i] = T_o_first[i];

	// glm::mat3x3 T_t = glm::transpose(T_o);
	// glm::vec3 temp_point = glm::vec3(1.0f,1.0f,-1.0f);
	glm::vec3 temp_point = glm::vec3(sigma_2,sigma_2,-1.0f);
	

	// 见pdf
	float distance = glm::dot(temp_point,(*T_t)[2] * (*T_t)[2]);
	temp_point *= 1 / (distance + 0.00001f); 

	point_image->x = glm::dot(temp_point,(*T_t)[0] * (*T_t)[2]); // 高斯的投影点不一定是bbox的中心点
	point_image->y = glm::dot(temp_point,(*T_t)[1] * (*T_t)[2]);
	
	// *point_image = glm::vec3(
	// 	glm::dot(temp_point,(*T_t)[0] * (*T_t)[2]),
	// 	glm::dot(temp_point,(*T_t)[1] * (*T_t)[2]),
	// 	glm::dot(temp_point,(*T_t)[2] * (*T_t)[2])
	// );
	float2 radius_square = {
		point_image->x * point_image->x - glm::dot(temp_point,(*T_t)[0] * (*T_t)[0]),
		point_image->y * point_image->y - glm::dot(temp_point,(*T_t)[1] * (*T_t)[1])
	};
	// glm::vec3 radius_square = (*point_image) * (*point_image) - glm::vec3(
	// 	glm::dot(temp_point,(*T_t)[0] * (*T_t)[0]),
	// 	glm::dot(temp_point,(*T_t)[1] * (*T_t)[1]),
	// 	glm::dot(temp_point,(*T_t)[2] * (*T_t)[2])
	// );
	// cout << glm::to_string(*T) << endl;
	float2 radi = { // 两根之差/2的平方
		glm::sqrt(max(radius_square.x,0.000001f)),
		glm::sqrt(max(radius_square.y,0.000001f))
	};
	radii[0] = radi.x;
	radii[1] = radi.y;
	// cout << glm::to_string(*T) << endl;
	// int radii;
	// return radii,center;
	// glm::mat4 projmatrix = glm::make_mat4(viewmatrix);
}



// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M, // 计算2dgs的radii，并根据radii计算深度
	const float* orig_points, //means3d
	const glm::vec3* scales,
	const float scale_modifier,
	const float sigma,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	// const float* cov3D_precomp,
	const float* colors_precomp,
	const glm::mat4* viewmatrix, //cuda
	const glm::mat3* projmatrix,
	const glm::vec3* cam_pos,
	const float* cam_intr,
	const int W, int H,
	// const float focal_x, float focal_y,
	// const float tan_fovx, float tan_fovy,
	float* radii,
	float2* points_xy_image,
	float* depths,
	float* normals,
	glm::mat3* KWH, // T
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	
	radii[2 * idx] = 0.0f;
	radii[2 * idx + 1] = 0.0f;
	
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	glm::vec4 p_view;

	if (!in_frustum(idx, orig_points, *viewmatrix, *projmatrix, prefiltered, &p_view))
		return;

	// Transform point by projecting
	// float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	// float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	// float p_w = 1.0f / (p_hom.w + 0.0000001f);
	// float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
	
	// glm::vec3 radi;
	// glm::vec3 means2D;
	compute2DGSBBox(
		sigma * sigma,
		*viewmatrix,
		*projmatrix,
		rotations[idx],
		scales[idx],
		orig_points + 3 * idx,
		&p_view,
		normals + 3 * idx,
		radii + 2 * idx,
		points_xy_image + idx,
		KWH + idx
	);
	// printf("%f,%f\n",radii[2 * idx], radii[2 * idx + 1]);
	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	// float mid = 0.5f * (cov.x + cov.z);
	// float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	// float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	// float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	// float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	float my_radius = sqrt(radii[2 * idx]*radii[2 * idx] + radii[2 * idx + 1]*radii[2 * idx + 1]);
	// float my_radius = max(radii[2 * idx], radii[2 * idx + 1]);
	uint2 rect_min, rect_max;
	getRect(points_xy_image[idx], my_radius, rect_min, rect_max, grid); // point_image为高斯的投影点
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	// radii[idx] = my_radius;
	// points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = { 0.0f, 0.0f, 0.0f, opacities[idx] };
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}






// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list, // 高斯点的索引
	int W, int H,
	const float2* __restrict__ points_xy_image, //!!!
	const float* __restrict__ means3D, // 3d高斯点
	const float sigma,
	const float* __restrict__ depths,
	const float* __restrict__ normals,
	// const float* __restrict__ cam_intr, 
	// const glm::vec4* __restrict__ quaternions, // [P, 4] 需要用到四元数生成tu,tv
	// const glm::vec3* __restrict__ scales, // tu, tv轴对应的尺度
	const glm::mat3x3* __restrict__ KWH, //!!!
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	glm::vec3* __restrict__ ADD_2,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	float* __restrict__ out_depth,
	float* __restrict__ out_normal,
	float* __restrict__ out_opacity,
	float* __restrict__ distort
)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	// pix_min,pix_max分别对应了当前像素对应tile的左上角和右下角的像素坐标
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y }; // 像素坐标从0开始
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	// 当前tile下tile索引的范围（按照深度排序），e.g.[10,50]
	// 注意10,50不是means3D的索引，其索引放在了binningState.point_list中

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x; // 该tile上有多少个高斯

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ glm::mat3x3 collected_KWH[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };
	float depth = 0.0f;
	float normal[3] = {0};
	bool depth_done = false;
	float A_N_Minus_1 = 0.0f;
	float D_N_Minus_1 = 0.0f;
	float D_2_N_Minus_1 = 0.0f;
	float distort_loss = 0.0f;
	


	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			collected_KWH[block.thread_rank()] = KWH[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;
			// 传递到这里需要每一个像素对应gs的权重
			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j]; // point_image
			// float2 dist2 = { xy.x - pixf.x, xy.y - pixf.y }; // xy平面上的距离
			float4 con_o = collected_conic_opacity[j];

			// float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			float g = computeLocalGaussian(
				sigma,
				// KWH[collected_id[j]],
				collected_KWH[j],
				pixf,
				xy
			);
			if (g < 0.000001f)
				continue;

			// if (power > 0.0f)
			// 	continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * g);
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}
			
			// Eq. (3) from 3D Gaussian splatting paper.
			const float omega = alpha * T;
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * omega;

			const float m = depths[collected_id[j]];

			distort_loss += omega * (m * m * A_N_Minus_1 + D_2_N_Minus_1 - 2 * m * D_N_Minus_1);

			A_N_Minus_1 += omega;
			D_N_Minus_1 += m * omega;
			D_2_N_Minus_1 += m * m * omega;
			// printf("%f,%f,%f\n",normal[0],normal[1],normal[2]);
			
			T = test_T;
			if(T < 0.5 && !depth_done){ // 透射率=0.5深度
				depth_done = true;
				// depth = depths[collected_id[j]];

				normal[0] += normals[collected_id[j] * 3] * omega;
				normal[1] += normals[collected_id[j] * 3 + 1] * omega;
				normal[2] += normals[collected_id[j] * 3 + 2] * omega;
			}
			depth += depths[collected_id[j]] * omega; // 深度期望
			
			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++){
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
			out_normal[ch * H * W + pix_id] = normal[ch];
		}
		out_depth[pix_id] = depth;
		out_opacity[pix_id] = 1 - T;

		ADD_2[pix_id].x = A_N_Minus_1;
		ADD_2[pix_id].y = D_N_Minus_1;
		ADD_2[pix_id].z = D_2_N_Minus_1;
		distort[pix_id] = distort_loss;
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges, // 
	const uint32_t* point_list,
	int W, int H,
	const float2* points_xy_image,
	const float* means3D,
	const float sigma,
	const float* depths,
	const float* normals,
	// const float* cam_intr,
	// const glm::vec4* quaternions,
	// const glm::vec3* scales,
	const glm::mat3x3* KWH,
	const float* colors,
	const float4* conic_opacity,
	glm::vec3* ADD_2,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	float* out_depth,
	float* out_normal,
	float* out_opacity,
	float* distort)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		points_xy_image,
		means3D,
		sigma,
		depths,
		normals,
		// cam_intr,
		// quaternions,
		// scales,
		KWH,
		colors,
		conic_opacity,
		ADD_2,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		out_depth,
		out_normal,
		out_opacity,
		distort);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const float sigma,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	// const float* cov3D_precomp,
	const float* colors_precomp,
	const glm::mat4* viewmatrix,
	const glm::mat3* projmatrix,
	const glm::vec3* cam_pos,
	const float* cam_intr,
	const int W, int H,
	// const float focal_x, float focal_y,
	// const float tan_fovx, float tan_fovy,
	float* radii,
	float2* means2D,
	float* depths,
	float* normals,
	glm::mat3* KWH,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{	
	// std::cout << "success" << std::endl;
	// std::cout << viewmatrix << std::endl;
	// // print_matrix3(viewmatrix);
	// // print_matrix3(projmatrix);
	// std::cout << "success" << std::endl;
	// std::cout << glm::to_string(*viewmatrix) << std::endl;
	// std::cout << glm::to_string(*projmatrix) << std::endl;
	// std::cout << viewmatrix
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		sigma,
		rotations,
		opacities,
		shs,
		clamped,
		// cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		cam_intr,
		W, H,
		// focal_x, focal_y,
		// tan_fovx, tan_fovy,
		radii,
		means2D,
		depths,
		normals,
		KWH,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered
		);
}