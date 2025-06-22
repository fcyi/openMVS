/*
* SceneTexture.cpp
*
* Copyright (c) 2014-2015 SEACAVE
*
* Author(s):
*
*      cDc <cdc.seacave@gmail.com>
*
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU Affero General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU Affero General Public License for more details.
*
* You should have received a copy of the GNU Affero General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>.
*
*
* Additional Terms:
*
*      You are required to preserve legal notices and author attributions in
*      that material or in the Appropriate Legal Notices displayed by works
*      containing it.
*/

#include "Common.h"
#include "Scene.h"
#include "RectsBinPack.h"
// connected components
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>

using namespace MVS;


// D E F I N E S ///////////////////////////////////////////////////

// uncomment to enable multi-threading based on OpenMP
#ifdef _USE_OPENMP
#define TEXOPT_USE_OPENMP
#endif

// uncomment to use SparseLU for solving the linear systems
// (should be faster, but not working on old Eigen)
#if !defined(EIGEN_DEFAULT_TO_ROW_MAJOR) || EIGEN_WORLD_VERSION>3 || (EIGEN_WORLD_VERSION==3 && EIGEN_MAJOR_VERSION>2)
#define TEXOPT_SOLVER_SPARSELU
#endif

// method used to try to detect outlier face views
// (should enable more consistent textures, but it is not working)
#define TEXOPT_FACEOUTLIER_NA 0
#define TEXOPT_FACEOUTLIER_MEDIAN 1
#define TEXOPT_FACEOUTLIER_GAUSS_DAMPING 2
#define TEXOPT_FACEOUTLIER_GAUSS_CLAMPING 3
#define TEXOPT_FACEOUTLIER TEXOPT_FACEOUTLIER_GAUSS_CLAMPING

// method used to find optimal view per face
#define TEXOPT_INFERENCE_LBP 1
#define TEXOPT_INFERENCE_TRWS 2
#define TEXOPT_INFERENCE TEXOPT_INFERENCE_LBP

// inference algorithm
#if TEXOPT_INFERENCE == TEXOPT_INFERENCE_LBP
#include "../Math/LBP.h"
namespace MVS {
typedef LBPInference::NodeID NodeID;
// Potts model as smoothness function
// 设置平滑cost,如果两个节点标签相同则cost=0,否则为MaxEnergy。其中若标签值为0，则表示空标签
// 目的是让相邻face的标签尽可能一致
LBPInference::EnergyType STCALL SmoothnessPotts(LBPInference::NodeID, LBPInference::NodeID, LBPInference::LabelID l1, LBPInference::LabelID l2) {
	return l1 == l2 && l1 != 0 && l2 != 0 ? LBPInference::EnergyType(0) : LBPInference::EnergyType(LBPInference::MaxEnergy);
}
}
#endif
#if TEXOPT_INFERENCE == TEXOPT_INFERENCE_TRWS
#include "../Math/TRWS/MRFEnergy.h"
namespace MVS {
// TRWS MRF energy using Potts model
typedef unsigned NodeID;
typedef unsigned LabelID;
typedef TypePotts::REAL EnergyType;
static const EnergyType MaxEnergy(1);
struct TRWSInference {
	typedef MRFEnergy<TypePotts> MRFEnergyType;
	typedef MRFEnergy<TypePotts>::Options MRFOptions;

	CAutoPtr<MRFEnergyType> mrf;
	CAutoPtrArr<MRFEnergyType::NodeId> nodes;

	inline TRWSInference() {}
	void Init(NodeID nNodes, LabelID nLabels) {
		mrf = new MRFEnergyType(TypePotts::GlobalSize(nLabels));
		nodes = new MRFEnergyType::NodeId[nNodes];
	}
	inline bool IsEmpty() const {
		return mrf == NULL;
	}
	inline void AddNode(NodeID n, const EnergyType* D) {
		nodes[n] = mrf->AddNode(TypePotts::LocalSize(), TypePotts::NodeData(D));
	}
	inline void AddEdge(NodeID n1, NodeID n2) {
		mrf->AddEdge(nodes[n1], nodes[n2], TypePotts::EdgeData(MaxEnergy));
	}
	EnergyType Optimize() {
		MRFOptions options;
		options.m_eps = 0.005;
		options.m_iterMax = 1000;
		#if 1
		EnergyType lowerBound, energy;
		mrf->Minimize_TRW_S(options, lowerBound, energy);
		#else
		EnergyType energy;
		mrf->Minimize_BP(options, energy);
		#endif
		return energy;
	}
	inline LabelID GetLabel(NodeID n) const {
		return mrf->GetSolution(nodes[n]);
	}
};
}
#endif


// S T R U C T S ///////////////////////////////////////////////////

typedef Mesh::Vertex Vertex;
typedef Mesh::VIndex VIndex;
typedef Mesh::Face Face;
typedef Mesh::FIndex FIndex;
typedef Mesh::TexCoord TexCoord;
typedef Mesh::TexIndex TexIndex;

typedef int MatIdx;
typedef Eigen::Triplet<float,MatIdx> MatEntry;
typedef Eigen::SparseMatrix<float,Eigen::ColMajor,MatIdx> SparseMat;

enum Mask {
	empty = 0,
	border = 128,
	interior = 255
};

struct MeshTexture {
	// used to render the surface to a view camera
	typedef TImage<cuint32_t> FaceMap;
	struct RasterMesh : TRasterMesh<RasterMesh> {
		typedef TRasterMesh<RasterMesh> Base;
		FaceMap& faceMap;
		FIndex idxFace;
		Image8U mask;
		bool validFace;

		RasterMesh(const Mesh::VertexArr& _vertices, const Camera& _camera, DepthMap& _depthMap, FaceMap& _faceMap)
			: Base(_vertices, _camera, _depthMap), faceMap(_faceMap) {}
		void Clear() {
			Base::Clear();
			faceMap.memset((uint8_t)NO_ID);
		}
		void Raster(const ImageRef& pt, const Point3f& bary) {
			const Point3f pbary(PerspectiveCorrectBarycentricCoordinates(bary));
			const Depth z(ComputeDepth(pbary));
			ASSERT(z > Depth(0));
			Depth& depth = depthMap(pt);
			if (depth == 0 || depth > z) {
				depth = z;
				faceMap(pt) = validFace && (validFace = (mask(pt) != 0)) ? idxFace : NO_ID;
			}
		}
	};

	// used to represent a pixel color
	typedef Point3f Color;
	typedef CLISTDEF0(Color) Colors;

	// used to store info about a face (view, quality)
	struct FaceData {
		IIndex idxView;// the view seeing this face
		float quality; // how well the face is seen by this view
		#if TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA
		Color color; // additionally store mean color (used to remove outliers)
		#endif
	};
	typedef cList<FaceData,const FaceData&,0,8,uint32_t> FaceDataArr; // store information about one face seen from several views
	typedef cList<FaceDataArr,const FaceDataArr&,2,1024,FIndex> FaceDataViewArr; // store data for all the faces of the mesh

	typedef cList<Mesh::FaceIdxArr, const Mesh::FaceIdxArr&,2,1024, FIndex> VirtualFaceIdxsArr; // store face indices for each virtual face

	// used to assign a view to a face
	typedef uint32_t Label;
	typedef cList<Label,Label,0,1024,FIndex> LabelArr;

	// represents a texture patch
	struct TexturePatch {
		Label label; // view index
		Mesh::FaceIdxArr faces; // indices of the faces contained by the patch
		RectsBinPack::Rect rect; // the bounding box in the view containing the patch
	};
	typedef cList<TexturePatch,const TexturePatch&,1,1024,FIndex> TexturePatchArr;

	// used to optimize texture patches
	// 用来优化纹理patch。其中存储接缝处顶点的所有相关信息
	struct SeamVertex {
		struct Patch {
			struct Edge {
				uint32_t idxSeamVertex; // 这个边的另一个顶点， the other vertex of this edge
				FIndex idxFace; // 在这个patch中包含该边的face id， the face containing this edge in this patch

				inline Edge() {}
				inline Edge(uint32_t _idxSeamVertex) : idxSeamVertex(_idxSeamVertex) {}
				inline bool operator == (uint32_t _idxSeamVertex) const {
					return (idxSeamVertex == _idxSeamVertex);
				}
			};
			typedef cList<Edge,const Edge&,0,4,uint32_t> Edges;

			uint32_t idxPatch; // 包含该顶点的patch的id， the patch containing this vertex
			Point2f proj; // 该顶点在这个patch的投影坐标， the projection of this vertex in this patch
			Edges edges; // 一般情况下，在流形的mesh中，每个patch中以某个点为起始的边会有两个， the edges starting from this vertex, contained in this patch (exactly two for manifold meshes)

			inline Patch() {}
			inline Patch(uint32_t _idxPatch) : idxPatch(_idxPatch) {}
			inline bool operator == (uint32_t _idxPatch) const {
				return (idxPatch == _idxPatch);
			}
		};
		typedef cList<Patch,const Patch&,1,4,uint32_t> Patches;  // 因为一个顶点可能对应多个patch，所以采用容器结构对与顶点对应的patch进行存储

		VIndex idxVertex; // 顶点的索引， the index of this vertex
		Patches patches; // 包含该顶点的所有patch， the patches meeting at this vertex (two or more)

		inline SeamVertex() {}
		inline SeamVertex(uint32_t _idxVertex) : idxVertex(_idxVertex) {}
		inline bool operator == (uint32_t _idxVertex) const {
			return (idxVertex == _idxVertex);
		}
		// 根据id取patch，若不存在于patches中，就往patches中添加一个新的与给定id相对应的patch
		Patch& GetPatch(uint32_t idxPatch) {
			const uint32_t idx(patches.Find(idxPatch));
			if (idx == NO_ID)
				return patches.emplace_back(idxPatch);
			return patches[idx];
		}
		// 根据patch id 进行排序
		inline void SortByPatchIndex(IndexArr& indices) const {
			indices.resize(patches.size());
			std::iota(indices.Begin(), indices.End(), 0);
			std::sort(indices.Begin(), indices.End(), [&](IndexArr::Type i0, IndexArr::Type i1) -> bool {
				return patches[i0].idxPatch < patches[i1].idxPatch;
			});
		}
	};
	typedef cList<SeamVertex,const SeamVertex&,1,256,uint32_t> SeamVertices;

	// used to iterate vertex labels
	struct PatchIndex {
		bool bIndex; // 记录顶点是否在边界，若在边界则为true，否则为false
		union {
			uint32_t idxPatch; // 顶点所在patch id
			uint32_t idxSeamVertex; // 顶点若在边界上则对应seamvertex（因为这个变量里面存放所有在边界上的顶点以及其他的边界相关信息）这个list的id是多少
		};
	};
	typedef CLISTDEF0(PatchIndex) PatchIndices;
	struct VertexPatchIterator {
		uint32_t idx;  // 记录的是当前处理的第几个patch
		uint32_t idxPatch;  // 当前的patch id
		const SeamVertex::Patches* pPatches;
		inline VertexPatchIterator(const PatchIndex& patchIndex, const SeamVertices& seamVertices) : idx(NO_ID) {
			// 若patch对应的顶点在边界上，则将其所对应的多个patch信息存放到pPatches中
			// 否则说明顶点仅对应一个patch，此时仅需要记录当前节点所对应的patch的id即可
			if (patchIndex.bIndex) {
				pPatches = &seamVertices[patchIndex.idxSeamVertex].patches;
			} else {
				idxPatch = patchIndex.idxPatch;
				pPatches = NULL;
			}
		}
		inline operator uint32_t () const {
			return idxPatch;
		}
		inline bool Next() {
			if (pPatches == NULL)
				return (idx++ == NO_ID);  // idx一开始为0，与NO_ID不等，此时相当于返回true，表示idxPatch中存储的patch的索引是有效的可以进行处理
			if (++idx >= pPatches->size())  // 表示节点对应的所有patch都已经过处理，所以idxPatch中存储的patch的索引不需要再进行考虑
				return false;
			idxPatch = (*pPatches)[idx].idxPatch;
			return true;
		}
	};

	// used to sample seam edges
	typedef TAccumulator<Color> AccumColor;
	typedef Sampler::Linear<float> Sampler;
	struct SampleImage {
		AccumColor accumColor;
		const Image8U3& image;
		const Sampler sampler;

		inline SampleImage(const Image8U3& _image) : image(_image), sampler() {}
		// sample the edge with linear weights
		void AddEdge(const TexCoord& p0, const TexCoord& p1) {
			const TexCoord p01(p1 - p0);
			const float length(norm(p01));
			ASSERT(length > 0.f);
			const int nSamples(ROUND2INT(MAXF(length, 1.f) * 2.f)-1);
			AccumColor edgeAccumColor;
			for (int s=0; s<nSamples; ++s) {
				const float len(static_cast<float>(s) / nSamples);
				const TexCoord samplePos(p0 + p01 * len);
				const Color color(image.sample<Sampler,Color>(sampler, samplePos));
				edgeAccumColor.Add(RGB2YCBCR(color), 1.f-len);
			}
			accumColor.Add(edgeAccumColor.Normalized(), length);
		}
		// returns accumulated color
		Color GetColor() const {
			return accumColor.Normalized();
		}
	};

	// used to interpolate adjustments color over the whole texture patch
	typedef TImage<Color> ColorMap;


public:
	MeshTexture(Scene& _scene, unsigned _nResolutionLevel=0, unsigned _nMinResolution=640);
	~MeshTexture();

	void ListVertexFaces();

	bool ListCameraFaces(FaceDataViewArr&, float fOutlierThreshold, int nIgnoreMaskLabel, const IIndexArr& views);

	#if TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA
	bool FaceOutlierDetection(FaceDataArr& faceDatas, float fOutlierThreshold) const;
	#endif
	
	void CreateVirtualFaces(const FaceDataViewArr& facesDatas, FaceDataViewArr& virtualFacesDatas, VirtualFaceIdxsArr& virtualFaces, unsigned minCommonCameras=2, float thMaxNormalDeviation=25.f) const;
	IIndexArr SelectBestView(const FaceDataArr& faceDatas, FIndex fid, unsigned minCommonCameras, float ratioAngleToQuality) const;

	bool FaceViewSelection(unsigned minCommonCameras, float fOutlierThreshold, float fRatioDataSmoothness, int nIgnoreMaskLabel, const IIndexArr& views);
	
	void CreateSeamVertices();
	void GlobalSeamLeveling();
	void LocalSeamLeveling();
	void GenerateTexture(bool bGlobalSeamLeveling, bool bLocalSeamLeveling, unsigned nTextureSizeMultiple, unsigned nRectPackingHeuristic, Pixel8U colEmpty, float fSharpnessWeight, int maxTextureSize);

	template <typename PIXEL>
	static inline PIXEL RGB2YCBCR(const PIXEL& v) {
		typedef typename PIXEL::Type T;
		return PIXEL(
			v[0] * T(0.299) + v[1] * T(0.587) + v[2] * T(0.114),
			v[0] * T(-0.168736) + v[1] * T(-0.331264) + v[2] * T(0.5) + T(128),
			v[0] * T(0.5) + v[1] * T(-0.418688) + v[2] * T(-0.081312) + T(128)
		);
	}
	template <typename PIXEL>
	static inline PIXEL YCBCR2RGB(const PIXEL& v) {
		typedef typename PIXEL::Type T;
		const T v1(v[1] - T(128));
		const T v2(v[2] - T(128));
		return PIXEL(
			v[0]/* * T(1) + v1 * T(0)*/ + v2 * T(1.402),
			v[0]/* * T(1)*/ + v1 * T(-0.34414) + v2 * T(-0.71414),
			v[0]/* * T(1)*/ + v1 * T(1.772)/* + v2 * T(0)*/
		);
	}


protected:
	static void ProcessMask(Image8U& mask, int stripWidth);
	static void PoissonBlending(const Image32F3& src, Image32F3& dst, const Image8U& mask, float bias=1.f);


public:
	const unsigned nResolutionLevel; // 多少倍来下采样图像用来贴图， how many times to scale down the images before mesh optimization
	const unsigned nMinResolution; // 用来贴图的图像的最小分辨率阈值，how many times to scale down the images before mesh optimization

	// store found texture patches
	TexturePatchArr texturePatches;

	// used to compute the seam leveling
	PairIdxArr seamEdges; // 不同纹理patch的相交边，由两个邻接面表示（两个面共享一条边）id， the (face-face) edges connecting different texture patches
	Mesh::FaceIdxArr components; // 存储每个face对应的纹理patch id; for each face, stores the texture patch index to which belongs
	IndexArr mapIdxPatch; // 无效纹理块被移除后，新的id与旧的映射， remap texture patch indices after invalid patches removal
	SeamVertices seamVertices; // 存储不同patch间的邻接edge以及边界处的顶点。 array of vertices on the border between two or more patches

	// valid the entire time
	Mesh::VertexFacesArr& vertexFaces; // 每个顶点包含的所有faces。 for each vertex, the list of faces containing it
	BoolArr& vertexBoundary; // 记录每个顶点是否是边界点。 for each vertex, stores if it is at the boundary or not
	Mesh::FaceFacesArr& faceFaces; // for each face, the list of adjacent faces, NO_ID for border edges (optional)
	Mesh::TexCoordArr& faceTexcoords; // 存储每个face的三个顶点的纹理坐标。 for each face, the texture-coordinates of the vertices
	Mesh::TexIndexArr& faceTexindices; // for each face, the texture-coordinates of the vertices
	Mesh::Image8U3Arr& texturesDiffuse; // 纹理图， texture containing the diffuse color

	// constant the entire time
	Mesh::VertexArr& vertices;
	Mesh::FaceArr& faces;
	ImageArr& images;

	Scene& scene; // the mesh vertices and faces
};

// creating an invalid mask for the given image corresponding to
// the invalid pixels generated during image correction for the lens distortion;
// the returned mask has the same size as the image and is set to zero for invalid pixels
static Image8U DetectInvalidImageRegions(const Image8U3& image)
{
	const cv::Scalar upDiff(3);
	const int flags(8 | (255 << 8));
	Image8U mask(image.rows + 2, image.cols + 2);
	mask.memset(0);
	Image8U imageGray;
	cv::cvtColor(image, imageGray, cv::COLOR_BGR2GRAY);
	if (imageGray(0, 0) == 0)
		cv::floodFill(imageGray, mask, cv::Point(0, 0), 255, NULL, cv::Scalar(0), upDiff, flags);
	if (imageGray(image.rows / 2, 0) == 0)
		cv::floodFill(imageGray, mask, cv::Point(0, image.rows / 2), 255, NULL, cv::Scalar(0), upDiff, flags);
	if (imageGray(image.rows - 1, 0) == 0)
		cv::floodFill(imageGray, mask, cv::Point(0, image.rows - 1), 255, NULL, cv::Scalar(0), upDiff, flags);
	if (imageGray(image.rows - 1, image.cols / 2) == 0)
		cv::floodFill(imageGray, mask, cv::Point(image.cols / 2, image.rows - 1), 255, NULL, cv::Scalar(0), upDiff, flags);
	if (imageGray(image.rows - 1, image.cols - 1) == 0)
		cv::floodFill(imageGray, mask, cv::Point(image.cols - 1, image.rows - 1), 255, NULL, cv::Scalar(0), upDiff, flags);
	if (imageGray(image.rows / 2, image.cols - 1) == 0)
		cv::floodFill(imageGray, mask, cv::Point(image.cols - 1, image.rows / 2), 255, NULL, cv::Scalar(0), upDiff, flags);
	if (imageGray(0, image.cols - 1) == 0)
		cv::floodFill(imageGray, mask, cv::Point(image.cols - 1, 0), 255, NULL, cv::Scalar(0), upDiff, flags);
	if (imageGray(0, image.cols / 2) == 0)
		cv::floodFill(imageGray, mask, cv::Point(image.cols / 2, 0), 255, NULL, cv::Scalar(0), upDiff, flags);
	mask = (mask(cv::Rect(1,1, imageGray.cols,imageGray.rows)) == 0);
	return mask;
}

MeshTexture::MeshTexture(Scene& _scene, unsigned _nResolutionLevel, unsigned _nMinResolution)
	:
	nResolutionLevel(_nResolutionLevel),
	nMinResolution(_nMinResolution),
	vertexFaces(_scene.mesh.vertexFaces),
	vertexBoundary(_scene.mesh.vertexBoundary),
	faceFaces(_scene.mesh.faceFaces),
	faceTexcoords(_scene.mesh.faceTexcoords),
	faceTexindices(_scene.mesh.faceTexindices),
	texturesDiffuse(_scene.mesh.texturesDiffuse),
	vertices(_scene.mesh.vertices),
	faces(_scene.mesh.faces),
	images(_scene.images),
	scene(_scene)
{
}
MeshTexture::~MeshTexture()
{
	vertexFaces.Release();
	vertexBoundary.Release();
	faceFaces.Release();
}

// extract array of triangles incident to each vertex
// and check each vertex if it is at the boundary or not
// 提取每个顶点包含的faces,并判断是否在边界
void MeshTexture::ListVertexFaces()
{
	scene.mesh.EmptyExtra();
	scene.mesh.ListIncidenteFaces();
	scene.mesh.ListBoundaryVertices();
	scene.mesh.ListIncidenteFaceFaces();
}

// extract array of faces viewed by each image
/**
 * @brief  提取被image看到的所有faces，计算每个face能看到的views 相关信息（view id 投影到对应view的梯度幅值加和及颜色均值）
 * 
 * @param[in] facesDatas  存储的是每个face，投影到对应views相关信息（view id,每个投影点对应图像梯度幅值及颜色均值）
 * @param[in] fOutlierThreshold 判断face投影的view是否是外点的阈值
 * @return true 
 * @return false 
 */
bool MeshTexture::ListCameraFaces(FaceDataViewArr& facesDatas, float fOutlierThreshold, int nIgnoreMaskLabel, const IIndexArr& _views)
{
	// create faces octree
	// 构建八叉树
	Mesh::Octree octree;
	Mesh::FacesInserter::CreateOctree(octree, scene.mesh);

	// extract array of faces viewed by each image
	// 提取被每个image看到的faces
	IIndexArr views(_views);
	if (views.empty()) {
		views.resize(images.size());
		std::iota(views.begin(), views.end(), IIndex(0));
	}
	facesDatas.resize(faces.size());
	Util::Progress progress(_T("Initialized views"), views.size());
	typedef float real;
	TImage<real> imageGradMag;  // 图像梯度幅值
	TImage<real>::EMat mGrad[2];  // 梯度x,y两个方向
	FaceMap faceMap;  // 图像像素对应的face id
	DepthMap depthMap;  // 图像像素对应的深度
	#ifdef TEXOPT_USE_OPENMP
	bool bAbort(false);
	#pragma omp parallel for private(imageGradMag, mGrad, faceMap, depthMap)
	// 每帧图像逐个处理
	for (int_t idx=0; idx<(int_t)views.size(); ++idx) {
		#pragma omp flush (bAbort)
		if (bAbort) {
			++progress;
			continue;
		}
		const IIndex idxView(views[(IIndex)idx]);
	#else
	for (IIndex idxView: views) {
	#endif
		Image& imageData = images[idxView];
		if (!imageData.IsValid()) {
			++progress;
			continue;
		}
		// load image
		// 加载图像，计算用于贴图的图像分辨率
		unsigned level(nResolutionLevel);
		const unsigned imageSize(imageData.RecomputeMaxResolution(level, nMinResolution));
		if ((imageData.image.empty() || MAXF(imageData.width,imageData.height) != imageSize) && !imageData.ReloadImage(imageSize)) {
			#ifdef TEXOPT_USE_OPENMP
			bAbort = true;
			#pragma omp flush (bAbort)
			continue;
			#else
			return false;
			#endif
		}
		// 更新相机参数
		imageData.UpdateCamera(scene.platforms);
		// compute gradient magnitude
		// 计算图像梯度幅值来作为图像质量
		imageData.image.toGray(imageGradMag, cv::COLOR_BGR2GRAY, true);
		cv::Mat grad[2];
		mGrad[0].resize(imageGradMag.rows, imageGradMag.cols);
		grad[0] = cv::Mat(imageGradMag.rows, imageGradMag.cols, cv::DataType<real>::type, (void*)mGrad[0].data());
		mGrad[1].resize(imageGradMag.rows, imageGradMag.cols);
		grad[1] = cv::Mat(imageGradMag.rows, imageGradMag.cols, cv::DataType<real>::type, (void*)mGrad[1].data());
		#if 1
		// 计算梯度
		cv::Sobel(imageGradMag, grad[0], cv::DataType<real>::type, 1, 0, 3, 1.0/8.0);
		cv::Sobel(imageGradMag, grad[1], cv::DataType<real>::type, 0, 1, 3, 1.0/8.0);
		#elif 1
		const TMatrix<real,3,5> kernel(CreateDerivativeKernel3x5());
		cv::filter2D(imageGradMag, grad[0], cv::DataType<real>::type, kernel);
		cv::filter2D(imageGradMag, grad[1], cv::DataType<real>::type, kernel.t());
		#else
		const TMatrix<real,5,7> kernel(CreateDerivativeKernel5x7());
		cv::filter2D(imageGradMag, grad[0], cv::DataType<real>::type, kernel);
		cv::filter2D(imageGradMag, grad[1], cv::DataType<real>::type, kernel.t());
		#endif
		// 计算梯度的幅值mag=sqrt(dx^2+dy^2)
		(TImage<real>::EMatMap)imageGradMag = (mGrad[0].cwiseAbs2()+mGrad[1].cwiseAbs2()).cwiseSqrt();
		// apply some blur on the gradient to lower noise/glossiness effects onto face-quality score
		cv::GaussianBlur(imageGradMag, imageGradMag, cv::Size(15, 15), 0, 0, cv::BORDER_DEFAULT);
		// select faces inside view frustum
		// 选择视锥中的faces
		Mesh::FaceIdxArr cameraFaces;  // 存储了视锥内所有faces
		Mesh::FacesInserter inserter(cameraFaces);
		const TFrustum<float,5> frustum(Matrix3x4f(imageData.camera.P), (float)imageData.width, (float)imageData.height);
		octree.Traverse(frustum, inserter);
		// project all triangles in this view and keep the closest ones
		// 投影在这个view中的faces及对应深度，只保留最近的（去除遮挡点）
		faceMap.create(imageData.GetSize());
		depthMap.create(imageData.GetSize());
		RasterMesh rasterer(vertices, imageData.camera, depthMap, faceMap);
		if (nIgnoreMaskLabel >= 0) {
			// import mask
			BitMatrix bmask;
			DepthEstimator::ImportIgnoreMask(imageData, imageData.GetSize(), (uint16_t)OPTDENSE::nIgnoreMaskLabel, bmask, &rasterer.mask);
		} else if (nIgnoreMaskLabel == -1) {
			// creating mask to discard invalid regions created during image radial undistortion
			rasterer.mask = DetectInvalidImageRegions(imageData.image);
			#if TD_VERBOSE != TD_VERBOSE_OFF
			if (VERBOSITY_LEVEL > 2)
				cv::imwrite(String::FormatString("umask%04d.png", idxView), rasterer.mask);
			#endif
		}
		rasterer.Clear();
		for (FIndex idxFace : cameraFaces) {
			rasterer.validFace = true;
			const Face& facet = faces[idxFace];
			rasterer.idxFace = idxFace;
			rasterer.Project(facet);
			if (!rasterer.validFace)
				rasterer.Project(facet);
		}
		// compute the projection area of visible faces
		// 计算当前帧可见faces的投影面积
		#if TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA
		CLISTDEF0IDX(uint32_t,FIndex) areas(faces.size());
		areas.Memset(0);
		#endif

		#ifdef TEXOPT_USE_OPENMP
		#pragma omp critical
		#endif
		{
		// faceQuality is influenced by :
		// + area: the higher the area the more gradient scores will be added to the face quality
		// + sharpness: sharper image or image resolution or how close is to the face will result in higher gradient on the same face
		//				ON GLOSS IMAGES it happens to have a high volatile sharpness depending on how the light reflects under different angles
		// + angle: low angle increases the surface area
		for (int j=0; j<faceMap.rows; ++j) {
			for (int i=0; i<faceMap.cols; ++i) {
				const FIndex& idxFace = faceMap(j,i);
				ASSERT((idxFace == NO_ID && depthMap(j,i) == 0) || (idxFace != NO_ID && depthMap(j,i) > 0));
				if (idxFace == NO_ID)
					continue;
				FaceDataArr& faceDatas = facesDatas[idxFace];
				#if TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA
				uint32_t& area = areas[idxFace];
				if (area++ == 0) {
				#else
				if (faceDatas.empty() || faceDatas.back().idxView != idxView) {
				#endif
					// create new face-data
					// 创建新的face-data
					FaceData& faceData = faceDatas.emplace_back();
					faceData.idxView = idxView;
					faceData.quality = imageGradMag(j,i);
					#if TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA
					faceData.color = imageData.image(j,i);
					#endif
				} else {
					// update face-data
					// 更新face-data
					ASSERT(!faceDatas.empty());
					FaceData& faceData = faceDatas.back();
					ASSERT(faceData.idxView == idxView);
					faceData.quality += imageGradMag(j,i);
					#if TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA
					faceData.color += Color(imageData.image(j,i));
					#endif
				}
			}
		}
		// adjust face quality with camera angle relative to face normal
		// tries to increase chances of a camera with perpendicular view on the surface (smoothened normals) to be selected
		FOREACH(idxFace, facesDatas) {
			FaceDataArr& faceDatas = facesDatas[idxFace];
			if (faceDatas.empty() || faceDatas.back().idxView != idxView)
				continue;
			const Face& f = faces[idxFace];
			const Vertex faceCenter((vertices[f[0]] + vertices[f[1]] + vertices[f[2]]) / 3.f);
			const Point3f camDir(Cast<Mesh::Type>(imageData.camera.C) - faceCenter);
			const Normal& faceNormal = scene.mesh.faceNormals[idxFace];
			const float cosFaceCam(MAXF(0.001f, ComputeAngle(camDir.ptr(), faceNormal.ptr())));
			faceDatas.back().quality *= SQUARE(cosFaceCam);
		}
		#if TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA
		// 计算颜色均值
		FOREACH(idxFace, areas) {
			const uint32_t& area = areas[idxFace];
			if (area > 0) {
				Color& color = facesDatas[idxFace].back().color;
				color = RGB2YCBCR(Color(color * (1.f/(float)area)));
			}
		}
		#endif
		}
		++progress;
	}
	#ifdef TEXOPT_USE_OPENMP
	if (bAbort)
		return false;
	#endif
	progress.close();

	#if TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA
	if (fOutlierThreshold > 0) {
		// try to detect outlier views for each face，检测外点view（在场景中face的view被动态物体遮挡比如行人）
		// (views for which the face is occluded by a dynamic object in the scene, ex. pedestrians)
		for (FaceDataArr& faceDatas: facesDatas)
			FaceOutlierDetection(faceDatas, fOutlierThreshold);
	}
	#endif
	return true;
}

// order the camera view scores with highest score first and return the list of first <minCommonCameras> cameras
// ratioAngleToQuality represents the ratio in witch we combine normal angle to quality for a face to obtain the selection score
//  - a ratio of 1 means only angle is considered
//  - a ratio of 0.5 means angle and quality are equally important
//  - a ratio of 0 means only camera quality is considered when sorting
IIndexArr MeshTexture::SelectBestView(const FaceDataArr& faceDatas, FIndex fid, unsigned minCommonCameras, float ratioAngleToQuality) const
{
	ASSERT(!faceDatas.empty());
	#if 1
	
	// compute scores based on the view quality and its angle to the face normal
	float maxQuality = 0;
	for (const FaceData& faceData: faceDatas)
		maxQuality = MAXF(maxQuality, faceData.quality);
	const Face& f = faces[fid];
	const Vertex faceCenter((vertices[f[0]] + vertices[f[1]] + vertices[f[2]]) / 3.f);
	CLISTDEF0IDX(float,IIndex) scores(faceDatas.size());
	FOREACH(idxFaceData, faceDatas) {
		const FaceData& faceData = faceDatas[idxFaceData];
		const Image& imageData = images[faceData.idxView];
		const Point3f camDir(Cast<Mesh::Type>(imageData.camera.C) - faceCenter);
		const Normal& faceNormal = scene.mesh.faceNormals[fid];
		const float cosFaceCam(ComputeAngle(camDir.ptr(), faceNormal.ptr()));
		scores[idxFaceData] = ratioAngleToQuality*cosFaceCam + (1.f-ratioAngleToQuality)*faceData.quality/maxQuality;
	}
	// and sort the scores from to highest to smallest to get the best overall cameras
	IIndexArr scorePodium(faceDatas.size());
	std::iota(scorePodium.begin(), scorePodium.end(), 0);
	scorePodium.Sort([&scores](IIndex i, IIndex j) {
		return scores[i] > scores[j];
	});

	#else
	
	// sort qualityPodium in relation to faceDatas[index].quality decreasing
	IIndexArr qualityPodium(faceDatas.size());
	std::iota(qualityPodium.begin(), qualityPodium.end(), 0);
	qualityPodium.Sort([&faceDatas](IIndex i, IIndex j) {
		return faceDatas[i].quality > faceDatas[j].quality;
	});

	// sort anglePodium in relation to face angle to camera increasing
	const Face& f = faces[fid];
	const Vertex faceCenter((vertices[f[0]] + vertices[f[1]] + vertices[f[2]]) / 3.f);
	CLISTDEF0IDX(float,IIndex) cameraAngles(0, faceDatas.size());
	for (const FaceData& faceData: faceDatas) {
		const Image& imageData = images[faceData.idxView];
		const Point3f camDir(Cast<Mesh::Type>(imageData.camera.C) - faceCenter);
		const Normal& faceNormal = scene.mesh.faceNormals[fid];
		const float cosFaceCam(ComputeAngle(camDir.ptr(), faceNormal.ptr()));
		cameraAngles.emplace_back(cosFaceCam);
	}
	IIndexArr anglePodium(faceDatas.size());
	std::iota(anglePodium.begin(), anglePodium.end(), 0);
	anglePodium.Sort([&cameraAngles](IIndex i, IIndex j) {
		return cameraAngles[i] > cameraAngles[j];
	});

	// combine podium scores to get overall podium
	// and sort the scores in smallest to highest to get the best overall camera for current virtual face
	CLISTDEF0IDX(float,IIndex) scores(faceDatas.size());
	scores.Memset(0);
	FOREACH(sIdx, faceDatas) {
		scores[anglePodium[sIdx]] += ratioAngleToQuality * (sIdx+1);
		scores[qualityPodium[sIdx]] += (1.f - ratioAngleToQuality) * (sIdx+1);
	}
	IIndexArr scorePodium(faceDatas.size());
	std::iota(scorePodium.begin(), scorePodium.end(), 0);
	scorePodium.Sort([&scores](IIndex i, IIndex j) {
		return scores[i] < scores[j];
	});
	
	#endif
	IIndexArr cameras(MIN(minCommonCameras, faceDatas.size()));
	FOREACH(i, cameras)
		cameras[i] = faceDatas[scorePodium[i]].idxView;
	return cameras;
}

static bool IsFaceVisible(const MeshTexture::FaceDataArr& faceDatas, const IIndexArr& cameraList) {
	size_t camFoundCounter(0);
	for (const MeshTexture::FaceData& faceData : faceDatas) {
		const IIndex cfCam = faceData.idxView;
		for (IIndex camId : cameraList) {
			if (cfCam == camId) {
				if (++camFoundCounter == cameraList.size())
					return true;	
				break;
			}
		}
	}
	return camFoundCounter == cameraList.size();
}

// build virtual faces with:
// - similar normal
// - high percentage of common images that see them
void MeshTexture::CreateVirtualFaces(const FaceDataViewArr& facesDatas, FaceDataViewArr& virtualFacesDatas, VirtualFaceIdxsArr& virtualFaces, unsigned minCommonCameras, float thMaxNormalDeviation) const
{
	const float ratioAngleToQuality(0.67f);
	const float cosMaxNormalDeviation(COS(FD2R(thMaxNormalDeviation)));
	Mesh::FaceIdxArr remainingFaces(faces.size());
	std::iota(remainingFaces.begin(), remainingFaces.end(), 0);
	std::vector<bool> selectedFaces(faces.size(), false);
	cQueue<FIndex, FIndex, 0> currentVirtualFaceQueue;
	std::unordered_set<FIndex> queuedFaces;
	do {
		const FIndex startPos = RAND() % remainingFaces.size();
		const FIndex virtualFaceCenterFaceID = remainingFaces[startPos];
		ASSERT(currentVirtualFaceQueue.IsEmpty());
		const Normal& normalCenter = scene.mesh.faceNormals[virtualFaceCenterFaceID];
		const FaceDataArr& centerFaceDatas = facesDatas[virtualFaceCenterFaceID];
		// select the common cameras
		Mesh::FaceIdxArr virtualFace;
		FaceDataArr virtualFaceDatas;
		if (centerFaceDatas.empty()) {
			virtualFace.emplace_back(virtualFaceCenterFaceID);
			selectedFaces[virtualFaceCenterFaceID] = true;
			const auto posToErase = remainingFaces.FindFirst(virtualFaceCenterFaceID);
			ASSERT(posToErase != Mesh::FaceIdxArr::NO_INDEX);
			remainingFaces.RemoveAtMove(posToErase);
		} else {
			const IIndexArr selectedCams = SelectBestView(centerFaceDatas, virtualFaceCenterFaceID, minCommonCameras, ratioAngleToQuality);
			currentVirtualFaceQueue.AddTail(virtualFaceCenterFaceID);
			queuedFaces.clear();
			do {
				const FIndex currentFaceId = currentVirtualFaceQueue.GetHead();
				currentVirtualFaceQueue.PopHead();
				// check for condition to add in current virtual face
				// normal angle smaller than thMaxNormalDeviation degrees
				const Normal& faceNormal = scene.mesh.faceNormals[currentFaceId];
				const float cosFaceToCenter(ComputeAngleN(normalCenter.ptr(), faceNormal.ptr()));
				if (cosFaceToCenter < cosMaxNormalDeviation)
					continue;
				// check if current face is seen by all cameras in selectedCams
				ASSERT(!selectedCams.empty());
				if (!IsFaceVisible(facesDatas[currentFaceId], selectedCams))
					continue;
				// remove it from remaining faces and add it to the virtual face
				{
					const auto posToErase = remainingFaces.FindFirst(currentFaceId);
					ASSERT(posToErase != Mesh::FaceIdxArr::NO_INDEX);
					remainingFaces.RemoveAtMove(posToErase);
					selectedFaces[currentFaceId] = true;
					virtualFace.push_back(currentFaceId);
				}
				// add all new neighbors to the queue
				const Mesh::FaceFaces& ffaces = faceFaces[currentFaceId];
				for (int i = 0; i < 3; ++i) {
					const FIndex fIdx = ffaces[i];
					if (fIdx == NO_ID)
						continue;
					if (!selectedFaces[fIdx] && queuedFaces.find(fIdx) == queuedFaces.end()) {
						currentVirtualFaceQueue.AddTail(fIdx);
						queuedFaces.emplace(fIdx);
					}
				}
			} while (!currentVirtualFaceQueue.IsEmpty());
			// compute virtual face quality and create virtual face
			for (IIndex idxView: selectedCams) {
				FaceData& virtualFaceData = virtualFaceDatas.emplace_back();
				virtualFaceData.quality = 0;
				virtualFaceData.idxView = idxView;
				#if TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA
				virtualFaceData.color = Point3f::ZERO;
				#endif
				unsigned processedFaces(0);
				for (FIndex fid : virtualFace) {
					const FaceDataArr& faceDatas = facesDatas[fid];
					for (FaceData& faceData: faceDatas) {
						if (faceData.idxView == idxView) {
							virtualFaceData.quality += faceData.quality;
							#if TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA
							virtualFaceData.color += faceData.color;
							#endif
							++processedFaces;
							break;
						}
					}
				}
				ASSERT(processedFaces > 0);
				virtualFaceData.quality /= processedFaces;
				#if TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA
				virtualFaceData.color /= processedFaces;
				#endif
			}
			ASSERT(!virtualFaceDatas.empty());
		}
		virtualFacesDatas.emplace_back(std::move(virtualFaceDatas));
		virtualFaces.emplace_back(std::move(virtualFace));
	} while (!remainingFaces.empty());
}

#if TEXOPT_FACEOUTLIER == TEXOPT_FACEOUTLIER_MEDIAN

// decrease the quality of / remove all views in which the face's projection
// has a much different color than in the majority of views
// 如果face在该view的投影与大多views中不同，降低view质量或者直接移除。
bool MeshTexture::FaceOutlierDetection(FaceDataArr& faceDatas, float thOutlier) const
{
	// consider as outlier if the absolute difference to the median is outside this threshold
	// 如果颜色值与中值的差的绝对值大于thOutlier则认为是外点
	if (thOutlier <= 0)
		thOutlier = 0.15f*255.f;

	// init colors array
	// 初始化颜色
	if (faceDatas.size() <= 3)
		return false;
	FloatArr channels[3];
	for (int c=0; c<3; ++c)
		channels[c].resize(faceDatas.size());
	FOREACH(i, faceDatas) {
		const Color& color = faceDatas[i].color;
		for (int c=0; c<3; ++c)
			channels[c][i] = color[c];
	}

	// find median，找中值
	for (int c=0; c<3; ++c)
		channels[c].Sort();
	const unsigned idxMedian(faceDatas.size() >> 1);
	Color median;
	for (int c=0; c<3; ++c)
		median[c] = channels[c][idxMedian];

	// abort if there are not at least 3 inliers
	int nInliers(0);
	BoolArr inliers(faceDatas.size());
	FOREACH(i, faceDatas) {
		const Color& color = faceDatas[i].color;
		for (int c=0; c<3; ++c) {
			if (ABS(median[c]-color[c]) > thOutlier) {
				inliers[i] = false;
				goto CONTINUE_LOOP;
			}
		}
		inliers[i] = true;
		++nInliers;
		CONTINUE_LOOP:;
	}
	if (nInliers == faceDatas.size())
		return true;
	if (nInliers < 3)
		return false;

	// remove outliers
	RFOREACH(i, faceDatas)
		if (!inliers[i])
			faceDatas.RemoveAt(i);
	return true;
}

#elif TEXOPT_FACEOUTLIER != TEXOPT_FACEOUTLIER_NA

// A multi-variate normal distribution which is NOT normalized such that the integral is 1
// - centered is the vector for which the function is to be evaluated with the mean subtracted [Nx1]
// - X is the vector for which the function is to be evaluated [Nx1]
// - mu is the mean around which the distribution is centered [Nx1]
// - covarianceInv is the inverse of the covariance matrix [NxN]
// return exp(-1/2 * (X-mu)^T * covariance_inv * (X-mu))
template <typename T, int N>
inline T MultiGaussUnnormalized(const Eigen::Matrix<T,N,1>& centered, const Eigen::Matrix<T,N,N>& covarianceInv) {
	return EXP(T(-0.5) * T(centered.adjoint() * covarianceInv * centered));
}
template <typename T, int N>
inline T MultiGaussUnnormalized(const Eigen::Matrix<T,N,1>& X, const Eigen::Matrix<T,N,1>& mu, const Eigen::Matrix<T,N,N>& covarianceInv) {
	return MultiGaussUnnormalized<T,N>(X - mu, covarianceInv);
}

// decrease the quality of / remove all views in which the face's projection
// has a much different color than in the majority of views
bool MeshTexture::FaceOutlierDetection(FaceDataArr& faceDatas, float thOutlier) const
{
	// reject all views whose gauss value is below this threshold
	if (thOutlier <= 0)
		thOutlier = 6e-2f;

	const float minCovariance(1e-3f); // if all covariances drop below this the outlier detection aborted

	const unsigned maxIterations(10);
	const unsigned minInliers(4);

	// init colors array
	if (faceDatas.size() <= minInliers)
		return false;
	Eigen::Matrix3Xd colorsAll(3, faceDatas.size());
	BoolArr inliers(faceDatas.size());
	FOREACH(i, faceDatas) {
		colorsAll.col(i) = ((const Color::EVec)faceDatas[i].color).cast<double>();
		inliers[i] = true;
	}

	// perform outlier removal; abort if something goes wrong
	// (number of inliers below threshold or can not invert the covariance)
	size_t numInliers(faceDatas.size());
	Eigen::Vector3d mean;
	Eigen::Matrix3d covariance;
	Eigen::Matrix3d covarianceInv;
	for (unsigned iter = 0; iter < maxIterations; ++iter) {
		// compute the mean color and color covariance only for inliers
		const Eigen::Block<Eigen::Matrix3Xd,3,Eigen::Dynamic,!Eigen::Matrix3Xd::IsRowMajor> colors(colorsAll.leftCols(numInliers));
		mean = colors.rowwise().mean();
		const Eigen::Matrix3Xd centered(colors.colwise() - mean);
		covariance = (centered * centered.transpose()) / double(colors.cols() - 1);

		// stop if all covariances gets very small
		if (covariance.array().abs().maxCoeff() < minCovariance) {
			// remove the outliers
			RFOREACH(i, faceDatas)
				if (!inliers[i])
					faceDatas.RemoveAt(i);
			return true;
		}

		// invert the covariance matrix
		// (FullPivLU not the fastest, but gives feedback about numerical stability during inversion)
		const Eigen::FullPivLU<Eigen::Matrix3d> lu(covariance);
		if (!lu.isInvertible())
			return false;
		covarianceInv = lu.inverse();

		// filter inliers
		// (all views with a gauss value above the threshold)
		numInliers = 0;
		bool bChanged(false);
		FOREACH(i, faceDatas) {
			const Eigen::Vector3d color(((const Color::EVec)faceDatas[i].color).cast<double>());
			const double gaussValue(MultiGaussUnnormalized<double,3>(color, mean, covarianceInv));
			bool& inlier = inliers[i];
			if (gaussValue > thOutlier) {
				// set as inlier
				colorsAll.col(numInliers++) = color;
				if (inlier != true) {
					inlier = true;
					bChanged = true;
				}
			} else {
				// set as outlier
				if (inlier != false) {
					inlier = false;
					bChanged = true;
				}
			}
		}
		if (numInliers == faceDatas.size())
			return true;
		if (numInliers < minInliers)
			return false;
		if (!bChanged)
			break;
	}

	#if TEXOPT_FACEOUTLIER == TEXOPT_FACEOUTLIER_GAUSS_DAMPING
	// select the final inliers
	const float factorOutlierRemoval(0.2f);
	covarianceInv *= factorOutlierRemoval;
	RFOREACH(i, faceDatas) {
		const Eigen::Vector3d color(((const Color::EVec)faceDatas[i].color).cast<double>());
		const double gaussValue(MultiGaussUnnormalized<double,3>(color, mean, covarianceInv));
		ASSERT(gaussValue >= 0 && gaussValue <= 1);
		faceDatas[i].quality *= gaussValue;
	}
	#endif
	#if TEXOPT_FACEOUTLIER == TEXOPT_FACEOUTLIER_GAUSS_CLAMPING
	// remove outliers
	RFOREACH(i, faceDatas)
		if (!inliers[i])
			faceDatas.RemoveAt(i);
	#endif
	return true;
}
#endif
/**
 * @brief  给每个face（三角网格）分配最佳视图view 
 * 
 * 如果face在该view的投影与大多views中不同，降低view质量或者直接移除。
 * @param[in] fOutlierThreshold     颜色差异阈值，用于face选择投影的view时，剔除与大多view不同的外点view
 * @param[in] fRatioDataSmoothness  控制平滑程度，越大越平滑
 * @return true 
 * @return false 
 */
bool MeshTexture::FaceViewSelection(unsigned minCommonCameras, float fOutlierThreshold, float fRatioDataSmoothness, int nIgnoreMaskLabel, const IIndexArr& views)
{
	// extract array of triangles incident to each vertex
	// 提取顶点所在的所有faces
	ListVertexFaces();

	// create texture patches
	// 创建纹理块
	{
		// compute face normals and smoothen them
		scene.mesh.SmoothNormalFaces();

		// list all views for each face
		// 列出每个face能被看到的所有views,记录看到该face的所有views信息，这些view将作为标签，face将作为节点，根据它们的对应关系可以构造出一个全局优化函数，并通过MRF进行求解
		FaceDataViewArr facesDatas;
		if (!ListCameraFaces(facesDatas, fOutlierThreshold, nIgnoreMaskLabel, views))
			return false;

		// create faces graph
		// 创建以face为节点的无向图（MRF用于无向图，贝叶斯概率图模型用于有向图），从而方便对face的邻域进行遍历
		typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> Graph;
		typedef boost::graph_traits<Graph>::edge_iterator EdgeIter;
		typedef boost::graph_traits<Graph>::out_edge_iterator EdgeOutIter;
		Graph graph;
		LabelArr labels;

		// construct and use virtual faces for patch creation instead of actual mesh faces;
		// the virtual faces are composed of coplanar triangles sharing same views
		const bool bUseVirtualFaces(minCommonCameras > 0);
		if (bUseVirtualFaces) {
			// 1) create FaceToVirtualFaceMap
			FaceDataViewArr virtualFacesDatas;
			VirtualFaceIdxsArr virtualFaces; // stores each virtual face as an array of mesh face ID
			CreateVirtualFaces(facesDatas, virtualFacesDatas, virtualFaces, minCommonCameras);
			Mesh::FaceIdxArr mapFaceToVirtualFace(faces.size()); // for each mesh face ID, store the virtual face ID witch contains it
			size_t controlCounter(0);
			FOREACH(idxVF, virtualFaces) {
				const Mesh::FaceIdxArr& vf = virtualFaces[idxVF];
				for (FIndex idxFace : vf) {
					mapFaceToVirtualFace[idxFace] = idxVF;
					++controlCounter;
				}
			}
			ASSERT(controlCounter == faces.size());
			// 2) create function to find virtual faces neighbors
			VirtualFaceIdxsArr virtualFaceNeighbors; { // for each virtual face, the list of virtual faces with at least one vertex in common
				virtualFaceNeighbors.resize(virtualFaces.size());
				FOREACH(idxVF, virtualFaces) {
					const Mesh::FaceIdxArr& vf = virtualFaces[idxVF];
					Mesh::FaceIdxArr& vfNeighbors = virtualFaceNeighbors[idxVF];
					for (FIndex idxFace : vf) {
						const Mesh::FaceFaces& adjFaces = faceFaces[idxFace];
						for (int i = 0; i < 3; ++i) {
							const FIndex fAdj(adjFaces[i]);
							if (fAdj == NO_ID)
								continue;
							if (mapFaceToVirtualFace[fAdj] == idxVF)
								continue;
							if (fAdj != idxFace && vfNeighbors.Find(mapFaceToVirtualFace[fAdj]) == Mesh::FaceIdxArr::NO_INDEX) {
								vfNeighbors.emplace_back(mapFaceToVirtualFace[fAdj]);
							}
						}
					}
				}
			}
			// 3) use virtual faces to build the graph
			// 4) assign images to virtual faces
			// 5) spread image ID to each mesh face from virtual face
			FOREACH(idxFace, virtualFaces) {
				MAYBEUNUSED const Mesh::FIndex idx((Mesh::FIndex)boost::add_vertex(graph));  // 向graph中添加节点
				ASSERT(idx == idxFace);
			}
			FOREACH(idxVirtualFace, virtualFaces) {
				// 取face的三个相邻faces，如果在边界可能只有1个或者2个
				const Mesh::FaceIdxArr& afaces = virtualFaceNeighbors[idxVirtualFace];
				for (FIndex idxVirtualFaceAdj: afaces) {  // 对相邻face进行遍历
					// 前面face已经处理过则跳过（因为face是根据id从小到大进行处理，而在无向图中，节点之间的邻接性是对称的，若获取到的相邻face的id小于当前face，则说明该相邻face之前已被处理过，并且相邻face与当前face之间的相邻关系也被考虑过），从而避免重复处理
					if (idxVirtualFace >= idxVirtualFaceAdj)
						continue;
					const bool bInvisibleFace(virtualFacesDatas[idxVirtualFace].empty());  // virtualFacesDatas中存放了每个face对应多少个相机能看到，若没有相机能看到，则说明该face不可见
					const bool bInvisibleFaceAdj(virtualFacesDatas[idxVirtualFaceAdj].empty());
					// 如果当前face和邻域face都没有可见的view则跳过
					if (bInvisibleFace || bInvisibleFaceAdj)
						continue;
					boost::add_edge(idxVirtualFace, idxVirtualFaceAdj, graph);  // 向图graph中添加边，该边的顶点为idxVirtualFace和idxVirtualFaceAdj
				}
			}
			ASSERT((Mesh::FIndex)boost::num_vertices(graph) == virtualFaces.size());
			// assign the best view to each face
			// 给每个面找一个最好的view
			labels.resize(faces.size()); {
				// normalize quality values
				// 归一化质量值，即每个face投影到其所可能对应的视图后，每个视图所对应的图像区域的梯度幅值的最大值。OpenMVS认为视图区域的梯度幅值越大，对应的质量越好
				float maxQuality(0);  // 计算最大质量值
				// 对每个face所对应的视图进行遍历，从而分别获取每个face所对应的最大质量值
				for (const FaceDataArr& faceDatas: virtualFacesDatas) {
					for (const FaceData& faceData: faceDatas)
						if (maxQuality < faceData.quality)
							maxQuality = faceData.quality;
				}
				Histogram32F hist(std::make_pair(0.f, maxQuality), 1000);  // 根据上述的质量构建一个直方图，其数值分布范围为[0, maxQuality]，并且将这个范围内的数据分为1000组进行统计
				for (const FaceDataArr& faceDatas: virtualFacesDatas) {
					for (const FaceData& faceData: faceDatas)
						hist.Add(faceData.quality);
				}
				// 归一化质量
				const float normQuality(hist.GetApproximatePermille(0.95f));
				// 马尔可夫随机场（概率无向图模型）解决labeling问题，优化方法LBP（loopy belief propagation algorithm）
				// 每个face都有k个标签，要求解的问题是选择一种标签组合使该方案发生的概率最大可以转化为最小能量求解问题：
				// 找到一组标签组合使得最终的cost最小 min(E)=Σc(xs)+Σc(xs,xt) xs是所有face（node）,xt是face的邻域
				// 具体有关马尔可夫介绍见课件
				#if TEXOPT_INFERENCE == TEXOPT_INFERENCE_LBP
				// initialize inference structures
				// Step 1 初始化inference,设置邻域和节点
				const LBPInference::EnergyType MaxEnergy(fRatioDataSmoothness*(LBPInference::EnergyType)LBPInference::MaxEnergy);
				LBPInference inference; {
					inference.SetNumNodes(virtualFaces.size());  // virtualFaces中包含每个连通域所包含的面片个数
					// 设置平滑cost,如果两个节点标签相同则cost=0,否则为MaxEnergy
					// 目的是让相邻face的标签尽可能一致
					inference.SetSmoothCost(SmoothnessPotts);  // 选择SmoothnessPotts模型用于产生平滑项对应的代价
					EdgeOutIter ei, eie;
					FOREACH(f, virtualFaces) {
						// 添加edge即每个face的邻域face
						for (boost::tie(ei, eie) = boost::out_edges(f, graph); ei != eie; ++ei) {
							ASSERT(f == (FIndex)ei->m_source);
							const FIndex fAdj((FIndex)ei->m_target);
							// 确保每个edge只添加一次，这是为了避免后续重复计算
							if (f < fAdj) // add edges only once
								inference.SetNeighbors(f, fAdj);
						}
						// set costs for label 0 (undefined)
						// 如果是未分配label的face设为0 cost为固定值MaxEnergy
						inference.SetDataCost((Label)0, f, MaxEnergy);
					}
				}

				// set data costs
				// Step 2 设置node的每个标签对应的cost 
				// set data costs for all labels (except label 0 - undefined)
				// 设置face 的每个label的cost，若是未分配label的face设为0，对应的cost为固定值MaxEnergy
				FOREACH(f, virtualFacesDatas) {
					const FaceDataArr& faceDatas = virtualFacesDatas[f];  // 索引为f的face所能看到的视图
					for (const FaceData& faceData: faceDatas) {
						// 有效标签从1开始，因为0是无标签的标记
						const Label label((Label)faceData.idxView+1);  // 视图所对应的id值
						// 归一化，将视图质量数值范围缩放到[0, 1]
						const float normalizedQuality(faceData.quality>=normQuality ? 1.f : faceData.quality/normQuality);
						// cost计算，质量越好代价越小
						const float dataCost((1.f-normalizedQuality)*MaxEnergy);
						inference.SetDataCost(label, f, dataCost);  // 此处的dataCost为视图梯度
					}
				}

				// assign the optimal view (label) to each face
				// (label 0 is reserved as undefined)
				// Step 3 调用能量最小优化函数，也就是LBP这个消息传播流程
				inference.Optimize();

				// extract resulting labeling
				// Step 4 提取labeling的结果
				LabelArr virtualLabels(virtualFaces.size());
				virtualLabels.Memset(0xFF);  // 将virtualLabels设为一个较大的范围
				FOREACH(l, virtualLabels) {
					const Label label(inference.GetLabel(l));  // 获取每个节点、面片所对应的最佳Label、视图
					ASSERT(label < images.size()+1);
					//  注意-1 ，在推理时label的有效值是从1开始的，0表示无效值，此处为了让label从0开始以方便进行索引所以进行-1操作
					if (label > 0)
						virtualLabels[l] = label-1;
				}
				FOREACH(l, labels) {
					labels[l] = virtualLabels[mapFaceToVirtualFace[l]];
				}
				#endif
			}

			graph.clear();
		}
		
		// create the graph of faces: each vertex is a face and the edges are the edges shared by the faces
		FOREACH(idxFace, faces) {
			MAYBEUNUSED const Mesh::FIndex idx((Mesh::FIndex)boost::add_vertex(graph));
			ASSERT(idx == idxFace);
		}
		FOREACH(idxFace, faces) {
			const Mesh::FaceFaces& afaces = faceFaces[idxFace];
			for (int v=0; v<3; ++v) {
				const FIndex idxFaceAdj = afaces[v];
				if (idxFaceAdj == NO_ID || idxFace >= idxFaceAdj)
					continue;
				const bool bInvisibleFace(facesDatas[idxFace].empty());
				const bool bInvisibleFaceAdj(facesDatas[idxFaceAdj].empty());
				if (bInvisibleFace || bInvisibleFaceAdj) {  // facesDatas中存放了每个face对应多少个相机能看到，若没有相机能看到，则说明该face不可见
					if (bInvisibleFace != bInvisibleFaceAdj)  // 若两个face一个能看到另一个不可见，则可认为这是边界，所以将这两个face的id记录在存储边界信息的列表seamEdges中
						seamEdges.emplace_back(idxFace, idxFaceAdj);
					continue;
				}
				boost::add_edge(idxFace, idxFaceAdj, graph);
			}
		}
		faceFaces.Release();
		ASSERT((Mesh::FIndex)boost::num_vertices(graph) == faces.size());

		// start patch creation starting directly from individual faces
		if (!bUseVirtualFaces) {
			// assign the best view to each face
			labels.resize(faces.size()); {
				// normalize quality values
				float maxQuality(0);
				for (const FaceDataArr& faceDatas: facesDatas) {
					for (const FaceData& faceData: faceDatas)
						if (maxQuality < faceData.quality)
							maxQuality = faceData.quality;
				}
				Histogram32F hist(std::make_pair(0.f, maxQuality), 1000);
				for (const FaceDataArr& faceDatas: facesDatas) {
					for (const FaceData& faceData: faceDatas)
						hist.Add(faceData.quality);
				}
				const float normQuality(hist.GetApproximatePermille(0.95f));

				#if TEXOPT_INFERENCE == TEXOPT_INFERENCE_LBP
				// initialize inference structures
				const LBPInference::EnergyType MaxEnergy(fRatioDataSmoothness*(LBPInference::EnergyType)LBPInference::MaxEnergy);
				LBPInference inference; {
					inference.SetNumNodes(faces.size());
					inference.SetSmoothCost(SmoothnessPotts);
					EdgeOutIter ei, eie;
					FOREACH(f, faces) {
						for (boost::tie(ei, eie) = boost::out_edges(f, graph); ei != eie; ++ei) {
							ASSERT(f == (FIndex)ei->m_source);
							const FIndex fAdj((FIndex)ei->m_target);
							if (f < fAdj) // add edges only once
								inference.SetNeighbors(f, fAdj);
						}
						// set costs for label 0 (undefined)
						inference.SetDataCost((Label)0, f, MaxEnergy);
					}
				}

				// set data costs for all labels (except label 0 - undefined)
				FOREACH(f, facesDatas) {
					const FaceDataArr& faceDatas = facesDatas[f];
					for (const FaceData& faceData: faceDatas) {
						const Label label((Label)faceData.idxView+1);
						const float normalizedQuality(faceData.quality>=normQuality ? 1.f : faceData.quality/normQuality);
						const float dataCost((1.f-normalizedQuality)*MaxEnergy);
						inference.SetDataCost(label, f, dataCost);
					}
				}

				// assign the optimal view (label) to each face
				// (label 0 is reserved as undefined)
				inference.Optimize();

				// extract resulting labeling
				labels.Memset(0xFF);
				FOREACH(l, labels) {
					const Label label(inference.GetLabel(l));
					ASSERT(label < images.size()+1);
					if (label > 0)
						labels[l] = label-1;
				}
				#endif

				// TRWS与LBP调用类似不再赘述
				#if TEXOPT_INFERENCE == TEXOPT_INFERENCE_TRWS
				// find connected components
				// 计算连通域个数是nComponents，每个face i所在的连通域的id是components[i]
				ASSERT((FIndex)boost::num_vertices(graph) == faces.size());
				components.resize(faces.size());
				const FIndex nComponents(boost::connected_components(graph, components.data()));

				// map face ID from global to component space
				// 将face的id从全局转到单个component空间
				typedef cList<NodeID, NodeID, 0, 128, NodeID> NodeIDs;
				NodeIDs nodeIDs(faces.size());  // 存储节点在component中的新id
				NodeIDs sizes(nComponents);     // 记录每个component中包含的face个数
				sizes.Memset(0);
				// 后续贴图是对每个连通域分别处理的，即后续在调用LBP进行处理时，这些连通域会被当做是单独的部分分开处理，所以需要对face进行重新编码，
				FOREACH(c, components)
					nodeIDs[c] = sizes[components[c]]++;   // 记录每个face在对应component中的新id

				// initialize inference structures
				const LabelID numLabels(images.size()+1);
				CLISTDEFIDX(TRWSInference, FIndex) inferences(nComponents);
				FOREACH(s, sizes) {
					const NodeID numNodes(sizes[s]);
					ASSERT(numNodes > 0);
					if (numNodes <= 1)
						continue;
					TRWSInference& inference = inferences[s];
					inference.Init(numNodes, numLabels);
				}

				// set data costs
				{
					// add nodes
					CLISTDEF0(EnergyType) D(numLabels);
					FOREACH(f, facesDatas) {
						TRWSInference& inference = inferences[components[f]];
						if (inference.IsEmpty())
							continue;
						D.MemsetValue(MaxEnergy);
						const FaceDataArr& faceDatas = facesDatas[f];
						for (const FaceData& faceData: faceDatas) {
							const Label label((Label)faceData.idxView);
							const float normalizedQuality(faceData.quality>=normQuality ? 1.f : faceData.quality/normQuality);
							const EnergyType dataCost(MaxEnergy*(1.f-normalizedQuality));
							D[label] = dataCost;
						}
						const NodeID nodeID(nodeIDs[f]);
						inference.AddNode(nodeID, D.Begin());
					}
					// add edges
					EdgeOutIter ei, eie;
					FOREACH(f, faces) {
						TRWSInference& inference = inferences[components[f]];
						if (inference.IsEmpty())
							continue;
						for (boost::tie(ei, eie) = boost::out_edges(f, graph); ei != eie; ++ei) {
							ASSERT(f == (FIndex)ei->m_source);
							const FIndex fAdj((FIndex)ei->m_target);
							ASSERT(components[f] == components[fAdj]);
							if (f < fAdj) // add edges only once
								inference.AddEdge(nodeIDs[f], nodeIDs[fAdj]);
						}
					}
				}

				// assign the optimal view (label) to each face
				#ifdef TEXOPT_USE_OPENMP
				#pragma omp parallel for schedule(dynamic)
				for (int i=0; i<(int)inferences.size(); ++i) {
				#else
				FOREACH(i, inferences) {
				#endif
					TRWSInference& inference = inferences[i];
					if (inference.IsEmpty())
						continue;
					inference.Optimize();
				}
				// extract resulting labeling
				// 提取label
				labels.Memset(0xFF);
				FOREACH(l, labels) {
					TRWSInference& inference = inferences[components[l]];  // OpenMVS中对于每个连通分量都构建一个推理器去进行处理，此处是选取每个推理器
					if (inference.IsEmpty())
						continue;
					const Label label(inference.GetLabel(nodeIDs[l]));
					ASSERT(label >= 0 && label < numLabels);
					if (label < images.size())
						labels[l] = label;
				}
				#endif
			}
		}

		// create texture patches
		// 创建纹理块
		{
			// divide graph in sub-graphs of connected faces having the same label
			// 划分有相同label的face
			EdgeIter ei, eie;
			const PairIdxArr::IDX startLabelSeamEdges(seamEdges.size());
			for (boost::tie(ei, eie) = boost::edges(graph); ei != eie; ++ei) {
				const FIndex fSource((FIndex)ei->m_source);
				const FIndex fTarget((FIndex)ei->m_target);
				ASSERT(components.empty() || components[fSource] == components[fTarget]);
				// 统计label不同的edge
				if (labels[fSource] != labels[fTarget])
					seamEdges.emplace_back(fSource, fTarget);
			}
			// 将graph中label不同的边剔除
			for (const PairIdx *pEdge=seamEdges.Begin()+startLabelSeamEdges, *pEdgeEnd=seamEdges.End(); pEdge!=pEdgeEnd; ++pEdge)
				boost::remove_edge(pEdge->i, pEdge->j, graph);

			// find connected components: texture patches
			// 重新查找连通块即为纹理块
			ASSERT((FIndex)boost::num_vertices(graph) == faces.size());
			components.resize(faces.size());
			const FIndex nComponents(boost::connected_components(graph, components.data()));

			// create texture patches;
			// last texture patch contains all faces with no texture
			// 构建纹理块，最后一个patch存放所有没有label的faces
			LabelArr sizes(nComponents);
			sizes.Memset(0);
			FOREACH(c, components)
				++sizes[components[c]];
			// +1增加一个Patch存放无纹理(label)的faces
			texturePatches.resize(nComponents+1);
			texturePatches.back().label = NO_ID;
			FOREACH(f, faces) {
				const Label label(labels[f]);
				const FIndex c(components[f]);
				TexturePatch& texturePatch = texturePatches[c];
				ASSERT(texturePatch.label == label || texturePatch.faces.empty());
				if (label == NO_ID) {
					texturePatch.label = NO_ID;
					texturePatches.back().faces.Insert(f);
				} else {
					if (texturePatch.faces.empty()) {
						texturePatch.label = label;
						texturePatch.faces.reserve(sizes[c]);
					}
					texturePatch.faces.Insert(f);
				}
			}
			// remove all patches with invalid label (except the last one)
			// and create the map from the old index to the new one
			// 移除所有patch中无效的label,同时记录顶点的新旧id
			mapIdxPatch.resize(nComponents);
			// mapIdxPatch:0,1,2...nComponents-1
			std::iota(mapIdxPatch.Begin(), mapIdxPatch.End(), 0);
			for (FIndex t = nComponents; t-- > 0; ) {
				if (texturePatches[t].label == NO_ID) {
					texturePatches.RemoveAtMove(t);
					mapIdxPatch.RemoveAtMove(t);
				}
			}
			const unsigned numPatches(texturePatches.size()-1);
			uint32_t idxPatch(0);
			for (IndexArr::IDX i=0; i<mapIdxPatch.size(); ++i) {
				while (i < mapIdxPatch[i])
					mapIdxPatch.InsertAt(i++, numPatches);
				mapIdxPatch[i] = idxPatch++;
			}
			while (mapIdxPatch.size() <= nComponents)
				mapIdxPatch.Insert(numPatches);
		}
	}
	return true;
}


// create seam vertices and edges
// 创建接缝的顶点和边。最终所有的处理结果都会被保存在seamVertices这个变量里面
void MeshTexture::CreateSeamVertices()
{
	// each vertex will contain the list of patches it separates,
	// except the patch containing invisible faces;
	// each patch contains the list of edges belonging to that texture patch, starting from that vertex
	// (usually there are pairs of edges in each patch, representing the two edges starting from that vertex separating two valid patches)
	// 每个顶点包含由它分割的patches（除去包含不可见faces的patch）
	// 每个patch包含属于纹理patch的以上述顶点起始的edge
	VIndex vs[2];
	uint32_t vs0[2], vs1[2];
	std::unordered_map<VIndex, uint32_t> mapVertexSeam;
	const unsigned numPatches(texturePatches.size()-1);  // 有效纹理块的个数。此处之所以-1，是因为texturePatches中最后一个数据对应的是无label的patch的信息，因为纹理坐标都分配好了，而它都没有参与，因为其与有效纹理块无关，故不需要考虑，
	for (const PairIdx& edge: seamEdges) {  // 对之前统计的边界信息进行遍历
		// store edge for the later seam optimization
		// 存储edge
		ASSERT(edge.i < edge.j);  
		// edge.i、edge.j分别对应边的起始点和终止点，因为每条邻接边都对应两个face，也就是它为两个face所共有，
		// 因此可以将边的两个端点分别从这两个face中取出来，而每个face都对应一种patch,这可能也是为什么可以通过边的顶点索引到其所对应的两个patch的原因
		const uint32_t idxPatch0(mapIdxPatch[components[edge.i]]);  // components中存放连通域中patch的id与其实际id之间的对应关系。
		const uint32_t idxPatch1(mapIdxPatch[components[edge.j]]);
		// 判断边所对应的两个face所属的patch的id是否相同，即其是否属于同一种patch，若是则说明它们属于同一个纹理块，即不属于接缝处，因此需要后续的处理
		ASSERT(idxPatch0 != idxPatch1 || idxPatch0 == numPatches);
		if (idxPatch0 == idxPatch1)
			continue;
		// 若边处于接缝处，则对其进行存储，以便后续处理
		seamVertices.ReserveExtra(2);
		scene.mesh.GetEdgeVertices(edge.i, edge.j, vs0, vs1);  // 获取边所对应的两个端点，并分别存放到vs0和vs1
		ASSERT(faces[edge.i][vs0[0]] == faces[edge.j][vs1[0]]);
		ASSERT(faces[edge.i][vs0[1]] == faces[edge.j][vs1[1]]);
		// 获取端点vs0所对应的顶点信息
		vs[0] = faces[edge.i][vs0[0]];
		vs[1] = faces[edge.i][vs0[1]];
		
		// 根据vs0所对应的顶点信息，构建接缝处的两个顶点变量seamVertex0、seamVertex1
		const auto itSeamVertex0(mapVertexSeam.emplace(std::make_pair(vs[0], seamVertices.size())));
		if (itSeamVertex0.second)
			seamVertices.emplace_back(vs[0]);
		SeamVertex& seamVertex0 = seamVertices[itSeamVertex0.first->second];

		const auto itSeamVertex1(mapVertexSeam.emplace(std::make_pair(vs[1], seamVertices.size())));
		if (itSeamVertex1.second)
			seamVertices.emplace_back(vs[1]);
		SeamVertex& seamVertex1 = seamVertices[itSeamVertex1.first->second];

		// 边所对应的两个端点分别对应着两个patch，patch0和patch1
		if (idxPatch0 < numPatches) {  // 处理patch0，计算其相关信息，即patch在接缝处与edge的端点相关的边，以及这些edge端点在patch0上的投影
			const TexCoord offset0(texturePatches[idxPatch0].rect.tl());  // patch0对应的纹理块的偏移量
			SeamVertex::Patch& patch00 = seamVertex0.GetPatch(idxPatch0);  // 往seamVertex0中加入patch0的id
			SeamVertex::Patch& patch10 = seamVertex1.GetPatch(idxPatch0);  // 往seamVertex1中加入patch0的id
			ASSERT(patch00.edges.Find(itSeamVertex1.first->second) == NO_ID);
			patch00.edges.emplace_back(itSeamVertex1.first->second).idxFace = edge.i;  // 往seamVertex0中加入接缝处的边上的对应端点
			patch00.proj = faceTexcoords[edge.i*3+vs0[0]]+offset0;  // 往seamVertex0中加入接缝处的边上的对应端点在patch0上的投影坐标
			ASSERT(patch10.edges.Find(itSeamVertex0.first->second) == NO_ID);
			patch10.edges.emplace_back(itSeamVertex0.first->second).idxFace = edge.i;  // 往seamVertex0中加入接缝处的边上的对应端点
			patch10.proj = faceTexcoords[edge.i*3+vs0[1]]+offset0;  // 往seamVertex1中加入接缝处的边上的对应端点在patch0上的投影坐标
		}
		if (idxPatch1 < numPatches) {  // 处理patch1
			const TexCoord offset1(texturePatches[idxPatch1].rect.tl());
			SeamVertex::Patch& patch01 = seamVertex0.GetPatch(idxPatch1);
			SeamVertex::Patch& patch11 = seamVertex1.GetPatch(idxPatch1);
			ASSERT(patch01.edges.Find(itSeamVertex1.first->second) == NO_ID);
			patch01.edges.emplace_back(itSeamVertex1.first->second).idxFace = edge.j;
			patch01.proj = faceTexcoords[edge.j*3+vs1[0]]+offset1;
			ASSERT(patch11.edges.Find(itSeamVertex0.first->second) == NO_ID);
			patch11.edges.emplace_back(itSeamVertex0.first->second).idxFace = edge.j;
			patch11.proj = faceTexcoords[edge.j*3+vs1[1]]+offset1;
		}
	}
	seamEdges.Release();
}

/**
 * @brief 全局颜色校正
 * 
 */
void MeshTexture::GlobalSeamLeveling()
{
	ASSERT(!seamVertices.empty());
	// 减一是最后一个patch是存放无label的faces
	const unsigned numPatches(texturePatches.size()-1);
	// Step 1 数据准备：记录每个顶点的patch id,并对处于patches间边界处的顶点进行标记
	// 有些顶点只对应一个patch，而有些顶点（尤其是边界处的顶点）则会对应多个patch
	// find the patch ID for each vertex
	// 存放每个顶点的patch id,如果顶点在seam上记录其在seamVertices中的id
	PatchIndices patchIndices(vertices.size());
	patchIndices.Memset(0);
	FOREACH(f, faces) {  // 遍历面片，每个面片都对应一个label(即对应的视图)
		const uint32_t idxPatch(mapIdxPatch[components[f]]);
		const Face& face = faces[f];
		for (int v=0; v<3; ++v)
			patchIndices[face[v]].idxPatch = idxPatch;  // 并且将face所对应的三个顶点所对应的patch的id分别存放到patchIndices里面
	}
	// 记录顶点在seamVertex的id
	FOREACH(i, seamVertices) {  // 遍历边界点
		const SeamVertex& seamVertex = seamVertices[i];
		ASSERT(!seamVertex.patches.empty());
		PatchIndex& patchIndex = patchIndices[seamVertex.idxVertex];  // 边界点所对应的patch的索引
		// 标记该顶点是否在纹理接缝上
		patchIndex.bIndex = true;
		patchIndex.idxSeamVertex = i;  // 记录每个patch所对应的边界信息，即其在存放边界信息的列表seamVertices中的位置
	}

	// assign a row index within the solution vector x to each vertex/patch
	// Step 2 分配一个行索引给每个顶点 后续构建索引
	// 即每个顶点在对应的patch在课件里公式（2）的矩阵形式（即公式（3））中的变量g里的所在行。
	// 一般顶点对应着一个patch，
	// 若顶点对应多个patch，即其处在多个patch的交接处，则其在这些Patch上的调整信息也会在g中被依次记录在一起。
	// 通过这个记录的位置信息，可以很方便地构造出公式（3）里的系数矩阵
	ASSERT(vertices.size() < static_cast<VIndex>(std::numeric_limits<MatIdx>::max()));
	MatIdx rowsX(0);
	typedef std::unordered_map<uint32_t,MatIdx> VertexPatch2RowMap;
	cList<VertexPatch2RowMap> vertpatch2rows(vertices.size());
	FOREACH(i, vertices) {
		const PatchIndex& patchIndex = patchIndices[i];  // 获取顶点所对应的patch
		VertexPatch2RowMap& vertpatch2row = vertpatch2rows[i];  // 第i个顶点所对应的patch在公式（3）中变量g里的位置
		if (patchIndex.bIndex) {  // 若顶点处于边界
			// vertex is part of multiple patches
			// 顶点是多个patch的一部分
			const SeamVertex& seamVertex = seamVertices[patchIndex.idxSeamVertex];  // 获取边界上的顶点对应的信息
			ASSERT(seamVertex.idxVertex == i);
			// 遍历边界点所对应的patch，并记录其在公式（3）中变量g里的位置
			for (const SeamVertex::Patch& patch: seamVertex.patches) {
				ASSERT(patch.idxPatch != numPatches);
				vertpatch2row[patch.idxPatch] = rowsX++;
			}
		} else
		if (patchIndex.idxPatch < numPatches) {  // 若顶点不在边界则只对应着一个patch
			// vertex is part of only one patch
			// 顶点是一个patch的一部分
			vertpatch2row[patchIndex.idxPatch] = rowsX++;
		}
	}
	// 参考论文Let There Be Color! Large-Scale Texturing of 3D Reconstructions公式2，3 min(g_t(A_t*A+Gamma_t*Gamma)g-2coeffB_t*A*g)+coeffB_t*coeffB
	// 求最小值，对g求导令其导数为0则：(A_t*A+Gamma_t*Gamma)g=A_t*coeffB
	// fill Tikhonov's Gamma matrix (regularization constraints)
	// 系数矩阵Gamma的构建，Gamma描述的是同一Patch内的顶点的调整量的差异
	const float lambda(0.1f);
	MatIdx rowsGamma(0);
	Mesh::VertexIdxArr adjVerts;
	CLISTDEF0(MatEntry) rows(0, vertices.size()*4);
	FOREACH(v, vertices) {  // 遍历顶点
		adjVerts.Empty();
		scene.mesh.GetAdjVertices(v, adjVerts);  // 获取v的邻域点（即one-ring邻域点），并记录在adjVerts中
		VertexPatchIterator itV(patchIndices[v], seamVertices);
		while (itV.Next()) {  // 依次处理顶点所对应的patch
			const uint32_t idxPatch(itV);  // 当前处理的patch的id
			if (idxPatch == numPatches)  // 由于patch是从0开始索引的，若idxPatch == numPatches，则表示当前顶点所对应的patch都考虑过了
				continue;
			const MatIdx col(vertpatch2rows[v].at(idxPatch));  // 获取顶点v在idxPatch对应的patch信息在公式（3）中变量g里行位置（或系数矩阵A或Gamma的列位置）
			for (const VIndex vAdj: adjVerts) {  // 遍历邻域点
				// 这是为了避免重复计算，因为此处考虑的点与邻域点之间的关系是对称的，也就是对应着无向边，若之前边所对应的邻域关系的相关信息已经计算过，就没有必要再进行计算。
				// 例如，若邻域对v0v1在当前顶点为v0时已经计算过，则当以v1作为当前顶点时，v0v1这个邻域对又会被考虑到，但没必要再次对其进行计算。
				if (v >= vAdj)
					continue;
				// 遍历邻域点对应的patch
				VertexPatchIterator itVAdj(patchIndices[vAdj], seamVertices);
				while (itVAdj.Next()) {
					const uint32_t idxPatchAdj(itVAdj);
					// 若当前点的patch与邻域点patch的id相同，则表明它们是同一个patch内的，
					// 存储对应的信息，rowsGamma表示在公式（3）中系数矩阵Gamma中的行数，每加入一对处于同一个patch内的点，rowsGamma就加1
					// 需要注意的是，对于边界处的顶点所构成的边（或邻域对）可能各对应着多个Patch，因此边、邻域对关于这些Patch的相关信息需要计算多遍，也就是此处的计算流程对于同一条边可能会经过多次计算（一般也就两次）
					if (idxPatch == idxPatchAdj) {
						const MatIdx colAdj(vertpatch2rows[vAdj].at(idxPatchAdj));
						// 此处之所以用lambda和-lambda，是为了构造论文中提及的目标函数中的优化项，(gvi-gvj)*lambda
						rows.emplace_back(rowsGamma, col, lambda);
						rows.emplace_back(rowsGamma, colAdj, -lambda);
						++rowsGamma;
					}
				}
			}
		}
	}
	ASSERT(rows.size()/2 < static_cast<IDX>(std::numeric_limits<MatIdx>::max()));

	SparseMat Gamma(rowsGamma, rowsX);  // 以稀疏矩阵的形式对Gamma矩阵进行存储
	Gamma.setFromTriplets(rows.Begin(), rows.End());
	rows.Empty();

	// fill the matrix A and the coefficients for the Vector b of the linear equation system
	// (A_t*A+Gamma_t*Gamma)g=A_t*coeffB
	// 计算A矩阵和b构建ax=b，其中，a = (A_t*A+Gamma_t*Gamma), b = A_t*coeffB
	// 系数矩阵A的构建，与系数矩阵Gamma的构建类似，A描述的是patch接缝处的节点在不同Patch对应的纹理块上的颜色差异（希望通过颜色调整量使其尽可能一致）
	IndexArr indices;
	Colors vertexColors;
	Colors coeffB;
	for (const SeamVertex& seamVertex: seamVertices) {
		if (seamVertex.patches.size() < 2)
			continue;
		seamVertex.SortByPatchIndex(indices);
		vertexColors.resize(indices.size());
		FOREACH(i, indices) {
			const SeamVertex::Patch& patch0 = seamVertex.patches[indices[i]];
			ASSERT(patch0.idxPatch < numPatches);
			SampleImage sampler(images[texturePatches[patch0.idxPatch].label].image);
			for (const SeamVertex::Patch::Edge& edge: patch0.edges) {
				const SeamVertex& seamVertex1 = seamVertices[edge.idxSeamVertex];
				const SeamVertex::Patches::IDX idxPatch1(seamVertex1.patches.Find(patch0.idxPatch));
				ASSERT(idxPatch1 != SeamVertex::Patches::NO_INDEX);
				const SeamVertex::Patch& patch1 = seamVertex1.patches[idxPatch1];
				sampler.AddEdge(patch0.proj, patch1.proj);
			}
			vertexColors[i] = sampler.GetColor();
		}
		const VertexPatch2RowMap& vertpatch2row = vertpatch2rows[seamVertex.idxVertex];
		for (IDX i=0; i<indices.size()-1; ++i) {
			const uint32_t idxPatch0(seamVertex.patches[indices[i]].idxPatch);
			const Color& color0 = vertexColors[i];
			const MatIdx col0(vertpatch2row.at(idxPatch0));
			for (IDX j=i+1; j<indices.size(); ++j) {
				const uint32_t idxPatch1(seamVertex.patches[indices[j]].idxPatch);
				const Color& color1 = vertexColors[j];
				const MatIdx col1(vertpatch2row.at(idxPatch1));
				ASSERT(idxPatch0 < idxPatch1);
				const MatIdx rowA((MatIdx)coeffB.size());
				coeffB.Insert(color1 - color0);
				ASSERT(ISFINITE(coeffB.back()));
				rows.emplace_back(rowA, col0,  1.f);
				rows.emplace_back(rowA, col1, -1.f);
			}
		}
	}
	ASSERT(coeffB.size() < static_cast<IDX>(std::numeric_limits<MatIdx>::max()));

	const MatIdx rowsA((MatIdx)coeffB.size());
	SparseMat A(rowsA, rowsX);
	A.setFromTriplets(rows.Begin(), rows.End());
	rows.Release();

	SparseMat Lhs(A.transpose() * A + Gamma.transpose() * Gamma);
	// CG uses only the lower triangle, so prune the rest and compress matrix
	// CG 仅使用下三角数据
	Lhs.prune([](const int& row, const int& col, const float&) -> bool {
		return col <= row;
	});

	// globally solve for the correction colors
	// 基于Eigen的共轭梯度法求解g
	Eigen::Matrix<float,Eigen::Dynamic,3,Eigen::RowMajor> colorAdjustments(rowsX, 3);  // 初始化要调整的颜色值，3表示RGB颜色通道，rowsX表示变量g的维数，也就是顶点的调整量的个数，因为g的每个分量都对应一个顶点在相应patch上的调整量
	{
		// init CG solver
		// 设置误差容忍度和迭代次数
		Eigen::ConjugateGradient<SparseMat, Eigen::Lower> solver;
		solver.setMaxIterations(1000);
		solver.setTolerance(0.0001f);
		solver.compute(Lhs);
		ASSERT(solver.info() == Eigen::Success);
		#ifdef TEXOPT_USE_OPENMP
		#pragma omp parallel for
		#endif
		for (int channel=0; channel<3; ++channel) {  // 逐个通道进行处理
			// init right hand side vector
			// 初始化向量b，即Rhs
			const Eigen::Map< Eigen::VectorXf, Eigen::Unaligned, Eigen::Stride<0,3> > b(coeffB.front().ptr()+channel, rowsA);
			const Eigen::VectorXf Rhs(SparseMat(A.transpose()) * b);
			// solve for x
			// 求解x
			const Eigen::VectorXf x(solver.solve(Rhs));
			ASSERT(solver.info() == Eigen::Success);
			// subtract mean since the system is under-constrained and
			// we need the solution with minimal adjustments
			// 减去均值是因为系统是无约束的。可能还是为了避免调整量计算过大，这个步骤和数据归一化很像。
			Eigen::Map< Eigen::VectorXf, Eigen::Unaligned, Eigen::Stride<0,3> >(colorAdjustments.data()+channel, rowsX) = x.array() - x.mean();
			DEBUG_LEVEL(3, "\tcolor channel %d: %d iterations, %g residual", channel, solver.iterations(), solver.error());
		}
	}

	// adjust texture patches using the correction colors
	// 调整patch
	#ifdef TEXOPT_USE_OPENMP
	#pragma omp parallel for schedule(dynamic)
	for (int i=0; i<(int)numPatches; ++i) {
	#else
	for (unsigned i=0; i<numPatches; ++i) {
	#endif
		const uint32_t idxPatch((uint32_t)i);
		TexturePatch& texturePatch = texturePatches[idxPatch];
		ColorMap imageAdj(texturePatch.rect.size());  // 用于存放通过插值获取的纹理块中的像素点的颜色调整量
		imageAdj.memset(0);
		// interpolate color adjustments over the whole patch
		// 整个patch插值调整颜色。也就是获得face对应的顶点的颜色调整量后，对面片中的位置进行插值（根据重心坐标进行插值,这种插值方式与线性插值很类似），获取face所覆盖的视图中每个像素的颜色调整量
		struct RasterPatch {
			const TexCoord* tri;
			Color colors[3];
			ColorMap& image;
			inline RasterPatch(ColorMap& _image) : image(_image) {}
			inline cv::Size Size() const { return image.size(); }
			inline void operator()(const ImageRef& pt, const Point3f& bary) {
				ASSERT(image.isInside(pt));
				image(pt) = colors[0]*bary.x + colors[1]*bary.y + colors[2]*bary.z;
			}
		} data(imageAdj);
		for (const FIndex idxFace: texturePatch.faces) {  // 遍历纹理块对应的face
			const Face& face = faces[idxFace];
			data.tri = faceTexcoords.Begin()+idxFace*3;  // face对应的纹理坐标
			for (int v=0; v<3; ++v)
				data.colors[v] = colorAdjustments.row(vertpatch2rows[face[v]].at(idxPatch));  // 记录调整后的颜色值（三角面片对应的三个顶点对应3个校正量）
			// render triangle and for each pixel interpolate the color adjustment
			// from the triangle corners using barycentric coordinates
			// 利用重心坐标插值三角内的像素颜色调整值。
			ColorMap::RasterizeTriangleBary(data.tri[0], data.tri[1], data.tri[2], data);
		}
		// dilate with one pixel width, in order to make sure patch border smooths out a little
		// 膨胀一个像素，确保patch边界平滑。
		imageAdj.DilateMean<1>(imageAdj, Color::ZERO);
		// apply color correction to the patch image
		cv::Mat image(images[texturePatch.label].image(texturePatch.rect));  // patch对应的图像
		for (int r=0; r<image.rows; ++r) {
			for (int c=0; c<image.cols; ++c) {
				const Color& a = imageAdj(r,c);  // 获取对应位置的颜色校正值
				if (a == Color::ZERO)
					continue;
				// 直接在image上对像素点v的颜色进行校正
				Pixel8U& v = image.at<Pixel8U>(r,c);
				const Color col(RGB2YCBCR(Color(v)));
				const Color acol(YCBCR2RGB(Color(col+a)));
				for (int p=0; p<3; ++p)
					v[p] = (uint8_t)CLAMP(ROUND2INT(acol[p]), 0, 255);
			}
		}
	}
}

// set to one in order to dilate also on the diagonal of the border
// 设置为1以便也在边界的对角线上扩展
// (normally not needed)
#define DILATE_EXTRA 0
/**
 * @brief 处理mask,局部融合只处理从mask边界往里stripWidth个像素的宽度（即是一个条状mask）
 * 
 * @param[in/out] mask  输入也是输出，是patch对应的mask
 * @param[in] stripWidth mask的宽度
 */
void MeshTexture::ProcessMask(Image8U& mask, int stripWidth)
{
	typedef Image8U::Type Type;

	// dilate and erode around the border,
	// in order to fill all gaps and remove outside pixels
	// 膨胀腐蚀边界。为了填充沟和移除外点像素，也就是通过先膨胀再腐蚀，将黑斑给滤除掉，并且保持边界不变
	// (due to imperfect overlay of the raster line border and raster faces)
	#define DILATEDIR(rd,cd) { \  // 膨胀操作，用于滤除小黑点
		Type& vi = mask(r+(rd),c+(cd)); \
		if (vi != border) \
			vi = interior; \
	}
	const int HalfSize(1);
	const int RowsEnd(mask.rows-HalfSize);
	const int ColsEnd(mask.cols-HalfSize);
	for (int r=HalfSize; r<RowsEnd; ++r) {
		for (int c=HalfSize; c<ColsEnd; ++c) {
			const Type v(mask(r,c));
			if (v != border)
				continue;
			#if DILATE_EXTRA
			for (int i=-HalfSize; i<=HalfSize; ++i) {
				const int rw(r+i);
				for (int j=-HalfSize; j<=HalfSize; ++j) {
					const int cw(c+j);
					Type& vi = mask(rw,cw);
					if (vi != border)
						vi = interior;
				}
			}
			#else
			DILATEDIR(-1, 0);
			DILATEDIR(1, 0);
			DILATEDIR(0, -1);
			DILATEDIR(0, 1);
			#endif
		}
	}
	#undef DILATEDIR
	#define ERODEDIR(rd,cd) { \  // 腐蚀操作，用于消除亮斑
		const int rl(r-(rd)), cl(c-(cd)), rr(r+(rd)), cr(c+(cd)); \
		const Type vl(mask.isInside(ImageRef(cl,rl)) ? mask(rl,cl) : uint8_t(empty)); \
		const Type vr(mask.isInside(ImageRef(cr,rr)) ? mask(rr,cr) : uint8_t(empty)); \
		if ((vl == border && vr == empty) || (vr == border && vl == empty)) { \
			v = empty; \
			continue; \
		} \
	}
	#if DILATE_EXTRA
	for (int i=0; i<2; ++i)
	#endif
	for (int r=0; r<mask.rows; ++r) {
		for (int c=0; c<mask.cols; ++c) {
			Type& v = mask(r,c);
			if (v != interior)
				continue;
			ERODEDIR(0, 1);
			ERODEDIR(1, 0);
			ERODEDIR(1, 1);
			ERODEDIR(-1, 1);
		}
	}
	#undef ERODEDIR

	// mark all interior pixels with empty neighbors as border
	// 标记所有内点中邻域是空的像素为边界，
	// 也就是将哪些存在数值为空的邻域点的内点标记为边界点
	for (int r=0; r<mask.rows; ++r) {
		for (int c=0; c<mask.cols; ++c) {
			Type& v = mask(r,c);
			if (v != interior)
				continue;
			if (mask(r-1,c) == empty ||
				mask(r,c-1) == empty ||
				mask(r+1,c) == empty ||
				mask(r,c+1) == empty)
				v = border;
		}
	}

	#if 0
	// mark all interior pixels with border neighbors on two sides as border
	{
	Image8U orgMask;
	mask.copyTo(orgMask);
	for (int r=0; r<mask.rows; ++r) {
		for (int c=0; c<mask.cols; ++c) {
			Type& v = mask(r,c);
			if (v != interior)
				continue;
			if ((orgMask(r+1,c+0) == border && orgMask(r+0,c+1) == border) ||
				(orgMask(r+1,c+0) == border && orgMask(r-0,c-1) == border) ||
				(orgMask(r-1,c-0) == border && orgMask(r+0,c+1) == border) ||
				(orgMask(r-1,c-0) == border && orgMask(r-0,c-1) == border))
				v = border;
		}
	}
	}
	#endif

	// compute the set of valid pixels at the border of the texture patch
	// 计算有效像素集合，即收集mask的边界像素
	#define ISEMPTY(mask, x,y) (mask(y,x) == empty)
	const int width(mask.width()), height(mask.height());
	typedef std::unordered_set<ImageRef> PixelSet;
	PixelSet borderPixels;
	for (int y=0; y<height; ++y) {
		for (int x=0; x<width; ++x) {
			if (ISEMPTY(mask, x,y))
				continue;
			// valid border pixels need no invalid neighbors
			// 有效的边界像素不需要无效的邻域，也就是说若mask四周上的像素其实就是边界像素，因此不需要根据邻域点进行判断
			if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {  // (x, y)处于mask的边界位置
				borderPixels.insert(ImageRef(x,y));
				continue;
			}
			// check the direct neighborhood of all invalid pixels
			// 检查所有无效像素的直接邻域（即四邻域），
			// 将哪些存在数值为空的邻域点的内点取出来作为边界点，并存储到borderPixels中
			for (int j=-1; j<=1; ++j) {
				for (int i=-1; i<=1; ++i) {
					// if the valid pixel has an invalid neighbor...
					const int xn(x+i), yn(y+j);
					// 判断(xn, yn)是否在图像内部，并且mask上的对应值是否为空
					if (ISINSIDE(xn, 0, width) &&
						ISINSIDE(yn, 0, height) &&
						ISEMPTY(mask, xn,yn)) {
						// add the pixel to the set of valid border pixels
						// 将该像素添加到有效的边界像素集
						borderPixels.insert(ImageRef(x,y));
						goto CONTINUELOOP;
					}
				}
			}
			CONTINUELOOP:;
		}
	}

	// iteratively erode all border pixels
	// 迭代腐蚀所有边界像素，也就是将需要的边界附近的像素设为空，即以mask的边界像素作为起点，向边界两边腐蚀stripWidth次
	{
	Image8U orgMask;
	mask.copyTo(orgMask);  // 将原本的mask信息保存在orgMask中，在对orgMask进行处理
	typedef std::vector<ImageRef> PixelVector;
	for (int s=0; s<stripWidth; ++s) {
		PixelVector emptyPixels(borderPixels.begin(), borderPixels.end());
		borderPixels.clear();
		// mark the new empty pixels as empty in the mask
		// 在mask中将新的空像素（即mask的上记录的边界像素）标记为空
		for (PixelVector::const_iterator it=emptyPixels.cbegin(); it!=emptyPixels.cend(); ++it)
			orgMask(*it) = empty;
		// find the set of valid pixels at the border of the valid area
		// 在有效区域的边界上找到有效像素集，也就是进行3*3的腐蚀
		for (PixelVector::const_iterator it=emptyPixels.cbegin(); it!=emptyPixels.cend(); ++it) {
			for (int j=-1; j<=1; j++) {
				for (int i=-1; i<=1; i++) {
					const int xn(it->x+i), yn(it->y+j);
					if (ISINSIDE(xn, 0, width) &&
						ISINSIDE(yn, 0, height) &&
						!ISEMPTY(orgMask, xn, yn))
						borderPixels.insert(ImageRef(xn,yn));
				}
			}
		}
	}
	#undef ISEMPTY

	// mark all remaining pixels empty in the mask
	// 将mask中剩余的像素标记为空，也就是记录处通过迭代腐蚀标记出的需要进行处理的边界附近的像素，这些部分在orgMask中标记为空，而非边界部分在orgMask中数值不为空，
	// 由于mask用于记录的是需要进行处理的边界附近的像素，因此需要将orgMask中不为空的位置处的像素值设为空，从而不对其进行处理
	// 由于此处是通过在orgMask上执行迭代腐蚀操作来获取边界附近的像素点，因此边界附近的像素点在orgMask上的数值都为空。
	// 之所以使用腐蚀来获取边界附近像素，是因为在图像上暗斑更为少见，根据有标识性、区别性。
	for (int y=0; y<height; ++y) {
		for (int x=0; x<width; ++x) {
			if (orgMask(y,x) != empty)
				mask(y,x) = empty;
		}
	}
	}

	// mark all border pixels，标记所有的边界像素，这个所谓的边界像素，其实是边界附近的像素所形成的带状区域的边界，因此需要给它们一个额外的边界标记
	for (PixelSet::const_iterator it=borderPixels.cbegin(); it!=borderPixels.cend(); ++it)
		mask(*it) = border;

	#if 0
	// dilate border
	// 边界膨胀
	{
	Image8U orgMask;
	mask.copyTo(orgMask);
	for (int r=HalfSize; r<RowsEnd; ++r) {
		for (int c=HalfSize; c<ColsEnd; ++c) {
			const Type v(orgMask(r, c));
			if (v != border)
				continue;
			for (int i=-HalfSize; i<=HalfSize; ++i) {
				const int rw(r+i);
				for (int j=-HalfSize; j<=HalfSize; ++j) {
					const int cw(c+j);
					Type& vi = mask(rw, cw);
					if (vi == empty)
						vi = border;
				}
			}
		}
	}
	}
	#endif
}

// 计算图像img在像素位置i处的拉普拉斯梯度
inline MeshTexture::Color ColorLaplacian(const Image32F3& img, int i) {
	const int width(img.width());
	return img(i-width) + img(i-1) + img(i+1) + img(i+width) - img(i)*4.f;
}

// 泊松融合参考博客介绍https://blog.csdn.net/hjimce/article/details/45716603  参考论文Poisson Image Editing
/**
 * @brief 泊松融合用来校正纹理接缝处点的颜色差异，泊松融合方程见课件推导
 * 泊松方程：Ax=b  A 是稀疏矩阵（对应边界的是1 对应内部是Laplace算子） x是mask对应的纹理校正后的颜色值 b是边界border颜色和中间点inter的散度值
 * @param[in] src 纹理块对应的图像
 * @param[in] dst 纹理块对应的图像边界颜色值已经处理过使用的接缝处边界是两边patch颜色均值，顶点是相关所有patch颜色均值
 * @param[in] mask 标记用来融合的纹理条包括内外边界border和内部inter。见课件图示
 * @param[in] bias 权重用来计算像素的Laplace时，控制src和dst占比，ColorLaplacian(src,i)*bias + ColorLaplacian(dst,i)*(1.f-bias))
 */
void MeshTexture::PoissonBlending(const Image32F3& src, Image32F3& dst, const Image8U& mask, float bias)
{
	ASSERT(src.width() == mask.width() && src.width() == dst.width());
	ASSERT(src.height() == mask.height() && src.height() == dst.height());
	ASSERT(src.channels() == 3 && dst.channels() == 3 && mask.channels() == 1);
	ASSERT(src.type() == CV_32FC3 && dst.type() == CV_32FC3 && mask.type() == CV_8U);

	#ifndef _RELEASE
	// check the mask border has no pixels marked as interior
	// 确认mask边界是否有像素被标记为内部，也就是检查mask是否有效
	for (int x=0; x<mask.cols; ++x)
		ASSERT(mask(0,x) != interior && mask(mask.rows-1,x) != interior);
	for (int y=0; y<mask.rows; ++y)
		ASSERT(mask(y,0) != interior && mask(y,mask.cols-1) != interior);
	#endif

	const int n(dst.area());  // 纹理块的面积，也就是长乘宽
	const int width(dst.width());

	// indices初始化
	TImage<MatIdx> indices(dst.size());
	indices.memset(0xff);
	// 泊松方程 Ax=b 
	MatIdx nnz(0);
	for (int i = 0; i < n; ++i)
		if (mask(i) != empty)
			indices(i) = nnz++;  // 对mask上要处理的边界附近的像素进行统计
	if (nnz <= 0)
		return;

	Colors coeffB(nnz);  // b
	CLISTDEF0(MatEntry) coeffA(0, nnz);  // A，其第i行对应着第i个像素处理前后的结果
	for (int i = 0; i < n; ++i) {
		switch (mask(i)) {
		case border: {  // 若像素点处于边界处，则颜色值不变，
			const MatIdx idx(indices(i));
			ASSERT(idx != -1);
			coeffA.emplace_back(idx, idx, 1.f);  // 即A[idx, :] = 0,  A[idx, idx] = 1
			// 边界处颜色不变直接使用之前计算的均值即dst对应的颜色
			coeffB[idx] = (const Color&)dst(i);
		} break;
		case interior: {  // 若像素点处于内部，则计算其拉普拉斯二阶梯度，即散度
			const MatIdx idxUp(indices(i - width));  // 上面的像素索引
			const MatIdx idxLeft(indices(i - 1));    // 左边的像素索引
			const MatIdx idxCenter(indices(i));      // 中心的像素索引
			const MatIdx idxRight(indices(i + 1));   // 右边的像素索引
			const MatIdx idxDown(indices(i + width));// 下面的像素索引
			// all indices should be either border conditions or part of the optimization
			// 所有索引应该是在边界或者内部待优化
			ASSERT(idxUp != -1 && idxLeft != -1 && idxCenter != -1 && idxRight != -1 && idxDown != -1);
			// [1,1,-4,1,1]Laplace算子，在系数矩阵coeffA中设置Laplace算子对应的权重参数部分
			// 即A[idxCenter, :] = 0, 
			// A[idxCenter, idxUp] = A[idxCenter, idxLeft] = A[idxCenter, idxRight] = A[idxCenter, idxDown] = 1,
			// A[idxCenter, idxCenter] = -4 
			coeffA.emplace_back(idxCenter, idxUp, 1.f);
			coeffA.emplace_back(idxCenter, idxLeft, 1.f);
			coeffA.emplace_back(idxCenter, idxCenter,-4.f);
			coeffA.emplace_back(idxCenter, idxRight, 1.f);
			coeffA.emplace_back(idxCenter, idxDown, 1.f);
			// set target coefficient
			// 内部点inter的散度计算，即Laplace算子使用，div(5)=[V(2)+V(4)+V(6)+V(8)]-4*V(5)
			// 之后通过权重bias将原图src与边界处理过的图dst进行融合，即bias*src+(1-bias)*dst
			coeffB[idxCenter] = (bias == 1.f ?
								 ColorLaplacian(src,i) :
								 ColorLaplacian(src,i)*bias + ColorLaplacian(dst,i)*(1.f-bias));
		} break;
		}
	}
	// 构建稀疏矩阵A
	SparseMat A(nnz, nnz);
	A.setFromTriplets(coeffA.Begin(), coeffA.End());  // 将coeffA中的数据复制给A
	coeffA.Release();
	// eigen稀疏矩阵求解
	#ifdef TEXOPT_SOLVER_SPARSELU
	// use SparseLU factorization
	// (faster, but not working if EIGEN_DEFAULT_TO_ROW_MAJOR is defined, bug inside Eigen)
	const Eigen::SparseLU< SparseMat, Eigen::COLAMDOrdering<MatIdx> > solver(A);
	#else
	// use BiCGSTAB solver
	const Eigen::BiCGSTAB< SparseMat, Eigen::IncompleteLUT<float> > solver(A);
	#endif
	ASSERT(solver.info() == Eigen::Success);
	for (int channel=0; channel<3; ++channel) {
		const Eigen::Map< Eigen::VectorXf, Eigen::Unaligned, Eigen::Stride<0,3> > b(coeffB.front().ptr()+channel, nnz);  // 将coeffB中的数据复制给b
		const Eigen::VectorXf x(solver.solve(b));  // 通过调用solver.solve()求解Ax=b，求解颜色校正值x
		ASSERT(solver.info() == Eigen::Success);
		for (int i = 0; i < n; ++i) {
			const MatIdx index(indices(i));  // 取像素在mask上的位置索引
			if (index != -1)
				dst(i)[channel] = x[index];  // 将校正后的颜色值应用在dst的相应像素位置上
		}
	}
}
/**
 * @brief 局部颜色校正：泊松融合  在纹理块边界处进行局部颜色校正，使得纹理块间过渡比较平滑。泊松纹理融合具体见课件
 * 
 */
void MeshTexture::LocalSeamLeveling()
{
	ASSERT(!seamVertices.empty());
	const unsigned numPatches(texturePatches.size()-1);  // 有效patch数目，此处之所以减1是因为texturePatches中最后一个对应的是无效patch

	// adjust texture patches locally, so that the border continues smoothly inside the patch
	// 局部调整纹理patch，使得边界平滑过渡到patch内部
	#ifdef TEXOPT_USE_OPENMP
	#pragma omp parallel for schedule(dynamic)
	for (int i=0; i<(int)numPatches; ++i) {
	#else
	for (unsigned i=0; i<numPatches; ++i) {  // 遍历patch
	#endif
		const uint32_t idxPatch((uint32_t)i);
		const TexturePatch& texturePatch = texturePatches[idxPatch];  // 获取patch
		// extract image
		// 取patch对应的image
		const Image8U3& image0(images[texturePatch.label].image);  // patch对应的图像
		Image32F3 image, imageOrg;
		image0(texturePatch.rect).convertTo(image, CV_32FC3, 1.0/255.0);
		image.copyTo(imageOrg);  // 将校正前的图像数据存放到imageOrg中，而image0中的图像数据后续会被进行局部纹理颜色校正
		// render patch coverage
		// 渲染patch对应的mask
		Image8U mask(image.size()); {
			mask.memset(0);
			struct RasterMesh {
				Image8U& image;
				inline void operator()(const ImageRef& pt) {
					ASSERT(image.isInside(pt));
					image(pt) = interior;
				}
			} data{mask};
			for (const FIndex idxFace: texturePatch.faces) {
				const TexCoord* tri = faceTexcoords.data()+idxFace*3;  // 获取patch块对应的纹理坐标，即对应面片的三个顶点在纹理图像中的位置
				ColorMap::RasterizeTriangle(tri[0], tri[1], tri[2], data);  // 进行栅格化插值，将mask所覆盖部分的插值结果存放到data中，其实就是获取patch所对应的边界及其内部区域并将其存放到mask中
			}
		}
		// render the patch border meeting neighbor patches
		// 渲染与邻域patch的边界
		const Sampler sampler;
		const TexCoord offset(texturePatch.rect.tl());  // patch相对在整个纹理图中的偏移量
		for (const SeamVertex& seamVertex0: seamVertices) {  // 处理接缝处的顶点seamVertex0
			if (seamVertex0.patches.size() < 2)  // 若seamVertex0所属的patch数目小于2，则直接跳过
				continue;
			// 若seamVertex0属于当前遍历的patch，则返回其在存储对应边界信息的结构体seamVertex0.patches中的索引，
			// 否则就跳过
			const uint32_t idxVertPatch0(seamVertex0.patches.Find(idxPatch));
			if (idxVertPatch0 == SeamVertex::Patches::NO_INDEX)
				continue;
			const SeamVertex::Patch& patch0 = seamVertex0.patches[idxVertPatch0];  // 取出存储边界信息的结构体中与当前patch相一致的patch，即patch0
			const TexCoord p0(patch0.proj-offset);  // 将投影坐标减去偏移量offset获取纹理坐标，该坐标以该patch的左上角点为坐标原点
			// for each edge of this vertex belonging to this patch...
			// 处理属于patch0的所有包含seamVertex0的edge，从而获取同处于当前patch且与seamVertex0共享一条边的另一个顶点
			for (const SeamVertex::Patch::Edge& edge0: patch0.edges) {
				// select the same edge leaving from the adjacent vertex
				// 选择远离相邻顶点的同一条边
				const SeamVertex& seamVertex1 = seamVertices[edge0.idxSeamVertex];  // 同处于当前patch且与seamVertex0共享一条边的另一个顶点seamVertex1
				const uint32_t idxVertPatch0Adj(seamVertex1.patches.Find(idxPatch));  // seamVertex1的是否属于当前patch，若属于则返回其在存储对应接缝信息的结构体seamVertex1.patches中的索引位置
				ASSERT(idxVertPatch0Adj != SeamVertex::Patches::NO_INDEX);
				const SeamVertex::Patch& patch0Adj = seamVertex1.patches[idxVertPatch0Adj];  // seamVertex1在其所对应且与当前patch一致的patch上的纹理坐标
				const TexCoord p0Adj(patch0Adj.proj-offset);  // 将投影坐标减去偏移量offset得到对应的纹理坐标，该坐标以该patch的左上角点为坐标原点
				// find the other patch sharing the same edge (edge with same adjacent vertex)
				// 找到共享同一个edge的其它patch，之后通过泊松融合对mask所对应的像素的颜色值进行处理。
				// 即对于patch边界的像素，即mask所覆盖部分的像素，其颜色值取左右相邻的两个patch的颜色均值。
				// 具体看课件
				FOREACH(idxVertPatch1, seamVertex0.patches) {
					if (idxVertPatch1 == idxVertPatch0)
						continue;
					const SeamVertex::Patch& patch1 = seamVertex0.patches[idxVertPatch1];
					const uint32_t idxEdge1(patch1.edges.Find(edge0.idxSeamVertex));
					if (idxEdge1 == SeamVertex::Patch::Edges::NO_INDEX)
						continue;
					// pi不用减去对应patch的偏移量原因是p1对应的就是view原图image1  而p0的imageOrg取得原图中的对应纹理块，所以坐标要减去纹理块起始坐标
					const TexCoord& p1(patch1.proj);
					// select the same edge belonging to the second patch leaving from the adjacent vertex
					const uint32_t idxVertPatch1Adj(seamVertex1.patches.Find(patch1.idxPatch));
					ASSERT(idxVertPatch1Adj != SeamVertex::Patches::NO_INDEX);
					const SeamVertex::Patch& patch1Adj = seamVertex1.patches[idxVertPatch1Adj];
					const TexCoord& p1Adj(patch1Adj.proj);
					// this is an edge separating two (valid) patches;
					// draw it on this patch as the mean color of the two patches
					// edge 分开了两个patch，计算edge在两个patch的平均值（边界条件）
					const Image8U3& image1(images[texturePatches[patch1.idxPatch].label].image);
					struct RasterPatch {
						Image32F3& image;
						Image8U& mask;
						const Image32F3& image0;
						const Image8U3& image1;
						const TexCoord p0, p0Dir;
						const TexCoord p1, p1Dir;
						const float length;
						const Sampler sampler;
						inline RasterPatch(Image32F3& _image, Image8U& _mask, const Image32F3& _image0, const Image8U3& _image1,
							const TexCoord& _p0, const TexCoord& _p0Adj, const TexCoord& _p1, const TexCoord& _p1Adj)
							: image(_image), mask(_mask), image0(_image0), image1(_image1),
							p0(_p0), p0Dir(_p0Adj-_p0), p1(_p1), p1Dir(_p1Adj-_p1), length((float)norm(p0Dir)), sampler() {}
						inline void operator()(const ImageRef& pt) {
							const float l((float)norm(TexCoord(pt)-p0)/length);
							// compute mean color
							const TexCoord samplePos0(p0 + p0Dir * l);
							const Color color0(image0.sample<Sampler,Color>(sampler, samplePos0));
							const TexCoord samplePos1(p1 + p1Dir * l);
							const Color color1(image1.sample<Sampler,Color>(sampler, samplePos1)/255.f);
							image(pt) = Color((color0 + color1) * 0.5f);
							// set mask edge also
							mask(pt) = border;
						}
					} data(image, mask, imageOrg, image1, p0, p0Adj, p1, p1Adj);
					Image32F3::DrawLine(p0, p0Adj, data);  // 利用DrawLine将mask所对应的两个patch之间接缝处上的像素进行颜色校正，即将该像素的颜色值设为其在两个patch(p0Adj以及data)上的颜色值的平均
					// skip remaining patches,
					// as a manifold edge is shared by maximum two face (one in each patch), which we found already
					break;
				}
			}
			// render the vertex at the patch border meeting neighbor patches
			// 渲染patch边界的顶点，计算所有包含该顶点patch，计算颜色均值
			AccumColor accumColor;
			// for each patch...
			for (const SeamVertex::Patch& patch: seamVertex0.patches) {  // 遍历seamVertex0所属的patch
				// add its view to the vertex mean color
				// 将邻接patch view颜色值加入均值计算中
				const Image8U3& img(images[texturePatches[patch.idxPatch].label].image);
				accumColor.Add(img.sample<Sampler,Color>(sampler, patch.proj)/255.f, 1.f);  // 通过.sample()方法以及获取边界点在纹理图像上的颜色值，并累加到accumColor中
			}
			const ImageRef pt(ROUND2INT(patch0.proj-offset));
			image(pt) = accumColor.Normalized();  // 通过.Normalized()方法求均值，并将结果存放在image里的对应位置
			mask(pt) = border;  // 在mask中将相应像素标记为边界，也就是在mask中记录哪些点事边界区域的顶点
		}
		// make sure the border is continuous and
		// keep only the exterior tripe of the given size
		// 确保边界是连续的，因为patch全局已经调整过了，所以局部只调整patch边界向里20个像素构成的边界带见论文Let There Be Color!中fig.5
		// 将边界所覆盖的区域中，非边界部分（即patch边界向里超过20个像素的部分所构成的部分）的调整值都设为无效值
		ProcessMask(mask, 20);
		// compute texture patch blending
		// 泊松融合
		PoissonBlending(imageOrg, image, mask);
		// apply color correction to the patch image
		// 应用校正的颜色到patch中
		cv::Mat imagePatch(image0(texturePatch.rect));
		for (int r=0; r<image.rows; ++r) {
			for (int c=0; c<image.cols; ++c) {
				if (mask(r,c) == empty)  // 若位置(r, c)处的像素不为边界区域或边界区域的顶点，则跳过
					continue;
				const Color& a = image(r,c);  // 获取image中位置(r, c)处的校正后的像素值
				Pixel8U& v = imagePatch.at<Pixel8U>(r,c);  // 获取纹理块在位置(r, c)处的颜色值
				for (int p=0; p<3; ++p)
					v[p] = (uint8_t)CLAMP(ROUND2INT(a[p]*255.f), 0, 255);  // 用a对v进行调整
			}
		}
	}
}
/**
 * @brief 生成纹理图
 * 
 * @param[in] bGlobalSeamLeveling  控制全局颜色校正的开关
 * @param[in] bLocalSeamLeveling   控制局部颜色校正的开关
 * @param[in] nTextureSizeMultiple 
 * @param[in] nRectPackingHeuristic 
 * @param[in] colEmpty             rgb颜色值用来填充纹理图上空缺部分的颜色
 */
void MeshTexture::GenerateTexture(bool bGlobalSeamLeveling, bool bLocalSeamLeveling, unsigned nTextureSizeMultiple, unsigned nRectPackingHeuristic, Pixel8U colEmpty, float fSharpnessWeight, int maxTextureSize)
{
	// project patches in the corresponding view and compute texture-coordinates and bounding-box
	// Step 1 投影patch到对应view上计算纹理坐标和包围盒。相当于是数据准备工作
	const int border(2);  // 纹理图的预留边界
	faceTexcoords.resize(faces.size()*3);  // faceTexcoords用于存储纹理坐标（属于像素坐标），纹理坐标size是faces的3倍，因为记录的是face的三个顶点的纹理坐标
	faceTexindices.resize(faces.size());
	#ifdef TEXOPT_USE_OPENMP
	const unsigned numPatches(texturePatches.size()-1);  // 减一是去掉最后一个没有label的patch
	#pragma omp parallel for schedule(dynamic)
	for (int_t idx=0; idx<(int_t)numPatches; ++idx) {
		TexturePatch& texturePatch = texturePatches[(uint32_t)idx];
	#else
	for (TexturePatch *pTexturePatch=texturePatches.Begin(), *pTexturePatchEnd=texturePatches.End()-1; pTexturePatch<pTexturePatchEnd; ++pTexturePatch) {  // 对texturePatches进行遍历，其中存放着每个视图所对应的face以及对应的图像索引
		TexturePatch& texturePatch = *pTexturePatch;
	#endif
		const Image& imageData = images[texturePatch.label];  // texturePatch.label表示纹理块所对应的图像ID，imageData表示纹理块所对应的图像数据
		// project vertices and compute bounding-box
		// 投影顶点，计算在图像上的所有投影点的包围盒
		AABB2f aabb(true);  // 用来计算纹理坐标包围盒
		for (const FIndex idxFace: texturePatch.faces) {  // 对纹理块所对应的face进行遍历
			const Face& face = faces[idxFace];
			// 指针操作访问每个face的纹理坐标地址
			TexCoord* texcoords = faceTexcoords.data()+idxFace*3;  // 因为每个面片包含3个顶点，在中对于属于同一个面片的顶点所对应的纹理坐标存储在一起，因此才进行乘3
			for (int i=0; i<3; ++i) {
				texcoords[i] = imageData.camera.ProjectPointP(vertices[face[i]]);  // 将面片的顶点投影到像素平面上，从而得到对应的纹理坐标
				ASSERT(imageData.image.isInsideWithBorder(texcoords[i], border));
				aabb.InsertFull(texcoords[i]);  // 对纹理坐标进行记录，以后续计算出一个面片在图像上所覆盖的区域的外接矩形框，这个矩形区域即为纹理块
			}
		}
		// compute relative texture coordinates
		// 计算相对纹理坐标：上面patch投影得到一个纹理块aabb取其坐标最大最小，将其放在最终的纹理图上，会相对在原图位置有一个偏移量offset
		ASSERT(imageData.image.isInside(Point2f(aabb.ptMin)));
		ASSERT(imageData.image.isInside(Point2f(aabb.ptMax)));
		// 计算纹理块最大包围矩形 rect.xy是纹理块在纹理图上的起始坐标。此处的borer是外接矩形的预留边界，也就是说包含预留边界的外接矩形框相比于原本的外接矩形框，其包含的范围在行方向和列方向上都多了2*border行、2*border列
		// 注意，此处纹理块预留边界是很重要的，若不预留边界（即将border设为0），最终渲染产生的纹理模型上是存在接缝（一般为黑色），而这个接缝在实际拍摄的场景中是不存在的，拍摄到的视图中也不存在。
		// 而预留边界之后，就不会出现这个现象。这可能是因为在渲染的时候，若不预留边界，在一些基于窗口处理时就会卡到纹理块边界，即部分窗口包含了纹理块之外的部分，这些部分一般表示为0值，进而引入黑边，
		// 进而导致这些纹理块所对应的face之间有裂开的视觉感受，这种裂开的视觉观感就是不存在的接缝，又被称为伪影干扰
		texturePatch.rect.x = FLOOR2INT(aabb.ptMin[0])-border;
		texturePatch.rect.y = FLOOR2INT(aabb.ptMin[1])-border;
		// 计算长和宽，加了两倍的border宽度，即取纹理块时多取一2个像素的宽度。原因是如果不在原图上多取一点，三维模型显示时纹理会有
		// 黑色接缝（实际是没有的只是渲染的时候如果刚好在边界会出现这种现象，所以一般会多取一点），大家可以自己设置为0看下实验效果
		texturePatch.rect.width = CEIL2INT(aabb.ptMax[0]-aabb.ptMin[0])+border*2;
		texturePatch.rect.height = CEIL2INT(aabb.ptMax[1]-aabb.ptMin[1])+border*2;
		ASSERT(imageData.image.isInside(texturePatch.rect.tl()));
		ASSERT(imageData.image.isInside(texturePatch.rect.br()));
		// rect.tl(top/left即左上角)指的是纹理块在原始图像中的起始坐标，也就是相对原点(0, 0)的偏移量
		const TexCoord offset(texturePatch.rect.tl());
		for (const FIndex idxFace: texturePatch.faces) {
			TexCoord* texcoords = faceTexcoords.data()+idxFace*3;
			// 因为刚计算的纹理坐标还是相对于原图的，所以需要减去起始点，让纹理块坐标从（0，0）点开始。不过这个偏移量offset也会被记录下来
			for (int v=0; v<3; ++v)
				texcoords[v] -= offset;
		}
	}
	{
		// init last patch to point to a small uniform color patch
		// 初始化最后一个无label的patch（也就是不知道该patch、face对应哪一张图像）将其纹理坐标设为一个固定的值，指向一个相同颜色。
		// 所以没有标签对应的mesh上的face的颜色都是统一的背景色
		TexturePatch& texturePatch = texturePatches.back();
		const int sizePatch(border*2+1);
		texturePatch.rect = cv::Rect(0,0, sizePatch,sizePatch);
		for (const FIndex idxFace: texturePatch.faces) {
			TexCoord* texcoords = faceTexcoords.data()+idxFace*3;
			for (int i=0; i<3; ++i)
				texcoords[i] = TexCoord(0.5f, 0.5f);
		}
	}

	// perform seam leveling
	// Step 2 处理纹理接缝，主要是首先进行全局颜色校正，然后在纹理块交界处进行泊松融合消除纹理块交界处的颜色差异。相当于进行全局和局部融合。
	if (texturePatches.size() > 2 && (bGlobalSeamLeveling || bLocalSeamLeveling)) {
		// create seam vertices and edges
		// 创建不同纹理块间连接处的顶点和边。因为全局颜色校正主要是对接缝处的顶点进行处理，因此需要准备好接缝处的信息
		CreateSeamVertices();

		// perform global seam leveling
		// 全局颜色校正
		if (bGlobalSeamLeveling) {
			TD_TIMER_STARTD();
			GlobalSeamLeveling();
			DEBUG_ULTIMATE("\tglobal seam leveling completed (%s)", TD_TIMER_GET_FMT().c_str());
		}

		// perform local seam leveling
		// 局部颜色校正
		if (bLocalSeamLeveling) {
			TD_TIMER_STARTD();
			LocalSeamLeveling();
			DEBUG_ULTIMATE("\tlocal seam leveling completed (%s)", TD_TIMER_GET_FMT().c_str());
		}
	}

	// merge texture patches with overlapping rectangles
	// Step 3 合并纹理块：如果两个纹理块label相同，且小的包含在大的里面则合并。合并小的纹理块，移除无效的纹理块
	for (unsigned i=0; i<texturePatches.size()-1; ++i) {
		TexturePatch& texturePatchBig = texturePatches[i];
		for (unsigned j=1; j<texturePatches.size(); ++j) {
			if (i == j)
				continue;
			TexturePatch& texturePatchSmall = texturePatches[j];
			// 如果label不同，不能合并
			if (texturePatchBig.label != texturePatchSmall.label)
				continue;
			// 如果小patch不被包含在大patch里也不能合并
			if (!RectsBinPack::IsContainedIn(texturePatchSmall.rect, texturePatchBig.rect))
				continue;
			// translate texture coordinates
			// 变换合并后，计算小patch对应的新的纹理坐标。
			const TexCoord offset(texturePatchSmall.rect.tl()-texturePatchBig.rect.tl());  // 计算两个patch在原始图像中起始点相对偏移量
			for (const FIndex idxFace: texturePatchSmall.faces) {
				TexCoord* texcoords = faceTexcoords.data()+idxFace*3;
				// 将小patch的纹理坐标变换到大path的纹理坐标系中
				for (int v=0; v<3; ++v)
					texcoords[v] += offset;
			}
			// join faces lists
			// 将小patch的faces合并到大的里面
			texturePatchBig.faces.JoinRemove(texturePatchSmall.faces);
			// remove the small patch
			// 在texturePatches移除小path
			texturePatches.RemoveAtMove(j--);
		}
	}

	// Step 4 create texture 创建纹理图,将来自于多个图像的每个纹理块集中到同一张纹理图中，更新新的纹理坐标，以方便索引。纹理坐标和顶点面信息会保存在obj中
	// 纹理图会保存在jpg/png等。mtl存储了一些贴图的材质信息具体见课件介绍，mtl是obj的存储文件格式之一，常见的obj的存储文件格式有obj,mtl,jpg。
	// ply存储点云和图像信息
	// 其中若采用obj存储的纹理贴图结果中，则vt表示纹理坐标、vn是法线、v表示顶点、f表示face，而ply文件存储的纹理贴图结果中，则全靠头文件（即.ply文件中一开始的几行描述信息来表示数据的具体存储字段、存储格式、数据类型）来记录这些信息
	// 此外，用.ply存储结果的话，纹理图会以png的格式存储，这两个文件一定要放在同一个目录下，因为.ply只记录该纹理图的名称，而没有纹理图所处的地址，所以若要使用纹理图时只会在.ply文件所处的目录中去找。
	// 若使用.obj存储，则会产生.obj（存放.mtl文件名以及vt、vn、v、f等信息），.obj.mtl（存放光照材质参数以及对应的纹理图名称），以及.png（存放纹理图）这三个文件，这三个文件要放在同一目录下
	// 最终保存的纹理贴图结果中除了纹理图外，还包括纹理块在纹理图上的纹理坐标以及mesh
	// 值得注意的是：这里我们会发现每个纹理块大小是不一样的所以最终看到的纹理图会有很多大小不同的纹理块
	{
		// arrange texture patches to fit the smallest possible texture image
		// 排列纹理块以组合最小尺寸纹理图像
		// 计算纹理图的大小，使其能够包含所有的纹理快
		RectsBinPack::RectWIdxArr unplacedRects(texturePatches.size());
		FOREACH(i, texturePatches) {
			if (maxTextureSize > 0 && (texturePatches[i].rect.width > maxTextureSize || texturePatches[i].rect.height > maxTextureSize)) {
			    DEBUG("error: a patch of size %u x %u does not fit the texture", texturePatches[i].rect.width, texturePatches[i].rect.height);
			    ABORT("the maximum texture size chosen cannot fit a patch");
			}
			unplacedRects[i] = {texturePatches[i].rect, i};
		}

		// pack patches: one pack per texture file
		CLISTDEF2IDX(RectsBinPack::RectWIdxArr, TexIndex) placedRects; {
			// increase texture size till all patches fit
			// 增加纹理图size,直到所有纹理块都包含进去，并且将纹理块从小到大进行排序，再依次存储到纹理图中。此外，将纹理块放入纹理图中时还对其进行旋转，以节省纹理图的大小、存储空间
			// 并且获取纹理块在纹理图中的纹理坐标位置
			// 具体实现可以参考装箱问题
			const unsigned typeRectsBinPack(nRectPackingHeuristic/100);
			const unsigned typeSplit((nRectPackingHeuristic-typeRectsBinPack*100)/10);
			const unsigned typeHeuristic(nRectPackingHeuristic%10);
			int textureSize = 0;
			while (!unplacedRects.empty()) {
				TD_TIMER_STARTD();
				if (textureSize == 0) {
					textureSize = RectsBinPack::ComputeTextureSize(unplacedRects, nTextureSizeMultiple);
					if (maxTextureSize > 0 && textureSize > maxTextureSize)
						textureSize = maxTextureSize;
				}

				RectsBinPack::RectWIdxArr newPlacedRects;
				switch (typeRectsBinPack) {
				case 0: {
					MaxRectsBinPack pack(textureSize, textureSize);
					newPlacedRects = pack.Insert(unplacedRects, (MaxRectsBinPack::FreeRectChoiceHeuristic)typeHeuristic);
					break; }
				case 1: {
					SkylineBinPack pack(textureSize, textureSize, typeSplit!=0);
					newPlacedRects = pack.Insert(unplacedRects, (SkylineBinPack::LevelChoiceHeuristic)typeHeuristic);
					break; }
				case 2: {
					GuillotineBinPack pack(textureSize, textureSize);
					newPlacedRects = pack.Insert(unplacedRects, false, (GuillotineBinPack::FreeRectChoiceHeuristic)typeHeuristic, (GuillotineBinPack::GuillotineSplitHeuristic)typeSplit);
					break; }
				default:
					ABORT("error: unknown RectsBinPack type");
				}
				DEBUG_ULTIMATE("\tpacking texture completed: %u initial patches, %u placed patches, %u texture-size, %u textures (%s)", texturePatches.size(), newPlacedRects.size(), textureSize, placedRects.size(), TD_TIMER_GET_FMT().c_str());

				if (textureSize == maxTextureSize || unplacedRects.empty()) {
					// create texture image，创建纹理图texturesDiffuse，其大小为textureSize*textureSize
					placedRects.emplace_back(std::move(newPlacedRects));
					texturesDiffuse.emplace_back(textureSize, textureSize).setTo(cv::Scalar(colEmpty.b, colEmpty.g, colEmpty.r));  // 纹理图中纹理块的初始化，一开始将对应位置的颜色值设为colEmpty，表示空的颜色值，即纹理图的背景色、底色
					textureSize = 0;
				} else {
					// try again with a bigger texture
					textureSize *= 2;
					if (maxTextureSize > 0)
						textureSize = std::max(textureSize, maxTextureSize);
					unplacedRects.JoinRemove(newPlacedRects);
				}
			}
		}

		#ifdef TEXOPT_USE_OPENMP
		#pragma omp parallel for schedule(dynamic)
		for (int_t i=0; i<(int_t)placedRects.size(); ++i) {
			for (int_t j=0; j<(int_t)placedRects[(TexIndex)i].size(); ++j) {
				const TexIndex idxTexture((TexIndex)i);
				const uint32_t idxPlacedPatch((uint32_t)j);
		#else
		FOREACH(idxTexture, placedRects) {  // 遍历所有的patch
			FOREACH(idxPlacedPatch, placedRects[idxTexture]) {
		#endif
				const TexturePatch& texturePatch = texturePatches[placedRects[idxTexture][idxPlacedPatch].patchIdx];
				const RectsBinPack::Rect& rect = placedRects[idxTexture][idxPlacedPatch].rect;  // 排序之后的patch在纹理图中对应的矩形位置
				// copy patch image，赋值patch图像
				ASSERT((rect.width == texturePatch.rect.width && rect.height == texturePatch.rect.height) ||
					(rect.height == texturePatch.rect.width && rect.width == texturePatch.rect.height));
				int x(0), y(1);
				if (texturePatch.label != NO_ID) {
					const Image& imageData = images[texturePatch.label];
					cv::Mat patch(imageData.image(texturePatch.rect));
					if (rect.width != texturePatch.rect.width) {  // 这说明进行过转置，因为纹理图中存放的纹理块都是一致的宽度大于高度，若不满足则会先对其进行转置，再进行存放
						// flip patch and texture-coordinates
						// 转置patch和对应纹理坐标
						patch = patch.t();
						x = 1; y = 0;
					}
					patch.copyTo(texturesDiffuse[idxTexture](rect));  // 将纹理块存放到纹理图的对应矩形区域上
				}
				// compute final texture coordinates
				// 计算最终的纹理坐标，并存储到faceTexcoords中
				const TexCoord offset(rect.tl());  // 纹理块在纹理图上的左上角点相对于纹理图原点的偏移量
				for (const FIndex idxFace: texturePatch.faces) {
					TexCoord* texcoords = faceTexcoords.data()+idxFace*3;
					faceTexindices[idxFace] = idxTexture;
					for (int v=0; v<3; ++v) {
						TexCoord& texcoord = texcoords[v];
						// translate
						// 纹理坐标变换
						texcoord = TexCoord(
							texcoord[x]+offset.x,
							texcoord[y]+offset.y
						);
					}
				}
			}
		}
		if (texturesDiffuse.size() == 1)
			faceTexindices.Release();
		// apply some sharpening
		if (fSharpnessWeight > 0) {
			constexpr double sigma = 1.5;
			for (auto &textureDiffuse: texturesDiffuse) {
			    Image8U3 blurryTextureDiffuse;
			    cv::GaussianBlur(textureDiffuse, blurryTextureDiffuse, cv::Size(), sigma);
			    cv::addWeighted(textureDiffuse, 1+fSharpnessWeight, blurryTextureDiffuse, -fSharpnessWeight, 0, textureDiffuse);
			}
		}
	}
}

// texture mesh
//  - minCommonCameras: generate texture patches using virtual faces composed of coplanar triangles sharing at least this number of views (0 - disabled, 3 - good value)
//  - fSharpnessWeight: sharpness weight to be applied on the texture (0 - disabled, 0.5 - good value)
//  - nIgnoreMaskLabel: label value to ignore in the image mask, stored in the MVS scene or next to each image with '.mask.png' extension (-1 - auto estimate mask for lens distortion, -2 - disabled)
/**
 * @brief 纹理贴图，首先给每个face（三角网格）选择一个视图（图像），然后生成纹理块（texture patch）。由于不同纹理块来自不同的
 *        图像故光照角度不同，所以不同patch间会有颜色差异，因此需要进行颜色校正：先全局校正整体颜色差异（globel）再局部调整接缝处的颜色差
 *        异(local seam leveling)
 * @param[in] nResolutionLevel     scale用于计算纹理贴图的图像分辨率 =image_size/2^nResolutionLevel
 * @param[in] nMinResolution       贴图最小分辨率阈值，与上述分辨率相比取最大值
 * @param[in] fOutlierThreshold    颜色差异阈值，用于face选择投影的view时，剔除与大多view不同的外点view
 * @param[in] fRatioDataSmoothness 平滑系数
 * @param[in] bGlobalSeamLeveling  控制是否全局纹理融合，bool型
 * @param[in] bLocalSeamLeveling   控制是否局部纹理融合，bool型
 * @param[in] nTextureSizeMultiple 
 * @param[in] nRectPackingHeuristic 
 * @param[in] colEmpty             rgb颜色值用来填充纹理图上空缺部分的颜色
 * @return true 
 * @return false 
 * 参考论文：Let There Be Color! Large-Scale Texturing of 3D Reconstructions
 */
bool Scene::TextureMesh(unsigned nResolutionLevel, unsigned nMinResolution, unsigned minCommonCameras, float fOutlierThreshold, float fRatioDataSmoothness,
	bool bGlobalSeamLeveling, bool bLocalSeamLeveling, unsigned nTextureSizeMultiple, unsigned nRectPackingHeuristic, Pixel8U colEmpty, float fSharpnessWeight,
	int nIgnoreMaskLabel, int maxTextureSize, const IIndexArr& views)
{
	MeshTexture texture(*this, nResolutionLevel, nMinResolution);

	// assign the best view to each face
	// Step 1 给每个face（三角网格）分配最佳视图view 
	{
		TD_TIMER_STARTD();
		if (!texture.FaceViewSelection(minCommonCameras, fOutlierThreshold, fRatioDataSmoothness, nIgnoreMaskLabel, views))
			return false;
		DEBUG_EXTRA("Assigning the best view to each face completed: %u faces (%s)", mesh.faces.size(), TD_TIMER_GET_FMT().c_str());
	}

	// generate the texture image and atlas
	// Step 2 生成纹理图像，并进行纹理颜色校正与融合
	{
		TD_TIMER_STARTD();
		texture.GenerateTexture(bGlobalSeamLeveling, bLocalSeamLeveling, nTextureSizeMultiple, nRectPackingHeuristic, colEmpty, fSharpnessWeight, maxTextureSize);
		DEBUG_EXTRA("Generating texture atlas and image completed: %u patches, %u image size, %u textures (%s)", texture.texturePatches.size(), mesh.texturesDiffuse[0].width(), mesh.texturesDiffuse.size(), TD_TIMER_GET_FMT().c_str());
	}

	return true;
} // TextureMesh
/*----------------------------------------------------------------*/
