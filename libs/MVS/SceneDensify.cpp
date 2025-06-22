/*
* SceneDensify.cpp
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
#include "SceneDensify.h"
#include "PatchMatchCUDA.h"
// MRF: view selection
#include "../Math/TRWS/MRFEnergy.h"

using namespace MVS;


// D E F I N E S ///////////////////////////////////////////////////

// uncomment to enable multi-threading based on OpenMP
#ifdef _USE_OPENMP
#define DENSE_USE_OPENMP
#endif


// S T R U C T S ///////////////////////////////////////////////////

// Dense3D data.events
enum EVENT_TYPE {
	EVT_FAIL = 0,  // 事件失败
	EVT_CLOSE,     // 事件关闭

	EVT_PROCESSIMAGE,  // 图像预处理

	EVT_ESTIMATEDEPTHMAP,  // 深度图重建
	EVT_OPTIMIZEDEPTHMAP,  // 深度图优化
	EVT_SAVEDEPTHMAP,      // 深度图保存

	EVT_FILTERDEPTHMAP,    // 深度图滤波
	EVT_ADJUSTDEPTHMAP,    // 深度图调整
};

// 通过下面的类来调用作者实现的事件
class EVTFail : public Event
{
public:
	EVTFail() : Event(EVT_FAIL) {}
};
class EVTClose : public Event
{
public:
	EVTClose() : Event(EVT_CLOSE) {}
};

class EVTProcessImage : public Event
{
public:
	IIndex idxImage;
	EVTProcessImage(IIndex _idxImage) : Event(EVT_PROCESSIMAGE), idxImage(_idxImage) {}
};

class EVTEstimateDepthMap : public Event
{
public:
	IIndex idxImage;
	EVTEstimateDepthMap(IIndex _idxImage) : Event(EVT_ESTIMATEDEPTHMAP), idxImage(_idxImage) {}
};
class EVTOptimizeDepthMap : public Event
{
public:
	IIndex idxImage;
	EVTOptimizeDepthMap(IIndex _idxImage) : Event(EVT_OPTIMIZEDEPTHMAP), idxImage(_idxImage) {}
};
class EVTSaveDepthMap : public Event
{
public:
	IIndex idxImage;
	EVTSaveDepthMap(IIndex _idxImage) : Event(EVT_SAVEDEPTHMAP), idxImage(_idxImage) {}
};

class EVTFilterDepthMap : public Event
{
public:
	IIndex idxImage;
	EVTFilterDepthMap(IIndex _idxImage) : Event(EVT_FILTERDEPTHMAP), idxImage(_idxImage) {}
};
class EVTAdjustDepthMap : public Event
{
public:
	IIndex idxImage;
	EVTAdjustDepthMap(IIndex _idxImage) : Event(EVT_ADJUSTDEPTHMAP), idxImage(_idxImage) {}
};
/*----------------------------------------------------------------*/


// convert the ZNCC score to a weight used to average the fused points
inline float Conf2Weight(float conf, Depth depth) {
	return 1.f/(MAXF(1.f-conf,0.03f)*depth*depth);
}
/*----------------------------------------------------------------*/


// S T R U C T S ///////////////////////////////////////////////////


DepthMapsData::DepthMapsData(Scene& _scene)
	:
	scene(_scene),
	arrDepthData(_scene.images.GetSize())
{
} // constructor

DepthMapsData::~DepthMapsData()
{
} // destructor
/*----------------------------------------------------------------*/


// globally choose the best target view for each image,
// trying in the same time the selected image pairs to cover the whole scene;
// the map of selected neighbors for each image is returned in neighborsMap.
// For each view a list of neighbor views ordered by number of shared sparse points and overlapped image area is given.
// Next a graph is formed such that the vertices are the views and two vertices are connected by an edge if the two views have each other as neighbors.
// For each vertex, a list of possible labels is created using the list of neighbor views and scored accordingly (the score is normalized by the average score).
// For each existing edge, the score is defined such that pairing the same two views for any two vertices is discouraged (a constant high penalty is applied for such edges).
// This primal-dual defined problem, even if NP hard, can be solved by a Belief Propagation like algorithm, obtaining in general a solution close enough to optimality.
// 为每个image在全局中选择一个最优的target view 保证所选的所有图像对能覆盖整个场景。

// Step 2_2 从reference image 选的有效的邻域views中选取一个最佳邻域用来计算depth
/**
 * @brief 每张图像都选出nMaxViews个邻域帧，
 * 		  我们现在需要给每张图像选择一个最佳邻域（最简单的方式就是直接用分数最大的邻域帧作为目标帧，
 *        但这种选择方式仅仅是从每一帧的角度进行的，不是一种全局的选择方式，因为某一帧可能会成为多个图像帧的分数最大的邻域帧，
 *        所以仅仅考虑分数最大，难以做到场景的全覆盖，这对场景重建的整体效果不好）来求解深度图（即立体匹配，也就是视差图计算）
 *        这个就是马尔科夫随机场的labeling问题（能量优化，一种全局优化的方法，此处的labeling，即标签指的是每帧的邻域帧的id，
 * 		  节点即为当前帧，如何为每个节点选择一个合适的标签，使得整体的代价最小）。
 *        首先构建无向图，每个节点(node)就是view，edge就是两个view连线，对每个view（node）它的标签label就是邻域views。
 *        node的cost（unary cost）就是之前计算的score，用平均score归一化后的值；
 *        edge的cost(pairwise cost)的定义是不鼓励两个view（node）的邻域(label)是一样的（若一样会给一个大的cost惩罚这种情况）,原因是我们希望选择的邻域覆盖整个场景。
 * @param[in] images    所有图像
 * @param[in] imagesMap 图像在用来计算深度的所有图像中的id与在所有图像中的id的对应map
 * @param[in] neighborsMap 每帧的n个邻域
 * @return true 
 * @return false 
 */
bool DepthMapsData::SelectViews(IIndexArr& images, IIndexArr& imagesMap, IIndexArr& neighborsMap)
{
	// find all pair of images valid for dense reconstruction
	// 查找所有用来稠密重建的有效的图像对
	typedef std::unordered_map<uint64_t,float> PairAreaMap;
	PairAreaMap edges;
	double totScore(0);
	unsigned numScores(0);
	FOREACH(i, images) {
		const IIndex idx(images[i]);
		ASSERT(imagesMap[idx] != NO_ID);
		const ViewScoreArr& neighbors(arrDepthData[idx].neighbors);
		ASSERT(neighbors.size() <= OPTDENSE::nMaxViews);
		// register edges
		// 每个帧与它的n个邻域可以组成n个edge（两个节点的连接就是一条边，每个帧与其对应的邻域帧都能组成一条边），并记录这两个帧的共视点覆盖的图像面积area。
		for (const ViewScore& neighbor: neighbors) {
			const IIndex idx2(neighbor.ID);
			ASSERT(imagesMap[idx2] != NO_ID);
			edges[MakePairIdx(idx,idx2)] = neighbor.area;  // 记录邻域帧对应的面积
			// 记录所有score的和和个数，方便后续计算平均值
			totScore += neighbor.score;
			++numScores;
		}
	}
	// 如果edge为空，返回失败。（一般不会发生）
	if (edges.empty())
		return false;
	// 计算平均值avgScore
	const float avgScore((float)(totScore/(double)numScores));

	// run global optimization
	// 运行全局优化，能量最小优化=min(datacost+smoothcost)
	const float fPairwiseMul = OPTDENSE::fPairwiseMul; // default 0.3
	const float fEmptyUnaryMult = 6.f; // 空标签cost的系数
	const float fEmptyPairwise = 8.f*OPTDENSE::fPairwiseMul; // edge上两个节点的标签是空的cost系数
	const float fSamePairwise = 24.f*OPTDENSE::fPairwiseMul; // edge上两个节点的标签是相同的cost系数，越大越平滑
	const IIndex _num_labels = OPTDENSE::nMaxViews+1; //  n个邻域和一个空的状态即空标签（因为在优化过程中可能会存在某些帧找不到其所对应的最佳邻域帧，这时候就可以给其赋予一个空标签，因为后续会针对这个空标签进行处理） N neighbors and an empty state
	const IIndex _num_nodes = images.size();       // 节点个数就是图像个数
	typedef MRFEnergy<TypeGeneral> MRFEnergyType;  // 马尔科夫随机场能量优化
	// MRF初始化
	CAutoPtr<MRFEnergyType> energy(new MRFEnergyType(TypeGeneral::GlobalSize()));
	// 节点初始化
	CAutoPtrArr<MRFEnergyType::NodeId> nodes(new MRFEnergyType::NodeId[_num_nodes]);
	typedef SEACAVE::cList<TypeGeneral::REAL, TypeGeneral::REAL, 0> EnergyCostArr;
	// unary costs: inverse proportional to the image pair score
	// 一元代价（每个节点的代价）：avgScore/score 。view选当前标签（score越大该邻域越合适）的代价，avgScore/score就越小即代价越小则该邻域就越合适
	// 节点和对应的cost计算
	EnergyCostArr arrUnary(_num_labels);
	for (IIndex n=0; n<_num_nodes; ++n) {
		const ViewScoreArr& neighbors(arrDepthData[images[n]].neighbors);
		// 每个节点有k个标签（即每个图像帧有k个邻域帧），在每个标签下都会产生一个能量值
		FOREACH(k, neighbors)
			arrUnary[k] = avgScore/neighbors[k].score; // use average score to normalize the values (not to depend so much on the number of features in the scene)
		arrUnary[neighbors.size()] = fEmptyUnaryMult*(neighbors.empty()?avgScore*0.01f:arrUnary[neighbors.size()-1]);
		// 加入节点
		nodes[n] = energy->AddNode(TypeGeneral::LocalSize(neighbors.size()+1), TypeGeneral::NodeData(arrUnary.data()));
	}
	// pairwise costs: as ratios between the area to be covered and the area actually covered
	// 成对代价（edge代价）：要覆盖的面积和实际覆盖的面积之间的比率（有点类似于上面通过平均分数值对分数进行归一化的操作，因为实际覆盖的面积越大越好）,其实就是节点选对应标签的面积越大代价越小，如果标签一致则设比较大的代价
	// edge上两个节点在对应标签下的areai,areaj，取当前edge的area，如果两标签不一致cost=area/areai + area/areaj 
	// 如果一致，cost=fSamePairwise
	EnergyCostArr arrPairwise(_num_labels*_num_labels);  // edge的两个节点，每个节点有n个label，故有n*n个组合
	for (PairAreaMap::const_reference edge: edges) {
		const PairIdx pair(edge.first);
		const float area(edge.second);  // 要覆盖的面积，即定义这个边的时候，也就是当前帧与此时考虑的邻域帧对应的面积值
		const ViewScoreArr& neighborsI(arrDepthData[pair.i].neighbors);
		const ViewScoreArr& neighborsJ(arrDepthData[pair.j].neighbors);
		arrPairwise.Empty();
		// 计算n*n种组合的代价分别是多少
		FOREACHPTR(pNj, neighborsJ) {  // 将pNj表示为实际选择的相邻帧，pNj->area即为对应的实际覆盖面积
			const IIndex i(pNj->ID);  // pair.j的neighbour的id，可能会取到pair.i
			const float areaJ(area/pNj->area);
			FOREACHPTR(pNi, neighborsI) {
				const IIndex j(pNi->ID);  // pair.i的neighbour的id，可能会取到pair.j
				const float areaI(area/pNi->area);
				//如果两标签不一致cost=area/areai + area/areaj ，如果一致，cost=fSamePairwise
				arrPairwise.Insert(pair.i == i && pair.j == j ? fSamePairwise : fPairwiseMul*(areaI+areaJ));
			}
			// 插入edge的start节点标签为空标签的cost
			arrPairwise.Insert(fEmptyPairwise+fPairwiseMul*areaJ);
		}
		// 插入edge的end节点的标签为空标签的cost
		for (const ViewScore& Ni: neighborsI) {
			const float areaI(area/Ni.area);
			arrPairwise.Insert(fPairwiseMul*areaI+fEmptyPairwise);
		}
		// 插入edge两个节点的标签均为空的cost
		arrPairwise.Insert(fEmptyPairwise*2);
		const IIndex nodeI(imagesMap[pair.i]);
		const IIndex nodeJ(imagesMap[pair.j]);
		// 添加边和cost
		energy->AddEdge(nodes[nodeI], nodes[nodeJ], TypeGeneral::EdgeData(TypeGeneral::GENERAL, arrPairwise.Begin()));
	}

	// minimize energy
	// 能量最小化求解每个节点的最佳label方案
	MRFEnergyType::Options options;
	options.m_eps = OPTDENSE::fOptimizerEps;
	options.m_iterMax = OPTDENSE::nOptimizerMaxIters;
	#ifndef _RELEASE
	// 打印参数设置
	options.m_printIter = 1;
	options.m_printMinIter = 1;
	#endif
	// 求解方法TRW_S （Tree-reweighted Message Passing ）或者BP（Belief Propagation）
	// 论文：convergent tree-reweighted message passing for energy minimization
	#if 1
	TypeGeneral::REAL energyVal, lowerBound;
	energy->Minimize_TRW_S(options, lowerBound, energyVal);
	#else
	TypeGeneral::REAL energyVal;
	energy->Minimize_BP(options, energyVal);
	#endif

	// extract optimized depth map
	//每个结点node表示一个图像，优化后都会有唯一的label即最佳邻域帧的id，用neighborsMap记录。
	neighborsMap.Resize(_num_nodes);
	for (IIndex n=0; n<_num_nodes; ++n) {
		const ViewScoreArr& neighbors(arrDepthData[images[n]].neighbors);
		IIndex& idxNeighbor = neighborsMap[n];
		const IIndex label((IIndex)energy->GetSolution(nodes[n]));
		ASSERT(label <= neighbors.GetSize());
		if (label == neighbors.GetSize()) {  // 对于当前帧，取到的最优标签是空标签的情况
			idxNeighbor = NO_ID; // empty
		} else {
			idxNeighbor = label;
			DEBUG_ULTIMATE("\treference image %3u paired with target image %3u (idx %2u)", images[n], neighbors[label].ID, label);
		}
	}

	// remove all images with no valid neighbors
	// 如果有帧的最佳邻域是无效（NO_ID）则从用来计算depth的image中移除。
	RFOREACH(i, neighborsMap) {
		if (neighborsMap[i] == NO_ID) {
			// remove image with no neighbors
			// 移除没有邻域的image
			for (IIndex& imageMap: imagesMap)
				if (imageMap != NO_ID && imageMap > i)
					--imageMap;
			imagesMap[images[i]] = NO_ID;
			images.RemoveAtMove(i);
			neighborsMap.RemoveAtMove(i);
		}
	}
	return !images.IsEmpty();
} // SelectViews
/*----------------------------------------------------------------*/

// compute visibility for the reference image (the first image in "images")
// and select the best views for reconstructing the depth-map;
// extract also all 3D points seen by the reference image
// Step 2.1 给reference image 选有效的邻域views
bool DepthMapsData::SelectViews(DepthData& depthData)
{
	// find and sort valid neighbor views
	const IIndex idxImage((IIndex)(&depthData-arrDepthData.Begin()));
	ASSERT(depthData.neighbors.IsEmpty());
	// 邻域选择
	if (scene.images[idxImage].neighbors.empty() &&
		!scene.SelectNeighborViews(idxImage, depthData.points, OPTDENSE::nMinViews, OPTDENSE::nMinViewsTrustPoint>1?OPTDENSE::nMinViewsTrustPoint:2, FD2R(OPTDENSE::fOptimAngle), OPTDENSE::nPointInsideROI))
		return false;
	depthData.neighbors.CopyOf(scene.images[idxImage].neighbors);

	// remove invalid neighbor views
	// 移除无效的邻域帧（从面积、尺度、角度这三个方面入手，设一些阈值来进行筛选，然而这些阈值的设置实际中很难把握，很容易设置不当，进而找不到有效的邻域帧，进而导致图像帧无法参与深度计算，一般要么少用，要么把阈值设置得宽松一些）
	// 此外，在scene.SelectNeighborViews()方法中，对当前图像的领域帧的分数进行了计算，一般不使用此处的滤波，直接使用分数最大的几个邻域帧作为参考帧即可（因为分数越大，邻域帧参考意义越好）。
	const float fMinArea(OPTDENSE::fMinArea);
	const float fMinScale(0.2f), fMaxScale(3.2f);
	const float fMinAngle(FD2R(OPTDENSE::fMinAngle));
	const float fMaxAngle(FD2R(OPTDENSE::fMaxAngle));
	// 邻域滤波
	if (!Scene::FilterNeighborViews(depthData.neighbors, fMinArea, fMinScale, fMaxScale, fMinAngle, fMaxAngle, OPTDENSE::nMaxViews)) {
		DEBUG_EXTRA("error: reference image %3u has no good images in view", idxImage);
		return false;
	}
	return true;
} // SelectViews
/*----------------------------------------------------------------*/

// select target image for the reference image (the first image in "images"),
// initialize images data, and initialize depth-map and normal-map;
// if idxNeighbor is not NO_ID, only the reference image and the given neighbor are initialized;
// if numNeighbors is not 0, only the first numNeighbors neighbors are initialized;
// otherwise all are initialized;
// if loadImages, the image data is also setup
// if loadDepthMaps is 1, the depth-maps are loaded from disk,
// if 0, the reference depth-map is initialized from sparse point cloud,
// and if -1, the depth-maps are not initialized
// returns false if there are no good neighbors to estimate the depth-map
/**
 * @brief 初始化用来计算深度图的图像对，如果最佳邻域是有效值则直接初始化，如果是NO_ID则在Step 2_1中计算的众多邻域把score>fMinScore
           存储到depthdata.images中
 * 
 * @param[in] depthData 单帧depth的数据结构，里面存储了用来计算的各种数据比如images（参考帧和邻域帧）等
 * @param[in] idxNeighbor  最佳邻域的Id
 * @param[in] numNeighbors 用来depthmap计算的邻域个数
 * @return true 
 * @return false 
 */
bool DepthMapsData::InitViews(DepthData& depthData, IIndex idxNeighbor, IIndex numNeighbors, bool loadImages, int loadDepthMaps)
{
	const IIndex idxImage((IIndex)(&depthData-arrDepthData.Begin()));
	ASSERT(!depthData.neighbors.IsEmpty());

	// set this image the first image in the array
	// 存放用于深度计算的图像对，images中第一帧是reference image，之后的是邻域帧
	depthData.images.Empty();
	depthData.images.Reserve(depthData.neighbors.GetSize()+1);
	depthData.images.AddEmpty();

	if (idxNeighbor != NO_ID) {
		// set target image as the given neighbor
		const ViewScore& neighbor = depthData.neighbors[idxNeighbor];
		DepthData::ViewData& viewTrg = depthData.images.AddEmpty();
		viewTrg.pImageData = &scene.images[neighbor.ID];  // 往images里放入邻域帧
		viewTrg.scale = neighbor.scale;  //scale 是之前计算的reference与neighbor的图像尺度因子，即reference与neighbour的分辨率比值（具体参见邻域帧选择那一步）
		viewTrg.camera = viewTrg.pImageData->camera;  // 保存邻域帧对应的相机参数
		if (loadImages) {
			// depth计算使用的都是灰度图
			viewTrg.pImageData->image.toGray(viewTrg.image, cv::COLOR_BGR2GRAY, true);
			// !!! 尺度化处理，保证参考帧与邻域帧的共视区域的分辨率是近似的。resize neighbor帧 即将neighbor scale到与reference相同的尺度参考论文：Multi-View stereo for community photo collections(5.1 rescaling views)
			if (DepthData::ViewData::ScaleImage(viewTrg.image, viewTrg.image, viewTrg.scale))
				viewTrg.camera = viewTrg.pImageData->GetCamera(scene.platforms, viewTrg.image.size());  // 对图像进行缩放之后，需要对相应的相机内参进行相应的尺度缩放，保证它们可以对应
		} else {
			if (DepthData::ViewData::NeedScaleImage(viewTrg.scale))
				viewTrg.camera = viewTrg.pImageData->GetCamera(scene.platforms, Image8U::computeResize(viewTrg.pImageData->image.size(), viewTrg.scale));
		}
		DEBUG_EXTRA("Reference image %3u paired with image %3u", idxImage, neighbor.ID);
	} else {
		// initialize all neighbor views too (global reconstruction is used)
		// 把neighbors中所有符合fMinScore选前numNeighbors个为邻域帧，OPTDENSE::fViewMinScoreRatio和OPTDENSE::fViewMinScore都可以进行人工调整
		const float fMinScore(MAXF(depthData.neighbors.First().score*OPTDENSE::fViewMinScoreRatio, OPTDENSE::fViewMinScore));
		FOREACH(idx, depthData.neighbors) {
			const ViewScore& neighbor = depthData.neighbors[idx];
			if ((numNeighbors && depthData.images.GetSize() > numNeighbors) ||
				(neighbor.score < fMinScore))  // 注意，邻域帧的分数都是降序排列的，所以一旦分数小于阈值，后续的邻域帧也就没必要再进行遍历了
				break;
			DepthData::ViewData& viewTrg = depthData.images.AddEmpty();
			viewTrg.pImageData = &scene.images[neighbor.ID];  // 往images里放入邻域帧
			viewTrg.scale = neighbor.scale;
			viewTrg.camera = viewTrg.pImageData->camera;
			if (loadImages) {
				viewTrg.pImageData->image.toGray(viewTrg.image, cv::COLOR_BGR2GRAY, true);
				if (DepthData::ViewData::ScaleImage(viewTrg.image, viewTrg.image, viewTrg.scale))  // 进行对应的尺度调整
					viewTrg.camera = viewTrg.pImageData->GetCamera(scene.platforms, viewTrg.image.size());
			} else {
				if (DepthData::ViewData::NeedScaleImage(viewTrg.scale))
					viewTrg.camera = viewTrg.pImageData->GetCamera(scene.platforms, Image8U::computeResize(viewTrg.pImageData->image.size(), viewTrg.scale));
			}
		}
		#if TD_VERBOSE != TD_VERBOSE_OFF
		// print selected views
		if (g_nVerbosityLevel > 2) {
			String msg;
			for (IIndex i=1; i<depthData.images.size(); ++i)
				msg += String::FormatString(" %3u(%.2fscl)", depthData.images[i].GetID(), depthData.images[i].scale);
			VERBOSE("Reference image %3u paired with %u views:%s (%u shared points)", idxImage, depthData.images.size()-1, msg.c_str(), depthData.points.GetSize());
		} else
		DEBUG_EXTRA("Reference image %3u paired with %u views", idxImage, depthData.images.size()-1);
		#endif
	}
	if (depthData.images.size() < 2) { // 只有参考帧没有对应的邻域帧的情况 
		depthData.images.Release();
		return false;
	}

	// initialize reference image as well
	// 初始化第一帧
	DepthData::ViewData& viewRef = depthData.images.front();
	viewRef.scale = 1;
	viewRef.pImageData = &scene.images[idxImage];
	viewRef.camera = viewRef.pImageData->camera;
	if (loadImages)
		viewRef.pImageData->image.toGray(viewRef.image, cv::COLOR_BGR2GRAY, true);

	// initialize views
	for (IIndex i=1; i<depthData.images.size(); ++i) {
		DepthData::ViewData& view = depthData.images[i];
		if (loadDepthMaps > 0) {
			// load known depth-map
			String imageFileName;
			IIndexArr IDs;
			cv::Size imageSize;
			Depth dMin, dMax;
			NormalMap normalMap;
			ConfidenceMap confMap;
			ViewsMap viewsMap;
			ImportDepthDataRaw(ComposeDepthFilePath(view.GetID(), "dmap"),
				imageFileName, IDs, imageSize, view.cameraDepthMap.K, view.cameraDepthMap.R, view.cameraDepthMap.C,
				dMin, dMax, view.depthMap, normalMap, confMap, viewsMap, 1);
			ASSERT(viewRef.image.size() == view.depthMap.size());
		}
		view.Init(viewRef.camera);
	}

	if (loadDepthMaps > 0) {
		// load known depth-map and normal-map
		String imageFileName;
		IIndexArr IDs;
		cv::Size imageSize;
		Camera camera;
		ConfidenceMap confMap;
		ViewsMap viewsMap;
		if (!ImportDepthDataRaw(ComposeDepthFilePath(viewRef.GetID(), "dmap"),
				imageFileName, IDs, imageSize, camera.K, camera.R, camera.C, depthData.dMin, depthData.dMax,
				depthData.depthMap, depthData.normalMap, confMap, viewsMap, 3))
			return false;
		ASSERT(viewRef.image.size() == depthData.depthMap.size());
		ASSERT(depthData.normalMap.empty() || viewRef.image.size() == depthData.normalMap.size());
		if (depthData.normalMap.empty()) {
			// estimate normal map
			EstimateNormalMap(viewRef.camera.K, depthData.depthMap, depthData.normalMap);
		}
	} else if (loadDepthMaps == 0) {
		// initialize depth and normal maps
		// 初始化深度图和法向量图

		// initialize the depth-map（此处的深度图初始化在PatchMatch中得到应用）
		// Step 3_2_1 PM:depth 初始化：先根据当前帧能看到的点云计算投影到depth上得到稀疏depth图，同时根据这些depth值计算最大最小值；
		// 利用CGAL中的三角网格化函数对稀疏点网格划分，然后再将其栅格化投影对应像素对空缺位置的depth进行插值；
		// 细节见InitDepthMap函数
		if (OPTDENSE::nMinViewsTrustPoint < 2 || depthData.points.empty()) {
			// compute depth range and initialize known depths, else random
			const Image8U::Size size(viewRef.image.size());
			depthData.depthMap.create(size); depthData.depthMap.memset(0);
			depthData.normalMap.create(size);
			if (depthData.points.empty()) {
				// all values will be initialized randomly
				depthData.dMin = 1e-1f;
				depthData.dMax = 1e+2f;
			} else {  // 深度初始化的简化版本
				// initialize with the sparse point-cloud
				//初始化depth法1:根据当前帧的points投影到depth上并在以投影点为中心,在窗口大小为2*nPixelArea+1内设置与投影点相同的depth.
				const int nPixelArea(2); // half windows size around a pixel to be initialize with the known depth
				depthData.dMin = FLT_MAX;
				depthData.dMax = 0;
				FOREACHPTR(pPoint, depthData.points) {  // 当前帧的稀疏点
					const PointCloud::Point& X = scene.pointcloud.points[*pPoint];       // 世界坐标系下的坐标
					const Point3 camX(viewRef.camera.TransformPointW2C(Cast<REAL>(X)));  // 相机坐标系下的坐标
					const ImageRef x(ROUND2INT(viewRef.camera.TransformPointC2I(camX))); // 像素坐标系下的坐标
					const float d((float)camX.z);  // 深度值
					// 计算以投影点为中心的窗口起始点
					const ImageRef sx(MAXF(x.x-nPixelArea,0), MAXF(x.y-nPixelArea,0));
					const ImageRef ex(MINF(x.x+nPixelArea,size.width-1), MINF(x.y+nPixelArea,size.height-1));
					// 将这个窗口内的所有像素都设为与投影点相同的深度值
					for (int y=sx.y; y<=ex.y; ++y) {
						for (int x=sx.x; x<=ex.x; ++x) {
							depthData.depthMap(y,x) = d;
							depthData.normalMap(y,x) = Normal::ZERO;
						}
					}
					if (depthData.dMin > d)
						depthData.dMin = d;
					if (depthData.dMax < d)
						depthData.dMax = d;
				}
				// 略微扩大下深度范围，因为稀疏特征点的深度值可能不能完整地涵盖整张图像上的像素点的深度范围
				depthData.dMin *= 0.9f;
				depthData.dMax *= 1.1f;
			}
		} else {
			ASSERT(!depthData.points.empty());
			// compute rough estimates using the sparse point-cloud
			//初始化法2:投影到depth上的稀疏点进行三角网格划分,在每个三角网格内进行栅格化并根据三个顶点所在的平面进行插值。这个步骤和SGM对深度图的初始化一样，可以参考相应代码和讲解
			InitDepthMap(depthData);
		}
	}
	return true;
} // InitViews
/*----------------------------------------------------------------*/

// roughly estimate depth and normal maps by triangulating the sparse point cloud
// and interpolating normal and depth for all pixels
// depth初始化，利用稀疏点投影到depth上，对这些点进行三角网格划分然后在每个三角面上插值得到每个像素的depth和法向量
bool DepthMapsData::InitDepthMap(DepthData& depthData)
{
	TD_TIMER_STARTD();

	ASSERT(depthData.images.GetSize() > 1 && !depthData.points.IsEmpty());
	const DepthData::ViewData& image(depthData.GetView());
	TriangulatePoints2DepthMap(image, scene.pointcloud, depthData.points, depthData.depthMap, depthData.normalMap, depthData.dMin, depthData.dMax, OPTDENSE::bAddCorners, OPTDENSE::bInitSparse);
	depthData.dMin *= 0.9f;
	depthData.dMax *= 1.1f;

	#if TD_VERBOSE != TD_VERBOSE_OFF
	// save rough depth map as image
	// 保存深度图
	if (g_nVerbosityLevel > 4) {
		ExportDepthMap(ComposeDepthFilePath(image.GetID(), "init.png"), depthData.depthMap);
		ExportNormalMap(ComposeDepthFilePath(image.GetID(), "init.normal.png"), depthData.normalMap);
		ExportPointCloud(ComposeDepthFilePath(image.GetID(), "init.ply"), *depthData.images.First().pImageData, depthData.depthMap, depthData.normalMap);
	}
	#endif

	DEBUG_ULTIMATE("Depth-map %3u roughly estimated from %u sparse points: %dx%d (%s)", image.GetID(), depthData.points.size(), image.image.width(), image.image.height(), TD_TIMER_GET_FMT().c_str());
	return true;
} // InitDepthMap
/*----------------------------------------------------------------*/


// initialize the confidence map (NCC score map) with the score of the current estimates
// 利用之前初始化的depth计算每个像素的score，并且获取每一个像素对应patch的法向量以及置信度（即分数、匹配代价）
void* STCALL DepthMapsData::ScoreDepthMapTmp(void* arg)
{
	DepthEstimator& estimator = *((DepthEstimator*)arg);
	IDX idx;  // 要处理的像素id
	//Thread::safeInc(estimator.idxPixel):使用了InterlockedIncrement对idxPixel进行锁定防止多线程访问冲突
    //InterlockedIncrement( &lReference );  // 对这个引用计数进行锁定并ADD 1 每调用一次加1
	while ((idx=(IDX)Thread::safeInc(estimator.idxPixel)) < estimator.coords.GetSize()) {
		const ImageRef& x = estimator.coords[idx];  // 要处理的像素坐标
		// patch准备，判断能否构建patch，主要是剔除边界；在reference图像上计算patch 值
		if (!estimator.PreparePixelPatch(x) || !estimator.FillPixelPatch()) {
			// 失败就设为0
			estimator.depthMap0(x) = 0;
			estimator.normalMap0(x) = Normal::ZERO;
			estimator.confMap0(x) = 2.f;
			continue;
		}
		// 获取x对应额深度值和法向量
		Depth& depth = estimator.depthMap0(x);
		Normal& normal = estimator.normalMap0(x);
		// viewDir指相机原点到x0所在相机坐标形成的向量（也就是相机光心与相应图像帧上的像素点x0相连产生的向量）
		const Normal viewDir(Cast<float>(static_cast<const Point3&>(estimator.X0)));
		if (!ISINSIDE(depth, estimator.dMin, estimator.dMax)) {
			// init with random values
			// 初始化
			depth = estimator.RandomDepth(estimator.dMinSqr, estimator.dMaxSqr);  // 随机初始化一个最小最大深度值形成的深度范围内的深度值
			normal = estimator.RandomNormal(viewDir);  // 沿着viewDir随机初始化一条法向量
		} else if (normal.dot(viewDir) >= 0) {
			// replace invalid normal with random values
			// 如果法线与view夹角小于90，则无效，因为这种情况下是看不到点的
			normal = estimator.RandomNormal(viewDir);
		}
		// patch代价计算
		// 利用当前初始深度图和normal计算当前帧与邻域帧的匹配代价wncc（以对应像素为中心的patch的匹配代价）,计算confidence
		// 代价越高，置信度越低
		ASSERT(ISEQUAL(norm(normal), 1.f));
		estimator.confMap0(x) = estimator.ScorePixel(depth, normal);
	}
	return NULL;
}

// run propagation and random refinement cycles
// 对每个像素进行depth的邻域传播和随机优化
void* STCALL DepthMapsData::EstimateDepthMapTmp(void* arg)
{
	DepthEstimator& estimator = *((DepthEstimator*)arg);
	IDX idx;
	while ((idx=(IDX)Thread::safeInc(estimator.idxPixel)) < estimator.coords.GetSize())
		estimator.ProcessPixel(idx);
	return NULL;
}

// remove all estimates with too big score and invert confidence map
// depth优化，主要是剔除score过大的点，
void* STCALL DepthMapsData::EndDepthMapTmp(void* arg)
{
	DepthEstimator& estimator = *((DepthEstimator*)arg);
	IDX idx;
	MAYBEUNUSED const float fOptimAngle(FD2R(OPTDENSE::fOptimAngle));
	while ((idx=(IDX)Thread::safeInc(estimator.idxPixel)) < estimator.coords.GetSize()) {
		const ImageRef& x = estimator.coords[idx];
		ASSERT(estimator.depthMap0(x) >= 0);
		Depth& depth = estimator.depthMap0(x);
		float& conf = estimator.confMap0(x);
		// check if the score is good enough
		// and that the cross-estimates is close enough to the current estimate
		// 判断score是否足够好，conf越大越不好
		if (depth <= 0 || conf >= OPTDENSE::fNCCThresholdKeep) {
			// used if gap-interpolation is active
			conf = 0;
			depth = 0;
			estimator.normalMap0(x) = Normal::ZERO;
		} else {
			#if 1
			// converted ZNCC [0-2] score, where 0 is best, to [0-1] confidence, where 1 is best
			// 将置信度invert，0-1 1是最好
			conf = conf>=1.f ? 0.f : 1.f-conf;
			#else  //不考虑
			#if 1
			FOREACH(i, estimator.images)
				estimator.scores[i] = ComputeAngle<REAL,float>(estimator.image0.camera.TransformPointI2W(Point3(x,depth)).ptr(), estimator.image0.camera.C.ptr(), estimator.images[i].view.camera.C.ptr());
			#if DENSE_AGGNCC == DENSE_AGGNCC_NTH
			const float fCosAngle(estimator.scores.GetNth(estimator.idxScore));
			#elif DENSE_AGGNCC == DENSE_AGGNCC_MEAN
			const float fCosAngle(estimator.scores.mean());
			#elif DENSE_AGGNCC == DENSE_AGGNCC_MIN
			const float fCosAngle(estimator.scores.minCoeff());
			#else
			const float fCosAngle(estimator.idxScore ?
				std::accumulate(estimator.scores.begin(), &estimator.scores.PartialSort(estimator.idxScore), 0.f) / estimator.idxScore :
				*std::min_element(estimator.scores.cbegin(), estimator.scores.cend()));
			#endif
			const float wAngle(MINF(POW(ACOS(fCosAngle)/fOptimAngle,1.5f),1.f));
			#else
			const float wAngle(1.f);
			#endif
			#if 1
			conf = wAngle/MAXF(conf,1e-2f);
			#else
			conf = wAngle/(depth*SQUARE(MAXF(conf,1e-2f)));
			#endif
			#endif
		}
	}
	return NULL;
}

DepthData DepthMapsData::ScaleDepthData(const DepthData& inputDeptData, float scale) {
	ASSERT(scale <= 1);
	if (scale == 1)
		return inputDeptData;
	DepthData rescaledDepthData(inputDeptData);
	FOREACH (idxView, rescaledDepthData.images) {
		DepthData::ViewData& viewData = rescaledDepthData.images[idxView];
		ASSERT(viewData.depthMap.empty() || viewData.image.size() == viewData.depthMap.size());
		cv::resize(viewData.image, viewData.image, cv::Size(), scale, scale, cv::INTER_AREA);
		viewData.camera = viewData.pImageData->camera;
		viewData.camera.K = viewData.camera.GetScaledK(viewData.pImageData->GetSize(), viewData.image.size());
		if (!viewData.depthMap.empty()) {
			cv::resize(viewData.depthMap, viewData.depthMap, viewData.image.size(), 0, 0, cv::INTER_AREA);
			viewData.cameraDepthMap = viewData.pImageData->camera;
			viewData.cameraDepthMap.K = viewData.cameraDepthMap.GetScaledK(viewData.pImageData->GetSize(), viewData.image.size());
		}
		viewData.Init(rescaledDepthData.images[0].camera);
	}
	if (!rescaledDepthData.depthMap.empty())
		cv::resize(rescaledDepthData.depthMap, rescaledDepthData.depthMap, cv::Size(), scale, scale, cv::INTER_NEAREST);
	if (!rescaledDepthData.normalMap.empty())
		cv::resize(rescaledDepthData.normalMap, rescaledDepthData.normalMap, cv::Size(), scale, scale, cv::INTER_NEAREST);
	return rescaledDepthData;
}

// estimate depth-map using propagation and random refinement with NCC score
// as in: "Accurate Multiple View 3D Reconstruction Using Patch-Based Stereo for Large-Scale Scenes", S. Shen, 2013
// The implementations follows closely the paper, although there are some changes/additions.
// Given two views of the same scene, we note as the "reference image" the view for which a depth-map is reconstructed, and the "target image" the other view.
// As a first step, the whole depth-map is approximated by interpolating between the available sparse points.
// Next, the depth-map is passed from top/left to bottom/right corner and the opposite sens for each of the next steps.
// For each pixel, first the current depth estimate is replaced with its neighbor estimates if the NCC score is better.
// Second, the estimate is refined by trying random estimates around the current depth and normal values, keeping the one with the best score.
// The estimation can be stopped at any point, and usually 2-3 iterations are enough for convergence.
// For each pixel, the depth and normal are scored by computing the NCC score between the patch in the reference image and the wrapped patch in the target image, as dictated by the homography matrix defined by the current values to be estimate.
// In order to ensure some smoothness while locally estimating each pixel, a bonus is added to the NCC score if the estimate for this pixel is close to the estimates for the neighbor pixels.
// Optionally, the occluded pixels can be detected by extending the described iterations to the target image and removing the estimates that do not have similar values in both views.
//  - nGeometricIter: current geometric-consistent estimation iteration (-1 - normal patch-match)
/**
 * @brief depth计算,采用patchMatch方法 
        原理细节主要是参考论文"Accurate Multiple View 3D Reconstruction Using Patch-Based Stereo for Large-Scale Scenes", S. Shen, 2013
 *      代价计算分两种方式一种是上述论文里面的简单的直接使用NCC来作为匹配代价；
		另一种是带权重的代价参考论文"PatchMatch Stereo - Stereo Matching with Slanted Support Windows"公式3
 *		以NCC为评价标准，使用传播和随机优化来估计深度图

 *      给定同一场景的两个视图，我们将重建深度图的视图记为“参考图像”，将另一个视图记为“目标图像”。
 *      第一步：在可用的稀疏点之间插值初始化深度图；
 *      第二步：传播优化：深度图从顶部/左侧传播到底部/右侧，并在接下来的每个步骤中传递相反的方向。
        传播过程中，对于每个像素，如果邻域的NCC评分较好，首先将当前的深度估计替换为它的邻居深度值。
        然后通过尝试对当前深度值做一个随机调整来细化重建的深度，并保留得分最高的那个，通常2-3次迭代就足够收敛了。
		（对于每个像素，深度和法线是通过计算参考图像中的patch与目标图像中被包裹的patch之间的NCC分数来进行评分的，这由待估计的当前值定义的单应性矩阵来决定。
		为了在局部估计每个像素时保证一定的平滑性，如果这个像素的估计接近于相邻像素的估计，那么NCC分数就会得到额外的奖励。
		可选地，通过将描述的迭代扩展到目标图像并删除在两个视图中不具有相似值的估计，可以检测被遮挡的像素。）
		第三步: 滤波
 * @param[in] idxImage 
 * @return true 
 * @return false 
 */
bool DepthMapsData::EstimateDepthMap(IIndex idxImage, int nGeometricIter)
{
	#ifdef _USE_CUDA
	if (pmCUDA) {
		pmCUDA->EstimateDepthMap(arrDepthData[idxImage]);
		return true;
	}
	#endif // _USE_CUDA

	TD_TIMER_STARTD();

	const unsigned nMaxThreads(scene.nMaxThreads);
	const unsigned iterBegin(nGeometricIter < 0 ? 0u : OPTDENSE::nEstimationIters+(unsigned)nGeometricIter);
	const unsigned iterEnd(nGeometricIter < 0 ? OPTDENSE::nEstimationIters : iterBegin+1);

	// init threads
	// 线程初始化
	ASSERT(nMaxThreads > 0);
	cList<DepthEstimator> estimators;
	estimators.reserve(nMaxThreads);
	cList<SEACAVE::Thread> threads;
	if (nMaxThreads > 1)
		threads.resize(nMaxThreads-1); // current thread is also used
	volatile Thread::safe_t idxPixel;

	// initialize depth and normal maps（最关键的一步）
	// Step 3_2_1 PM:depth 初始化
	// 初始化深度图和法向量图的具体操作可以参见SceneDensify.cpp中的DepthMapsData::InitViews()方法
	// 法向量很重要，因为它是Patch（切平面）的法向量
	// 置信度和匹配代价相关，代价越小，置信度越高
	// Multi-Resolution : 
	DepthData& fullResDepthData(arrDepthData[idxImage]);
	const unsigned totalScaleNumber(nGeometricIter < 0 ? OPTDENSE::nSubResolutionLevels : 0u);
	DepthMap lowResDepthMap;
	NormalMap lowResNormalMap;
	#if DENSE_NCC == DENSE_NCC_WEIGHTED
	DepthEstimator::WeightMap weightMap0;
	#else
	Image64F imageSum0;
	#endif
	DepthMap currentSizeResDepthMap;
	for (unsigned scaleNumber = totalScaleNumber+1; scaleNumber-- > 0; ) {
		// initialize
		float scale = 1.f / POWI(2, scaleNumber);
		DepthData currentDepthData(ScaleDepthData(fullResDepthData, scale));
		DepthData& depthData(scaleNumber==0 ? fullResDepthData : currentDepthData);
		ASSERT(depthData.images.size() > 1);
		const DepthData::ViewData& image(depthData.images.front());
		ASSERT(!image.image.empty() && !depthData.images[1].image.empty());
		const Image8U::Size size(image.image.size());
		if (scaleNumber != totalScaleNumber) {
			cv::resize(lowResDepthMap, depthData.depthMap, size, 0, 0, OPTDENSE::nIgnoreMaskLabel >= 0 ? cv::INTER_NEAREST : cv::INTER_LINEAR);
			cv::resize(lowResNormalMap, depthData.normalMap, size, 0, 0, cv::INTER_NEAREST);
			depthData.depthMap.copyTo(currentSizeResDepthMap);
		}
		else if (totalScaleNumber > 0) {
			fullResDepthData.depthMap.release();
			fullResDepthData.normalMap.release();
			fullResDepthData.confMap.release();
		}
		depthData.confMap.create(size);

		// init integral images and index to image-ref map for the reference data
		// 初始化积分图和参考图的索引map
		// 权重初始化,size为图像大小，将其尺寸resize到有效像素的数目
		#if DENSE_NCC == DENSE_NCC_WEIGHTED
		weightMap0.clear();
		weightMap0.resize(size.area()-(size.width+1)*DepthEstimator::nSizeHalfWindow);
		#else
		//计算积分图
		cv::integral(image.image, imageSum0, CV_64F);
		#endif
		if (prevDepthMapSize != size || OPTDENSE::nIgnoreMaskLabel >= 0) {
			BitMatrix mask;
			if (OPTDENSE::nIgnoreMaskLabel >= 0 && DepthEstimator::ImportIgnoreMask(*image.pImageData, depthData.depthMap.size(), (uint16_t)OPTDENSE::nIgnoreMaskLabel, mask))
				depthData.ApplyIgnoreMask(mask);
			//                        1 2 4 7
			// 1 2 4 7 5 3 6 8 9 >    3 5 8
			//                        6 9
			// depth坐标索引转换成之字形，方便后续迭代直接使用，
			// 迭代传播优化时的传播路线，偶次迭代，从右下到左上，奇次迭代，从左上到右下
			// coords中存储的就是迭代传播优化时所用的z字形的索引路线对应的坐标
			DepthEstimator::MapMatrix2ZigzagIdx(size, coords, mask, MAXF(64,(int)nMaxThreads*8));
			#if 0 && !defined(_RELEASE)
			// show pixels to be processed
			Image8U cmask(size);
			cmask.memset(0);
			for (const DepthEstimator::MapRef& x: coords)
				cmask(x.y, x.x) = 255;
			cmask.Show("cmask");
			#endif
			prevDepthMapSize = size;
		}

		// initialize the reference confidence map (NCC score map) with the score of the current estimates
		// Step 3_2_2 PM:score confidence初始化，预先计算reference帧每个像素的ncc相关计算
		{
			// create working threads
			// 创建工作线程
			idxPixel = -1;
			ASSERT(estimators.empty());
			while (estimators.size() < nMaxThreads) {
				// estimators初始化
				estimators.emplace_back(iterBegin, depthData, idxPixel,
					#if DENSE_NCC == DENSE_NCC_WEIGHTED
					weightMap0,
					#else
					imageSum0,
					#endif
					coords);
				estimators.Last().lowResDepthMap = currentSizeResDepthMap;
			}
			ASSERT(estimators.size() == threads.size()+1);
			FOREACH(i, threads)
				threads[i].start(ScoreDepthMapTmp, &estimators[i]);
			ScoreDepthMapTmp(&estimators.back());
			// wait for the working threads to close
			// 等待线程关闭
			FOREACHPTR(pThread, threads)
				pThread->join();
			estimators.clear();
			#if TD_VERBOSE != TD_VERBOSE_OFF
			// save rough depth map as image
			// 保存深度图
			if (g_nVerbosityLevel > 4 && nGeometricIter < 0) {
				ExportDepthMap(ComposeDepthFilePath(image.GetID(), "rough.png"), depthData.depthMap);
				ExportNormalMap(ComposeDepthFilePath(image.GetID(), "rough.normal.png"), depthData.normalMap);
				ExportPointCloud(ComposeDepthFilePath(image.GetID(), "rough.ply"), *depthData.images.First().pImageData, depthData.depthMap, depthData.normalMap);
			}
			#endif
		}

		// run propagation and random refinement cycles on the reference data
		// Step 3_2_3 PM:depth迭代优化： 邻域传播和随机优化。在初始化深度图和置信度图的基础上对每个像素点的深度值进行优化，使其更接近真实值
		// 需要注意的是，虽然迭代传播在PatchMatch Stereo - Stereo Matching with slanded support windows提出，
		// 但是OpenMVS的作者采用的是Accurate Multiple View 3D Reconstruction Using Patch-Based Stereo for Large-Scale Scenes
		// 相比之下，前一篇论文在迭代传播时使用空间传播、视图传播、时序传播以及随机分配，而后者则仅使用空间传播和随机分配，
		// 因为后一篇作者经过实验发现这样做不仅节省性能，而且效果也不差，并且作者指出，虽然没有使用视图传播和时序传播，但是由于在
		// 计算完每张深度图之后进行深度信息融合时会使用帧间一致性检查（类似于视差计算的一致性检查），
		// 即判断当前帧上的像素深度与该像素投影到邻域帧上的位置的深度是否一致，若不一致就将其进行剔除，使其不参与后续的重建过程。
		// 因此是否进行视图传播与时序传播对最终的结果影响不大，但若想获取完整的深度图，则可以用上视图传播和时序传播来提高深度估计的质量，
		// 而若仅仅是用深度估计来进行三维重建，不进行视图传播和时序传播是可以的，而且毕竟后续也会根据视图之间的信息进行滤波，以剔除深度噪声，
		// 所以虽然不使用视图传播和时序传播会导致每一帧上的像素点的深度估计不是那么准确，存在着一定的噪声，但是对于重建没有影响
		for (unsigned iter=iterBegin; iter<iterEnd; ++iter) {
			// create working threads
			// 线程启动
			idxPixel = -1;
			ASSERT(estimators.empty());
			while (estimators.size() < nMaxThreads) {
				estimators.emplace_back(iter, depthData, idxPixel,
					#if DENSE_NCC == DENSE_NCC_WEIGHTED
					weightMap0,
					#else
					imageSum0,
					#endif
					coords);
				estimators.Last().lowResDepthMap = currentSizeResDepthMap;
			}
			ASSERT(estimators.size() == threads.size()+1);
			// 启用多线程计算
			FOREACH(i, threads)
				threads[i].start(EstimateDepthMapTmp, &estimators[i]);
			EstimateDepthMapTmp(&estimators.back());
			// wait for the working threads to close
			// 等待线程结束
			FOREACHPTR(pThread, threads)
				pThread->join();
			estimators.clear();
			#if 1 && TD_VERBOSE != TD_VERBOSE_OFF
			// save intermediate depth map as image
			if (g_nVerbosityLevel > 4) {
				String path(ComposeDepthFilePath(image.GetID(), "iter")+String::ToString(iter));
				if (nGeometricIter >= 0)
					path += String::FormatString(".geo%d", nGeometricIter);
				ExportDepthMap(path+".png", depthData.depthMap);
				ExportNormalMap(path+".normal.png", depthData.normalMap);
				ExportPointCloud(path+".ply", *depthData.images.First().pImageData, depthData.depthMap, depthData.normalMap);
			}
			#endif
		}

		// remember sub-resolution estimates for next iteration
		if (scaleNumber > 0) {
			lowResDepthMap = depthData.depthMap;
			lowResNormalMap = depthData.normalMap;
		}
	}

	DepthData& depthData(fullResDepthData);
	// remove all estimates with too big score and invert confidence map
	// Step 3_2_4 PM: 滤波，去除score大的点 invert(也就是对置信度图进行反转，因为上面获取的所谓的置信度图其实是代价图，
	// 因为其与代价成正比，而实际情况置信度图应该与代价图呈反比，所以需要对计算得到的置信度图进行反转) 
	// 置信度图0-1 1是最优
	{
		const float fNCCThresholdKeep(OPTDENSE::fNCCThresholdKeep);
		if (nGeometricIter < 0 && OPTDENSE::nEstimationGeometricIters)
			OPTDENSE::fNCCThresholdKeep *= 1.333f;
		// create working threads
		idxPixel = -1;
		ASSERT(estimators.empty());
		while (estimators.size() < nMaxThreads)
			estimators.emplace_back(0, depthData, idxPixel,
				#if DENSE_NCC == DENSE_NCC_WEIGHTED
				weightMap0,
				#else
				imageSum0,
				#endif
				coords);
		ASSERT(estimators.size() == threads.size()+1);
		FOREACH(i, threads)
			threads[i].start(EndDepthMapTmp, &estimators[i]);
		EndDepthMapTmp(&estimators.back());
		// wait for the working threads to close。线程关闭
		FOREACHPTR(pThread, threads)
			pThread->join();
		estimators.clear();
		OPTDENSE::fNCCThresholdKeep = fNCCThresholdKeep;
	}

	// 保存深度估计结果
	DEBUG_EXTRA("Depth-map for image %3u %s: %dx%d (%s)", depthData.images.front().GetID(),
		depthData.images.size() > 2 ?
			String::FormatString("estimated using %2u images", depthData.images.size()-1).c_str() :
			String::FormatString("with image %3u estimated", depthData.images[1].GetID()).c_str(),
		depthData.depthMap.cols, depthData.depthMap.rows, TD_TIMER_GET_FMT().c_str());
	return true;
} // EstimateDepthMap
/*----------------------------------------------------------------*/


// filter out small depth segments from the given depth map
// 从给定的深度图中过滤出小的深度段
bool DepthMapsData::RemoveSmallSegments(DepthData& depthData)
{
	const float fDepthDiffThreshold(OPTDENSE::fDepthDiffThreshold*0.7f);
	unsigned speckle_size = OPTDENSE::nSpeckleSize;
	DepthMap& depthMap = depthData.depthMap;
	NormalMap& normalMap = depthData.normalMap;
	ConfidenceMap& confMap = depthData.confMap;
	ASSERT(!depthMap.empty());
	const ImageRef size(depthMap.size());

	// allocate memory on heap for dynamic programming arrays
	// 在堆上为动态编程数组分配内存
	TImage<bool> done_map(size, false);  // 用于标记深度图中的每个像素是否被处理过，若已经处理过，则对应像素位置的标记为True，可以跳过
	CAutoPtrArr<ImageRef> seg_list(new ImageRef[size.x*size.y]);  // 用于存放得到的每个连通域里面的像素坐标
	unsigned seg_list_count;
	unsigned seg_list_curr;
	ImageRef neighbor[4];

	// for all pixels do
	// 逐像素处理
	for (int u=0; u<size.x; ++u) {
		for (int v=0; v<size.y; ++v) {
			// if the first pixel in this segment has been already processed => skip
			// 如果这个段中的第一个像素已经被处理=>跳过
			if (done_map(v,u))
				continue;

			// init segment list (add first element
			// and set it to be the next element to check)
			// 初始化 分割list
			seg_list[0] = ImageRef(u,v);
			seg_list_count = 1;  // 用于记录连通域中所包含的像素数目
			seg_list_curr  = 0;

			// add neighboring segments as long as there
			// are none-processed pixels in the seg_list;
			// none-processed means: seg_list_curr<seg_list_count
			// 只要seg_list中有未处理的像素，就添加相邻的分割块，一直到这个分割块中的像素被处理完为止
			while (seg_list_curr < seg_list_count) {
				// get address of current pixel in this segment
				// 取当前像素在这个分割块中的地址以及其所对应的深度值
				const ImageRef addr_curr(seg_list[seg_list_curr]);
				const Depth& depth_curr = depthMap(addr_curr);

				if (depth_curr>0) {
					// fill list with neighbor positions
					// 用邻域像素填充list
					neighbor[0] = ImageRef(addr_curr.x-1, addr_curr.y  );
					neighbor[1] = ImageRef(addr_curr.x+1, addr_curr.y  );
					neighbor[2] = ImageRef(addr_curr.x  , addr_curr.y-1);
					neighbor[3] = ImageRef(addr_curr.x  , addr_curr.y+1);

					// for all neighbors do
					// 处理每一个邻域
					for (int i=0; i<4; ++i) {
						// get neighbor pixel address
						// 取邻域坐标
						const ImageRef& addr_neighbor(neighbor[i]);
						// check if neighbor is inside image
						// 确认邻域是否在图像内
						if (addr_neighbor.x>=0 && addr_neighbor.y>=0 && addr_neighbor.x<size.x && addr_neighbor.y<size.y) {
							// check if neighbor has not been added yet
							// 确认邻域是否已经被处理过
							bool& done = done_map(addr_neighbor);
							if (!done) {
								// check if the neighbor is valid and similar to the current pixel
								// 确认邻域是否属于当前分割块
								// (belonging to the current segment)
								// 获取邻域像素坐标对应的深度值，并判断邻域像素的深度值与当前像素的深度值是否相似，若相似则认为当前像素与邻域像素处于同一个连通域中，
								// 并且由于这个邻域像素在此处会被考虑与当前像素的相似性，因此会被标记为处理过
								const Depth& depth_neighbor = depthMap(addr_neighbor);
								if (depth_neighbor>0 && IsDepthSimilar(depth_curr, depth_neighbor, fDepthDiffThreshold)) {
									// add neighbor coordinates to segment list
									// 如果属于则加到分割块的list中
									seg_list[seg_list_count++] = addr_neighbor;
									// set neighbor pixel in done_map to "done"
									// (otherwise a pixel may be added 2 times to the list, as
									//  neighbor of one pixel and as neighbor of another pixel)
									// 标记邻域已被处理过
									done = true;
								}
							}
						}
					}
				}

				// set current pixel in seg_list to "done"
				// 在seg列表中设置当前像素为“已完成”，并且令索引增加，以用于处理分割快所包含的下一个像素
				++seg_list_curr;

				// set current pixel in done_map to "done"
				// 标记当前像素已被处理过
				done_map(addr_curr) = true;
			} // end: while (seg_list_curr < seg_list_count)

			// if segment NOT large enough => invalidate pixels
			// 如果分割块大小不够大（此处使用speckle_size来表示分割块的大小），就认为是无效的将其剔除
			if (seg_list_count < speckle_size) {
				// for all pixels in current segment invalidate pixels
				// 把无效的像素深度都置为0
				for (unsigned i=0; i<seg_list_count; ++i) {
					depthMap(seg_list[i]) = 0;
					if (!normalMap.empty()) normalMap(seg_list[i]) = Normal::ZERO;
					if (!confMap.empty()) confMap(seg_list[i]) = 0;
				}
			}
		}
	}

	return true;
} // RemoveSmallSegments
/*----------------------------------------------------------------*/

// try to fill small gaps in the depth map
// 填充小的洞
bool DepthMapsData::GapInterpolation(DepthData& depthData)
{
	const float fDepthDiffThreshold(OPTDENSE::fDepthDiffThreshold*2.5f);
	unsigned nIpolGapSize = OPTDENSE::nIpolGapSize;
	DepthMap& depthMap = depthData.depthMap;
	NormalMap& normalMap = depthData.normalMap;
	ConfidenceMap& confMap = depthData.confMap;
	ASSERT(!depthMap.empty());
	const ImageRef size(depthMap.size());

	// 1. Row-wise: 处理行
	// for each row do
	for (int v=0; v<size.y; ++v) {
		// init counter
		unsigned count = 0;

		// for each element of the row do 处理每一行的每个像素
		for (int u=0; u<size.x; ++u) {
			// get depth of this location
			// 取深度值
			const Depth& depth = depthMap(v,u);

			// if depth not valid => count and skip it
			// 无效跳过，并记录
			if (depth <= 0) {
				++count;  // 相当于统计孔洞在x方向（即水平方向）上的大小
				continue;
			}
			if (count == 0)
				continue;

			// check if speckle is small enough
			// 判断洞是否足够小，因为对于大孔洞即使是使用填充，填充的深度值是很不准的
			// and value in range
			if (count <= nIpolGapSize && (unsigned)u > count) {
				// first value index for interpolation
				// 第一个要插值的索引
				int u_curr(u-count);
				const int u_first(u_curr-1);  // 当前要填充的孔洞在x方向上最左端位置旁边的具有非零深度值的位置
				// compute mean depth
				// 计算洞的两端深度的平均深度，若两端的深度值相似，才会考虑利用它们对孔洞内的像素位置的深度进行插值
				// 此时的depth相当于是当前要填充的孔洞在x方向上最右端位置旁边的非零深度值
				const Depth& depthFirst = depthMap(v,u_first);
				if (IsDepthSimilar(depthFirst, depth, fDepthDiffThreshold)) {
					#if 0
					// set all values with the average
					const Depth avg((depthFirst+depth)*0.5f);
					do {
						depthMap(v,u_curr) = avg;
					} while (++u_curr<u);						
					#else
					// interpolate values
					// 线性插值
					const Depth diff((depth-depthFirst)/(count+1));
					Depth d(depthFirst);
					const float c(confMap.empty() ? 0.f : MINF(confMap(v,u_first), confMap(v,u)));
					if (normalMap.empty()) {
						do {
							depthMap(v,u_curr) = (d+=diff);
							if (!confMap.empty()) confMap(v,u_curr) = c;
						} while (++u_curr<u);						
					} else {
						Point2f dir1, dir2;
						Normal2Dir(normalMap(v,u_first), dir1);
						Normal2Dir(normalMap(v,u), dir2);
						const Point2f dirDiff((dir2-dir1)/float(count+1));
						do {
							depthMap(v,u_curr) = (d+=diff);
							dir1 += dirDiff;
							Dir2Normal(dir1, normalMap(v,u_curr));
							if (!confMap.empty()) confMap(v,u_curr) = c;
						} while (++u_curr<u);						
					}
					#endif
				}
			}

			// reset counter
			count = 0;
		}
	}

	// 2. Column-wise: 处理每一列同上
	// for each column do
	for (int u=0; u<size.x; ++u) {

		// init counter
		unsigned count = 0;

		// for each element of the column do
		for (int v=0; v<size.y; ++v) {
			// get depth of this location
			const Depth& depth = depthMap(v,u);

			// if depth not valid => count and skip it
			if (depth <= 0) {
				++count;
				continue;
			}
			if (count == 0)
				continue;

			// check if gap is small enough
			// and value in range
			if (count <= nIpolGapSize && (unsigned)v > count) {
				// first value index for interpolation
				int v_curr(v-count);
				const int v_first(v_curr-1);
				// compute mean depth
				const Depth& depthFirst = depthMap(v_first,u);
				if (IsDepthSimilar(depthFirst, depth, fDepthDiffThreshold)) {
					#if 0
					// set all values with the average
					const Depth avg((depthFirst+depth)*0.5f);
					do {
						depthMap(v_curr,u) = avg;
					} while (++v_curr<v);						
					#else
					// interpolate values
					const Depth diff((depth-depthFirst)/(count+1));
					Depth d(depthFirst);
					const float c(confMap.empty() ? 0.f : MINF(confMap(v_first,u), confMap(v,u)));
					if (normalMap.empty()) {
						do {
							depthMap(v_curr,u) = (d+=diff);
							if (!confMap.empty()) confMap(v_curr,u) = c;
						} while (++v_curr<v);						
					} else {
						Point2f dir1, dir2;
						Normal2Dir(normalMap(v_first,u), dir1);
						Normal2Dir(normalMap(v,u), dir2);
						const Point2f dirDiff((dir2-dir1)/float(count+1));
						do {
							depthMap(v_curr,u) = (d+=diff);
							dir1 += dirDiff;
							Dir2Normal(dir1, normalMap(v_curr,u));
							if (!confMap.empty()) confMap(v_curr,u) = c;
						} while (++v_curr<v);						
					}
					#endif
				}
			}

			// reset counter
			count = 0;
		}
	}
	return true;
} // GapInterpolation
/*----------------------------------------------------------------*/


// filter depth-map, one pixel at a time, using confidence based fusion or neighbor pixels
// 逐像素滤波，利用邻域信息（即邻域深度）和置信度进行调整.如果bAdjust == True，则滤波后会修改原depth值（邻域投影当前得到depth如果与原深度相似则加和取平均代替原来depth）
// 如果不adjust，则直接根据计算有效views（即邻域投影当前得到depth与原深度相似的邻域帧）如果足够多则保留原depth否则置为0
bool DepthMapsData::FilterDepthMap(DepthData& depthDataRef, const IIndexArr& idxNeighbors, bool bAdjust)
{
	TD_TIMER_STARTD();

	// count valid neighbor depth-maps
	// 判断depth的邻域是否足够后续滤波
	ASSERT(depthDataRef.IsValid() && !depthDataRef.IsEmpty());
	const IIndex N = idxNeighbors.GetSize();
	ASSERT(OPTDENSE::nMinViewsFilter > 0 && scene.nCalibratedImages > 1);
	const IIndex nMinViews(MINF(OPTDENSE::nMinViewsFilter,scene.nCalibratedImages-1));
	// 帧间一致性计算所用的阈值，若邻域的个数小于这个阈值，则无法进行帧间一致性检查
	// 这个数值太大，则计算量会增加，太小，则容易达不到所需的滤波效果（一般可以在2、3、4、5中选取）
	const IIndex nMinViewsAdjust(MINF(OPTDENSE::nMinViewsFilterAdjust,scene.nCalibratedImages-1));
	if (N < nMinViews || N < nMinViewsAdjust) {
		DEBUG("error: depth map %3u can not be filtered", depthDataRef.GetView().GetID());
		return false;
	}

	// project all neighbor depth-maps to this image
	// depthDataRef的所有邻域的depth和conf，投影到当前帧。
	const DepthData::ViewData& imageRef = depthDataRef.images.First();  // 要进行帧间滤波的当前帧
	const Image8U::Size sizeRef(depthDataRef.depthMap.size());  // 要进行帧间滤波的当前帧的大小
	const Camera& cameraRef = imageRef.camera;
	DepthMapArr depthMaps(N);  // N个邻域投影在当前帧的深度图
	ConfidenceMapArr confMaps(N);  // 同上置信度
	FOREACH(n, depthMaps) {
		DepthMap& depthMap = depthMaps[n];
		depthMap.create(sizeRef);
		depthMap.memset(0);
		ConfidenceMap& confMap = confMaps[n];
		if (bAdjust) {
			confMap.create(sizeRef);
			confMap.memset(0);
		}
		const IIndex idxView = depthDataRef.neighbors[idxNeighbors[(IIndex)n]].ID;  //邻域ID
		const DepthData& depthData = arrDepthData[idxView];      //邻域depth相关数据
		const Camera& camera = depthData.images.First().camera;  //邻域相机内外参数
		const Image8U::Size size(depthData.depthMap.size());     // 邻域帧的大小
		for (int i=0; i<size.height; ++i) {
			for (int j=0; j<size.width; ++j) {
				const ImageRef x(j,i);
				const Depth depth(depthData.depthMap(x));
				if (depth == 0)
					continue;
				ASSERT(depth > 0);
				const Point3 X(camera.TransformPointI2W(Point3(x.x,x.y,depth)));  //计算邻域在世界坐标系下的xyz坐标
				const Point3 camX(cameraRef.TransformPointW2C(X));  //投影到当前帧ref的相机坐标系下
				if (camX.z <= 0)  // camX.z就是邻域像素投影到当前ref的相机坐标系下的深度值，若其小于等于0，则表示该深度值无效
					continue;
				#if 0
				// set depth on the rounded image projection only
				// 只投影到取整像素
				const ImageRef xRef(ROUND2INT(cameraRef.TransformPointC2I(camX)));
				if (!depthMap.isInside(xRef))
					continue;
				Depth& depthRef(depthMap(xRef));
				if (depthRef != 0 && depthRef < camX.z)
					continue;
				depthRef = camX.z;
				if (bAdjust)
					confMap(xRef) = depthData.confMap(x);
				#else
				// set depth on the 4 pixels around the image projection
				// 计算得到的像素坐标一般不一定是整数，所以上下取整取四个相邻像素得到四个投影坐标
				const Point2 imgX(cameraRef.TransformPointC2I(camX));  // 投影到像素坐标
				const ImageRef xRefs[4] = {
					ImageRef(FLOOR2INT(imgX.x), FLOOR2INT(imgX.y)),
					ImageRef(FLOOR2INT(imgX.x), CEIL2INT(imgX.y)),
					ImageRef(CEIL2INT(imgX.x), FLOOR2INT(imgX.y)),
					ImageRef(CEIL2INT(imgX.x), CEIL2INT(imgX.y))
				};
				for (int p=0; p<4; ++p) {
					const ImageRef& xRef = xRefs[p];
					// 判断是否在图像范围内
					if (!depthMap.isInside(xRef))
						continue;
					Depth& depthRef(depthMap(xRef));
					// 如果当前坐标已经被投影过（depthRef != 0）且深度图比现在投影的深度值小（depthRef < (Depth)camX.z）则不再投影。
					// 因为只选择靠相机比较近的深度（因为在空间中存在遮挡问题，距离相机越近，被遮挡的可能性越低。认为如果深度比当前大的是遮挡部分投影的）
					if (depthRef != 0 && depthRef < (Depth)camX.z)
						continue;
					depthRef = (Depth)camX.z;
					// 保留对应邻域帧的置信度值，因为后续会根据置信度值进行权重累加
					if (bAdjust)
						confMap(xRef) = depthData.confMap(x);
				}
				#endif
			}
		}
		#if TD_VERBOSE != TD_VERBOSE_OFF
		if (g_nVerbosityLevel > 3)
			ExportDepthMap(MAKE_PATH(String::FormatString("depthRender%04u.%04u.png", depthDataRef.GetView().GetID(), idxView)), depthMap);
		#endif
	}

	const float thDepthDiff(OPTDENSE::fDepthDiffThreshold*1.2f);
	DepthMap newDepthMap(sizeRef);  // 存放滤波后的深度信息
	ConfidenceMap newConfMap(sizeRef);  // 存放滤波后的置信度值
	#if TD_VERBOSE != TD_VERBOSE_OFF
	size_t nProcessed(0), nDiscarded(0);
	#endif
	if (bAdjust) {
		// average similar depths, and decrease confidence if depths do not agree
		// (inspired by: "Real-Time Visibility-Based Fusion of Depth Maps", Merrell, 2007)
		// 深度图调整
		for (int i=0; i<sizeRef.height; ++i) {
			for (int j=0; j<sizeRef.width; ++j) {
				const ImageRef xRef(j,i);
				const Depth depth(depthDataRef.depthMap(xRef));  // 要滤波的当前帧的深度值
				if (depth == 0) {
					newDepthMap(xRef) = 0;
					newConfMap(xRef) = 0;
					continue;
				}
				ASSERT(depth > 0);
				#if TD_VERBOSE != TD_VERBOSE_OFF
				++nProcessed;
				#endif
				// update best depth and confidence estimate with all estimates
				// 更新最好的depth和置信度用邻域投影到当前帧的depth
				float posConf(depthDataRef.confMap(xRef)), negConf(0);  // 要滤波的当前帧的置信度值
				Depth avgDepth(depth*posConf);
				unsigned nPosViews(0), nNegViews(0);  //有效深度的view个数，无效个数
				unsigned n(N);
				// 循环处理N个邻域
				do {
					const Depth d(depthMaps[--n](xRef));  // 当前帧上的像素在邻域帧上的投影位置的深度d
					if (d == 0) {
						// 如果加和小于最小views个数则直接认为depth不准把refer的对应depth踢掉设为0
						if (nPosViews + nNegViews + n < nMinViews)
							goto DiscardDepth;
						continue;
					}
					ASSERT(d > 0);
					// 判断refer的深度d与在当前帧上的深度值depth的差值是否小于阈值
					if (IsDepthSimilar(depth, d, thDepthDiff)) {
						// average similar depths
						// 平均相似深度图
						const float c(confMaps[n](xRef));
						avgDepth += d*c;  // 与置信度相乘类似带权重累加即为均值
						posConf += c;
						++nPosViews;  //有效像素加1
					} else {
						// penalize confidence
						// 如果比当前大认为是遮挡
						if (depth > d) {
							// occlusion
							negConf += confMaps[n](xRef);
						} else {
							// free-space violation
							// 比d小，因为我们只信任靠近相机的深度值，所以还是取当前深度在邻域投影的值对应的置信度
							const DepthData& depthData = arrDepthData[depthDataRef.neighbors[idxNeighbors[n]].ID];  // 取出邻域帧上对应的深度信息
							const Camera& camera = depthData.images.First().camera;  // 获取邻域帧的相机参数
							// 将当前帧上的像素点xRef投影到世界坐标系下，再投影到邻域帧的相机坐标系下
							const Point3 X(cameraRef.TransformPointI2W(Point3(xRef.x,xRef.y,depth)));
							const ImageRef x(ROUND2INT(camera.TransformPointW2I(X)));
							// 获取当前帧上的像素点在邻域帧上投影位置的置信度
							if (depthData.confMap.isInside(x)) {
								const float c(depthData.confMap(x));
								negConf += (c > 0 ? c : confMaps[n](xRef));
							} else
								negConf += confMaps[n](xRef);
						}
						++nNegViews;  // 无效的views加1
					}
				} while (n);
				ASSERT(nPosViews+nNegViews >= nMinViews);
				// if enough good views and positive confidence...
				// 如果有效的邻域足够多，置信度ok则认为是内点，更新depth
				if (nPosViews >= nMinViewsAdjust && posConf > negConf && ISINSIDE(avgDepth/=posConf, depthDataRef.dMin, depthDataRef.dMax)) {
					// consider this pixel an inlier
					newDepthMap(xRef) = avgDepth;
					newConfMap(xRef) = posConf - negConf;  // 不知道为何要使用这种方式来定义新的置信度，可能是因为新的置信度与对应的有效置信度成正比，与对应的无效置信度成反比
				} else {
					// consider this pixel an outlier
					// 否则设为0
					DiscardDepth:
					newDepthMap(xRef) = 0;
					newConfMap(xRef) = 0;
					#if TD_VERBOSE != TD_VERBOSE_OFF
					++nDiscarded;
					#endif
				}
			}
		}
	} else {
		// 如果不调整depth值则直接根据
		// remove depth if it does not agree with enough neighbors
		// 如果邻域与当前depth相似有一定的数量则保留否则剔除
		const float thDepthDiffStrict(OPTDENSE::fDepthDiffThreshold*0.8f);
		const unsigned nMinGoodViewsProc(75), nMinGoodViewsDeltaProc(65);
		const unsigned nDeltas(4);
		const unsigned nMinViewsDelta(nMinViews*(nDeltas-2));
		const ImageRef xDs[nDeltas] = { ImageRef(-1,0), ImageRef(1,0), ImageRef(0,-1), ImageRef(0,1) };
		for (int i=0; i<sizeRef.height; ++i) {
			for (int j=0; j<sizeRef.width; ++j) {
				const ImageRef xRef(j,i);
				const Depth depth(depthDataRef.depthMap(xRef));
				if (depth == 0) {
					newDepthMap(xRef) = 0;
					newConfMap(xRef) = 0;
					continue;
				}
				ASSERT(depth > 0);
				#if TD_VERBOSE != TD_VERBOSE_OFF
				++nProcessed;
				#endif
				// check if very similar with the neighbors projected to this pixel
				// 判断邻域投影到当前像素的深度是否与原深度相似
				{
					unsigned nGoodViews(0);
					unsigned nViews(0);
					unsigned n(N);
					do {
						const Depth d(depthMaps[--n](xRef));
						if (d > 0) {
							// valid view
							// 如果深度差值小于阈值则有效
							++nViews;
							if (IsDepthSimilar(depth, d, thDepthDiffStrict)) {
								// agrees with this neighbor
								++nGoodViews;
							}
						}
					} while (n);
					// 如果有效view小于最小views或者小于最小goodview则直接丢弃
					// 可以通过控制nMinViews、nMinGoodViewsProc来控制滤波后的深度图质量
					// 例如，nMinViews越大，被滤除掉的点越多，保留下来的深度值越准确
					if (nGoodViews < nMinViews || nGoodViews < nViews*nMinGoodViewsProc/100) {
						#if TD_VERBOSE != TD_VERBOSE_OFF
						++nDiscarded;
						#endif
						newDepthMap(xRef) = 0;
						newConfMap(xRef) = 0;
						continue;
					}
				}
				// check if similar with the neighbors projected around this pixel
				// 判断投影到当前像素的邻域的depth是否与原depth相似
				{
					unsigned nGoodViews(0);
					unsigned nViews(0);
					for (unsigned d=0; d<nDeltas; ++d) {
						const ImageRef xDRef(xRef+xDs[d]);
						unsigned n(N);
						do {
							const Depth d(depthMaps[--n](xDRef));
							if (d > 0) {
								// valid view
								++nViews;
								if (IsDepthSimilar(depth, d, thDepthDiff)) {
									// agrees with this neighbor
									// 相似加1
									++nGoodViews;
								}
							}
						} while (n);
					}
					if (nGoodViews < nMinViewsDelta || nGoodViews < nViews*nMinGoodViewsDeltaProc/100) {
						#if TD_VERBOSE != TD_VERBOSE_OFF
						++nDiscarded;
						#endif
						newDepthMap(xRef) = 0;
						newConfMap(xRef) = 0;
						continue;
					}
				}
				// enough good views, keep it
				// 有足够多的views则直接保留原来的depth
				newDepthMap(xRef) = depth;
				newConfMap(xRef) = depthDataRef.confMap(xRef);
			}
		}
	}
	if (!SaveDepthMap(ComposeDepthFilePath(imageRef.GetID(), "filtered.dmap"), newDepthMap) ||
		!SaveConfidenceMap(ComposeDepthFilePath(imageRef.GetID(), "filtered.cmap"), newConfMap))
		return false;

	DEBUG("Depth map %3u filtered using %u other images: %u/%u depths discarded (%s)",
		imageRef.GetID(), N, nDiscarded, nProcessed, TD_TIMER_GET_FMT().c_str());
	return true;
} // FilterDepthMap
/*----------------------------------------------------------------*/


// fuse all depth-maps by simply projecting them in a 3D point cloud
// in the world coordinate space
void DepthMapsData::MergeDepthMaps(PointCloud& pointcloud, bool bEstimateColor, bool bEstimateNormal)
{
	TD_TIMER_STARTD();

	// estimate total number of 3D points that will be generated
	size_t nPointsEstimate(0);
	for (const DepthData& depthData: arrDepthData)
		if (depthData.IsValid())
			nPointsEstimate += (size_t)depthData.depthMap.size().area()*7/10;

	// fuse all depth-maps
	size_t nDepthMaps(0), nDepths(0);
	pointcloud.points.reserve(nPointsEstimate);
	pointcloud.pointViews.reserve(nPointsEstimate);
	if (bEstimateColor)
		pointcloud.colors.reserve(nPointsEstimate);
	if (bEstimateNormal)
		pointcloud.normals.reserve(nPointsEstimate);
	Util::Progress progress(_T("Merged depth-maps"), arrDepthData.size());
	GET_LOGCONSOLE().Pause();
	FOREACH(idxImage, arrDepthData) {
		TD_TIMER_STARTD();
		DepthData& depthData = arrDepthData[idxImage];
		ASSERT(depthData.GetView().GetLocalID(scene.images) == idxImage);
		if (!depthData.IsValid())
			continue;
		if (depthData.IncRef(ComposeDepthFilePath(depthData.GetView().GetID(), "dmap")) == 0)
			return;
		ASSERT(!depthData.IsEmpty());
		const DepthData::ViewData& image = depthData.GetView();
		const size_t nNumPointsPrev(pointcloud.points.size());
		for (int i=0; i<depthData.depthMap.rows; ++i) {
			for (int j=0; j<depthData.depthMap.cols; ++j) {
				// ignore invalid depth
				const ImageRef x(j,i);
				const Depth depth(depthData.depthMap(x));
				if (depth == 0)
					continue;
				ASSERT(ISINSIDE(depth, depthData.dMin, depthData.dMax));
				// create the corresponding 3D point
				pointcloud.points.emplace_back(image.camera.TransformPointI2W(Point3(Cast<float>(x),depth)));
				pointcloud.pointViews.emplace_back().push_back(idxImage);
				if (bEstimateColor)
					pointcloud.colors.emplace_back(image.pImageData->image(x));
				if (bEstimateNormal)
					depthData.GetNormal(x, pointcloud.normals.emplace_back());
				++nDepths;
			}
		}
		depthData.DecRef();
		++nDepthMaps;
		ASSERT(pointcloud.points.size() == pointcloud.pointViews.size());
		DEBUG_ULTIMATE("Depths map for reference image %3u merged using %u depths maps: %u new points (%s)",
			idxImage, depthData.images.size()-1, pointcloud.points.size()-nNumPointsPrev, TD_TIMER_GET_FMT().c_str());
		progress.display(idxImage+1);
	}
	GET_LOGCONSOLE().Play();
	progress.close();

	DEBUG_EXTRA("Depth-maps merged: %u depth-maps, %u depths, %u points (%d%%%%) (%s)",
		nDepthMaps, nDepths, pointcloud.points.size(), ROUND2INT(100.f*pointcloud.points.size()/nDepths), TD_TIMER_GET_FMT().c_str());
} // MergeDepthMaps
/*----------------------------------------------------------------*/

// fuse all valid depth-maps in the same 3D point cloud;
// join points very likely to represent the same 3D point and
// filter out points blocking the view
// depth融合：将所有有效depth融合为一个点云并带有views信息（这个view信息表示每个点云来自于哪些图像帧的深度信息，对于下一步的曲面重建十分重要）
// 参考Accurate Multiple View 3D Reconstruction Using Patch-Based Stereo for Large-Scale Scenes D部分
// 和视差一致性检查很相似
void DepthMapsData::FuseDepthMaps(PointCloud& pointcloud, bool bEstimateColor, bool bEstimateNormal)
{
	TD_TIMER_STARTD();

	struct Proj {  // 用于进行投影的结构体
		union {
			uint32_t idxPixel;
			struct {
				uint16_t x, y; // image pixel coordinates
			};
		};
		inline Proj() {}
		inline Proj(uint32_t _idxPixel) : idxPixel(_idxPixel) {}
		inline Proj(const ImageRef& ir) : x(ir.x), y(ir.y) {}
		inline ImageRef GetCoord() const { return ImageRef(x,y); }
	};
	typedef SEACAVE::cList<Proj,const Proj&,0,4,uint32_t> ProjArr;
	typedef SEACAVE::cList<ProjArr,const ProjArr&,1,65536> ProjsArr;

	// find best connected images
	// 找具有有效depth的帧 并记录存储到connection中记录ID和邻域个数
	IndexScoreArr connections(scene.images.size());
	size_t nPointsEstimate(0);
	bool bNormalMap(true);
	#ifdef DENSE_USE_OPENMP
	bool bAbort(false);
	#pragma omp parallel for shared(connections, nPointsEstimate, bNormalMap, bAbort)
	for (int64_t i=0; i<(int64_t)scene.images.size(); ++i) {
		#pragma omp flush (bAbort)
		if (bAbort)
			continue;
		const IIndex idxImage((IIndex)i);
	#else
	FOREACH(idxImage, scene.images) {
	#endif
		IndexScore& connection = connections[idxImage];
		DepthData& depthData = arrDepthData[idxImage];
		// 判断深度图是否有效
		if (!depthData.IsValid()) {
			connection.idx = NO_ID;
			connection.score = 0;
			continue;
		}
		// 深度图加载
		const String fileName(ComposeDepthFilePath(depthData.GetView().GetID(), "dmap"));
		if (depthData.IncRef(fileName) == 0) {
			#ifdef DENSE_USE_OPENMP
			bAbort = true;
			#pragma omp flush (bAbort)
			continue;
			#else
			return;
			#endif
		}
		ASSERT(!depthData.IsEmpty());
		connection.idx = idxImage;
		connection.score = (float)scene.images[idxImage].neighbors.size();  // 一般认为，邻域帧越多，对应的深度估计越好
		if (bEstimateNormal && depthData.normalMap.empty()) {
			EstimateNormalMap(depthData.images.front().camera.K, depthData.depthMap, depthData.normalMap);
			if (!depthData.Save(fileName)) {
				#ifdef DENSE_USE_OPENMP
				bAbort = true;
				#pragma omp flush (bAbort)
				continue;
				#else
				return;
				#endif
			}
		}
		#ifdef DENSE_USE_OPENMP
		#pragma omp critical
		#endif
		{
		// 统计所有的深度图融合之后可能得到的点云个数，简单乘以系数0.5*0.3大概估计有效点云的个数
		// depthData.depthMap.area()表示深度图depthData.depthMap所包含的像素数目，
		// 若深度图中的每个像素位置上的深度都对点云有贡献，那么就可能产生depthData.depthMap.area()个点云，
		// 然而，实际上一张深度图中的点不太可能全都对产生点云有贡献，因此此处引入两个系数来进行调整，
		// 这其实是一个内存预分配的过程，系数设大设小对最终的结果没有影响
		nPointsEstimate += ROUND2INT(depthData.depthMap.area()*(0.5f/*valid*/*0.3f/*new*/));
		if (depthData.normalMap.empty())
			bNormalMap = false;
		}
	}
	#ifdef DENSE_USE_OPENMP
	if (bAbort)
		return;
	#endif
	// 根据score进行排序升序，优先处理邻域最多的帧
	connections.Sort();
	while (!connections.empty() && connections.back().score <= 0)
		connections.pop_back();
	if (connections.empty()) {
		DEBUG("error: no valid depth-maps found");
		return;
	}

	// fuse all depth-maps, processing the best connected images first
	// 融合所有depth，首先处理连接最好的image（邻域最多）
	const unsigned nMinViewsFuse(MINF(OPTDENSE::nMinViewsFuse, scene.images.size()));  // 最小融合帧的个数
	const float normalError(COS(FD2R(OPTDENSE::fNormalDiffThreshold)));  //法线误差
	CLISTDEF0(Depth*) invalidDepths(0, 32);
	size_t nDepths(0);
	typedef TImage<cuint32_t> DepthIndex;
	typedef cList<DepthIndex> DepthIndexArr;
	DepthIndexArr arrDepthIdx(scene.images.size());  // 用来记录depth是否融合，如果已经融合则记录融合对应的point的id否则No_ID
	ProjsArr projs(0, nPointsEstimate);  // 存放的是point在每个view中的投影坐标
	if (bEstimateNormal && !bNormalMap)
		bEstimateNormal = false;
	pointcloud.points.reserve(nPointsEstimate);  // 存放顶点坐标
	pointcloud.pointViews.reserve(nPointsEstimate);  // 每个点的views id(能看到该点的image)
	pointcloud.pointWeights.reserve(nPointsEstimate);  // 每个点的view的权重
	if (bEstimateColor)  // 是否对颜色进行重建，一般不进行计算
		pointcloud.colors.reserve(nPointsEstimate);
	if (bEstimateNormal)  // 是否对法线进行重建，一般不进行计算
		pointcloud.normals.reserve(nPointsEstimate);
	Util::Progress progress(_T("Fused depth-maps"), connections.size());
	GET_LOGCONSOLE().Pause();
	// 逐帧融合
	for (const IndexScore& connection: connections) {
		TD_TIMER_STARTD();
		const uint32_t idxImage(connection.idx);
		const DepthData& depthData(arrDepthData[idxImage]);
		ASSERT(!depthData.images.empty() && !depthData.neighbors.empty());
		// 初始化邻域arrDepthIdx
		for (const ViewScore& neighbor: depthData.neighbors) {
			DepthIndex& depthIdxs = arrDepthIdx[neighbor.ID];
			if (!depthIdxs.empty())
				continue;
			const DepthData& depthDataB(arrDepthData[neighbor.ID]);
			if (depthDataB.IsEmpty())
				continue;
			depthIdxs.create(depthDataB.depthMap.size());
			depthIdxs.memset((uint8_t)NO_ID);
		}
		ASSERT(!depthData.IsEmpty());
		// 初始化当前帧depthIdxs
		const Image8U::Size sizeMap(depthData.depthMap.size());  // 深度图的大小
		const Image& imageData = *depthData.images.front().pImageData;  // 深度图对应的图像颜色数据
		ASSERT(&imageData-scene.images.data() == idxImage);
		DepthIndex& depthIdxs = arrDepthIdx[idxImage];  // 用来记录depth是否已经被融合过或被剔除，因为处理邻域帧的时候也会修改这部分的数值，对其进行记录可以避免重复计算
		if (depthIdxs.empty()) {
			depthIdxs.create(Image8U::Size(imageData.width, imageData.height));
			depthIdxs.memset((uint8_t)NO_ID);
		}
		const size_t nNumPointsPrev(pointcloud.points.size());  //不参与计算log用的
		// 逐点融合depth
		for (int i=0; i<sizeMap.height; ++i) {
			for (int j=0; j<sizeMap.width; ++j) {
				const ImageRef x(j,i);  // 对应的像素坐标
				const Depth depth(depthData.depthMap(x));
				if (depth == 0)
					continue;
				++nDepths;
				ASSERT(ISINSIDE(depth, depthData.dMin, depthData.dMax));
				uint32_t& idxPoint = depthIdxs(x);  // 对应的点云的id
				// 如果当前点深度已经被处理过，则跳过
				if (idxPoint != NO_ID)
					continue;
				// create the corresponding 3D point
				// 计算新插入三维点的id
				idxPoint = (uint32_t)pointcloud.points.size();
				PointCloud::Point& point = pointcloud.points.emplace_back();
				// uv d转世界坐标系下的xyz
				point = imageData.camera.TransformPointI2W(Point3(Point2f(x),depth));
				// 当前点的views计算
				PointCloud::ViewArr& views = pointcloud.pointViews.emplace_back();
				// 首先插入当前帧id,这个肯定是能看到的view，后续view从idxImage的邻域中选择
				views.emplace_back(idxImage);
				// 计算插入的view的权重
				PointCloud::WeightArr& weights = pointcloud.pointWeights.emplace_back();
				// 将x的置信度转weight
				REAL confidence(weights.emplace_back(Conf2Weight(depthData.confMap.empty() ? 1.f : depthData.confMap(x),depth)));
				// 把uv坐标存入projs
				ProjArr& pointProjs = projs.emplace_back();
				pointProjs.emplace_back(Proj(x));
				// 计算对应的法向量
				const PointCloud::Normal normal(bNormalMap ? Cast<Normal::Type>(imageData.camera.R.t()*Cast<REAL>(depthData.normalMap(x))) : Normal(0,0,-1));
				ASSERT(ISEQUAL(norm(normal), 1.f));
				// check the projection in the neighbor depth-maps。判断当前位置在邻域帧上的深度估计是否也是合适的，若合适，则会对邻域帧的ID进行记录
				Point3 X(point*confidence);
				Pixel32F C(Cast<float>(imageData.image(x))*confidence);
				PointCloud::Normal N(normal*confidence);
				invalidDepths.clear();  //记录无效depth
				for (const ViewScore& neighbor: depthData.neighbors) {
					const IIndex idxImageB(neighbor.ID);
					// 邻域depth
					DepthData& depthDataB = arrDepthData[idxImageB];
					if (depthDataB.IsEmpty())
						continue;
					// 邻域image
					const Image& imageDataB = scene.images[idxImageB];
					// 将point投影到邻域idxImageB的相机坐标系下
					const Point3f pt(imageDataB.camera.ProjectPointP3(point));
					if (pt.z <= 0)
						continue;
					// point投影到邻域帧得到像素坐标xB
					const ImageRef xB(ROUND2INT(pt.x/pt.z), ROUND2INT(pt.y/pt.z));
					DepthMap& depthMapB = depthDataB.depthMap;
					if (!depthMapB.isInside(xB))
						continue;
					Depth& depthB = depthMapB(xB);
					if (depthB == 0)
						continue;
					uint32_t& idxPointB = arrDepthIdx[idxImageB](xB);
					// 如果xB像素已经被融合过则跳过
					if (idxPointB != NO_ID)
						continue;
					// 如果point投影到邻域的depth（pt/z）比邻域本身的depthB相似且法线相似，则把邻域的view信息插入point的views
					if (IsDepthSimilar(pt.z, depthB, OPTDENSE::fDepthDiffThreshold)) {
						// check if normals agree
						const PointCloud::Normal normalB(bNormalMap ? Cast<Normal::Type>(imageDataB.camera.R.t()*Cast<REAL>(depthDataB.normalMap(xB))) : Normal(0,0,-1));
						ASSERT(ISEQUAL(norm(normalB), 1.f));
						if (normal.dot(normalB) > normalError) {
							// add view to the 3D point
							ASSERT(views.FindFirst(idxImageB) == PointCloud::ViewArr::NO_INDEX);
							const float confidenceB(Conf2Weight(depthDataB.confMap.empty() ? 1.f : depthDataB.confMap(xB),depthB));
							const IIndex idx(views.InsertSort(idxImageB));
							weights.InsertAt(idx, confidenceB);
							pointProjs.InsertAt(idx, Proj(xB));
							// 记录depth对应的融合后的point的id，避免后续重复处理
							idxPointB = idxPoint;
							X += imageDataB.camera.TransformPointI2W(Point3(Point2f(xB),depthB))*REAL(confidenceB);
							if (bEstimateColor)
								C += Cast<float>(imageDataB.image(xB))*confidenceB;
							if (bEstimateNormal)
								N += normalB*confidenceB;
							confidence += confidenceB;
							continue;
						}
					}
					// 如果比邻域本身深度值小，则邻域深度是被遮挡的，则丢弃depthB
					if (pt.z < depthB) {
						// discard depth
						invalidDepths.emplace_back(&depthB);
					}
				}
				// 如果点的总的views小于最小view则剔除原先插入的point及相关参数
				if (views.size() < nMinViewsFuse) {
					// remove point
					FOREACH(v, views) {
						const IIndex idxImageB(views[v]);
						const ImageRef x(pointProjs[v].GetCoord());
						ASSERT(arrDepthIdx[idxImageB].isInside(x) && arrDepthIdx[idxImageB](x).idx != NO_ID);
						arrDepthIdx[idxImageB](x).idx = NO_ID;
					}
					// .pop_back()剔除掉容器中的最后一个元素
					projs.pop_back();
					pointcloud.pointWeights.pop_back();
					pointcloud.pointViews.pop_back();
					pointcloud.points.pop_back();
				} else {
					// this point is valid, store it
					// 点是有效则存储
					const REAL nrm(REAL(1)/confidence);
					point = X*nrm;
					ASSERT(ISFINITE(point));
					if (bEstimateColor)
						pointcloud.colors.emplace_back((C*(float)nrm).cast<uint8_t>());
					if (bEstimateNormal)
						pointcloud.normals.emplace_back(normalized(N*(float)nrm));
					// invalidate all neighbor depths that do not agree with it
					for (Depth* pDepth: invalidDepths)
						*pDepth = 0;
				}
			}
		}
		ASSERT(pointcloud.points.size() == pointcloud.pointViews.size() && pointcloud.points.size() == pointcloud.pointWeights.size() && pointcloud.points.size() == projs.size());
		DEBUG_ULTIMATE("Depths map for reference image %3u fused using %u depths maps: %u new points (%s)", idxImage, depthData.images.size()-1, pointcloud.points.size()-nNumPointsPrev, TD_TIMER_GET_FMT().c_str());
		progress.display(&connection-connections.data());
	}
	GET_LOGCONSOLE().Play();
	progress.close();
	arrDepthIdx.Release();

	DEBUG_EXTRA("Depth-maps fused and filtered: %u depth-maps, %u depths, %u points (%d%%%%) (%s)",
		connections.size(), nDepths, pointcloud.points.size(), ROUND2INT((100.f*pointcloud.points.size())/nDepths), TD_TIMER_GET_FMT().c_str());
	// 法线计算可选 后续计算不需要
	if (bEstimateNormal && !pointcloud.points.empty() && pointcloud.normals.empty()) {
		// estimate normal also if requested (quite expensive if normal-maps not available)
		// 计算法线，如果没有必要就不用计算因为比较费时
		TD_TIMER_STARTD();
		pointcloud.normals.resize(pointcloud.points.size());
		const int64_t nPoints((int64_t)pointcloud.points.size());
		#ifdef DENSE_USE_OPENMP
		#pragma omp parallel for
		#endif
		for (int64_t i=0; i<nPoints; ++i) {
			PointCloud::WeightArr& weights = pointcloud.pointWeights[i];
			ASSERT(!weights.empty());
			IIndex idxView(0);
			float bestWeight = weights.front();
			for (IIndex idx=1; idx<weights.size(); ++idx) {
				const PointCloud::Weight& weight = weights[idx];
				if (bestWeight < weight) {
					bestWeight = weight;
					idxView = idx;
				}
			}
			const DepthData& depthData(arrDepthData[pointcloud.pointViews[i][idxView]]);
			ASSERT(depthData.IsValid() && !depthData.IsEmpty());
			// 法线计算
			depthData.GetNormal(projs[i][idxView].GetCoord(), pointcloud.normals[i]);
		}
		DEBUG_EXTRA("Normals estimated for the dense point-cloud: %u normals (%s)", pointcloud.GetSize(), TD_TIMER_GET_FMT().c_str());
	}

	// release all depth-maps
	// 内存释放，释放所有的depth
	for (DepthData& depthData: arrDepthData)
		if (depthData.IsValid())
			depthData.DecRef();
} // FuseDepthMaps
/*----------------------------------------------------------------*/



// S T R U C T S ///////////////////////////////////////////////////
// 法向量计算，启动多线程
DenseDepthMapData::DenseDepthMapData(Scene& _scene, int _nFusionMode)
	: scene(_scene), depthMaps(_scene), idxImage(0), sem(1), nEstimationGeometricIter(-1), nFusionMode(_nFusionMode)
{
	if (nFusionMode < 0) {
		STEREO::SemiGlobalMatcher::CreateThreads(scene.nMaxThreads);
		if (nFusionMode == -1)
			OPTDENSE::nOptimize = 0;
	}
}

// 析构 线程释放
DenseDepthMapData::~DenseDepthMapData()
{
	if (nFusionMode < 0)
		STEREO::SemiGlobalMatcher::DestroyThreads();
}

void DenseDepthMapData::SignalCompleteDepthmapFilter()
{
	ASSERT(idxImage > 0);
	if (Thread::safeDec(idxImage) == 0)
		sem.Signal((unsigned)images.GetSize()*2);
}
/*----------------------------------------------------------------*/



// S T R U C T S ///////////////////////////////////////////////////

static void* DenseReconstructionEstimateTmp(void*);
static void* DenseReconstructionFilterTmp(void*);
/**
 * @brief 深度计算
 * 
 * @param[in] nFusionMode   控制参数：-1采用SGM/tSGM计算，只输出视差图 -2 计算加融合视差图 0是深度图计算和融合 1采用patchMatch方式 只输出深度图
 * <0是用SGM计算视差的方式计算深度，>=0用PatchMatch方式计算深度
 * @return true  计算成功
 * @return false 深度图计算失败
 */
bool Scene::DenseReconstruction(int nFusionMode, bool bCrop2ROI, float fBorderROI)
{
	DenseDepthMapData data(*this, nFusionMode);

	// estimate depth-maps
	// depth map fusion mode (-2 - fuse disparity-maps, -1 - export disparity-maps only, 0 - depth-maps & fusion, 1 - export depth-maps only)
	// Step 1 深度图计算两种方式：patchMatch nFusionMode=1；SGM/tSGM nFusionMode=-1 通过nFusionMode控制可以通过这个参数设置对
	// 这两种算法进行效果性能对比
	if (!ComputeDepthMaps(data))
		return false;
	if (ABS(nFusionMode) == 1)
		return true;

	// fuse all depth-maps
	// Step 2 将所有depth融合为一个带views信息的点云。每个点记录所有能看到的view的ID信息
	// 深度图到网格模型有两类方法：
	// 深度图 -> 深度融合产生点云 -> 网格划分产生网格模型（例如，OpenMVS）
	// 深度图 -> 基于TSDF产生网格模型（一般要求所用的深度信息质量较好，否则重建的效果很差，例如，KinectFusion）
	pointcloud.Release();
	if (OPTDENSE::nMinViewsFuse < 2) {
		// merge depth-maps
		data.depthMaps.MergeDepthMaps(pointcloud, OPTDENSE::nEstimateColors == 2, OPTDENSE::nEstimateNormals == 2);
	} else {
		// fuse depth-maps
		data.depthMaps.FuseDepthMaps(pointcloud, OPTDENSE::nEstimateColors == 2, OPTDENSE::nEstimateNormals == 2);
	}
	#if TD_VERBOSE != TD_VERBOSE_OFF
	if (g_nVerbosityLevel > 2) {
		// print number of points with 3+ views
		size_t nPoints1m(0), nPoints2(0), nPoints3p(0);
		FOREACHPTR(pViews, pointcloud.pointViews) {
			switch (pViews->GetSize())
			{
			case 0:
			case 1:
				++nPoints1m;
				break;
			case 2:
				++nPoints2;
				break;
			default:
				++nPoints3p;
			}
		}
		VERBOSE("Dense point-cloud composed of:\n\t%u points with 1- views\n\t%u points with 2 views\n\t%u points with 3+ views", nPoints1m, nPoints2, nPoints3p);
	}
	#endif
	// Step 3，4 点云颜色和法线计算（可选，不建议，耗时）
	if (!pointcloud.IsEmpty()) {
		if (bCrop2ROI && IsBounded()) {
			TD_TIMER_START();
			const size_t numPoints = pointcloud.GetSize();
			const OBB3f ROI(fBorderROI == 0 ? obb : (fBorderROI > 0 ? OBB3f(obb).EnlargePercent(fBorderROI) : OBB3f(obb).Enlarge(-fBorderROI)));
			pointcloud.RemovePointsOutside(ROI);
			VERBOSE("Point-cloud trimmed to ROI: %u points removed (%s)",
				numPoints-pointcloud.GetSize(), TD_TIMER_GET_FMT().c_str());
		}
		if (pointcloud.colors.IsEmpty() && OPTDENSE::nEstimateColors == 1)
			EstimatePointColors(images, pointcloud);
		if (pointcloud.normals.IsEmpty() && OPTDENSE::nEstimateNormals == 1)
			EstimatePointNormals(images, pointcloud);
	}

	if (OPTDENSE::bRemoveDmaps) {
		// delete all depth-map files
		FOREACH(i, images) {
			const DepthData& depthData = data.depthMaps.arrDepthData[i];
			if (!depthData.IsValid())
				continue;
			File::deleteFile(ComposeDepthFilePath(depthData.GetView().GetID(), "dmap"));
		}
	}
	return true;
} // DenseReconstruction
/*----------------------------------------------------------------*/

// do first half of dense reconstruction: depth map computation
// results are saved to "data"
// 稠密重建的第一步：计算结果保存在data中
/**
 * @brief 深度图计算patch/sgm,tsgm
 * 
 * @param[in/out] data 存储深度计算需要的数据和计算结果：深度图，此外还包含线程控制、计算模式等参数、数据设置
 * @return true 
 * @return false 
 */
bool Scene::ComputeDepthMaps(DenseDepthMapData& data)
{
	// compute point-cloud from the existing mesh
	if (!mesh.IsEmpty() && !ImagesHaveNeighbors()) {
		SampleMeshWithVisibility();
		mesh.Release();
	}
	
	// compute point-cloud from the existing mesh
	if (IsEmpty() && !ImagesHaveNeighbors()) {
		VERBOSE("warning: empty point-cloud, rough neighbor views selection based on image pairs baseline");
		EstimateNeighborViewsPointCloud();
	}

	{
	// maps global view indices to our list of views to be processed
	// view在所有图像中的索引与要处理的图像list中索引对应关系。
	IIndexArr imagesMap;

	// prepare images for dense reconstruction (load if needed)
	// Step 1 数据准备：load 图像，对图像进行筛选去除无效图像,并根据传入的参数nResolutionLeval对load的图像做resize,对应相机参数做同样调整。
	{
		TD_TIMER_START();
		data.images.Reserve(images.GetSize());  // 注意，用于存储用于计算深度信息的图像的id，此处分配的空间并不一定会全用到，因为其中有些图像会被置为无效
		imagesMap.Resize(images.GetSize());     // 用于记录计算深度图的图像中，哪些可用，哪些不可用，对于参与计算深度的图像，记录其在data.images中的id是多少
		#ifdef DENSE_USE_OPENMP
		bool bAbort(false);
		#pragma omp parallel for shared(data, bAbort)
		for (int_t ID=0; ID<(int_t)images.GetSize(); ++ID) {
			#pragma omp flush (bAbort)
			if (bAbort)
				continue;
			const IIndex idxImage((IIndex)ID);
		#else
		FOREACH(idxImage, images) {  
		#endif
			// skip invalid, uncalibrated or discarded images
			// 跳过无效的，未标定的 或被丢弃的图像
			Image& imageData = images[idxImage];
			// isvalid 判断imageData的poseID是否等于NO_ID（无效）。我们也可以在外部传入数据时如果有些帧不想参与计算可以设置其为NO_ID
			if (!imageData.IsValid()) {
				#ifdef DENSE_USE_OPENMP
				#pragma omp critical
				#endif
				imagesMap[idxImage] = NO_ID;
				continue;
			}
			// map image index
			// 计算图像的map
			#ifdef DENSE_USE_OPENMP
			#pragma omp critical
			#endif
			{
				// 由于data.images中存储的是参与深度计算的图像索引，因此每次往其中插入数据（参与深度计算的图像索引）前，其大小就可视作新插入数据在data.images中的索引位置
				imagesMap[idxImage] = data.images.GetSize();
				data.images.Insert(idxImage);
			}
			// reload image at the appropriate resolution，
			// imagesize-图像最长边，nResolutionLevel==0->使用原图进行深度计算
			// 用于计算深度图的图像的最大分辨率计算方式：imagesize=max(width,height), nMaxResolution=imagesize/(2^nResolutionLevel)
			//                                    if(nMaxResolution<OPTDENSE::nMinResolution),从level为0开始找到最开始大于OPTDENSE::nMinResolution的值作为nMaxResolution;
			//                                    最后 nMaxResolution=min(nMaxResolution,OPTDENSE::nMaxResolution)
			unsigned nResolutionLevel(OPTDENSE::nResolutionLevel);
			const unsigned nMaxResolution(imageData.RecomputeMaxResolution(nResolutionLevel, OPTDENSE::nMinResolution, OPTDENSE::nMaxResolution));
			// 根据计算的分辨率对图像进行resize
			if (!imageData.ReloadImage(nMaxResolution)) {
				#ifdef DENSE_USE_OPENMP
				bAbort = true;
				#pragma omp flush (bAbort)
				continue;
				#else
				return false;
				#endif
			}
			//根据resize的图像对相机内参做同样调整
			imageData.UpdateCamera(platforms);
			// print image camera
			DEBUG_ULTIMATE("K%d = \n%s", idxImage, cvMat2String(imageData.camera.K).c_str());
			DEBUG_LEVEL(3, "R%d = \n%s", idxImage, cvMat2String(imageData.camera.R).c_str());
			DEBUG_LEVEL(3, "C%d = \n%s", idxImage, cvMat2String(imageData.camera.C).c_str());
		}
		#ifdef DENSE_USE_OPENMP
		if (bAbort || data.images.IsEmpty()) {
		#else
		if (data.images.IsEmpty()) {
		#endif
			VERBOSE("error: preparing images for dense reconstruction failed (errors loading images)");
			return false;
		}
		VERBOSE("Preparing images for dense reconstruction completed: %d images (%s)", images.GetSize(), TD_TIMER_GET_FMT().c_str());
	}

	// select images to be used for dense reconstruction
	// Step 2 给每帧图像选择一个最佳参考帧用来计算depth
	// （对于每一个图像帧，立体视觉的深度图计算至少需要有一个参考帧，
	// 此处实现的sgm以及patchMatch都只考虑了一个参考帧，所以需要从邻域帧中选择一个最佳的参考帧）。
	// 主要是利用每帧对应的points和帧间夹角信息来筛选邻域帧。
	{
		TD_TIMER_START();
		// for each image, find all useful neighbor views
		// Step 2.1 给每一帧图像找到所有满足条件的邻域帧（有三个条件分别与面积、尺度以及角度相关，具体参见代码）
		IIndexArr invalidIDs;
		#ifdef DENSE_USE_OPENMP
		#pragma omp parallel for shared(data, invalidIDs)
		for (int_t ID=0; ID<(int_t)data.images.GetSize(); ++ID) {
			const IIndex idx((IIndex)ID);
		#else
		FOREACH(idx, data.images) {
		#endif
			const IIndex idxImage(data.images[idx]);
			ASSERT(imagesMap[idxImage] != NO_ID);
			DepthData& depthData(data.depthMaps.arrDepthData[idxImage]);
			// 如果没有选到足够的邻域就将该帧设为无效不进行depth计算。具体选帧算法见scene.cpp中的SelectNeighborViews
			if (!data.depthMaps.SelectViews(depthData)) {
				#ifdef DENSE_USE_OPENMP
				#pragma omp critical
				#endif
				invalidIDs.InsertSort(idx);
			}
		}
		// 无效帧移除
		RFOREACH(i, invalidIDs) {
			const IIndex idx(invalidIDs[i]);
			imagesMap[data.images.Last()] = idx;
			imagesMap[data.images[idx]] = NO_ID;
			data.images.RemoveAt(idx);
		}
		// Step 2.2 globally select a target view for each reference image
		// 从邻域帧中选择最佳帧用来深度恢复，具体见当前cpp的SelectViews函数
		if (OPTDENSE::nNumViews == 1 && !data.depthMaps.SelectViews(data.images, imagesMap, data.neighborsMap)) {
			VERBOSE("error: no valid images to be dense reconstructed");
			return false;
		}
		ASSERT(!data.images.IsEmpty());
		VERBOSE("Selecting images for dense reconstruction completed: %d images (%s)", data.images.GetSize(), TD_TIMER_GET_FMT().c_str());
	}
	}

	#ifdef _USE_CUDA
	// initialize CUDA
	if (CUDA::desiredDeviceID >= -1 && data.nFusionMode >= 0) {
		data.depthMaps.pmCUDA = new PatchMatchCUDA(CUDA::desiredDeviceID);
		if (CUDA::devices.IsEmpty())
			data.depthMaps.pmCUDA.Release();
		else
			data.depthMaps.pmCUDA->Init(false);
	}
	#endif // _USE_CUDA

	// initialize the queue of images to be processed
	// Step 3 深度计算（也就是稠密重建），分多线程和单线程，主要是通过事件队列实现整个 working 流程。
	// 事件队列，多线程
	const int nOptimize(OPTDENSE::nOptimize);
	if (OPTDENSE::nEstimationGeometricIters && data.nFusionMode >= 0)
		OPTDENSE::nOptimize = 0;
	data.idxImage = 0;  // 要处理的当前帧的ID
	ASSERT(data.events.IsEmpty());
	data.events.AddEvent(new EVTProcessImage(0));  // 将图像处理加入事件队列中（线程启动之后，最先处理的就是第0帧，即ID为0的图像帧），以方便线程调用
	// start working threads
	// 启动工作线程
	data.progress = new Util::Progress("Estimated depth-maps", data.images.GetSize());
	GET_LOGCONSOLE().Pause();
	if (nMaxThreads > 1) {
		// multi-thread execution
		// ??? 为什么depth计算只用两个线程
		// ??? 可能是因为当前帧计算的depth需要传播给邻域帧做初始化，帧之间是有关联性的。
		cList<SEACAVE::Thread> threads(2);
		FOREACHPTR(pThread, threads)
			pThread->start(DenseReconstructionEstimateTmp, (void*)&data);
		FOREACHPTR(pThread, threads)
			pThread->join();
	} else {
		// single-thread execution
		// 单线程计算，如果想调试可以设线程数为1使用该函数
		DenseReconstructionEstimate((void*)&data);
	}
	GET_LOGCONSOLE().Play();
	if (!data.events.IsEmpty())
		return false;
	data.progress.Release();

	if (data.nFusionMode >= 0) {
		#ifdef _USE_CUDA
		// initialize CUDA
		if (data.depthMaps.pmCUDA && OPTDENSE::nEstimationGeometricIters) {
			data.depthMaps.pmCUDA->Release();
			data.depthMaps.pmCUDA->Init(true);
		}
		#endif // _USE_CUDA
		while (++data.nEstimationGeometricIter < (int)OPTDENSE::nEstimationGeometricIters) {
			// initialize the queue of images to be geometric processed
			if (data.nEstimationGeometricIter+1 == (int)OPTDENSE::nEstimationGeometricIters)
				OPTDENSE::nOptimize = nOptimize;
			data.idxImage = 0;
			ASSERT(data.events.IsEmpty());
			data.events.AddEvent(new EVTProcessImage(0));
			// start working threads
			data.progress = new Util::Progress("Geometric-consistent estimated depth-maps", data.images.GetSize());
			GET_LOGCONSOLE().Pause();
			if (nMaxThreads > 1) {
				// multi-thread execution
				cList<SEACAVE::Thread> threads(2);
				FOREACHPTR(pThread, threads)
					pThread->start(DenseReconstructionEstimateTmp, (void*)&data);
				FOREACHPTR(pThread, threads)
					pThread->join();
			} else {
				// single-thread execution
				DenseReconstructionEstimate((void*)&data);
			}
			GET_LOGCONSOLE().Play();
			if (!data.events.IsEmpty())
				return false;
			data.progress.Release();
			// replace raw depth-maps with the geometric-consistent ones
			for (IIndex idx: data.images) {
				const DepthData& depthData(data.depthMaps.arrDepthData[idx]);
				if (!depthData.IsValid())
					continue;
				const String rawName(ComposeDepthFilePath(depthData.GetView().GetID(), "dmap"));
				File::deleteFile(rawName);
				File::renameFile(ComposeDepthFilePath(depthData.GetView().GetID(), "geo.dmap"), rawName);
			}
		}
		data.nEstimationGeometricIter = -1;
	}

	// Step 4 深度图优化：主要是对depth进行滤波（即稠密重建滤波，帧间滤波）去噪填充一些孔洞
	// 帧间滤波，类似于SGM中的帧间一致性检查，不过帧间一致性检查只用了左右目的图像帧，即基于左图计算右图的视差，再基于右图计算左图的视差
	// 要求对于同一个像素在这两张视差图上的视差值应该要一致（即左图上的点的视差值与其投影到右图上的位置的视差值需要一致，反之亦然），
	// 而此处的帧间滤波则涉及到多个邻域帧，但是原理也是考虑当前帧上的像素在邻域帧上的投影位置的深度信息是否一致，
	// 由于涉及到多个邻域帧，因此对应的一致性检查方式也需要进行调整，例如当多个领域帧中大部分投影位置的深度信息都一致时可以认为该位置的深度估计是准确的，
	// 并且在当前帧和相邻帧上剔除掉深度估计不够准确的像素点
	// 具体实现参看代码。
	if ((OPTDENSE::nOptimize & OPTDENSE::ADJUST_FILTER) != 0) {
		// initialize the queue of depth-maps to be filtered
		data.sem.Clear();
		data.idxImage = data.images.GetSize();
		ASSERT(data.events.IsEmpty());
		FOREACH(i, data.images)
			data.events.AddEvent(new EVTFilterDepthMap(i));
		// start working threads
		// 线程启动
		data.progress = new Util::Progress("Filtered depth-maps", data.images.GetSize());
		GET_LOGCONSOLE().Pause();
		if (nMaxThreads > 1) {
			// multi-thread execution
			// 多线程
			cList<SEACAVE::Thread> threads(MINF(nMaxThreads, (unsigned)data.images.GetSize()));
			FOREACHPTR(pThread, threads)
				pThread->start(DenseReconstructionFilterTmp, (void*)&data);
			FOREACHPTR(pThread, threads)
				pThread->join();
		} else {
			// single-thread execution
			// 单线程
			DenseReconstructionFilter((void*)&data);
		}
		GET_LOGCONSOLE().Play();
		if (!data.events.IsEmpty())
			return false;
		data.progress.Release();
	}
	return true;
} // ComputeDepthMaps
/*----------------------------------------------------------------*/

void* DenseReconstructionEstimateTmp(void* arg) {
	const DenseDepthMapData& dataThreads = *((const DenseDepthMapData*)arg);
	dataThreads.scene.DenseReconstructionEstimate(arg);
	return NULL;
}

// initialize the dense reconstruction with the sparse point cloud
/**
 * @brief 深度图计算的主流程各个环节线程启动，包括depth初始化：主要是通过稀疏点进行插值优化 depth计算 depth优化
 * 
 * @param[in/out] pData 输入输出数据，所有深度计算相关数据
 */
void Scene::DenseReconstructionEstimate(void* pData)
{
	DenseDepthMapData& data = *((DenseDepthMapData*)pData);
	while (true) {  // 计算所有图像帧的深度图
		CAutoPtr<Event> evt(data.events.GetEvent());
		switch (evt->GetID()) {
		// Step 3_1 depth data 初始化
		case EVT_PROCESSIMAGE: {
			const EVTProcessImage& evtImage = *((EVTProcessImage*)(Event*)evt);
			if (evtImage.idxImage >= data.images.size()) {
				if (nMaxThreads > 1) {
					// close working threads
					// idxImage超过图像总数，说明所有计算已经完成，线程关闭
					data.events.AddEvent(new EVTClose);
				}
				return;
			}
			// select views to reconstruct the depth-map for this image
			// 为当前帧选择邻域帧（也就是前面所说的目标帧，即最佳的邻域帧）来计算深度图
			const IIndex idx = data.images[evtImage.idxImage];
			DepthData& depthData(data.depthMaps.arrDepthData[idx]);
			const bool depthmapComputed(data.nFusionMode < 0 || (data.nFusionMode >= 0 && data.nEstimationGeometricIter < 0 && File::access(ComposeDepthFilePath(data.scene.images[idx].ID, "dmap"))));
			// initialize images pair: reference image and the best neighbor view
			ASSERT(data.neighborsMap.IsEmpty() || data.neighborsMap[evtImage.idxImage] != NO_ID);
			//初始化用来计算深度图的图像对，如果最佳邻为空，则从neighbors中根据score选取不超过nNumViews neighbor views，若不为空就选用当前帧对应的最佳邻域帧（在邻域帧选择中已经获取）
			if (!data.depthMaps.InitViews(depthData, data.neighborsMap.IsEmpty()?NO_ID:data.neighborsMap[evtImage.idxImage], OPTDENSE::nNumViews, !depthmapComputed, depthmapComputed ? -1 : (data.nEstimationGeometricIter >= 0 ? 1 : 0))) {
				// process next image
				// 如果当前帧没有找到邻域，则无法计算深度，直接跳过处理下一帧图像（safeInc 每次调用都会对data.idxImage加1）
				data.events.AddEvent(new EVTProcessImage((IIndex)Thread::safeInc(data.idxImage)));
				break;
			}
			// try to load already compute depth-map for this image
			// 尝试加载当前帧的深度图，如果有就不用再计算，直接进入优化环节
			if (depthmapComputed && data.nFusionMode >= 0) {
				if (OPTDENSE::nOptimize & OPTDENSE::OPTIMIZE) {
					if (!depthData.Load(ComposeDepthFilePath(depthData.GetView().GetID(), "dmap"))) {
						VERBOSE("error: invalid depth-map '%s'", ComposeDepthFilePath(depthData.GetView().GetID(), "dmap").c_str());
						exit(EXIT_FAILURE);
					}
					// optimize depth-map
					// 优化深度图
					data.events.AddEventFirst(new EVTOptimizeDepthMap(evtImage.idxImage));
				}
				// process next image
				// 如果当前深度不需要优化，则开始处理下一帧图像
				data.events.AddEvent(new EVTProcessImage((uint32_t)Thread::safeInc(data.idxImage)));
			} else {
				// estimate depth-map
				// 计算深度图
				data.events.AddEventFirst(new EVTEstimateDepthMap(evtImage.idxImage));  // 将深度图计算加入到事件队列中
			}
			break; }
		// Step 3_2 depth 计算:用两种算法实现,一种是基于patchMatch计算深度图实现（主要参考Accurate Multiple View 3D Reconstruction Using Patch-Based Stereo for Large-Scale Scenes）
		// 一种是基于SGM计算视差图实现（参考Accurate and Efficient Stereo Processing by Semi-Global Matching and Mutual Information）
		case EVT_ESTIMATEDEPTHMAP: {
			const EVTEstimateDepthMap& evtImage = *((EVTEstimateDepthMap*)(Event*)evt);
			// request next image initialization to be performed while computing this depth-map
			// 计算当前帧深度时，下一帧图像初始化同时在做，不知道这是不是使用两个线程的原因之一。
			data.events.AddEvent(new EVTProcessImage((uint32_t)Thread::safeInc(data.idxImage)));
			// extract depth map
			// 提取深度
			data.sem.Wait();
			if (data.nFusionMode >= 0) {
				// extract depth-map using Patch-Match algorithm
				// Step 3_2_1 patch match算法
				data.depthMaps.EstimateDepthMap(data.images[evtImage.idxImage], data.nEstimationGeometricIter);
			} else {
				// extract disparity-maps using SGM algorithm
				// Step 3_2_2 SGM算法
				if (data.nFusionMode == -1) {  // 只进行深度图计算
					data.sgm.Match(*this, data.images[evtImage.idxImage], OPTDENSE::nNumViews);
				} else {  // 除了深度图计算外还会进行融合
					// fuse existing disparity-maps
					// 融合现存的视差图
					const IIndex idx(data.images[evtImage.idxImage]);
					DepthData& depthData(data.depthMaps.arrDepthData[idx]);
					data.sgm.Fuse(*this, data.images[evtImage.idxImage], OPTDENSE::nNumViews, 2, depthData.depthMap, depthData.confMap);
					// 计算法线，一般用不到为减少内存消耗尽量不算
					if (OPTDENSE::nEstimateNormals == 2)
						EstimateNormalMap(depthData.images.front().camera.K, depthData.depthMap, depthData.normalMap);
					depthData.dMin = ZEROTOLERANCE<float>(); depthData.dMax = FLT_MAX;
				}
			}
			data.sem.Signal();
			if (OPTDENSE::nOptimize & OPTDENSE::OPTIMIZE) {
				// optimize depth-map
				// 优化深度图
				data.events.AddEventFirst(new EVTOptimizeDepthMap(evtImage.idxImage));
			} else {
				// save depth-map
				// 保存深度图
				data.events.AddEventFirst(new EVTSaveDepthMap(evtImage.idxImage));
			}
			break; }
		// Step 3_3 depth 优化 移除小的连通域segment（这一步可能会产生一些孔洞，此处的小连通域指的是孤立的小块或者是一块较大的连通域中的部分突变的小块） 填充小洞gap（单张深度图处理）
		// 单帧滤波
		case EVT_OPTIMIZEDEPTHMAP: {
			const EVTOptimizeDepthMap& evtImage = *((EVTOptimizeDepthMap*)(Event*)evt);
			const IIndex idx = data.images[evtImage.idxImage];
			DepthData& depthData(data.depthMaps.arrDepthData[idx]);
			#if TD_VERBOSE != TD_VERBOSE_OFF
			// save depth map as image
			// 将深度图保存为图像
			if (g_nVerbosityLevel > 3)
				ExportDepthMap(ComposeDepthFilePath(depthData.GetView().GetID(), "raw.png"), depthData.depthMap);
			#endif
			// apply filters
			// 滤波，将一些离散的深度值滤波剔除
			if (OPTDENSE::nOptimize & (OPTDENSE::REMOVE_SPECKLES)) {
				TD_TIMER_START();
				if (data.depthMaps.RemoveSmallSegments(depthData)) {
					DEBUG_ULTIMATE("Depth-map %3u filtered: remove small segments (%s)", depthData.GetView().GetID(), TD_TIMER_GET_FMT().c_str());
				}
			}
			// 滤波，将深度图孔洞填充
			if (OPTDENSE::nOptimize & (OPTDENSE::FILL_GAPS)) {
				TD_TIMER_START();
				if (data.depthMaps.GapInterpolation(depthData)) {
					DEBUG_ULTIMATE("Depth-map %3u filtered: gap interpolation (%s)", depthData.GetView().GetID(), TD_TIMER_GET_FMT().c_str());
				}
			}
			// save depth-map
			// 保存深度图
			data.events.AddEventFirst(new EVTSaveDepthMap(evtImage.idxImage));
			break; }
		// Step 3_4 depth 保存 一般用不到
		case EVT_SAVEDEPTHMAP: {
			const EVTSaveDepthMap& evtImage = *((EVTSaveDepthMap*)(Event*)evt);
			const IIndex idx = data.images[evtImage.idxImage];
			DepthData& depthData(data.depthMaps.arrDepthData[idx]);
			#if TD_VERBOSE != TD_VERBOSE_OFF
			// save depth map as image
			// 将深度图保存为图像
			if (g_nVerbosityLevel > 2) {
				ExportDepthMap(ComposeDepthFilePath(depthData.GetView().GetID(), "png"), depthData.depthMap);
				ExportConfidenceMap(ComposeDepthFilePath(depthData.GetView().GetID(), "conf.png"), depthData.confMap);
				ExportPointCloud(ComposeDepthFilePath(depthData.GetView().GetID(), "ply"), *depthData.images.First().pImageData, depthData.depthMap, depthData.normalMap);
				if (g_nVerbosityLevel > 4) {
					ExportNormalMap(ComposeDepthFilePath(depthData.GetView().GetID(), "normal.png"), depthData.normalMap);
					depthData.confMap.Save(ComposeDepthFilePath(depthData.GetView().GetID(), "conf.pfm"));
				}
			}
			#endif
			// save compute depth-map for this image
			// 保存计算的深度图，dmap里面包含了置信度、法向量以及深度值这三个量，可以根据需要自定义保存的格式和内容，例如可以将深度信息保存为整型，而非此处的浮点型
			if (!depthData.depthMap.empty())
				depthData.Save(ComposeDepthFilePath(depthData.GetView().GetID(), data.nEstimationGeometricIter < 0 ? "dmap" : "geo.dmap"));
			depthData.ReleaseImages();
			depthData.Release();
			data.progress->operator++();
			break; }

		case EVT_CLOSE: {
			return; }

		default:
			ASSERT("Should not happen!" == NULL);
		}
	}
} // DenseReconstructionEstimate
/*----------------------------------------------------------------*/

void* DenseReconstructionFilterTmp(void* arg) {
	DenseDepthMapData& dataThreads = *((DenseDepthMapData*)arg);
	dataThreads.scene.DenseReconstructionFilter(arg);
	return NULL;
}

// filter estimated depth-maps
// 深度图滤波
void Scene::DenseReconstructionFilter(void* pData)
{
	DenseDepthMapData& data = *((DenseDepthMapData*)pData);
	CAutoPtr<Event> evt;
	while ((evt=data.events.GetEvent(0)) != NULL) {
		switch (evt->GetID()) {
		case EVT_FILTERDEPTHMAP: {
			const EVTFilterDepthMap& evtImage = *((EVTFilterDepthMap*)(Event*)evt);
			const IIndex idx = data.images[evtImage.idxImage];
			DepthData& depthData(data.depthMaps.arrDepthData[idx]);
			if (!depthData.IsValid()) {
				data.SignalCompleteDepthmapFilter();
				break;
			}
			// make sure all depth-maps are loaded
			// depth加载，确保所有的深度图被加载，也就是准备好需要进行滤波的图像帧以及它们对应的邻域帧
			depthData.IncRef(ComposeDepthFilePath(depthData.GetView().GetID(), "dmap"));
			const unsigned numMaxNeighbors(8);
			IIndexArr idxNeighbors(0, depthData.neighbors.GetSize());
			FOREACH(n, depthData.neighbors) {
				const IIndex idxView = depthData.neighbors[n].ID;
				DepthData& depthDataPair = data.depthMaps.arrDepthData[idxView];
				if (!depthDataPair.IsValid())
					continue;
				if (depthDataPair.IncRef(ComposeDepthFilePath(depthDataPair.GetView().GetID(), "dmap")) == 0) {
					// signal error and terminate
					data.events.AddEventFirst(new EVTFail);
					return;
				}
				idxNeighbors.Insert(n);
				if (idxNeighbors.GetSize() == numMaxNeighbors)
					break;
			}
			// filter the depth-map for this image
			// depth滤波
			if (data.depthMaps.FilterDepthMap(depthData, idxNeighbors, OPTDENSE::bFilterAdjust)) {
				// load the filtered maps after all depth-maps were filtered
				data.events.AddEvent(new EVTAdjustDepthMap(evtImage.idxImage));
			}
			// unload referenced depth-maps
			FOREACHPTR(pIdxNeighbor, idxNeighbors) {
				const IIndex idxView = depthData.neighbors[*pIdxNeighbor].ID;
				DepthData& depthDataPair = data.depthMaps.arrDepthData[idxView];
				depthDataPair.DecRef();
			}
			depthData.DecRef();
			data.SignalCompleteDepthmapFilter();
			break; }
		// 一些文件操作，例如数据加载、保存等
		case EVT_ADJUSTDEPTHMAP: {
			const EVTAdjustDepthMap& evtImage = *((EVTAdjustDepthMap*)(Event*)evt);
			const IIndex idx = data.images[evtImage.idxImage];
			DepthData& depthData(data.depthMaps.arrDepthData[idx]);
			ASSERT(depthData.IsValid());
			data.sem.Wait();
			// load filtered maps
			if (depthData.IncRef(ComposeDepthFilePath(depthData.GetView().GetID(), "dmap")) == 0 ||
				!LoadDepthMap(ComposeDepthFilePath(depthData.GetView().GetID(), "filtered.dmap"), depthData.depthMap) ||
				!LoadConfidenceMap(ComposeDepthFilePath(depthData.GetView().GetID(), "filtered.cmap"), depthData.confMap))
			{
				// signal error and terminate
				data.events.AddEventFirst(new EVTFail);
				return;
			}
			ASSERT(depthData.GetRef() == 1);
			File::deleteFile(ComposeDepthFilePath(depthData.GetView().GetID(), "filtered.dmap").c_str());
			File::deleteFile(ComposeDepthFilePath(depthData.GetView().GetID(), "filtered.cmap").c_str());
			#if TD_VERBOSE != TD_VERBOSE_OFF
			// save depth map as image
			if (g_nVerbosityLevel > 2) {
				ExportDepthMap(ComposeDepthFilePath(depthData.GetView().GetID(), "filtered.png"), depthData.depthMap);
				ExportPointCloud(ComposeDepthFilePath(depthData.GetView().GetID(), "filtered.ply"), *depthData.images.First().pImageData, depthData.depthMap, depthData.normalMap);
			}
			#endif
			// save filtered depth-map for this image
			// 保存滤波的图像
			depthData.Save(ComposeDepthFilePath(depthData.GetView().GetID(), "dmap"));
			depthData.DecRef();
			data.progress->operator++();
			break; }

		case EVT_FAIL: {
			data.events.AddEventFirst(new EVTFail);
			return; }

		default:
			ASSERT("Should not happen!" == NULL);
		}
	}
} // DenseReconstructionFilter
/*----------------------------------------------------------------*/

// filter point-cloud based on camera-point visibility intersections
// 点云滤波，主要是利用可见性进行滤波 一般用不到不再注释
void Scene::PointCloudFilter(int thRemove)
{
	TD_TIMER_STARTD();

	typedef TOctree<PointCloud::PointArr,PointCloud::Point::Type,3,uint32_t> Octree;
	struct Collector {
		typedef Octree::IDX_TYPE IDX;
		typedef PointCloud::Point::Type Real;
		typedef TCone<Real,3> Cone;
		typedef TSphere<Real,3> Sphere;
		typedef TConeIntersect<Real,3> ConeIntersect;

		Cone cone;
		const ConeIntersect coneIntersect;
		const PointCloud& pointcloud;
		IntArr& visibility;
		PointCloud::Index idxPoint;
		Real distance;
		int weight;
		#ifdef DENSE_USE_OPENMP
		uint8_t pcs[sizeof(CriticalSection)];
		#endif

		Collector(const Cone::RAY& ray, Real angle, const PointCloud& _pointcloud, IntArr& _visibility)
			: cone(ray, angle), coneIntersect(cone), pointcloud(_pointcloud), visibility(_visibility)
		#ifdef DENSE_USE_OPENMP
		{ new(pcs) CriticalSection; }
		~Collector() { reinterpret_cast<CriticalSection*>(pcs)->~CriticalSection(); }
		inline CriticalSection& GetCS() { return *reinterpret_cast<CriticalSection*>(pcs); }
		#else
		{}
		#endif
		inline void Init(PointCloud::Index _idxPoint, const PointCloud::Point& X, int _weight) {
			const Real thMaxDepth(1.02f);
			idxPoint =_idxPoint;
			const PointCloud::Point::EVec D((PointCloud::Point::EVec&)X-cone.ray.m_pOrig);
			distance = D.norm();
			cone.ray.m_vDir = D/distance;
			cone.maxHeight = MaxDepthDifference(distance, thMaxDepth);
			weight = _weight;
		}
		inline bool Intersects(const Octree::POINT_TYPE& center, Octree::Type radius) const {
			return coneIntersect(Sphere(center, radius*Real(SQRT_3)));
		}
		inline void operator() (const IDX* idices, IDX size) {
			const Real thSimilar(0.01f);
			Real dist;
			FOREACHRAWPTR(pIdx, idices, size) {
				const PointCloud::Index idx(*pIdx);
				if (coneIntersect.Classify(pointcloud.points[idx], dist) == VISIBLE && !IsDepthSimilar(distance, dist, thSimilar)) {
					if (dist > distance)
						visibility[idx] += pointcloud.pointViews[idx].size();
					else
						visibility[idx] -= weight;
				}
			}
		}
	};
	typedef CLISTDEF2(Collector) Collectors;

	// create octree to speed-up search
	Octree octree(pointcloud.points, [](Octree::IDX_TYPE size, Octree::Type /*radius*/) {
		return size > 128;
	});
	IntArr visibility(pointcloud.GetSize()); visibility.Memset(0);
	Collectors collectors; collectors.reserve(images.size());
	FOREACH(idxView, images) {
		const Image& image = images[idxView];
		const Ray3f ray(Cast<float>(image.camera.C), Cast<float>(image.camera.Direction()));
		const float angle(float(image.ComputeFOV(0)/image.width));
		collectors.emplace_back(ray, angle, pointcloud, visibility);
	}

	// run all camera-point visibility intersections
	Util::Progress progress(_T("Point visibility checks"), pointcloud.GetSize());
	#ifdef DENSE_USE_OPENMP
	#pragma omp parallel for //schedule(dynamic)
	for (int64_t i=0; i<(int64_t)pointcloud.GetSize(); ++i) {
		const PointCloud::Index idxPoint((PointCloud::Index)i);
	#else
	FOREACH(idxPoint, pointcloud.points) {
	#endif
		const PointCloud::Point& X = pointcloud.points[idxPoint];
		const PointCloud::ViewArr& views = pointcloud.pointViews[idxPoint];
		for (PointCloud::View idxView: views) {
			Collector& collector = collectors[idxView];
			#ifdef DENSE_USE_OPENMP
			Lock l(collector.GetCS());
			#endif
			collector.Init(idxPoint, X, (int)views.size());
			octree.Collect(collector, collector);
		}
		++progress;
	}
	progress.close();

	#if TD_VERBOSE != TD_VERBOSE_OFF
	if (g_nVerbosityLevel > 2) {
		// print visibility stats
		UnsignedArr counts(0, 64);
		for (int views: visibility) {
			if (views > 0)
				continue;
			while (counts.size() <= IDX(-views))
				counts.push_back(0);
			++counts[-views];
		}
		String msg;
		msg.reserve(64*counts.size());
		FOREACH(c, counts)
			if (counts[c])
				msg += String::FormatString("\n\t% 3u - % 9u", c, counts[c]);
		VERBOSE("Visibility lengths (%u points):%s", pointcloud.GetSize(), msg.c_str());
		// save outlier points
		PointCloud pc;
		RFOREACH(idxPoint, pointcloud.points) {
			if (visibility[idxPoint] <= thRemove) {
				pc.points.push_back(pointcloud.points[idxPoint]);
				pc.colors.push_back(pointcloud.colors[idxPoint]);
			}
		}
		pc.Save(MAKE_PATH("scene_dense_outliers.ply"));
	}
	#endif

	// filter points
	const size_t numInitPoints(pointcloud.GetSize());
	RFOREACH(idxPoint, pointcloud.points) {
		if (visibility[idxPoint] <= thRemove)
			pointcloud.RemovePoint(idxPoint);
	}

	DEBUG_EXTRA("Point-cloud filtered: %u/%u points (%d%%%%) (%s)", pointcloud.points.size(), numInitPoints, ROUND2INT((100.f*pointcloud.points.GetSize())/numInitPoints), TD_TIMER_GET_FMT().c_str());
} // PointCloudFilter
/*----------------------------------------------------------------*/
