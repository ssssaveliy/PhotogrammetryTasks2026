#include "descriptor_matcher.h"

#include "flann_factory.h"

#include <opencv2/flann/miniflann.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

void phg::DescriptorMatcher::filterMatchesRatioTest(const std::vector<std::vector<cv::DMatch>> &matches,
                                                    std::vector<cv::DMatch> &filtered_matches)
{
    const float ratio_thresh = 0.75f;

    filtered_matches.clear();
    filtered_matches.reserve(matches.size());

    for (const auto &m : matches) {
        if (m.size() < 2) {
            continue;
        }

        const cv::DMatch &m0 = m[0];
        const cv::DMatch &m1 = m[1];

        if (m0.queryIdx < 0 || m0.trainIdx < 0) {
            continue;
        }
        if (!std::isfinite(m0.distance) || !std::isfinite(m1.distance)) {
            continue;
        }
        if (m1.distance <= 0.0f) {
            continue;
        }

        if (m0.distance < ratio_thresh * m1.distance) {
            filtered_matches.push_back(m0);
        }
    }
}

void phg::DescriptorMatcher::filterMatchesClusters(const std::vector<cv::DMatch> &matches,
                                                   const std::vector<cv::KeyPoint> keypoints_query,
                                                   const std::vector<cv::KeyPoint> keypoints_train,
                                                   std::vector<cv::DMatch> &filtered_matches)
{
    filtered_matches.clear();

    const int total_neighbours = 5; 
    const int consistent_matches = 3;
    const float radius_limit_scale = 2.f;

    const int n_matches = (int)matches.size();
    if (n_matches < total_neighbours) {
        filtered_matches = matches;
        return;
    }

    cv::Mat points_query(n_matches, 2, CV_32FC1);
    cv::Mat points_train(n_matches, 2, CV_32FC1);
    for (int i = 0; i < n_matches; ++i) {
        const int qi = matches[i].queryIdx;
        const int ti = matches[i].trainIdx;

        if (qi < 0 || qi >= (int)keypoints_query.size() || ti < 0 || ti >= (int)keypoints_train.size()) {
            points_query.at<float>(i, 0) = 0.0f;
            points_query.at<float>(i, 1) = 0.0f;
            points_train.at<float>(i, 0) = 0.0f;
            points_train.at<float>(i, 1) = 0.0f;
            continue;
        }

        points_query.at<float>(i, 0) = keypoints_query[qi].pt.x;
        points_query.at<float>(i, 1) = keypoints_query[qi].pt.y;
        points_train.at<float>(i, 0) = keypoints_train[ti].pt.x;
        points_train.at<float>(i, 1) = keypoints_train[ti].pt.y;
    }

    const int ntrees = 1;
    const int nchecks = 32;
    std::shared_ptr<cv::flann::IndexParams> index_params = flannKdTreeIndexParams(ntrees);
    std::shared_ptr<cv::flann::SearchParams> search_params = flannKsTreeSearchParams(nchecks);

    std::shared_ptr<cv::flann::Index> index_query = flannKdTreeIndex(points_query, index_params);
    std::shared_ptr<cv::flann::Index> index_train = flannKdTreeIndex(points_train, index_params);

    cv::Mat indices_query(n_matches, total_neighbours, CV_32SC1);
    cv::Mat distances2_query(n_matches, total_neighbours, CV_32FC1);
    cv::Mat indices_train(n_matches, total_neighbours, CV_32SC1);
    cv::Mat distances2_train(n_matches, total_neighbours, CV_32FC1);

    index_query->knnSearch(points_query, indices_query, distances2_query, total_neighbours, *search_params);
    index_train->knnSearch(points_train, indices_train, distances2_train, total_neighbours, *search_params);

    float radius2_query = 0.0f;
    float radius2_train = 0.0f;
    {
        std::vector<float> max_d2_q(n_matches);
        std::vector<float> max_d2_t(n_matches);
        for (int i = 0; i < n_matches; ++i) {
            max_d2_q[i] = distances2_query.at<float>(i, total_neighbours - 1);
            max_d2_t[i] = distances2_train.at<float>(i, total_neighbours - 1);
        }

        const int median_pos = n_matches / 2;
        std::nth_element(max_d2_q.begin(), max_d2_q.begin() + median_pos, max_d2_q.end());
        std::nth_element(max_d2_t.begin(), max_d2_t.begin() + median_pos, max_d2_t.end());

        radius2_query = max_d2_q[median_pos] * radius_limit_scale * radius_limit_scale;
        radius2_train = max_d2_t[median_pos] * radius_limit_scale * radius_limit_scale;
        if (!(radius2_query > 0.0f)) radius2_query = std::numeric_limits<float>::infinity();
        if (!(radius2_train > 0.0f)) radius2_train = std::numeric_limits<float>::infinity();
    }

    filtered_matches.reserve(n_matches);

    for (int i = 0; i < n_matches; ++i) {
        std::array<int, (size_t)total_neighbours> neigh_q{};
        int n_q = 0;
        for (int k = 0; k < total_neighbours; ++k) {
            const float d2 = distances2_query.at<float>(i, k);
            if (d2 <= radius2_query) {
                neigh_q[(size_t)n_q++] = indices_query.at<int>(i, k);
            }
        }

        int inter = 0;
        for (int k = 0; k < total_neighbours; ++k) {
            const float d2 = distances2_train.at<float>(i, k);
            if (d2 <= radius2_train) {
                const int id = indices_train.at<int>(i, k);
                for (int j = 0; j < n_q; ++j) {
                    if (neigh_q[(size_t)j] == id) {
                        ++inter;
                        break;
                    }
                }
            }
        }

        if (inter >= consistent_matches) {
            filtered_matches.push_back(matches[i]);
        }
    }
}
