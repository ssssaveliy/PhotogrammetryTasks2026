#include <iostream>
#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "flann_matcher.h"
#include "flann_factory.h"

phg::FlannMatcher::FlannMatcher()
{
    const int ntrees = 4;
    const int nchecks = 32;

    index_params = flannKdTreeIndexParams(ntrees);
    search_params = flannKsTreeSearchParams(nchecks);
}

void phg::FlannMatcher::train(const cv::Mat &train_desc)
{
    if (train_desc.rows < 2) {
        throw std::runtime_error("FlannMatcher::train : needed at least 2 train descriptors");
    }

    if (train_desc.type() != CV_32F) {
        train_desc.convertTo(train_desc_, CV_32F);
    } else {
        train_desc_ = train_desc;
    }

    flann_index = flannKdTreeIndex(train_desc_, index_params);
}

void phg::FlannMatcher::knnMatch(const cv::Mat &query_desc,
                                 std::vector<std::vector<cv::DMatch>> &matches,
                                 int k) const
{
    if (!flann_index) {
        throw std::runtime_error("FlannMatcher::knnMatch : matcher is not trained");
    }
    if (!search_params) {
        throw std::runtime_error("FlannMatcher::knnMatch : invalid search_params");
    }
    if (k <= 0) {
        throw std::runtime_error("FlannMatcher::knnMatch : invalid k");
    }

    cv::Mat query32f;
    if (query_desc.type() != CV_32F) {
        query_desc.convertTo(query32f, CV_32F);
    } else {
        query32f = query_desc;
    }

    const int n_query = query32f.rows;
    matches.assign(n_query, {});

    cv::Mat indices(n_query, k, CV_32SC1);
    cv::Mat dists2(n_query, k, CV_32FC1);

    flann_index->knnSearch(query32f, indices, dists2, k, *search_params);

    for (int qi = 0; qi < n_query; ++qi) {
        std::vector<cv::DMatch> &dst = matches[qi];
        dst.resize(k);

        for (int j = 0; j < k; ++j) {
            cv::DMatch m;
            m.queryIdx = qi;
            m.trainIdx = indices.at<int>(qi, j);
            m.imgIdx = 0;

            float d2 = dists2.at<float>(qi, j);
            m.distance = (d2 > 0.0f) ? std::sqrt(d2) : 0.0f;

            dst[j] = m;
        }

        std::sort(dst.begin(), dst.end(),
                  [](const cv::DMatch &a, const cv::DMatch &b) { return a.distance < b.distance; });
    }
}
