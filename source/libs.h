#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/ml.hpp>

#include <vector>
#include <unordered_map>

#include <algorithm>
#include <cmath>
#include <numeric>

#include <string>
#include <sstream>
#include <fstream>

#include <filesystem>
namespace fs = std::filesystem;

#include <chrono>
using namespace std::chrono;

using namespace cv;
using namespace std;

// Thêm header vào file này để sử dụng các hàm utilities
#include "utils.h"

struct myRecord {
    string Name;
    string Label;
    string URL;
    Mat descriptors;
};