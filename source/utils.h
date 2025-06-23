#pragma once
#include "libs.h"

// DESCRIPTORS COMPUTING
// Hàm tính toán histogram cho một ảnh trong không gian màu BGR
Mat myCalcHistogram(const string& imagePath);

// Hàm lấy descriptors sử dụng sift
Mat computeSIFTDescriptors(const Ptr<SIFT>& sift, const string& imagePath, const Size& size);

// Hàm lấy descriptors sử dụng orb
Mat computeORBDescriptors(const Ptr<ORB>& orb, const string& imagePath, const Size& size);

// UTILS
// Hàm phân tách thành một vector chuỗi dựa trên dấu phân cách
vector<string> split(const string& s, char delimiter);

// Hàm chuyển đổi ma trận histogram thành chuỗi (lưu ý với hàm myCalcHistogram cho ra [1 x 768] [cols x rows])
// Tương ứng với 768 cột và 1 hàng
string histogramToString(const Mat& hist);

// Chuyển đổi vector<pair<string, double>> thành vector<Mat>
vector<Mat> myCvt_pair_to_Mat(const vector<pair<string, double>>& imagePaths);

// Hiển thị hình ảnh của một vector ảnh nhưng trong cùng 1 cửa sổ
void showImages(const vector<Mat>& images, const string& nameWindow);

// Hàm lấy tên file từ URL
string getFileNameFromFilePath(const string& filePath);

// Hàm đọc CSV và trả về một unordered_map với key là picture name và value là label
unordered_map<string, string> getNameLabelFromCSV(const string& csvFilePath);
