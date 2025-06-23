#pragma once
#include "libs.h"

// Interface cho SIFT và ORB, chỉ gồm các phương thức thuần ảo
class LocalFeature {
public:
    virtual vector<pair<string, double>> searchLocal(const string& queryImagePath, int n, const Size& size) = 0;
    // MAP
    // Tổng hợp n = 3, 5, 11, 21 ghi lại MAP vào một file CSV
    // Có các cột là: URL,Label,N=3,N=5,N=11,N=21
    virtual void computeAndWriteMAP(const string& queryImagePath, const Size& size, ofstream& outputFile) = 0;
    virtual void processWriteMap(const string& MAPcsvPath, const Size& size, int maxRows) = 0;

    // Áp dụng kmeans vào để thực hiện truy vấn
    // Hàm tạo từ điển
    virtual Mat createCodebook(int clusterCount) = 0;
    // Gán đặc trưng vào cụm gần nhất
    virtual Mat mapFeaturesToCodewords(const Mat& descriptors, const Mat& dictionary) = 0;
    // Tạo biểu diễn histogram cho mỗi ảnh
    virtual vector<Mat> createHistograms(const Mat& dictionary) = 0;
    // Truy vấn sử dụng Kmeans (khởi tạo random)
    virtual vector<pair<string, double>> searchLocalKmeans(const string& queryImagePath, int n, const Size& size,
        const Mat& codebook, const vector<Mat>& histograms) = 0;
    // Ghi Map cho SiftKmeans
    virtual void computeAndWriteMAP_Kmeans(const string& queryImagePath, const Size& size, ofstream& outputFile,
        const Mat& codebook, const vector<Mat>& histograms) = 0;
    // Ghi MAP cho tất cả ảnh trong records sử dụng SiftKmeans
    virtual void processWriteMap_Kmeans(const string& MAPcsvPath, const Size& size, int maxRows,
        const Mat& codebook, const vector<Mat>& histograms) = 0;
};

class SIFTFeature : public LocalFeature {
private:
    vector<myRecord> records;
public:
    SIFTFeature(const vector<myRecord>& records) : records(records) {};

    //Normal
    vector<pair<string, double>> searchLocal(const string& queryImagePath, int n, const Size& size) override;
    void computeAndWriteMAP(const string& queryImagePath, const Size& size, ofstream& outputFile) override;
    void processWriteMap(const string& MAPcsvPath, const Size& size, int maxRows) override;

    //Kmeans
    Mat createCodebook(int clusterCount);
    Mat mapFeaturesToCodewords(const Mat& descriptors, const Mat& dictionary);
    vector<Mat> createHistograms(const Mat& dictionary);
    vector<pair<string, double>> searchLocalKmeans(const string& queryImagePath, int n, const Size& size,
        const Mat& codebook, const vector<Mat>& histograms);
    void computeAndWriteMAP_Kmeans(const string& queryImagePath, const Size& size, ofstream& outputFile,
        const Mat& codebook, const vector<Mat>& histograms);
    void processWriteMap_Kmeans(const string& MAPcsvPath, const Size& size, int maxRows,
        const Mat& codebook, const vector<Mat>& histograms);
};

class ORBFeature : public LocalFeature {
private:
    vector<myRecord> records;
public:
    ORBFeature(const vector<myRecord>& records) : records(records) {};

    //Normal
    vector<pair<string, double>> searchLocal(const string& queryImagePath, int n, const Size& size) override;
    void computeAndWriteMAP(const string& queryImagePath, const Size& size, ofstream& outputFile) override;
    void processWriteMap(const string& MAPcsvPath, const Size& size, int maxRows) override;

    //Kmeans
    Mat createCodebook(int clusterCount);
    Mat mapFeaturesToCodewords(const Mat& descriptors, const Mat& dictionary);
    vector<Mat> createHistograms(const Mat& dictionary);
    vector<pair<string, double>> searchLocalKmeans(const string& queryImagePath, int n, const Size& size,
        const Mat& codebook, const vector<Mat>& histograms);
    void computeAndWriteMAP_Kmeans(const string& queryImagePath, const Size& size, ofstream& outputFile,
        const Mat& codebook, const vector<Mat>& histograms);
    void processWriteMap_Kmeans(const string& MAPcsvPath, const Size& size, int maxRows,
        const Mat& codebook, const vector<Mat>& histograms);
};

// Factory cho local feature, trong hàm main, sử dụng factory để thực hiện truy vấn
class LocalFeatureFactory {
public:
    LocalFeature* getFeature(const string& featureType, const vector<myRecord>& records) {
        if (featureType == "sift") {
            return new SIFTFeature(records);
        }
        else if (featureType == "orb") {
            return new ORBFeature(records);
        }
        return nullptr;
    }
};