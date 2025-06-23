#pragma once
#include "libs.h"

class GlobalFeature {
public:
    virtual vector<pair<string, double>> searchGlobal(const string& queryImagePath, int n) = 0;
    virtual void computeAndWriteMAP(const string& queryImagePath, ofstream& outputFile) = 0;
    virtual void processWriteMap(const string& MAPcsvPath, int maxRows) = 0;
};

class HistogramFeature : public GlobalFeature {
private:
    vector<myRecord> records;
public:
    HistogramFeature(const vector<myRecord>& records) : records(records) {};

    vector<pair<string, double>> searchGlobal(const string& queryImagePath, int n) override;
    void computeAndWriteMAP(const string& queryImagePath, ofstream& outputFile) override;
    void processWriteMap(const string& MAPcsvPath, int maxRows) override;
};

// Factory cho Global feature, trong hàm main, sử dụng factory để thực hiện truy vấn
class GlobalFeatureFactory {
public:
    GlobalFeature* getFeature(const string& featureType, const vector<myRecord>& records) {
        if (featureType == "histogram") {
            return new HistogramFeature(records);
        }
        else if (featureType == "correlogram") {
            cout << "Not yet implement";
            return nullptr;
        }
        return nullptr;
    }
};

// Hàm để tính tổng của giá trị nhỏ nhất trong các bin tương ứng của hai histogram (độ đo)
double calcSimilarityScore_hist(const Mat& hist1, const Mat& hist2);