#pragma once
#include "libs.h"

// Thực hiện làm cơ sở dữ liệu bao gồm tạo ra cơ sở dữ liệu từ ảnh và load cơ sở dữ liệu lên
class myDatabase {
private:
	string imageFolderPath;
public:
	// Khởi tạo
	myDatabase(const string& imageFolderPath) : imageFolderPath(imageFolderPath) {};

	// HISTOGRAM
	// Hàm tạo file CSV với cột name, label, url và histogram
	void createCsv_featuresHist(const string& inputCsvFilePath, const string& outputCsvFilePath);
	// Hàm đọc dữ liệu từ file CSV vào vector myRecord
	vector<myRecord> makeRecordsForHist(const string& filePath);

	// SIFT & ORB
	// Lưu vector descriptors vào file dat
	void saveDescriptorsToDat(const string& descriptorsFilePath, const vector<Mat>& descriptors);
	// Thực hiện tính sau đó lưu vector descriptors vào file dat, các đường dẫn đến ảnh vào file txt
	void create_descriptorsAndURL(const string& descriptorsFilePath, const string& URLfilePath, const Size& size, const string& queryType);
	// Đọc file dat lấy ra vector ma trận descriptors và url để lấy đường dẫn
	vector<pair<string, Mat>> read_featureAndURL(const string& descriptorsFilePath, const string& URLfilePath);
	// Hàm tạo records từ vector cặp URL và descriptors
	vector<myRecord> makeRecords(vector<pair<string, Mat>> URL_descriptorsList, const string& CSVFilePath);

	// KMEANS
	// Lưu ma trận vào file dat
	void saveMatToDat(const string& centersFilePath, const Mat& centers);
	// Đọc ma trận từ file dat
	Mat readMatFromDat(const string& centersFilePath);
};
